"""
mynsepulse swing engine — auth + health service.

Exists for one reason: Upstox access tokens die at 03:30 IST every day and
cannot be refreshed, so a human has to approve a fresh authorisation once
daily. This serves the two endpoints that flow needs, plus a health page
that tells you at a glance whether today's token is in place.

Deliberately minimal. No market data is served from here — the scanner
runs as a scheduled job and talks to Supabase directly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from upstox_client import (
    IST,
    TokenStore,
    UpstoxCredentials,
    authorization_url,
    exchange_auth_code,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("auth-server")

app = FastAPI(title="mynsepulse swing engine", docs_url=None, redoc_url=None)

STATE_PATH = Path("/data/oauth_state.json")
STATE_TTL = timedelta(minutes=10)

store = TokenStore()


def creds() -> UpstoxCredentials:
    try:
        return UpstoxCredentials(
            api_key=os.environ["UPSTOX_API_KEY"],
            api_secret=os.environ["UPSTOX_API_SECRET"],
            redirect_uri=os.environ["UPSTOX_REDIRECT_URI"],
        )
    except KeyError as missing:
        raise HTTPException(503, f"Server misconfigured: {missing} not set")


# ---------------------------------------------------------------------
# Internal auth — same lesson as the NSE Pulse endpoints. Never a
# publishable key, always a timing-safe compare, fail closed.
# ---------------------------------------------------------------------
def require_internal_key(request: Request) -> None:
    expected = os.environ.get("INTERNAL_API_KEY")
    if not expected:
        raise HTTPException(503, "INTERNAL_API_KEY not configured")
    provided = request.headers.get("x-internal-key", "")
    if not hmac.compare_digest(
        hashlib.sha256(provided.encode()).digest(),
        hashlib.sha256(expected.encode()).digest(),
    ):
        raise HTTPException(401, "Unauthorized")


# ---------------------------------------------------------------------
# CSRF state — single use, short lived
# ---------------------------------------------------------------------
def issue_state() -> str:
    state = secrets.token_urlsafe(32)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "state": state,
        "expires_at": (datetime.now(IST) + STATE_TTL).isoformat(),
    }))
    return state


def consume_state(provided: str) -> bool:
    """Single use: the state is destroyed whether or not it matched."""
    if not STATE_PATH.exists():
        return False
    try:
        row = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return False
    finally:
        STATE_PATH.unlink(missing_ok=True)

    if datetime.fromisoformat(row.get("expires_at", "1970-01-01T00:00:00+05:30")) < datetime.now(IST):
        log.warning("OAuth state expired")
        return False
    return hmac.compare_digest(row.get("state", ""), provided)


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/auth/upstox/login")
def login():
    """Tap this once a day. Redirects to Upstox for approval."""
    return RedirectResponse(authorization_url(creds(), state=issue_state()), status_code=302)


@app.get("/auth/upstox/callback", response_class=HTMLResponse)
def callback(code: str | None = Query(None), state: str | None = Query(None)):
    if not code or not state:
        return HTMLResponse(_page("Login failed", "Upstox did not return a code."), 400)

    if not consume_state(state):
        log.warning("OAuth callback with bad or expired state")
        return HTMLResponse(_page("Login failed", "State mismatch. Start again."), 400)

    try:
        token = exchange_auth_code(creds(), code)
    except Exception as exc:
        log.error("Token exchange failed: %s", exc)
        return HTMLResponse(_page("Login failed", "Token exchange rejected."), 502)

    store.write(token)
    expires = store.read().get("expires_at", "unknown")
    log.info("Upstox token stored, valid until %s", expires)
    return HTMLResponse(_page("Connected", f"Token valid until {expires}."))


# ---------------------------------------------------------------------
# Long-running jobs
#
# The backfill has to run in this process: it needs the Upstox token on
# the /data volume, and a Railway volume attaches to one service only.
# It runs on a background thread so the HTTP request returns immediately
# — a 20-minute request would be killed by the proxy long before the job
# finished. Progress is read back from the ingestion_runs table.
# ---------------------------------------------------------------------
_job_lock = threading.Lock()
_job_state: dict = {"name": None, "running": False, "started_at": None,
                    "finished_at": None, "error": None}


def _run_job(name: str, fn) -> None:
    try:
        fn()
        with _job_lock:
            _job_state.update(running=False, finished_at=datetime.now(IST).isoformat(),
                              error=None)
        log.info("Job %s finished", name)
    except Exception:
        detail = traceback.format_exc(limit=6)
        with _job_lock:
            _job_state.update(running=False, finished_at=datetime.now(IST).isoformat(),
                              error=detail)
        log.error("Job %s failed:\n%s", name, detail)


@app.post("/jobs/cold-start")
def cold_start_job(request: Request):
    """Kicks off the full ingestion. Idempotent and resumable if re-run."""
    require_internal_key(request)

    if store.valid_token() is None:
        raise HTTPException(409, "No valid Upstox token — connect Upstox first")

    with _job_lock:
        if _job_state["running"]:
            raise HTTPException(409, f"Job already running: {_job_state['name']}")
        _job_state.update(name="cold_start", running=True,
                          started_at=datetime.now(IST).isoformat(),
                          finished_at=None, error=None)

    from ingest import cold_start
    threading.Thread(target=_run_job, args=("cold_start", cold_start), daemon=True).start()
    return {"ok": True, "started": True}


@app.get("/signals")
def signals(request: Request):
    """
    Current signals for the dashboard, grouped into score bands.

    Band labels are NOT conviction claims. Until resolved outcomes exist
    the hit rate is null and the UI must say so — a score band asserting a
    success probability nothing has measured is exactly the badge the
    intraday desk had to remove.
    """
    require_internal_key(request)
    from ingest import connect

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select s.symbol, y.company_name, y.industry, s.as_of_date,
                       s.setup_type, s.pattern, s.entry_trigger, s.stop_loss,
                       s.t1, s.t2, s.r_multiple_t1, s.qty_suggested, s.risk_amount,
                       s.score_total, s.score_breakdown, s.regime_state,
                       s.notes, s.status, s.expires_on, s.base_low,
                       s.base_start_date, s.pivot_bar_date
                from signals s
                join symbols y on y.symbol = s.symbol
                where s.status in ('pending','triggered')
                order by s.score_total desc
            """)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

            cur.execute("select state, breadth_above_50dma, vix, distribution_days, as_of "
                        "from market_regime order by as_of desc limit 1")
            reg = cur.fetchone()

            # Realised hit rate per band, once outcomes exist.
            cur.execute("""
                select case when s.score_total >= 80 then 'high'
                            when s.score_total >= 65 then 'medium'
                            else 'low' end as band,
                       count(*) filter (where o.r_realised is not null) as resolved,
                       count(*) filter (where o.r_realised > 0) as wins
                from signals s
                left join signal_outcomes o on o.signal_id = s.id
                group by 1
            """)
            calib = {r[0]: {"resolved": r[1], "wins": r[2],
                            "hit_rate": (r[2] / r[1]) if r[1] and r[1] >= 20 else None}
                     for r in cur.fetchall()}
    finally:
        conn.close()

    def band(score):
        return "high" if score >= 80 else "medium" if score >= 65 else "low"

    for r in rows:
        r["band"] = band(float(r["score_total"]))
        for k in ("entry_trigger", "stop_loss", "t1", "t2", "r_multiple_t1",
                  "score_total", "risk_amount", "base_low"):
            if r.get(k) is not None:
                r[k] = float(r[k])
        for k in ("as_of_date", "expires_on", "base_start_date", "pivot_bar_date"):
            if r.get(k) is not None:
                r[k] = str(r[k])

    return {
        "regime": {"state": reg[0], "breadth_above_50dma": float(reg[1]) if reg and reg[1] is not None else None,
                   "vix": float(reg[2]) if reg and reg[2] is not None else None,
                   "distribution_days": reg[3], "as_of": str(reg[4])} if reg else None,
        "calibration": {b: calib.get(b, {"resolved": 0, "wins": 0, "hit_rate": None})
                        for b in ("high", "medium", "low")},
        "signals": rows,
        "counts": {b: sum(1 for r in rows if r["band"] == b)
                   for b in ("high", "medium", "low")},
    }


@app.post("/jobs/scan")
def scan_job(request: Request):
    """
    Runs the gate scan and returns the candidate list synchronously.

    Unlike the backfill this is seconds, not minutes — one query and some
    pandas — so there is no reason to hide it behind a background thread
    and a polling loop.
    """
    require_internal_key(request)
    from scan import run_scan
    try:
        return run_scan()
    except Exception as exc:
        log.error("Scan failed: %s", exc)
        raise HTTPException(500, f"Scan failed: {exc}")


@app.get("/jobs/status")
def jobs_status(request: Request):
    require_internal_key(request)
    with _job_lock:
        current = dict(_job_state)

    runs, counts = [], {}
    try:
        from ingest import connect
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("select job, started_at, finished_at, status, rows_written, "
                            "error from ingestion_runs order by id desc limit 12")
                runs = [
                    {"job": r[0], "started_at": str(r[1]), "finished_at": str(r[2]),
                     "status": r[3], "rows_written": r[4],
                     "error": (r[5][:300] if r[5] else None)}
                    for r in cur.fetchall()
                ]
                cur.execute("select (select count(*) from symbols), "
                            "(select count(*) from ohlcv_daily), "
                            "(select count(*) from corporate_actions), "
                            "(select count(*) from surveillance)")
                s, o, c, v = cur.fetchone()
                counts = {"symbols": s, "ohlcv_daily": o,
                          "corporate_actions": c, "surveillance": v}
        finally:
            conn.close()
    except Exception as exc:
        counts = {"error": str(exc)[:200]}

    return {"current": current, "recent_runs": runs, "row_counts": counts}


@app.post("/auth/upstox/disconnect")
def disconnect(request: Request):
    """
    Clears the stored token. Protected by the internal key: the UI must
    call this through a Lovable server function, never from the browser,
    so the key stays server-side. Worst case if it leaked is a forced
    re-login, but a forced re-login at 16:10 means a missed scan.
    """
    require_internal_key(request)
    was_connected = store.valid_token() is not None
    store.path.unlink(missing_ok=True)
    log.info("Upstox token cleared (was_connected=%s)", was_connected)
    return {"ok": True, "was_connected": was_connected}


@app.get("/health")
def health():
    """Public but says nothing sensitive: no token, no expiry, just fit/unfit."""
    return {"ok": True, "authenticated": store.valid_token() is not None}


@app.get("/status")
def status(request: Request):
    require_internal_key(request)
    row = store.read()
    return {
        "authenticated": store.valid_token() is not None,
        "issued_at": row.get("issued_at"),
        "expires_at": row.get("expires_at"),
        "now_ist": datetime.now(IST).isoformat(),
        "paper_mode": os.environ.get("PAPER_MODE", "true"),
        "universe": os.environ.get("UNIVERSE"),
    }


def _page(title: str, detail: str) -> str:
    return f"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font:16px system-ui;background:#0b0f14;color:#e6edf3;display:grid;
place-items:center;height:100vh;margin:0;text-align:center}}
.c{{max-width:32rem;padding:2rem}}h1{{color:#2dd4bf;font-size:1.25rem}}
p{{color:#8b949e;line-height:1.6}}</style>
<div class=c><h1>{title}</h1><p>{detail}</p></div>"""
