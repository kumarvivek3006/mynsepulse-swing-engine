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

from fastapi import Body, FastAPI, HTTPException, Query, Request
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


@app.on_event("startup")
def _start_scheduler() -> None:
    try:
        import scheduler
        scheduler.start()
    except Exception:
        # A broken scheduler must not stop the auth server from serving —
        # without it you cannot log in to fix anything.
        log.exception("Scheduler failed to start")


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
                with latest as (
                    select distinct on (symbol) symbol, adj_close, trade_date
                    from ohlcv_daily order by symbol, trade_date desc
                )
                select s.id, s.symbol, y.company_name, y.industry, s.as_of_date,
                       s.setup_type, s.pattern, s.entry_trigger, s.stop_loss,
                       s.t1, s.t2, s.r_multiple_t1, s.qty_suggested, s.risk_amount,
                       s.score_total, s.score_breakdown, s.regime_state,
                       s.notes, s.status, s.expires_on, s.base_low,
                       s.base_start_date, s.pivot_bar_date,
                       l.adj_close as last_close, l.trade_date as last_bar,
                       o.entry_date, o.entry_price, o.exit_date, o.exit_price,
                       o.exit_reason, o.r_realised, o.max_favourable_r, o.max_adverse_r
                from signals s
                join symbols y on y.symbol = s.symbol
                left join latest l on l.symbol = s.symbol
                left join signal_outcomes o on o.signal_id = s.id
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

    # 20 EMA and the latest confirmed swing low, for the symbols on screen
    # only — a handful of queries, not five hundred.
    ema20_by_symbol: dict[str, float] = {}
    trail_by_symbol: dict[str, float] = {}
    if rows:
        conn2 = connect()
        try:
            with conn2.cursor() as cur:
                for sym in {r["symbol"] for r in rows}:
                    cur.execute("""
                        select adj_close, adj_low from ohlcv_daily
                        where symbol = %s order by trade_date desc limit 60
                    """, (sym,))
                    bars = cur.fetchall()
                    if len(bars) < 25:
                        continue
                    closes = [float(b[0]) for b in reversed(bars)]
                    lows = [float(b[1]) for b in reversed(bars)]

                    k = 2 / 21
                    ema = closes[0]
                    for px in closes[1:]:
                        ema = px * k + ema * (1 - k)
                    ema20_by_symbol[sym] = ema

                    # Most recent low with two higher lows either side.
                    for i in range(len(lows) - 3, 2, -1):
                        w = lows[i - 2:i + 3]
                        if lows[i] == min(w):
                            trail_by_symbol[sym] = lows[i]
                            break
        finally:
            conn2.close()

    def band(score):
        return "high" if score >= 80 else "medium" if score >= 65 else "low"

    for r in rows:
        r["id"] = str(r["id"])
        r["band"] = band(float(r["score_total"]))

        # Live progress against the levels. Everything here is derived from
        # prices that exist, not from a projection.
        lc = float(r["last_close"]) if r.get("last_close") is not None else None
        entry = float(r["entry_trigger"])
        stop, t1 = float(r["stop_loss"]), float(r["t1"])
        risk = entry - stop
        taken = r.get("entry_price") is not None

        prog = {"last_close": lc, "taken": taken}
        if lc is not None:
            prog["pct_to_entry"] = round((entry / lc - 1) * 100, 2)
            basis = float(r["entry_price"]) if taken else entry
            prog["r_now"] = round((lc - basis) / risk, 2) if risk > 0 else None
            if taken:
                prog["state"] = ("stopped" if lc <= stop else
                                 "target_hit" if lc >= t1 else "open")
            else:
                prog["state"] = ("triggered" if lc >= entry else
                                 "invalidated" if lc <= stop else "waiting")
        # Advisories. Every one is derived from a price that traded, and
        # from the levels already published — never a fresh projection.
        adv = []
        t2 = float(r["t2"]) if r.get("t2") is not None else None
        ema20 = ema20_by_symbol.get(r["symbol"])
        trail = trail_by_symbol.get(r["symbol"])

        if taken and lc is not None:
            entry_px = float(r["entry_price"])
            if lc <= stop:
                adv.append({"kind": "stop_hit", "severity": "high",
                            "text": "Price closed at or below the stop."})
            elif lc >= t1:
                adv.append({"kind": "target_hit", "severity": "good",
                            "text": f"T1 {t1} reached. Trail rather than hold for T2."})
                if t2:
                    adv.append({"kind": "trail_target", "severity": "info",
                                "text": f"T2 {t2} is the remaining objective."})
            if prog.get("r_now") is not None and prog["r_now"] >= 1:
                adv.append({"kind": "trail_breakeven", "severity": "info",
                            "text": "Beyond 1R — move the stop to breakeven."})
            if trail and trail > stop:
                adv.append({"kind": "trail_stop", "severity": "info",
                            "text": f"Swing-low trail now at {round(trail, 2)}."})
            if ema20 and lc < ema20 and lc > stop:
                adv.append({"kind": "early_exit", "severity": "warn",
                            "text": "Closed below the 20 EMA — momentum failing."})
            if lc < entry_px and prog.get("r_now", 0) <= -0.5:
                adv.append({"kind": "early_exit", "severity": "warn",
                            "text": "Half the risk is gone with no progress."})
        elif lc is not None:
            if lc <= stop:
                adv.append({"kind": "invalidated", "severity": "warn",
                            "text": "Broke the stop before triggering. Setup is void."})
            elif lc >= entry:
                adv.append({"kind": "triggered", "severity": "good",
                            "text": "Trading through the entry trigger."})
            elif prog.get("pct_to_entry") is not None and prog["pct_to_entry"] <= 1.0:
                adv.append({"kind": "approaching", "severity": "info",
                            "text": f"Within {prog['pct_to_entry']}% of the trigger."})

        r["advisories"] = adv
        r["progress"] = prog
        for k in ("entry_trigger", "stop_loss", "t1", "t2", "r_multiple_t1",
                  "score_total", "risk_amount", "base_low", "last_close",
                  "entry_price", "exit_price", "r_realised",
                  "max_favourable_r", "max_adverse_r"):
            if r.get(k) is not None:
                r[k] = float(r[k])
        for k in ("as_of_date", "expires_on", "base_start_date", "pivot_bar_date",
                  "last_bar", "entry_date", "exit_date"):
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


@app.get("/daily-log")
def daily_log(request: Request, day: str | None = Query(None)):
    """
    Everything the engine produced on a date, including setups that were
    never taken and those that expired. The dashboard shows what to trade;
    this is the record of what it said, which is what makes the score
    bands auditable later.
    """
    require_internal_key(request)
    from ingest import connect

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select coalesce(%s::date, max(as_of_date)) from signals", (day,))
            target = cur.fetchone()[0]
            if target is None:
                return {"day": None, "signals": [], "rejections": [], "regime": None}

            cur.execute("""
                select s.id, s.symbol, y.company_name, s.setup_type, s.pattern,
                       s.entry_trigger, s.stop_loss, s.t1, s.t2, s.r_multiple_t1,
                       s.score_total, s.status, s.regime_state,
                       o.entry_price, o.exit_price, o.exit_reason, o.r_realised
                from signals s
                join symbols y on y.symbol = s.symbol
                left join signal_outcomes o on o.signal_id = s.id
                where s.as_of_date = %s
                order by s.score_total desc
            """, (target,))
            cols = [d[0] for d in cur.description]
            sigs = [dict(zip(cols, r)) for r in cur.fetchall()]

            cur.execute("""
                select coalesce(failed_gate, 'passed') as gate, reason_code, count(*)
                from gate_log where as_of_date = %s
                group by 1, 2 order by 3 desc
            """, (target,))
            rejections = [{"gate": r[0], "reason": r[1], "count": r[2]}
                          for r in cur.fetchall()]

            cur.execute("select state, breadth_above_50dma, vix, distribution_days "
                        "from market_regime where as_of = %s", (target,))
            reg = cur.fetchone()

            cur.execute("select distinct as_of_date from signals order by 1 desc limit 30")
            days = [str(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()

    for s_ in sigs:
        s_["id"] = str(s_["id"])
        for k in ("entry_trigger", "stop_loss", "t1", "t2", "r_multiple_t1",
                  "score_total", "entry_price", "exit_price", "r_realised"):
            if s_.get(k) is not None:
                s_[k] = float(s_[k])

    return {
        "day": str(target),
        "available_days": days,
        "regime": {"state": reg[0], "breadth_above_50dma": float(reg[1]) if reg and reg[1] is not None else None,
                   "vix": float(reg[2]) if reg and reg[2] is not None else None,
                   "distribution_days": reg[3]} if reg else None,
        "signals": sigs,
        "rejections": rejections,
    }


@app.get("/my-trades")
def my_trades(request: Request):
    """Taken trades, open and closed, with realised R and running stats."""
    require_internal_key(request)
    from ingest import connect

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                with latest as (
                    select distinct on (symbol) symbol, adj_close
                    from ohlcv_daily order by symbol, trade_date desc
                )
                select s.id, s.symbol, y.company_name, s.setup_type, s.pattern,
                       s.entry_trigger, s.stop_loss, s.t1, s.t2, s.score_total,
                       s.status, s.as_of_date, l.adj_close as last_close,
                       o.entry_date, o.entry_price, o.exit_date, o.exit_price,
                       o.exit_reason, o.r_realised
                from signals s
                join symbols y on y.symbol = s.symbol
                join signal_outcomes o on o.signal_id = s.id
                left join latest l on l.symbol = s.symbol
                where o.entry_price is not null
                order by (o.exit_date is not null), o.entry_date desc
            """)
            cols = [d[0] for d in cur.description]
            trades = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()

    open_t, closed = [], []
    for t in trades:
        t["id"] = str(t["id"])
        for k in ("entry_trigger", "stop_loss", "t1", "t2", "score_total",
                  "last_close", "entry_price", "exit_price", "r_realised"):
            if t.get(k) is not None:
                t[k] = float(t[k])
        for k in ("as_of_date", "entry_date", "exit_date"):
            if t.get(k) is not None:
                t[k] = str(t[k])

        risk = t["entry_price"] - t["stop_loss"]
        if t.get("exit_price") is None:
            t["r_now"] = round((t["last_close"] - t["entry_price"]) / risk, 2) \
                if t.get("last_close") and risk > 0 else None
            open_t.append(t)
        else:
            closed.append(t)

    resolved = [t["r_realised"] for t in closed if t.get("r_realised") is not None]
    stats = {
        "open": len(open_t),
        "closed": len(closed),
        "wins": sum(1 for r in resolved if r > 0),
        "losses": sum(1 for r in resolved if r <= 0),
        "hit_rate": round(sum(1 for r in resolved if r > 0) / len(resolved), 3)
        if resolved else None,
        "expectancy_r": round(sum(resolved) / len(resolved), 3) if resolved else None,
        "total_r": round(sum(resolved), 2) if resolved else None,
    }
    return {"open": open_t, "closed": closed, "stats": stats}


@app.post("/signals/take")
def take_signal(request: Request, body: dict = Body(...)):
    """
    Mark a recommendation as actually taken.

    Records the fill price the user got, not the trigger price we
    published. Slippage between the two is the difference between a
    backtest and a track record, and only the real fill can validate the
    score bands later.
    """
    require_internal_key(request)
    signal_id = body.get("signal_id")
    if not signal_id:
        raise HTTPException(400, "signal_id required")

    from ingest import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select entry_trigger, stop_loss from signals where id = %s",
                        (signal_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "signal not found")

            entry_price = float(body.get("entry_price") or row[0])
            cur.execute("update signals set status = 'triggered' where id = %s", (signal_id,))
            cur.execute("""
                insert into signal_outcomes (signal_id, entry_date, entry_price)
                values (%s, current_date, %s)
                on conflict (signal_id) do update set
                    entry_date = excluded.entry_date,
                    entry_price = excluded.entry_price
            """, (signal_id, entry_price))
        conn.commit()
        return {"ok": True, "signal_id": signal_id, "entry_price": entry_price}
    finally:
        conn.close()


@app.post("/signals/close")
def close_signal(request: Request, body: dict = Body(...)):
    """Record the exit and the realised R. This is what calibrates the bands."""
    require_internal_key(request)
    signal_id = body.get("signal_id")
    exit_price = body.get("exit_price")
    if not signal_id or exit_price is None:
        raise HTTPException(400, "signal_id and exit_price required")

    from ingest import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select s.entry_trigger, s.stop_loss, o.entry_price
                from signals s left join signal_outcomes o on o.signal_id = s.id
                where s.id = %s
            """, (signal_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "signal not found")

            entry = float(row[2]) if row[2] is not None else float(row[0])
            risk = entry - float(row[1])
            r_realised = round((float(exit_price) - entry) / risk, 3) if risk > 0 else None

            cur.execute("update signals set status = %s where id = %s",
                        (body.get("exit_reason") or "closed", signal_id))
            cur.execute("""
                insert into signal_outcomes
                    (signal_id, entry_price, exit_date, exit_price, exit_reason, r_realised)
                values (%s, %s, current_date, %s, %s, %s)
                on conflict (signal_id) do update set
                    exit_date = excluded.exit_date,
                    exit_price = excluded.exit_price,
                    exit_reason = excluded.exit_reason,
                    r_realised = excluded.r_realised
            """, (signal_id, entry, exit_price, body.get("exit_reason"), r_realised))
        conn.commit()
        return {"ok": True, "r_realised": r_realised}
    finally:
        conn.close()


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
    mode = request.query_params.get("mode", "postclose")
    try:
        return run_scan(mode=mode)
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


@app.get("/jobs/schedule")
def schedule_status(request: Request):
    require_internal_key(request)
    import scheduler
    return scheduler.status()


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
