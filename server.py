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


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
SETTINGS_DEFAULTS = {"capital": 0.0, "risk_pct": 2.5,
                     "scale_out_pct": 50.0, "ma_exit_period": 20.0}


def _read_settings(cur) -> dict:
    """
    Settings are a convenience, not a dependency.

    If engine_settings is missing or unreadable, fall back to defaults
    rather than raising. Position sizing is optional — an unrun migration
    should not blank every recommendation on the dashboard, which is
    exactly what it did the first time.
    """
    try:
        cur.execute("select key, value from engine_settings")
        rows = dict(cur.fetchall())
    except Exception as exc:
        cur.connection.rollback()
        log.warning("engine_settings unreadable (%s); using defaults", exc)
        return {**SETTINGS_DEFAULTS, "available": False}

    return {
        "capital": float(rows.get("capital") or 0),
        "risk_pct": float(rows.get("risk_pct") or 2.5),
        "scale_out_pct": float(rows.get("scale_out_pct") or 50),
        "ma_exit_period": float(rows.get("ma_exit_period") or 20),
        "available": True,
    }


def size_position(capital: float, risk_pct: float, entry: float,
                  stop: float) -> dict:
    """
    qty = floor(capital * risk_pct% / (entry - stop))

    No caps. Position value is reported alongside so an oversized position
    is visible rather than silent: a very tight stop can produce a
    quantity worth more than the capital it is sized from, which is
    arithmetically correct and practically unfillable.
    """
    risk_per_share = entry - stop
    if capital <= 0 or risk_per_share <= 0:
        return {"qty": None, "risk_per_share": None, "risk_amount": None,
                "position_value": None, "exceeds_capital": False,
                "pct_of_capital": None}

    budget = capital * (risk_pct / 100.0)
    qty = int(budget // risk_per_share)
    position_value = qty * entry
    return {
        "qty": qty,
        "risk_per_share": round(risk_per_share, 2),
        "risk_amount": round(qty * risk_per_share, 2),
        "position_value": round(position_value, 2),
        "exceeds_capital": position_value > capital,
        "pct_of_capital": round(position_value / capital * 100, 1),
    }


@app.get("/settings")
def get_settings(request: Request):
    require_internal_key(request)
    from ingest import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            return _read_settings(cur)
    finally:
        conn.close()


@app.put("/settings")
def put_settings(request: Request, body: dict = Body(...)):
    """Capital and per-trade risk. Neither affects scanning or scoring."""
    require_internal_key(request)
    from ingest import connect

    updates = {}
    if "capital" in body:
        capital = float(body["capital"] or 0)
        if capital < 0:
            raise HTTPException(400, "capital cannot be negative")
        updates["capital"] = capital
    if "risk_pct" in body:
        risk = float(body["risk_pct"] or 0)
        if not 0 < risk <= 100:
            raise HTTPException(400, "risk_pct must be between 0 and 100")
        updates["risk_pct"] = risk
    if "scale_out_pct" in body:
        pct = float(body["scale_out_pct"] or 0)
        if not 0 <= pct <= 100:
            raise HTTPException(400, "scale_out_pct must be between 0 and 100")
        updates["scale_out_pct"] = pct
    if "ma_exit_period" in body:
        period = float(body["ma_exit_period"] or 0)
        if period not in (10, 20, 50):
            raise HTTPException(400, "ma_exit_period must be 10, 20 or 50")
        updates["ma_exit_period"] = period
    if not updates:
        raise HTTPException(400, "nothing to update")


    conn = connect()
    try:
        with conn.cursor() as cur:
            for key, value in updates.items():
                cur.execute("""
                    insert into engine_settings (key, value, updated_at)
                    values (%s, to_jsonb(%s::numeric), now())
                    on conflict (key) do update set
                        value = excluded.value, updated_at = now()
                """, (key, value))
            conn.commit()
            return _read_settings(cur)
    finally:
        conn.close()


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

            settings = _read_settings(cur)

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
    day_change: dict[str, float] = {}
    delivery: dict[str, dict] = {}
    if rows:
        conn2 = connect()
        try:
            with conn2.cursor() as cur:
                for sym in {r["symbol"] for r in rows}:
                    cur.execute("""
                        select adj_close, adj_low, adj_high from ohlcv_daily
                        where symbol = %s order by trade_date desc limit 60
                    """, (sym,))
                    bars = cur.fetchall()
                    if len(bars) < 25:
                        continue
                    closes = [float(b[0]) for b in reversed(bars)]
                    lows = [float(b[1]) for b in reversed(bars)]

                    # The session's own move, from the two most recent closes.
                    # Without this a card outside market hours had no change
                    # figure at all and the UI rendered a meaningless 0.00%.
                    if len(closes) >= 2 and closes[-2] > 0:
                        day_change[sym] = round((closes[-1] / closes[-2] - 1) * 100, 2)

                    # Delivery trend. Diagnostic only — no gate reads it.
                    cur.execute("""
                        select delivery_pct from delivery_daily
                        where symbol = %s and delivery_pct is not null
                        order by trade_date desc limit 40
                    """, (sym,))
                    dvals = [float(r[0]) for r in cur.fetchall()]
                    if len(dvals) >= 20:
                        recent = sum(dvals[:10]) / 10
                        prior = sum(dvals[10:]) / len(dvals[10:])
                        delivery[sym] = {
                            "latest": round(dvals[0], 2),
                            "avg_10d": round(recent, 2),
                            "avg_prior": round(prior, 2),
                            "expanding": bool(recent > prior * 1.1),
                        }

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

        # Promote the two provenance flags out of notes so the UI does not
        # have to reach into a jsonb blob. Purely additive — nothing else
        # reads them and no existing field changes.
        _notes = r.get("notes") or {}
        if isinstance(_notes, str):
            try:
                _notes = json.loads(_notes)
            except (ValueError, TypeError):
                _notes = {}
        r["is_add_on"] = bool(_notes.get("is_add_on"))
        r["is_transition"] = bool(_notes.get("is_transition"))
        r["is_provisional"] = bool(_notes.get("is_provisional"))
        # t2 can be genuinely absent now (no manufactured pad). t2_basis lets
        # the UI say "no second target — new-high breakout" rather than
        # rendering an empty field that looks like missing data.
        r["t2_basis"] = _notes.get("t2_basis")
        r["is_new_opportunity"] = bool(_notes.get("is_new_opportunity"))

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
        # Sizing is computed here, on read, rather than stored at scan time.
        # Editing capital then updates every card immediately instead of
        # waiting for the next scan. Levels are read, never modified.
        r["sizing"] = size_position(settings["capital"], settings["risk_pct"],
                                    entry, stop)
        r["qty_suggested"] = r["sizing"]["qty"]
        r["risk_amount"] = r["sizing"]["risk_amount"]

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
        "settings": settings,
        "regime": {"state": reg[0], "breadth_above_50dma": float(reg[1]) if reg and reg[1] is not None else None,
                   "vix": float(reg[2]) if reg and reg[2] is not None else None,
                   "distribution_days": reg[3], "as_of": str(reg[4])} if reg else None,
        "calibration": {b: calib.get(b, {"resolved": 0, "wins": 0, "hit_rate": None})
                        for b in ("high", "medium", "low")},
        "signals": rows,
        "counts": {b: sum(1 for r in rows if r["band"] == b)
                   for b in ("high", "medium", "low")},
    }


@app.get("/signals/live")
def signals_live(request: Request):
    """
    Live LTP and progress for open signals only.

    The rest of the desk runs on end-of-day bars, which is correct for
    deriving levels — but an armed setup sitting 2% under its trigger is
    only actionable if you can see where price actually is. This fetches
    quotes for the handful of symbols currently on screen, not the whole
    universe.

    Outside market hours Upstox returns the last traded price, which is the
    previous close. That is reported honestly via `is_live` rather than
    dressed up as a live tick.
    """
    require_internal_key(request)
    from ingest import connect

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select s.id, s.symbol, y.upstox_instrument_key,
                       s.entry_trigger, s.stop_loss, s.t1,
                       o.entry_price
                from signals s
                join symbols y on y.symbol = s.symbol
                left join signal_outcomes o on o.signal_id = s.id
                where s.status in ('pending','triggered')
                  and y.upstox_instrument_key is not null
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {"is_live": False, "as_of": datetime.now(IST).isoformat(), "quotes": []}

    now = datetime.now(IST)
    weekday = now.weekday() < 5
    minutes = now.hour * 60 + now.minute
    market_open = weekday and 555 <= minutes <= 945      # 09:15-15:45 IST

    quotes: dict = {}
    error = None
    try:
        from upstox_client import UpstoxClient
        client = UpstoxClient(store=store)
        quotes = client.quotes([r[2] for r in rows])
    except Exception as exc:
        # A failed quote fetch must not blank the dashboard — the cards
        # still hold valid levels from the last close.
        error = str(exc)[:200]
        log.warning("Live quote fetch failed: %s", error)

    # Upstox keys the quotes response by "NSE_EQ:RELIANCE" — exchange and
    # trading symbol — not by the instrument_key that was sent. Each quote
    # object does carry instrument_token, which IS the instrument_key, so
    # index on that. Matching on the key's ISIN suffix never worked because
    # the response keys do not contain the ISIN at all.
    by_token: dict[str, dict] = {}
    by_symbol: dict[str, dict] = {}
    for response_key, value in (quotes or {}).items():
        if not isinstance(value, dict):
            continue
        token = value.get("instrument_token") or value.get("instrument_key")
        if token:
            by_token[token] = value
        name = value.get("symbol") or response_key.split(":")[-1]
        if name:
            by_symbol[str(name).upper()] = value

    def find(key: str, symbol: str) -> dict:
        return by_token.get(key) or by_symbol.get(symbol.upper()) or {}

    out, unmatched = [], []
    for sig_id, symbol, key, entry, stop, t1, entry_price in rows:
        q = find(key, symbol)
        ltp = q.get("last_price") or q.get("ltp")
        if ltp is None:
            unmatched.append(symbol)
            continue
        ltp = float(ltp)
        entry, stop, t1 = float(entry), float(stop), float(t1)
        prev_close = q.get("close_price") or (q.get("ohlc") or {}).get("close")
        risk = entry - stop
        basis = float(entry_price) if entry_price is not None else entry

        out.append({
            "signal_id": str(sig_id),
            "symbol": symbol,
            "ltp": round(ltp, 2),
            "change_pct": round((ltp / float(prev_close) - 1) * 100, 2)
            if prev_close else None,
            "pct_to_entry": round((entry / ltp - 1) * 100, 2),
            "r_now": round((ltp - basis) / risk, 2) if risk > 0 else None,
            "state": ("stopped" if entry_price is not None and ltp <= stop else
                      "target_hit" if entry_price is not None and ltp >= t1 else
                      "open" if entry_price is not None else
                      "triggered" if ltp >= entry else
                      "invalidated" if ltp <= stop else "waiting"),
        })

    if unmatched:
        log.warning("No quote matched for %d/%d symbols: %s",
                    len(unmatched), len(rows), unmatched[:10])

    return {
        "is_live": market_open and not error and bool(out),
        "market_open": market_open,
        "as_of": now.isoformat(),
        "error": error,
        # Surfaced so a silent matching failure is visible rather than
        # looking indistinguishable from "the market is closed".
        "matched": len(out),
        "requested": len(rows),
        "unmatched": unmatched[:10],
        "quotes": out,
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


def _trail_levels(bars: list, entry_price: float, original_stop: float,
                  t1: float, t2: float | None, pivot: float | None = None,
                  atr_floor_mult: float = 0.75,
                  live_price: float | None = None) -> dict:
    """
    Trail to prices that actually printed. Nothing derived by formula.

    The earlier version offered breakeven and a Chandelier ATR stop. Both
    are manufactured: your entry price is an accounting fact the market has
    no memory of, and "highest high minus 2.5 ATR" is a number that has
    never traded. Neither is a level anyone is defending, so neither
    belongs as a stop.

    Candidates are structural only, taken from bars since entry:

      * confirmed swing lows        - a low with two higher lows either side
      * the breakout pivot          - old resistance becomes support once
                                      price has moved clear of it
      * low of a high-volume up bar - where buyers demonstrably stepped in
      * low of an unfilled gap up   - the gap edge is real support until filled

    ATR is used ONLY as a noise filter, never as the level itself: a
    candidate closer than 0.75 ATR to the current price would be inside
    daily noise. Same rule the entry stop uses.

    Of the survivors, the HIGHEST is taken - the tightest real level - and
    it can never sit below the original stop.

    bars: [(trade_date, close, high, low, volume)] oldest -> newest
    """
    if len(bars) < 6:
        return {"suggested_stop": original_stop, "basis": "original_stop",
                "raised": False, "gain_vs_original": 0.0, "r_now": None,
                "r_locked": -1.0, "next_target": t1,
                "target_stage": "running_to_t1", "atr14": None,
                "candidates": [], "risk_free_available": False}

    closes = [float(b[1]) for b in bars]
    highs = [float(b[2]) for b in bars]
    lows = [float(b[3]) for b in bars]
    vols = [float(b[4] or 0) for b in bars]
    # Levels come from closed bars; only the "where is price now" comparisons
    # use the live tick. Substituting a live price into the swing-low or
    # volume logic would corrupt structure that has already formed.
    last = float(live_price) if live_price is not None else closes[-1]

    risk = entry_price - original_stop
    r_now = ((last - entry_price) / risk) if risk > 0 else None

    atr = None
    if len(bars) >= 15:
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                   abs(lows[i] - closes[i - 1])) for i in range(1, len(bars))]
        atr = trs[0]
        for tr in trs[1:]:
            atr = (atr * 13 + tr) / 14

    candidates: list[tuple[float, str, str]] = []

    # Confirmed swing lows, most recent first.
    for i in range(len(lows) - 3, 1, -1):
        window = lows[i - 2:i + 3]
        if len(window) == 5 and lows[i] == min(window):
            candidates.append((lows[i], "swing_low",
                               f"swing low of {bars[i][0]}"))
            if len([c for c in candidates if c[1] == "swing_low"]) >= 3:
                break

    # The breakout pivot: former resistance, now support.
    if pivot and last > pivot:
        candidates.append((pivot, "breakout_pivot",
                           "the pivot the stock broke out through"))

    # Low of the heaviest up bar since entry - where buyers showed up.
    up_bars = [(i, vols[i]) for i in range(1, len(bars)) if closes[i] > closes[i - 1]]
    if up_bars:
        i = max(up_bars, key=lambda x: x[1])[0]
        candidates.append((lows[i], "high_volume_bar_low",
                           f"low of the heaviest up day ({bars[i][0]})"))

    # Unfilled gap up: the gap edge holds until it is filled.
    for i in range(len(bars) - 1, 0, -1):
        if lows[i] > highs[i - 1] and min(lows[i:]) > highs[i - 1]:
            candidates.append((highs[i - 1], "unfilled_gap",
                               f"top of the unfilled gap from {bars[i][0]}"))
            break

    # Noise filter, and never below the stop already in force.
    floor = atr * atr_floor_mult if atr else 0.0
    viable = [(p, kind, why) for p, kind, why in candidates
              if p >= original_stop and (last - p) >= floor]

    if viable:
        price, kind, why = max(viable, key=lambda c: c[0])
        suggested, basis, rationale = price * 0.999, kind, why
    else:
        suggested, basis = original_stop, "original_stop"
        rationale = "no structural level yet clears the noise floor"

    next_target = t1 if last < t1 else (t2 if t2 and last < t2 else None)

    return {
        "suggested_stop": round(suggested, 2),
        "basis": basis,
        "rationale": rationale,
        "raised": round(suggested, 2) > round(original_stop, 2),
        "gain_vs_original": round(suggested - original_stop, 2),
        "r_now": round(r_now, 2) if r_now is not None else None,
        "r_locked": round((suggested - entry_price) / risk, 2) if risk > 0 else 0.0,
        "next_target": next_target,
        "target_stage": ("beyond_t2" if next_target is None else
                         "running_to_t2" if next_target == t2 else "running_to_t1"),
        "atr14": round(atr, 2) if atr else None,
        # Informational: past 1R the risk COULD be removed. Deliberately not
        # offered as a stop level, because breakeven is not a real level.
        "risk_free_available": bool(r_now is not None and r_now >= 1.0),
        "candidates": [{"price": round(p, 2), "kind": k, "why": w}
                       for p, k, w in sorted(viable, key=lambda c: -c[0])[:4]],
    }


def _exit_plan(bars: list, trade: dict, settings: dict, trail: dict) -> dict:
    """
    The scale-out and moving-average exit that discretionary swing traders
    actually run, kept honest about which part is a level and which is a
    signal.

      * Scale out a configurable share at T1. Near-universal practice, and
        the piece the engine was missing entirely — it treated every trade
        as all-in until stopped or targeted.

      * Trail the remainder on structural levels (see _trail_levels).

      * Watch for a CLOSE below the moving average. Practitioners do not
        rest a stop order at a moving average; they exit the next session
        after a close below it. So this is reported as a signal to act on,
        never as a stop price — which is why it does not appear in the
        trail candidates.
    """
    period = int(settings.get("ma_exit_period") or 20)
    scale_pct = float(settings.get("scale_out_pct") or 50)

    closes = [float(b[1]) for b in bars] if bars else []
    ema = None
    if len(closes) >= period:
        k = 2 / (period + 1)
        ema = closes[0]
        for px in closes[1:]:
            ema = px * k + ema * (1 - k)

    last = closes[-1] if closes else None
    qty = trade.get("sizing", {}).get("qty")
    t1, entry = trade.get("t1"), trade.get("entry_price")
    already_scaled = trade.get("scaled_qty") is not None

    scale_qty = int(qty * scale_pct / 100) if qty and scale_pct else None
    t1_reached = bool(last is not None and t1 and last >= t1)

    stage = ("scaled" if already_scaled else
             "at_target" if t1_reached else
             "running")

    ma_break = bool(ema is not None and last is not None and last < ema)

    actions = []
    if t1_reached and not already_scaled and scale_qty:
        actions.append({
            "kind": "scale_out", "severity": "good",
            "text": f"T1 reached. Practitioner default is to sell {scale_pct:.0f}% "
                    f"({scale_qty} shares) and trail the rest.",
        })
    if ma_break:
        actions.append({
            "kind": "ma_exit_signal", "severity": "warn",
            "text": f"Closed below the {period} EMA ({round(ema, 2)}). "
                    "The common rule is to exit the remainder next session — "
                    "this is a signal to act on, not a resting stop.",
        })
    if already_scaled:
        actions.append({
            "kind": "runner", "severity": "info",
            "text": f"Runner only. Trailing on {trail.get('basis', 'structure')}.",
        })

    return {
        "stage": stage,
        "scale_out_pct": scale_pct,
        "scale_out_qty": scale_qty,
        "t1_reached": t1_reached,
        "ma_period": period,
        "ma_value": round(ema, 2) if ema is not None else None,
        "ma_break": ma_break,
        "scaled_qty": trade.get("scaled_qty"),
        "scaled_price": trade.get("scaled_price"),
        "actions": actions,
    }


def _thesis_health(bars: list, nifty: list, entry_price: float, stop: float,
                   pivot: float | None, entry_date, atr_hint: float | None,
                   live_price: float | None = None) -> dict:
    """
    Assess whether the reason for the trade still holds.

    Every check reads bars that printed. None of it forecasts. The output
    is a list of concrete deteriorations, not a confidence number — "closed
    back below the pivot" is actionable, "confidence 0.42" is not.

    bars: [(trade_date, close, high, low, volume)] oldest -> newest
    """
    if len(bars) < 25:
        return {"state": "unknown", "reasons": [],
                "note": "Not enough bars since entry to assess."}

    closes = [float(b[1]) for b in bars]
    highs = [float(b[2]) for b in bars]
    lows = [float(b[3]) for b in bars]
    vols = [float(b[4] or 0) for b in bars]
    # Live where the question is "where is price now"; closed bars for the
    # EMA, the volume profile and the high/low sequence, which are history.
    last = float(live_price) if live_price is not None else closes[-1]

    k = 2 / 21
    ema20 = closes[0]
    for px in closes[1:]:
        ema20 = px * k + ema20 * (1 - k)

    reasons: list[dict] = []

    if last < ema20:
        reasons.append({"kind": "below_20ema", "weight": 2,
                        "text": f"Closed below the 20 EMA ({round(ema20, 2)}) — "
                                "the trend guide it was riding."})

    # A breakout that returns inside its base has failed, whatever the P&L.
    if pivot and last < pivot:
        reasons.append({"kind": "back_below_pivot", "weight": 3,
                        "text": f"Back below the breakout pivot ({round(pivot, 2)}). "
                                "The breakout has failed."})

    if len(highs) >= 4 and highs[-1] < highs[-2] < highs[-3]:
        reasons.append({"kind": "lower_highs", "weight": 1,
                        "text": "Three consecutive lower highs — momentum rolling over."})

    # Distribution: declines carrying more volume than advances.
    recent = list(zip(closes[-11:], vols[-11:]))
    # Unchanged closes are neither accumulation nor distribution; counting
    # them as down days made a flat series look like heavy selling.
    up_vol = sum(v for i, (c, v) in enumerate(recent[1:], 1) if c > recent[i - 1][0])
    down_vol = sum(v for i, (c, v) in enumerate(recent[1:], 1) if c < recent[i - 1][0])
    if down_vol > up_vol * 1.5 and down_vol > 0:
        reasons.append({"kind": "distribution", "weight": 2,
                        "text": "Down days are carrying more volume than up days — "
                                "supply is arriving."})

    # Relative strength since entry. A stock lagging the index it should be
    # leading has lost the reason it was selected.
    if nifty and len(nifty) >= len(bars):
        idx = [float(n) for n in nifty[-len(bars):]]
        if idx[0] > 0 and closes[0] > 0:
            stock_move = last / closes[0] - 1
            index_move = idx[-1] / idx[0] - 1
            if stock_move < index_move - 0.02:
                reasons.append({"kind": "rs_lost", "weight": 2,
                                "text": f"Underperforming the Nifty since entry by "
                                        f"{round((index_move - stock_move) * 100, 1)}%."})

    risk = entry_price - stop
    if risk > 0 and last <= stop:
        reasons.append({"kind": "stop_breached", "weight": 4,
                        "text": f"Trading at or below the stop ({round(stop, 2)}). "
                                "The trade is invalidated."})
    elif risk > 0 and (last - stop) < risk * 0.35:
        remaining = round((last - stop) / risk * 100)
        reasons.append({"kind": "near_stop", "weight": 3,
                        "text": f"Only {remaining}% of the original risk remains "
                                "before the stop."})

    # Time stop: capital tied up with nothing to show for it.
    if entry_date:
        held = (bars[-1][0] - entry_date).days
        if held >= 15 and risk > 0 and (last - entry_price) / risk < 0.5:
            reasons.append({"kind": "time_stop", "weight": 1,
                            "text": f"{held} days held with under 0.5R of progress."})

    severity = sum(r["weight"] for r in reasons)
    state = "broken" if severity >= 5 else "weakening" if severity >= 2 else "intact"
    return {"state": state, "severity": severity, "reasons": reasons,
            "ema20": round(ema20, 2)}


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
                       s.status, s.as_of_date, s.base_low, s.pivot_bar_date,
                       s.entry_trigger as pivot, l.adj_close as last_close,
                       o.entry_date, o.entry_price, o.exit_date, o.exit_price,
                       o.exit_reason, o.r_realised,
                       o.scaled_qty, o.scaled_price, o.scaled_date
                from signals s
                join symbols y on y.symbol = s.symbol
                join signal_outcomes o on o.signal_id = s.id
                left join latest l on l.symbol = s.symbol
                where o.entry_price is not null
                order by (o.exit_date is not null), o.entry_date desc
            """)
            cols = [d[0] for d in cur.description]
            trades = [dict(zip(cols, r)) for r in cur.fetchall()]
            settings = _read_settings(cur)
    finally:
        conn.close()

    # Bars for the held symbols only, plus the index for the RS check.
    bars_by_symbol: dict[str, list] = {}
    nifty_closes: list = []
    conn = connect()
    try:
        with conn.cursor() as cur:
            symbols = {t["symbol"] for t in trades if t.get("exit_price") is None}
            for sym in symbols:
                cur.execute("""
                    select trade_date, adj_close, adj_high, adj_low, volume
                    from ohlcv_daily where symbol = %s
                    order by trade_date desc limit 60
                """, (sym,))
                bars_by_symbol[sym] = list(reversed(cur.fetchall()))
            if symbols:
                cur.execute("""
                    select adj_close from ohlcv_daily where symbol = 'NIFTY50'
                    order by trade_date desc limit 60
                """)
                nifty_closes = [r[0] for r in reversed(cur.fetchall())]
    finally:
        conn.close()

    # Live prices for open positions only — a handful of symbols. Thesis and
    # trail were computed purely from end-of-day bars, so a position could
    # break its stop or lose the 20 EMA mid-session and this page would not
    # know until the next close.
    live_by_symbol: dict[str, float] = {}
    live_error = None
    open_symbols = [t["symbol"] for t in trades if t.get("exit_price") is None]
    if open_symbols:
        try:
            from upstox_client import UpstoxClient
            conn2 = connect()
            try:
                with conn2.cursor() as cur:
                    cur.execute("select symbol, upstox_instrument_key from symbols "
                                "where symbol = any(%s) "
                                "and upstox_instrument_key is not null",
                                (open_symbols,))
                    keymap = dict(cur.fetchall())
            finally:
                conn2.close()

            if keymap:
                quotes = UpstoxClient(store=store).quotes(list(keymap.values()))
                by_token = {}
                for rk, v in (quotes or {}).items():
                    if isinstance(v, dict):
                        tok = v.get("instrument_token") or v.get("instrument_key")
                        if tok:
                            by_token[tok] = v
                        name = v.get("symbol") or rk.split(":")[-1]
                        by_token.setdefault(str(name).upper(), v)
                for sym, key in keymap.items():
                    q = by_token.get(key) or by_token.get(sym.upper()) or {}
                    ltp = q.get("last_price") or q.get("ltp")
                    if ltp is not None:
                        live_by_symbol[sym] = float(ltp)
        except Exception as exc:
            # Fall back to stored bars. The levels remain valid; only the
            # freshness of the comparison is lost.
            live_error = str(exc)[:200]
            log.warning("Live prices unavailable for my-trades: %s", live_error)

    open_t, closed = [], []
    for t in trades:
        t["id"] = str(t["id"])
        for k in ("entry_trigger", "stop_loss", "t1", "t2", "score_total",
                  "last_close", "entry_price", "exit_price", "r_realised",
                  "base_low", "pivot", "scaled_price"):
            if t.get(k) is not None:
                t[k] = float(t[k])
        for k in ("as_of_date", "entry_date", "exit_date"):
            if t.get(k) is not None:
                t[k] = str(t[k])

        risk = t["entry_price"] - t["stop_loss"]
        if t.get("exit_price") is None:
            live = live_by_symbol.get(t["symbol"])
            t["live_price"] = live
            t["price_basis"] = "live" if live is not None else "last_close"
            current = live if live is not None else t.get("last_close")

            t["r_now"] = round((current - t["entry_price"]) / risk, 2) \
                if current and risk > 0 else None

            entry_dt = None
            if t.get("entry_date"):
                entry_dt = datetime.strptime(t["entry_date"], "%Y-%m-%d").date()
            # Recomputed against the price actually paid, not the published
            # trigger. Display only — it changes nothing about the position.
            t["sizing"] = size_position(settings["capital"], settings["risk_pct"],
                                        t["entry_price"], t["stop_loss"])

            t["trail"] = _trail_levels(
                bars_by_symbol.get(t["symbol"], []), t["entry_price"],
                t["stop_loss"], t["t1"], t.get("t2"),
                pivot=t.get("entry_trigger"), live_price=live)

            t["exit_plan"] = _exit_plan(bars_by_symbol.get(t["symbol"], []),
                                        t, settings, t["trail"])

            t["thesis"] = _thesis_health(
                bars_by_symbol.get(t["symbol"], []), nifty_closes,
                t["entry_price"], t["stop_loss"],
                t.get("pivot"), entry_dt, None, live_price=live)
            open_t.append(t)
        else:
            closed.append(t)

    # Portfolio exposure. Visibility only — nothing in the scan, the gates,
    # scoring or signal generation reads any of this.
    capital = settings["capital"]
    open_risk = sum(t["sizing"]["risk_amount"] for t in open_t
                    if t.get("sizing", {}).get("risk_amount"))
    open_value = sum(t["sizing"]["position_value"] for t in open_t
                     if t.get("sizing", {}).get("position_value"))
    exposure = {
        "capital": capital,
        "risk_pct_per_trade": settings["risk_pct"],
        "open_positions": len(open_t),
        "open_risk": round(open_risk, 2) if open_risk else 0.0,
        "open_risk_pct": round(open_risk / capital * 100, 2) if capital > 0 else None,
        "deployed_value": round(open_value, 2) if open_value else 0.0,
        "deployed_pct": round(open_value / capital * 100, 1) if capital > 0 else None,
        "exceeds_capital": capital > 0 and open_value > capital,
    }

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
    return {"open": open_t, "closed": closed, "stats": stats,
            "exposure": exposure, "settings": settings,
            "live": {"symbols_priced": len(live_by_symbol),
                     "symbols_open": len(open_symbols),
                     "error": live_error}}


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


@app.post("/signals/scale-out")
def scale_out_signal(request: Request, body: dict = Body(...)):
    """Record a partial exit. The position stays open on the remainder."""
    require_internal_key(request)
    signal_id = body.get("signal_id")
    qty = body.get("qty")
    price = body.get("price")
    if not signal_id or qty is None or price is None:
        raise HTTPException(400, "signal_id, qty and price required")

    from ingest import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                update signal_outcomes
                   set scaled_qty = %s, scaled_price = %s, scaled_date = current_date
                 where signal_id = %s and entry_price is not null
            """, (int(qty), float(price), signal_id))
            if cur.rowcount == 0:
                raise HTTPException(404, "no open trade for that signal")
        conn.commit()
        return {"ok": True, "scaled_qty": int(qty), "scaled_price": float(price)}
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
    Start a scan on a background thread and return immediately.

    It used to run synchronously, which was fine while it was seconds of
    pandas. Then the freshness check began fetching missing bars inside the
    request, and a 500-symbol backfill takes minutes — past the proxy
    timeout, surfacing as a 524 with no way to tell whether the scan had
    actually run.

    The summary is written to engine_settings and read back through
    /jobs/status, so nothing is lost by not returning it here.
    """
    require_internal_key(request)
    mode = request.query_params.get("mode", "postclose")

    with _job_lock:
        if _job_state["running"]:
            raise HTTPException(409, f"Job already running: {_job_state['name']}")
        _job_state.update(name=f"scan:{mode}", running=True,
                          started_at=datetime.now(IST).isoformat(),
                          finished_at=None, error=None)

    def run():
        from ingest import connect as _connect
        from scan import run_scan
        summary = run_scan(mode=mode)
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    insert into engine_settings (key, value, updated_at)
                    values ('last_scan_summary', %s::jsonb, now())
                    on conflict (key) do update set
                        value = excluded.value, updated_at = now()
                """, (json.dumps(summary),))
            conn.commit()
        finally:
            conn.close()
        return summary

    threading.Thread(target=_run_job, args=(f"scan:{mode}", run),
                     daemon=True).start()
    return {"ok": True, "started": True, "mode": mode}


@app.post("/jobs/fundamentals")
def fundamentals_job(request: Request):
    """
    Ingest promoter holding and quarterly P&L. Weekly cadence — the data
    changes once a quarter and it is ~1000 calls through NSE's fragile path.
    """
    require_internal_key(request)
    with _job_lock:
        if _job_state["running"]:
            raise HTTPException(409, f"Job already running: {_job_state['name']}")
        _job_state.update(name="fundamentals", running=True,
                          started_at=datetime.now(IST).isoformat(),
                          finished_at=None, error=None)

    def run():
        from fundamentals import sync_quarterly_results, sync_shareholding
        from ingest import _run_log, connect as _connect

        conn = _connect()
        try:
            for name, fn in (("sync_shareholding", sync_shareholding),
                             ("sync_quarterly_results", sync_quarterly_results)):
                try:
                    result = fn(conn)
                    _run_log(conn, name, "success", result.get("written", 0))
                    log.info("%s: %s", name, result)
                except Exception as exc:
                    conn.rollback()
                    _run_log(conn, name, "failed", 0, str(exc)[:500])
                    raise
        finally:
            conn.close()

    threading.Thread(target=_run_job, args=("fundamentals", run), daemon=True).start()
    return {"ok": True, "started": True}


@app.get("/jobs/fundamentals/probe")
def fundamentals_probe(request: Request, symbol: str = Query("RELIANCE")):
    """
    Fetch one symbol and report the real response shape.

    Field names on these NSE endpoints are undocumented. Running 500 symbols
    against guessed keys would write hundreds of null rows that look like
    real data — so confirm the shape on one symbol first.
    """
    require_internal_key(request)
    from fundamentals import probe
    try:
        return probe(symbol)
    except Exception as exc:
        raise HTTPException(500, f"Probe failed: {exc}")


@app.post("/jobs/delivery")
def delivery_job(request: Request):
    """
    Ingest delivery percentage. Additive — no gate, score or signal reads it.

    Also runs automatically in the post-close slot; this is for backfilling.
    """
    require_internal_key(request)
    days = int(request.query_params.get("days", "30"))
    # Explicit range for a genuine historical backfill — see sync_delivery's
    # docstring for why a bare `days` bump cannot do this once any delivery
    # data already exists.
    start_str = request.query_params.get("start")
    end_str = request.query_params.get("end")
    start_dt = datetime.strptime(start_str, "%Y-%m-%d").date() if start_str else None
    end_dt = datetime.strptime(end_str, "%Y-%m-%d").date() if end_str else None

    with _job_lock:
        if _job_state["running"]:
            raise HTTPException(409, f"Job already running: {_job_state['name']}")
        _job_state.update(name="delivery", running=True,
                          started_at=datetime.now(IST).isoformat(),
                          finished_at=None, error=None)

    def run():
        from ingest import _run_log, connect as _connect, sync_delivery
        from nse_client import NSEClient
        conn = _connect()
        try:
            written = sync_delivery(conn, NSEClient(), days=days,
                                    start_date=start_dt, end_date=end_dt)
            _run_log(conn, "sync_delivery", "success", written)
        finally:
            conn.close()

    threading.Thread(target=_run_job, args=("delivery", run), daemon=True).start()
    return {"ok": True, "started": True, "days": days,
            "start": start_str, "end": end_str}


@app.post("/jobs/backtest")
def backtest_job(request: Request):
    """
    Replay the pipeline over history and score the result.

    Long-running by nature — a few years across 500 symbols — so it runs on
    a background thread and writes to backtest_runs / backtest_trades.
    Nothing in the live path reads those tables.
    """
    require_internal_key(request)
    from datetime import date as _date

    years = float(request.query_params.get("years", "2"))
    step = int(request.query_params.get("step", "1"))
    to_date = _date.today()
    from_date = to_date - timedelta(days=int(365 * years))

    with _job_lock:
        if _job_state["running"]:
            raise HTTPException(409, f"Job already running: {_job_state['name']}")
        _job_state.update(name="backtest", running=True,
                          started_at=datetime.now(IST).isoformat(),
                          finished_at=None, error=None)

    def run():
        from backtest import run_backtest, summarise
        from ingest import connect as _connect

        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    insert into backtest_runs (from_date, to_date, status, params)
                    values (%s, %s, 'running', %s) returning id
                """, (from_date, to_date,
                      json.dumps({"years": years, "step": step})))
                run_id = cur.fetchone()[0]
            conn.commit()
        finally:
            conn.close()

        try:
            result = run_backtest(from_date, to_date, step=step)
            from backtest import (compare_base_strategies, compare_exits,
                                 diagnose_quality_quartile, simulate_portfolio,
                                 split_sample)
            metrics = summarise(result["trades"])
            metrics["split_sample"] = split_sample(result["trades"])
            metrics["quality_diagnosis_q4"] = diagnose_quality_quartile(
                result["trades"], "q4")
            metrics["base_strategy_comparison"] = compare_base_strategies(
                result["trades"], result.get("base_strategy_trades", {}))
            metrics["exit_variants"] = compare_exits(result["trades"])

            # Portfolio construction: concentrated book, strongest first.
            # Pre-specified combinations only — not a sweep.
            metrics["portfolio"] = {
                "all_signals_no_cap": simulate_portfolio(
                    result["trades"], max_positions=9999, min_rs=0),
                "top8_any_rs": simulate_portfolio(
                    result["trades"], max_positions=8, min_rs=0),
                "top8_rs80plus": simulate_portfolio(
                    result["trades"], max_positions=8, min_rs=80),
                "top5_rs80plus": simulate_portfolio(
                    result["trades"], max_positions=5, min_rs=80),
            }

            conn = _connect()
            try:
                with conn.cursor() as cur:
                    for t in result["trades"]:
                        cur.execute("""
                            insert into backtest_trades
                                (run_id, symbol, signal_date, setup_type, pattern,
                                 score_total, band, regime, entry_trigger, stop_loss,
                                 t1, t2, r_planned, entry_date, entry_price,
                                 exit_date, exit_price, exit_reason, r_realised,
                                 max_favourable_r, max_adverse_r, bars_held)
                            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                                    %s,%s,%s,%s,%s,%s,%s)
                        """, (run_id, t["symbol"], t["signal_date"], t["setup_type"],
                              t["pattern"], t["score_total"], t["band"], t["regime"],
                              t["entry_trigger"], t["stop_loss"], t["t1"], t.get("t2"),
                              t["r_planned"], t.get("entry_date"), t.get("entry_price"),
                              t.get("exit_date"), t.get("exit_price"),
                              t.get("exit_reason"), t.get("r_realised"),
                              t.get("max_favourable_r"), t.get("max_adverse_r"),
                              t.get("bars_held")))
                    cur.execute("""
                        update backtest_runs set finished_at = now(), status = 'success',
                               universe = %s, signals = %s, metrics = %s
                         where id = %s
                    """, (result["universe"], len(result["trades"]),
                          json.dumps(metrics), run_id))
                conn.commit()
            finally:
                conn.close()
            log.info("Backtest %s complete: %d signals", run_id, len(result["trades"]))
        except Exception as exc:
            conn = _connect()
            try:
                with conn.cursor() as cur:
                    cur.execute("update backtest_runs set finished_at = now(), "
                                "status = 'failed', error = %s where id = %s",
                                (str(exc)[:500], run_id))
                conn.commit()
            finally:
                conn.close()
            raise

    threading.Thread(target=_run_job, args=("backtest", run), daemon=True).start()
    return {"ok": True, "started": True, "from": str(from_date), "to": str(to_date)}


@app.get("/jobs/backtest")
def backtest_results(request: Request):
    """Latest backtest run with its metrics."""
    require_internal_key(request)
    from ingest import connect

    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                select id, started_at, finished_at, from_date, to_date,
                       universe, signals, status, metrics, error
                from backtest_runs order by id desc limit 1
            """)
            row = cur.fetchone()
            if not row:
                return {"run": None}
            cols = ["id","started_at","finished_at","from_date","to_date",
                    "universe","signals","status","metrics","error"]
            run = dict(zip(cols, row))
            for k in ("started_at","finished_at","from_date","to_date"):
                if run.get(k) is not None:
                    run[k] = str(run[k])
            run["id"] = int(run["id"])
    finally:
        conn.close()
    return {"run": run}


@app.get("/jobs/status")
def jobs_status(request: Request):
    require_internal_key(request)
    with _job_lock:
        current = dict(_job_state)

    runs, counts, last_scan_summary = [], {}, None
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
                cur.execute("select value from engine_settings "
                            "where key = 'last_scan_summary'")
                row = cur.fetchone()
                last_scan_summary = row[0] if row else None

                cur.execute("select (select count(*) from symbols), "
                            "(select count(*) from ohlcv_daily), "
                            "(select count(*) from corporate_actions), "
                            "(select count(*) from surveillance), "
                            "(select count(*) from shareholding), "
                            "(select count(*) from fundamentals_quarterly), "
                            "(select count(distinct symbol) from shareholding), "
                            "(select max(period_end) from fundamentals_quarterly), "
                            "(select max(trade_date) from ohlcv_daily), "
                            "(select max(as_of_date) from signals)")
                s, o, c, v, sh, fq, shs, latest, last_bar, last_scan = cur.fetchone()
                counts = {"symbols": s, "ohlcv_daily": o,
                          "corporate_actions": c, "surveillance": v,
                          "shareholding": sh, "fundamentals_quarterly": fq,
                          "fundamentals_symbols": shs,
                          "fundamentals_latest_period": str(latest) if latest else None,
                          "latest_bar_date": str(last_bar) if last_bar else None,
                          "last_scan_date": str(last_scan) if last_scan else None}
        finally:
            conn.close()
    except Exception as exc:
        counts = {"error": str(exc)[:200]}

    return {"current": current, "recent_runs": runs, "row_counts": counts,
            "last_scan_summary": last_scan_summary}


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
