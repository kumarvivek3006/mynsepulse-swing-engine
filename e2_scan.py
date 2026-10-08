"""
e2_scan.py — the unattended Engine 2 scan. One implementation behind the three scheduled slots.

  premarket  08:00 IST  refresh surveillance / corporate actions, make sure the prior session's bars
                        are in, run all 20 detectors on the COMPLETED prior session -> armed watchlist
  intraday   hourly     10:10..15:10 IST: today's FORMING bar for the whole universe (+ index, VIX),
                        all 20 detectors. Provisional: never enters the forward record
  postclose  18:30 IST  waits for the day's bars (the live postclose loads them), then the definitive
                        scan on the completed daily bar -> tomorrow's watchlist

It runs in a child process (see e2_service.run_scan_slot) so a crash or an out-of-memory kill cannot
take the web service down. It records every run — success, skip or failure — in two places:

  engine2_last_scan_summary   the latest run, any mode, with full diagnostics
  engine2_scan_runs           the last RUNS_KEEP runs, compact

SHADOW: nothing here writes to `signals`. The live scan engine is untouched. `publishing` in the
summary says so on every run; publishing is the cutover step, not a scheduled-scan step.

FAIL LOUD
  token invalid           -> reason token_invalid, before any Upstox call
  bars behind             -> reason bars_not_current (after waiting / one self-refresh)
  forming bars too few    -> reason intraday_coverage_low
  index bar missing       -> reason index_intraday_unavailable (without it the calendar cannot move)
  a detector that throws  -> counted and named in detector_errors, logged at ERROR, the rest continue
  zero candidates         -> a VALID result (a defensive market is allowed to have none); recorded as such

Exit code: 0 for success or a deliberate skip, 2 for a failed slot. The last stdout line is
`E2SCAN_RESULT <json>`; everything else goes to stderr / the Railway log.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone

log = logging.getLogger("engine2.scan")

IST = timezone(timedelta(hours=5, minutes=30))
SUMMARY_KEY, RUNS_KEY = "engine2_last_scan_summary", "engine2_scan_runs"
RUNS_KEEP = 80
MODES = ("premarket", "intraday", "postclose")

FRESH_MIN_PCT = float(os.environ.get("E2_FRESH_MIN_PCT", "90"))              # share of the universe with the expected bar
INTRADAY_MIN_COVERAGE = float(os.environ.get("E2_INTRADAY_MIN_COVERAGE", "0.80"))
POSTCLOSE_WAIT_MIN = float(os.environ.get("E2_POSTCLOSE_WAIT_MIN", "20"))   # for the live postclose to load bars
POLL_SEC = float(os.environ.get("E2_POLL_SEC", "45"))
INDEX_FALLBACK_KEYS = ("NIFTY500", "NIFTY50")


class ScanFailed(RuntimeError):
    """A slot that could not do its job. `reason` is the specific, stable code the operator reads."""

    def __init__(self, reason: str, detail: dict | None = None):
        super().__init__(reason)
        self.reason, self.detail = reason, detail or {}


# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------
def _get(conn, key):
    with conn.cursor() as cur:
        cur.execute("select value from engine_settings where key = %s", (key,))
        r = cur.fetchone()
    return r[0] if r else None


def _put(conn, key, value) -> None:
    with conn.cursor() as cur:
        cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (key, json.dumps(value, default=str)))
    conn.commit()


def now_ist() -> datetime:
    return datetime.now(IST)


def compact(summary: dict) -> dict:
    """The run without its bulky lists — what goes in the ring buffer and in _last_runs."""
    return {k: v for k, v in summary.items() if k not in ("watchlist", "detector_error_samples")}


def record(conn, summary: dict) -> None:
    """Write the run. A recording failure is logged loudly and never hides the run's own outcome."""
    try:
        if summary.get("status") not in ("skipped_holiday", "skipped_muhurat_intraday"):
            _put(conn, SUMMARY_KEY, summary)             # a skip must not overwrite the last real scan
        runs = list(_get(conn, RUNS_KEY) or [])
        runs.append(compact(summary))
        _put(conn, RUNS_KEY, runs[-RUNS_KEEP:])
    except Exception:                                    # noqa: BLE001
        log.exception("Could not persist the engine-2 scan summary (the scan outcome stands)")
        try:
            conn.rollback()
        except Exception:
            pass


def _standalone(summary: dict) -> None:
    from ingest import connect
    conn = connect()
    try:
        record(conn, summary)
    finally:
        conn.close()


def record_failure(mode: str, reason: str, detail: dict | None = None, trigger: str = "scheduled") -> dict:
    """For callers that fail before (or without) the child process: same record, same shape."""
    s = {"engine": "engine2", "mode": mode, "slot": f"engine2_{mode}", "trigger": trigger, "status": "failed",
         "reason": reason, "detail": detail or {}, "finished_at": now_ist().isoformat(timespec="seconds"),
         "publishing": _publishing(0)}
    log.error("Engine 2 %s slot FAILED: %s %s", mode, reason, detail or "")
    try:
        _standalone(s)
    except Exception:                                    # noqa: BLE001
        log.exception("Could not record the %s failure", mode)
    return s


def record_skip(mode: str, result: dict, trigger: str = "scheduled") -> dict:
    s = {"engine": "engine2", "mode": mode, "slot": f"engine2_{mode}", "trigger": trigger,
         "status": result.get("status", "skipped"), "reason": result.get("reason"),
         "finished_at": now_ist().isoformat(timespec="seconds"), "publishing": _publishing(0)}
    try:
        _standalone(s)
    except Exception:                                    # noqa: BLE001
        log.exception("Could not record the %s skip", mode)
    return s


def _publishing(would_publish: int, pub: dict | None = None) -> dict:
    import engine_mode
    mode = engine_mode.current()
    if mode == engine_mode.ENGINE2 and pub is not None:
        return {"engine_mode": mode, "channel": "signals table + last_scan_summary",
                "published": pub.get("inserted", 0) + pub.get("updated", 0), "would_publish": int(would_publish), **pub}
    if mode == engine_mode.ENGINE2:
        return {"engine_mode": mode, "channel": "signals table + last_scan_summary", "published": 0, "would_publish": int(would_publish),
                "note": "Engine 2 is the publishing engine, and this slot did not reach the publish step."}
    return {"engine_mode": mode, "channel": "none", "published": 0, "would_publish": int(would_publish),
            "note": "SHADOW: ENGINE_MODE=legacy, so the original engine publishes and Engine 2 only records."}


# ----------------------------------------------------------------------
# data freshness
# ----------------------------------------------------------------------
def _token_ok() -> bool:
    from upstox_client import TokenStore
    return TokenStore().valid_token() is not None


def expected_session(conn) -> date:
    """Most recent session whose CLOSING bar must exist — holiday-aware, evaluated on the IST clock."""
    from market_calendar import expected_last_session
    return expected_last_session(conn, now_ist().replace(tzinfo=None))


def freshness(conn, symbols: list[str], expected: date) -> dict:
    with conn.cursor() as cur:
        cur.execute("select count(*) filter (where d >= %s) from "
                    "(select symbol, max(trade_date) d from ohlcv_daily where symbol = any(%s) group by symbol) t",
                    (expected, symbols))
        have = int(cur.fetchone()[0])
        cur.execute("select max(trade_date) from ohlcv_daily where symbol = 'NIFTY50'")
        ix = cur.fetchone()[0]
    pct = round(100.0 * have / max(len(symbols), 1), 1)
    return {"expected_session": str(expected), "index_latest": str(ix) if ix else None, "symbols_current": have,
            "universe": len(symbols), "pct_current": pct,
            "ok": bool(ix is not None and ix >= expected and pct >= FRESH_MIN_PCT)}


def ensure_fresh(conn, mode: str, symbols: list[str], notes: dict) -> dict:
    """
    Bars through the expected session are in, or this raises with the specific reason.
    No Upstox call is made unless bars are actually behind AND the token is valid.
    """
    expected = expected_session(conn)
    fr = freshness(conn, symbols, expected)
    notes["freshness"] = {"initial": fr}
    if fr["ok"]:
        return fr

    if not _token_ok():                                  # nobody can fetch the missing bars: stop now, do not wait
        raise ScanFailed("token_invalid", {"freshness": fr, "action": "Log in to Upstox; the bars cannot be refreshed."})

    if mode == "postclose":                              # the live postclose slot is loading these at the same time
        t0, deadline = time.time(), time.time() + POSTCLOSE_WAIT_MIN * 60
        while time.time() < deadline:
            time.sleep(POLL_SEC)
            fr = freshness(conn, symbols, expected)
            if fr["ok"]:
                notes["freshness"].update(waited_min=round((time.time() - t0) / 60, 1), final=fr)
                return fr
        notes["freshness"]["waited_min"] = round((time.time() - t0) / 60, 1)

    import scan as live_scan                              # incremental and idempotent: fetches only what is behind
    try:
        notes["freshness"]["self_refresh"] = live_scan.ensure_bars_current(conn)
    except Exception as e:                               # noqa: BLE001
        notes["freshness"]["self_refresh"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    fr = freshness(conn, symbols, expected)
    notes["freshness"]["final"] = fr
    if not fr["ok"]:
        raise ScanFailed("bars_not_current", {"freshness": fr})
    return fr


def refresh_overnight_inputs(notes: dict) -> None:
    """Premarket only: surveillance and corporate actions move overnight and both can disqualify a setup."""
    from ingest import connect, sync_corporate_actions, sync_surveillance
    from nse_client import NSEClient
    nse, conn, out = NSEClient(), connect(), {}
    try:
        for name, fn in (("surveillance", lambda: sync_surveillance(conn, nse)),
                         ("corporate_actions", lambda: sync_corporate_actions(conn, nse, years=1))):
            try:
                out[name] = {"ok": True, "rows": fn()}
            except Exception as e:                       # noqa: BLE001 — stale inputs are reported, the scan still runs
                try:
                    conn.rollback()
                except Exception:
                    pass
                log.error("Engine 2 premarket: %s refresh failed: %s", name, e)
                out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    finally:
        conn.close()
    notes["overnight_refresh"] = out


# ----------------------------------------------------------------------
# intraday: today's forming bar for the whole universe
# ----------------------------------------------------------------------
def bar_from_candles(candles: list, today: date) -> dict | None:
    """Upstox returns the session's candles newest-first. None when they are not today's."""
    if not candles:
        return None
    if str(candles[0][0])[:10] != today.isoformat() or str(candles[-1][0])[:10] != today.isoformat():
        return None
    return {"open": float(candles[-1][1]), "high": max(float(c[2]) for c in candles),
            "low": min(float(c[3]) for c in candles), "close": float(candles[0][4]),
            "volume": float(sum(int(c[5] or 0) for c in candles))}


def attach_forming_bars(conn, data, today: date, notes: dict) -> None:
    """Append today's partial bar to every symbol, the index and the VIX. Volume is NOT projected."""
    import pandas as pd
    from upstox_client import TokenExpired, UpstoxClient

    syms = sorted(data.frames)
    want = list(syms) + [data.index_symbol or "NIFTY50", "INDIAVIX"]
    with conn.cursor() as cur:
        cur.execute("select symbol, upstox_instrument_key from symbols where symbol = any(%s)", (want,))
        keys = {s: k for s, k in cur.fetchall() if k}
    client = UpstoxClient()
    ts = pd.Timestamp(today)
    t0 = time.time()

    def fetch(sym: str):
        k = keys.get(sym)
        if not k:
            return None, "no_instrument_key"
        try:
            return bar_from_candles(client.intraday_today(k), today), None
        except TokenExpired:
            raise ScanFailed("token_invalid", {"during": "forming-bar fetch", "fetched_before_failure": fetched})
        except Exception as e:                           # noqa: BLE001 — one symbol failing is counted, not fatal
            return None, f"{type(e).__name__}: {e}"[:120]

    fetched, no_bar, errors = 0, 0, {}
    for sym in syms:
        bar, err = fetch(sym)
        if err:
            errors.setdefault(err, []).append(sym)
            continue
        if bar is None:
            no_bar += 1
            continue
        df = data.frames[sym]
        if len(df) and pd.Timestamp(df["trade_date"].iloc[-1]).normalize() == ts:
            df = df.iloc[:-1]
        row = pd.DataFrame([{"trade_date": ts, **bar}])
        data.frames[sym] = pd.concat([df, row], ignore_index=True)
        fetched += 1

    coverage = fetched / max(len(syms), 1)
    notes["forming_bars"] = {"fetched": fetched, "no_bar_yet": no_bar, "universe": len(syms), "coverage": round(coverage, 3),
                             "fetch_errors": {k: {"n": len(v), "e.g.": v[:3]} for k, v in errors.items()},
                             "fetch_sec": round(time.time() - t0, 1),
                             "volume": "partial session volume, not projected"}
    if coverage < INTRADAY_MIN_COVERAGE:
        raise ScanFailed("intraday_coverage_low", notes["forming_bars"])

    # the index decides the calendar: without today's index bar no symbol can be 'current'
    ix_sym = data.index_symbol or "NIFTY50"
    ix_bar, ix_err = fetch(ix_sym)
    if ix_bar is None:
        raise ScanFailed("index_intraday_unavailable", {"index": ix_sym, "error": ix_err or "no candles for today"})
    s = data.index_close.copy()
    s.loc[ts] = ix_bar["close"]
    data.index_close = s.sort_index()
    notes["forming_bars"]["index"] = {"symbol": ix_sym, "close": ix_bar["close"]}

    vx_bar, vx_err = fetch("INDIAVIX")
    if vx_bar is not None and data.vix is not None:
        v = data.vix.copy()
        v.loc[ts] = vx_bar["close"]
        data.vix = v.sort_index()
        notes["forming_bars"]["vix"] = {"close": vx_bar["close"]}
    else:                                                # regime carries yesterday's VIX forward; say so
        notes["forming_bars"]["vix"] = {"carried_forward": True, "error": vx_err}


# ----------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------
def _validation_running(conn) -> bool:
    try:
        pr = _get(conn, "engine2_progress") or {}
        return pr.get("state") == "running"
    except Exception:                                    # noqa: BLE001
        return False


def build_summary(mode: str, out: dict, notes: dict, started: datetime, trigger: str, pub: dict | None = None) -> dict:
    import e2_backtest as BT
    from e2_detectors import DETECTORS
    raw = out.get("candidates_raw_by_detector") or {}
    errs = out.get("detector_errors") or {}
    reg = out.get("regime") or {}
    picks = out.get("picks") or []
    s = {
        "engine": "engine2", "code_version": BT.CODE_VERSION, "mode": mode, "slot": f"engine2_{mode}", "trigger": trigger,
        "status": "success", "reason": None, "provisional": bool(out.get("provisional")),
        "as_of": out.get("as_of"), "started_at": started.isoformat(timespec="seconds"),
        "finished_at": now_ist().isoformat(timespec="seconds"),
        "duration_sec": round((now_ist() - started).total_seconds(), 1),
        "universe": (out.get("universe") or {}).get("scanned", out.get("universe_symbols")),
        "universe_detail": out.get("universe"),
        "symbols_scanned": out.get("universe_symbols", 0) - out.get("stale_symbols", 0) - out.get("symbols_too_short", 0),
        "stale_symbols": out.get("stale_symbols"), "symbols_too_short": out.get("symbols_too_short"),
        "detectors_run": out.get("detectors_run"),
        "detector_candidates": {d: int(raw.get(d, 0)) for d in DETECTORS},          # every detector listed, zeros included
        "detectors_firing": sum(1 for d in DETECTORS if raw.get(d, 0)),
        "detector_errors": errs, "detector_error_samples": out.get("detector_error_samples") or {},
        "degraded": bool(errs),
        "gated": out.get("gated"), "candidates": out.get("candidates", 0),
        "signals": out.get("selected", 0), "not_selected_reasons": out.get("not_selected_reasons"),
        "regime": {k: reg.get(k) for k in ("state", "score", "size_mult", "max_positions", "breadth", "vix")},
        "validation_running_concurrently": notes.pop("validation_running", False),
        "publishing": _publishing(out.get("selected", 0), pub),
        "watchlist": [{"symbol": p["symbol"], "detector": p["detector"], "score": p["score"], "pivot": p["pivot"],
                       "entry_ref": p["plan"]["entry_ref"], "stop": p["plan"]["stop"], "t1": p["plan"]["t1"],
                       "t2": p["plan"]["t2"], "shares": p["plan"]["shares"], "sector": p["sector"]} for p in picks[:25]],
        **notes,
    }
    if not out.get("candidates"):
        s["note"] = (f"Zero candidates in a {reg.get('state')} regime across {s['detectors_run']} detectors — "
                     "a valid result, recorded, not an alarm.")
    return s


def run(mode: str, trigger: str = "scheduled") -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    from ingest import connect, kill_switch_active
    started, notes = now_ist(), {}
    conn = connect()
    try:
        try:
            if kill_switch_active(conn):
                s = {"engine": "engine2", "mode": mode, "slot": f"engine2_{mode}", "trigger": trigger,
                     "status": "skipped_kill_switch", "reason": "kill_switch_active", "publishing": _publishing(0),
                     "finished_at": now_ist().isoformat(timespec="seconds")}
                record(conn, s)
                return s

            import e2_backtest as BT
            import e2_live
            from universe import load_scan_universe

            uni = load_scan_universe(conn)
            notes["validation_running"] = _validation_running(conn)
            if mode == "premarket":
                refresh_overnight_inputs(notes)
            if mode == "intraday":
                if not _token_ok():                        # before any Upstox call
                    raise ScanFailed("token_invalid", {"action": "Log in to Upstox (the token expires 03:30 IST daily)."})
                ensure = expected_session(conn)            # yesterday's close must be in; today's bar is the forming one
                fr = freshness(conn, uni.symbols, ensure)
                notes["freshness"] = {"initial": fr}
                if not fr["ok"]:
                    ensure_fresh(conn, mode, uni.symbols, notes)
            else:
                ensure_fresh(conn, mode, uni.symbols, notes)

            data = BT.load_from_db(conn, window_bars=60)
            today = now_ist().date()
            if mode == "intraday":
                attach_forming_bars(conn, data, today, notes)

            out = e2_live.scan(conn, mode, data=data)

            want = today if mode == "intraday" else expected_session(conn)
            behind = out.get("as_of") != str(want) if mode == "intraday" else str(out.get("as_of")) < str(want)
            if behind:                                    # data AHEAD of the expected session (a late manual run) is fine
                raise ScanFailed("scan_not_on_expected_session", {"scanned_as_of": out.get("as_of"), "expected": str(want)})
            errs = out.get("detector_errors") or {}
            for name, n in errs.items():
                log.error("Engine 2 %s: detector %s threw on %d symbols: %s", mode, name, n,
                          (out.get("detector_error_samples") or {}).get(name))
            if errs and len(errs) >= int(out.get("detectors_run") or 0):
                raise ScanFailed("all_detectors_failed", {"detector_errors": errs})

            import engine_mode
            pub = None
            if engine_mode.is_engine2():                  # CUTOVER: Engine 2 publishes; legacy mode never reaches this
                import e2_publish
                try:
                    pub = e2_publish.publish(conn, out, mode, now_ist().date())
                except Exception as pe:                   # noqa: BLE001 — nothing was written (one transaction)
                    raise ScanFailed("publish_failed", {"error": f"{type(pe).__name__}: {pe}"[:300]})
                if pub.get("aborted"):
                    raise ScanFailed(pub["aborted"], {"note": "kill switch active: nothing was published"})

            s = build_summary(mode, out, notes, started, trigger, pub)
            record(conn, s)
            if pub is not None:                           # the key the UI reads (/jobs/status); engine2_last_scan_summary stays separate
                import scan as live_scan
                live_scan.store_summary(conn, e2_publish.ui_summary(s, out, pub))
            log.info("Engine 2 %s scan ok: as_of %s, universe %s, %d/%d detectors fired, %d candidates, %d selected%s",
                     mode, s["as_of"], s["universe"], s["detectors_firing"], s["detectors_run"], s["candidates"], s["signals"],
                     f", {len(errs)} detector(s) errored" if errs else "")
            return s
        except ScanFailed as f:
            try:
                conn.rollback()
            except Exception:
                pass
            s = {"engine": "engine2", "mode": mode, "slot": f"engine2_{mode}", "trigger": trigger, "status": "failed",
                 "reason": f.reason, "detail": f.detail, **notes, "started_at": started.isoformat(timespec="seconds"),
                 "finished_at": now_ist().isoformat(timespec="seconds"), "publishing": _publishing(0)}
            s.pop("validation_running", None)
            log.error("Engine 2 %s slot FAILED: %s %s", mode, f.reason, json.dumps(f.detail, default=str)[:400])
            record(conn, s)
            return s
        except Exception as e:                           # noqa: BLE001 — anything else is a failed slot with its own reason
            try:
                conn.rollback()
            except Exception:
                pass
            log.exception("Engine 2 %s slot crashed", mode)
            s = {"engine": "engine2", "mode": mode, "slot": f"engine2_{mode}", "trigger": trigger, "status": "failed",
                 "reason": f"exception: {type(e).__name__}: {e}"[:300], **notes,
                 "started_at": started.isoformat(timespec="seconds"),
                 "finished_at": now_ist().isoformat(timespec="seconds"), "publishing": _publishing(0)}
            s.pop("validation_running", None)
            record(conn, s)
            return s
    finally:
        conn.close()


if __name__ == "__main__":                               # the child process
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mode = sys.argv[1] if len(sys.argv) > 1 else "postclose"
    trig = sys.argv[2] if len(sys.argv) > 2 else "cli"
    result = run(mode, trig)
    print("E2SCAN_RESULT " + json.dumps(compact(result), default=str), flush=True)
    sys.exit(2 if result.get("status") == "failed" else 0)
