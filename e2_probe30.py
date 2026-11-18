"""
e2_probe30.py — is 30-minute (and 15-minute) HISTORY served by Upstox, and how far back? (owner's order of 7 Nov 12:38 IST: open_range_breakout needs
the first 30-minute range; if Upstox provides it, that detector can be built; if not, the candidate is blocked.)

Two things, both logged with the prefix E2PRB|:
  1. the verdict the engine's own intraday probe stored earlier (engine_settings.intraday_probe_latest), if there is one;
  2. a fresh probe at 15 and 30 minutes over lookbacks out to ~4 years, when a valid Upstox token is present (the token is valid only after the owner's
     daily login; without one this says so and nothing is guessed).
Read-only apart from storing the fresh verdict under engine_settings.engine2_probe30.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("engine2.probe30")
KEY = "engine2_probe30"
LOG_PREFIX = "E2PRB|"
INTERVALS = (15, 30)
LOOKBACKS = (1, 7, 30, 90, 180, 365, 540, 730, 1095, 1460)


SAMPLE_KEY = "engine2_probe_sample"
SAMPLE_N = 15
SAMPLE_LOOKBACKS = (7, 90, 365, 730, 1095, 1460)
SPAN_DAYS = (30, 90)                 # one request over a window of this many calendar days: does the API accept it, and does it return every session?


def pick_sample(conn, n: int = SAMPLE_N) -> list[dict]:
    """n symbols spread evenly across the liquidity ranking (median traded value, last 20 sessions), from the most to the least liquid active symbol."""
    with conn.cursor() as cur:
        cur.execute("""select o.symbol, s.upstox_instrument_key, s.listing_date,
                              percentile_cont(0.5) within group (order by o.close * o.volume) as tv
                       from ohlcv_daily o join symbols s using (symbol)
                       where s.is_active and s.upstox_instrument_key is not null
                         and o.trade_date >= (select max(trade_date) from ohlcv_daily) - 28
                       group by o.symbol, s.upstox_instrument_key, s.listing_date
                       having count(*) >= 10 order by tv desc""")
        rows = cur.fetchall()
    if not rows:
        return []
    k = min(n, len(rows))
    idx = sorted({int(round(i * (len(rows) - 1) / max(k - 1, 1))) for i in range(k)})
    return [{"symbol": rows[i][0], "key": rows[i][1], "listing_date": rows[i][2], "liquidity_rank": i + 1, "of": len(rows)} for i in idx]


def _sessions_in(payload) -> int:
    try:
        return len({str(c[0])[:10] for c in (payload.get("data") or {}).get("candles") or []})
    except Exception:                                                                  # noqa: BLE001
        return 0


def run_sample(conn=None, trigger: str = "manual", client=None) -> dict:
    """15-minute history for a SAMPLE of symbols across the liquidity range (owner, 18 Nov): is coverage broad enough to backfill the universe?
    Also times the requests, tests a 30- and 90-day span in one request, and prices a full backfill against the client's own rate limits."""
    from datetime import date, timedelta
    from ingest import connect
    import data_probe
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running"}
    try:
        if client is None:
            from upstox_client import TokenStore, UpstoxClient
            if TokenStore().valid_token() is None:
                out["ran"], out["reason"] = False, "no valid Upstox token (the daily login has not happened since the 03:30 IST expiry)"
                out["state"] = "done"
                client = None
            else:
                client = UpstoxClient()
        if client is not None:
            today = date.today()
            dates = data_probe.pick_probe_dates(conn, today, SAMPLE_LOOKBACKS)
            sample = pick_sample(conn)
            out["ran"], out["lookback_dates"], out["sample"] = True, [str(d) for d in dates], []
            secs = []
            for sm in sample:
                cells = []
                for d in dates:
                    if sm["listing_date"] and d < sm["listing_date"]:
                        cells.append({"date": str(d), "status": "not_listed_yet"})
                        continue
                    t1 = time.time()
                    try:
                        payload = client._get(data_probe.intraday_path(sm["key"], 15, d))
                        cell = data_probe.summarise_candles(payload, 15)
                    except Exception as exc:                                              # noqa: BLE001
                        if type(exc).__name__ == "TokenExpired":
                            raise
                        cell = {"ok": False, "candles": 0, "error": str(exc)[:120]}
                    secs.append(time.time() - t1)
                    cells.append({"date": str(d), "status": "ok" if cell.get("ok") else "short", "candles": cell.get("candles"), "expected": cell.get("expected"),
                                  "error": cell.get("error")})
                sm2 = {"symbol": sm["symbol"], "liquidity_rank": f"{sm['liquidity_rank']}/{sm['of']}", "cells": cells}
                tested = [c for c in cells if c["status"] != "not_listed_yet"]
                sm2["full_days"], sm2["tested_days"] = sum(c["status"] == "ok" for c in tested), len(tested)
                out["sample"].append(sm2)
            span = []
            for sm in sample[:: max(1, len(sample) // 4)][:4]:                              # four symbols across the range
                for days in SPAN_DAYS:
                    end = dates[0] if dates else today
                    t1 = time.time()
                    try:
                        payload = client._get(f"/v3/historical-candle/{sm['key']}/minutes/15/{end.isoformat()}/{(end - timedelta(days=days - 1)).isoformat()}")
                        status = payload.get("status") if isinstance(payload, dict) else "non-dict"
                        span.append({"symbol": sm["symbol"], "calendar_days": days, "status": status, "sessions_returned": _sessions_in(payload),
                                     "error": None if status == "success" else str(payload.get("errors") or payload.get("message"))[:160]})
                    except Exception as exc:                                                  # noqa: BLE001
                        if type(exc).__name__ == "TokenExpired":
                            raise
                        span.append({"symbol": sm["symbol"], "calendar_days": days, "status": "error", "error": str(exc)[:160]})
                    secs.append(time.time() - t1)
            out["span_tests"] = span
            tested = sum(s["tested_days"] for s in out["sample"])
            full = sum(s["full_days"] for s in out["sample"])
            out["coverage"] = {"symbol_dates_tested": tested, "full": full, "share_full": round(full / tested, 3) if tested else None,
                               "symbols_fully_covered": sum(1 for s in out["sample"] if s["tested_days"] and s["full_days"] == s["tested_days"]), "symbols": len(out["sample"])}
            ok_span = [x for x in span if x.get("status") == "success" and x.get("sessions_returned", 0) >= 15]
            best_days = max([x["calendar_days"] for x in ok_span], default=1)
            from upstox_client import REQ_PER_30MIN, REQ_PER_DAY
            universe = 497
            per_symbol = -(-1460 // best_days)
            reqs = universe * per_symbol
            out["backfill_estimate"] = {"universe": universe, "years": 4, "request_span_days": best_days, "requests_per_symbol": per_symbol, "requests_total": reqs,
                                        "limit_per_30min": REQ_PER_30MIN, "limit_per_day": REQ_PER_DAY, "days_at_daily_limit": round(reqs / REQ_PER_DAY, 1),
                                        "mean_request_seconds": round(sum(secs) / len(secs), 2) if secs else None,
                                        "rows_if_stored": universe * 1008 * 25, "note": "rows = 497 symbols x 4 years x 25 fifteen-minute bars a session; storage is a new build"}
            out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("15-minute sample probe failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (SAMPLE_KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the sample probe")
    try:
        for line in render_sample(out).splitlines():
            for i in range(0, max(len(line), 1), 1400):
                log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
    finally:
        if own:
            conn.close()
    return out


def render_sample(d: dict) -> str:
    L = [f"# 15-minute sample probe  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    if not d.get("ran"):
        L.append(f"sample probe ran: False reason: {d.get('reason')}")
        return "\n".join(L)
    L.append(f"sample probe ran: True; lookback dates {d.get('lookback_dates')}")
    for s in d.get("sample") or []:
        L.append(f"{s['symbol']} (liquidity rank {s['liquidity_rank']}): full {s['full_days']}/{s['tested_days']}  " +
                 "; ".join(f"{c['date']} {c['status']}" + (f" n{c.get('candles')}/{c.get('expected')}" if c.get("candles") is not None else "") + (f" ({c['error']})" if c.get("error") else "")
                           for c in s["cells"] if c["status"] != "ok"))
    L.append("coverage: " + json.dumps(d.get("coverage")))
    L.append("span tests (one request over N calendar days): " + json.dumps(d.get("span_tests"), default=str))
    L.append("backfill estimate: " + json.dumps(d.get("backfill_estimate")))
    return "\n".join(L)


def stored_verdict(conn) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("select value, updated_at from engine_settings where key = 'intraday_probe_latest'")
        row = cur.fetchone()
    if not row or not isinstance(row[0], dict):
        return None
    v = row[0]
    return {"stored_at": str(row[1]), "symbol": v.get("symbol"), "probed_at": v.get("probed_at"), "verdict": v.get("verdict"),
            "error": v.get("error"), "dates_probed": v.get("dates_probed")}


def run(conn=None, trigger: str = "manual", client=None) -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running"}
    try:
        out["stored_probe"] = stored_verdict(conn)
        try:
            if client is None:
                from upstox_client import TokenStore, UpstoxClient
                if TokenStore().valid_token() is None:
                    out["fresh_probe"] = {"ran": False, "reason": "no valid Upstox token (the daily login has not happened since the 03:30 IST expiry)"}
                    client = None
                else:
                    client = UpstoxClient()
            if client is not None:
                import data_probe
                res = data_probe.probe_intraday(client, conn, "RELIANCE", INTERVALS, lookbacks=LOOKBACKS)
                out["fresh_probe"] = {"ran": True, "symbol": res.get("symbol"), "error": res.get("error"), "verdict": res.get("verdict"),
                                      "cells": res.get("cells")}
        except Exception as e:                                                         # noqa: BLE001
            log.exception("fresh intraday probe failed")
            out["fresh_probe"] = {"ran": False, "reason": f"{type(e).__name__}: {e}"[:300]}
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("probe30 failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the probe30 result")
    try:
        for line in render(out).splitlines():
            for i in range(0, max(len(line), 1), 1400):
                log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
    finally:
        if own:
            conn.close()
    return out


def render(d: dict) -> str:
    L = [f"# Intraday history probe  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    L.append("stored probe (engine_settings.intraday_probe_latest): " + json.dumps(d.get("stored_probe"), default=str))
    fp = d.get("fresh_probe") or {}
    L.append(f"fresh probe ran: {fp.get('ran')}" + (f" reason: {fp.get('reason')}" if fp.get("reason") else ""))
    if fp.get("ran"):
        L.append("verdict: " + json.dumps(fp.get("verdict"), default=str))
        for itv, rows in (fp.get("cells") or {}).items():
            L.append(f"{itv}-minute: " + "; ".join(f"{r['date']} n{r.get('candles')}/{r.get('expected')}" + ("" if r.get("ok") else f" ({r.get('error') or 'partial'})") for r in rows))
    return "\n".join(L)
