"""
e2_t1cov.py — TIER 1 DATA COVERAGE (owner's order of 26 Nov 00:34 IST). READ-ONLY. Nothing here scores, gates or publishes.

Before the four score modifiers (catalyst, delivery %, sector RS, volume confirmation) are built, this answers what the database can
actually support over the 3-year backtest window:
  delivery_daily   rows, symbols, first/last date, rows per year, share of ohlcv symbol-days that have a delivery %, null share
  events_calendar  'results' rows, symbols, first/last date, rows per year, past vs future dates (a backtest needs PAST dates)
  symbols          how many of the universe map to one of the 11 sectors (e2_sector.sector_of), and the count per sector
  ohlcv_daily      first/last date and symbols (the window the modifiers would be tested over)
Output: one E2T1COV| log line per section (JSON), plus the same dict stored in engine_settings key e2_t1cov_latest.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("engine2.t1cov")
LOG_PREFIX = "E2T1COV|"
KEY = "e2_t1cov_latest"


def _one(cur, sql, params=None):
    cur.execute(sql, params or ())
    return cur.fetchone()


def _section(fn, cur, conn):
    try:
        return fn(cur)
    except Exception as e:                                          # noqa: BLE001  a missing table is a coverage fact, not a crash
        conn.rollback()
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _ohlcv(cur):
    r = _one(cur, "select min(trade_date), max(trade_date), count(distinct symbol), count(*) from ohlcv_daily "
                  "where symbol not in ('NIFTY50','NIFTY500','INDIAVIX')")
    return {"first": str(r[0]), "last": str(r[1]), "symbols": r[2], "rows": r[3]}


def _delivery(cur):
    r = _one(cur, "select min(trade_date), max(trade_date), count(distinct symbol), count(distinct trade_date), count(*), "
                  "count(*) filter (where delivery_pct is null) from delivery_daily")
    out = {"first": str(r[0]), "last": str(r[1]), "symbols": r[2], "dates": r[3], "rows": r[4], "null_pct_rows": r[5]}
    cur.execute("select extract(year from trade_date)::int, count(*), count(distinct symbol) from delivery_daily group by 1 order by 1")
    out["by_year"] = {str(y): {"rows": n, "symbols": s} for y, n, s in cur.fetchall()}
    r = _one(cur, "select count(*) from ohlcv_daily o where o.symbol not in ('NIFTY50','NIFTY500','INDIAVIX') "
                  "and o.trade_date >= (select min(trade_date) from delivery_daily)")
    covered = _one(cur, "select count(*) from ohlcv_daily o join delivery_daily d on d.symbol = o.symbol and d.trade_date = o.trade_date "
                        "where d.delivery_pct is not null")
    out["ohlcv_symbol_days_since_first_delivery"] = r[0]
    out["of_which_with_delivery_pct"] = covered[0]
    out["coverage"] = round(covered[0] / r[0], 3) if r[0] else None
    r3 = _one(cur, "select count(*) from ohlcv_daily where symbol not in ('NIFTY50','NIFTY500','INDIAVIX') "
                   "and trade_date >= (select max(trade_date) from ohlcv_daily) - interval '3 years'")
    c3 = _one(cur, "select count(*) from ohlcv_daily o join delivery_daily d on d.symbol = o.symbol and d.trade_date = o.trade_date "
                   "where d.delivery_pct is not null and o.trade_date >= (select max(trade_date) from ohlcv_daily) - interval '3 years'")
    out["coverage_last_3y"] = round(c3[0] / r3[0], 3) if r3[0] else None
    return out


def _events(cur):
    r = _one(cur, "select min(event_date), max(event_date), count(distinct symbol), count(*), "
                  "count(*) filter (where event_date < current_date), count(*) filter (where event_date >= current_date) "
                  "from events_calendar where event_type = 'results'")
    out = {"first": str(r[0]), "last": str(r[1]), "symbols": r[2], "rows": r[3], "past_rows": r[4], "future_rows": r[5]}
    cur.execute("select extract(year from event_date)::int, count(*), count(distinct symbol) from events_calendar "
                "where event_type = 'results' group by 1 order by 1")
    out["by_year"] = {str(y): {"rows": n, "symbols": s} for y, n, s in cur.fetchall()}
    cur.execute("select event_type, count(*) from events_calendar group by 1 order by 1")
    out["all_event_types"] = {t: n for t, n in cur.fetchall()}
    return out


def _sectors(cur):
    import e2_sector
    cur.execute("select symbol, industry, sector from symbols where is_active and coalesce(series,'') <> 'INDEX'")
    rows = cur.fetchall()
    counts: dict[str, int] = {}
    unmapped = []
    for sym, ind, sec in rows:
        s = e2_sector.sector_of(ind, sec)
        if s is None:
            unmapped.append(sym)
        else:
            counts[s] = counts.get(s, 0) + 1
    return {"active_symbols": len(rows), "mapped": len(rows) - len(unmapped), "unmapped": len(unmapped),
            "per_sector": counts, "unmapped_sample": unmapped[:20]}


def compute(conn) -> dict:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "note": "Read-only coverage probe; nothing scored or changed."}
    with conn.cursor() as cur:
        for name, fn in (("ohlcv_daily", _ohlcv), ("delivery_daily", _delivery), ("events_results", _events), ("sectors", _sectors)):
            out[name] = _section(fn, cur, conn)
    return out


def log_sections(out: dict) -> int:
    n = 0
    for k, v in out.items():
        line = json.dumps({k: v}, default=str)
        for i in range(0, len(line), 1400):
            log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
            n += 1
    return n


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"state": "running", "trigger": trigger}
    try:
        out.update(compute(conn))
        out["state"] = "done"
    except Exception as e:                                          # noqa: BLE001
        log.exception("tier 1 coverage probe failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                               # noqa: BLE001
        conn.rollback()
        log.exception("could not store the tier 1 coverage probe")
    try:
        log_sections(out)
    finally:
        if own:
            conn.close()
    return out
