"""
data_quality.py — Step 1.6: make bad data VISIBLE before it becomes a signal.

Checks over the scan universe:

  stale      a symbol whose newest bar is older than the last session the
             exchange calendar says should exist (counted in SESSIONS, so a
             weekend or holiday is never "stale")
  gaps       sessions inside a symbol's own history with no bar. A "session" is
             derived from the DATA (a date on which at least half the live
             universe has a bar), not from the holiday table, because that table
             holds the current year only; using it would flag every historical
             holiday as a missing bar.
  bad ticks  a bar that cannot be real: high < low, open/close outside
             [low, high], a non-positive price, negative volume, or an unusable
             adjustment factor (NULL, zero or negative). The adj_* columns are
             GENERATED as raw * adj_factor, so a separate range check on them
             would be redundant; the factor is the one adjustment failure that
             can silently zero or null every adjusted price the engine reads.
  fundamentals  symbols with no fundamentals, or whose newest period is too old
             (the aggregate "latest period is fresh" test in run_scan passes if
             ANY symbol is current; this is the per-symbol view).

REPORT-ONLY. Nothing here filters, blocks or alters a scan. It runs after the
postclose scan and records what it found. Severity is provisional: the
thresholds below are starting values and are to be recalibrated from the first
two weeks of real reports (principle: calibrate, don't set arbitrarily).
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def _now() -> datetime:
    """Wall clock in IST. market_calendar.expected_last_session compares the
    time of day with the 15:30 close, so it must be given IST: a container
    running UTC would read 19:00 IST as 13:30, decide today's close has not
    happened, and expect YESTERDAY's bar, so a failed postclose would pass."""
    return datetime.now(IST)


# ---- provisional policy (env-overridable; recalibrate after ~2 weeks) ----
STALE_CRITICAL_PCT = float(os.environ.get("DQ_STALE_CRITICAL_PCT", "3"))
GAP_WINDOW_YEARS = float(os.environ.get("DQ_GAP_WINDOW_YEARS", "4"))
BAD_TICK_WINDOW_SESSIONS = int(os.environ.get("DQ_BAD_TICK_WINDOW", "520"))
FUNDAMENTALS_STALE_DAYS = int(os.environ.get("DQ_FUNDAMENTALS_STALE_DAYS", "150"))
SESSION_COVERAGE = 0.5       # a date is a session if >= half of live symbols have a bar
SAMPLE = 15                  # examples kept per category


# ---------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------
STALE_SQL = """
    select u.symbol, max(o.trade_date) as last_bar
    from unnest(%(syms)s::text[]) as u(symbol)
    left join ohlcv_daily o on o.symbol = u.symbol
    group by u.symbol
"""

GAPS_SQL = """
    with u as (select unnest(%(syms)s::text[]) as symbol),
    b as (
        select o.symbol, min(o.trade_date) as lo, max(o.trade_date) as hi
        from ohlcv_daily o join u on u.symbol = o.symbol
        where o.trade_date >= %(start)s
        group by o.symbol
    ),
    cnt as (
        select o.trade_date, count(*) as n
        from ohlcv_daily o join u on u.symbol = o.symbol
        where o.trade_date >= %(start)s
        group by o.trade_date
    ),
    sessions as (
        select c.trade_date
        from cnt c
        where c.n >= %(cov)s * (select count(*) from b where b.lo <= c.trade_date
                                                          and b.hi >= c.trade_date)
    ),
    missing as (
        select b.symbol, s.trade_date
        from b
        join sessions s on s.trade_date between b.lo and b.hi
        left join ohlcv_daily o on o.symbol = b.symbol and o.trade_date = s.trade_date
        where o.symbol is null
    )
    select symbol, count(*) as n, array_agg(trade_date order by trade_date) as dates
    from missing group by symbol order by n desc, symbol
"""

SESSIONS_SQL = """
    with u as (select unnest(%(syms)s::text[]) as symbol),
    b as (select o.symbol, min(o.trade_date) lo, max(o.trade_date) hi
          from ohlcv_daily o join u on u.symbol = o.symbol
          where o.trade_date >= %(start)s group by o.symbol),
    cnt as (select o.trade_date, count(*) n from ohlcv_daily o join u on u.symbol = o.symbol
            where o.trade_date >= %(start)s group by o.trade_date)
    select count(*) from cnt c
    where c.n >= %(cov)s * (select count(*) from b where b.lo <= c.trade_date and b.hi >= c.trade_date)
"""

# Every bar in the window for the universe, classified. One pass.
BAD_TICKS_SQL = """
    select o.symbol, o.trade_date,
           array_remove(array[
             case when o.high < o.low then 'high_below_low' end,
             case when o.close > o.high or o.close < o.low then 'close_outside_range' end,
             case when o.open  > o.high or o.open  < o.low then 'open_outside_range' end,
             case when least(o.open, o.high, o.low, o.close) <= 0 then 'non_positive_price' end,
             case when o.volume < 0 then 'negative_volume' end,
             case when o.adj_factor is null or o.adj_factor <= 0 then 'bad_adj_factor' end
           ], null) as problems,
           (o.volume = 0) as zero_volume
    from ohlcv_daily o
    where o.symbol = any(%(syms)s)
      and o.trade_date >= %(start)s
"""

FUNDAMENTALS_SQL = """
    select u.symbol, max(f.period_end) as latest
    from unnest(%(syms)s::text[]) as u(symbol)
    left join fundamentals_quarterly f on f.symbol = u.symbol
    group by u.symbol
"""


# ---------------------------------------------------------------------
# Pure policy (testable without a database)
# ---------------------------------------------------------------------
def sessions_behind(last_bar: date, expected_last: date, session_dates: list[date]) -> int:
    """Sessions strictly after last_bar up to and including expected_last."""
    return sum(1 for d in session_dates if last_bar < d <= expected_last)


def classify(report: dict) -> dict:
    """
    Severity from a populated report. critical = something that can silently
    corrupt a signal or means the feed is down; warn = worth a look; ok.
    """
    reasons_c, reasons_w = [], []
    st = report["stale"]
    if st["pct"] > STALE_CRITICAL_PCT:
        reasons_c.append(f"{st['count']} symbols ({st['pct']}%) are behind the last session "
                         f"(> {STALE_CRITICAL_PCT}%): feed outage or missed refresh")
    elif st["count"]:
        reasons_w.append(f"{st['count']} symbols behind the last session")
    if st.get("no_bars"):
        reasons_w.append(f"{len(st['no_bars'])} symbols have no bars at all")

    mw = report["gaps"].get("market_wide_missing_recent", [])
    if mw:
        reasons_c.append(f"market-wide missing session(s) vs the calendar: {mw}")
    if report["gaps"]["symbols_with_gaps"]:
        reasons_w.append(f"{report['gaps']['symbols_with_gaps']} symbols have interior gaps "
                         f"({report['gaps']['missing_bars']} bars)")

    bt = report["bad_ticks"]
    if bt["latest_session"]:
        reasons_c.append(f"{bt['latest_session']} bad tick(s) in the latest session")
    if bt["bars"]:
        reasons_w.append(f"{bt['bars']} bad ticks in the last {bt['window_sessions']} sessions")

    fu = report["fundamentals"]
    if fu["no_data"] or fu["stale"]:
        reasons_w.append(f"fundamentals: {fu['no_data']} symbols with none, {fu['stale']} stale "
                         f"(> {FUNDAMENTALS_STALE_DAYS} days)")

    sev = "critical" if reasons_c else ("warn" if reasons_w else "ok")
    return {"severity": sev, "critical": reasons_c, "warnings": reasons_w}


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_checks(conn, symbols: list[str] | None = None, as_of: date | None = None) -> dict:
    from market_calendar import expected_last_session
    from universe import load_scan_universe

    if symbols is None:
        symbols = load_scan_universe(conn).symbols
    expected = as_of or expected_last_session(conn, now=_now())
    start_gaps = expected - timedelta(days=int(365 * GAP_WINDOW_YEARS))

    with conn.cursor() as cur:
        cur.execute(STALE_SQL, {"syms": symbols})
        last_bars = {r[0]: r[1] for r in cur.fetchall()}

    # sessions for the staleness count: data-derived dates in the last 30 days
    with conn.cursor() as cur:
        cur.execute("""select o.trade_date from ohlcv_daily o
                       where o.symbol = any(%(s)s) and o.trade_date >= %(d)s
                       group by o.trade_date
                       having count(*) >= %(c)s * %(n)s order by 1""",
                    {"s": symbols, "d": expected - timedelta(days=45),
                     "c": SESSION_COVERAGE, "n": len(symbols)})
        recent_sessions = [r[0] for r in cur.fetchall()]

    no_bars, behind = [], []
    for sym in symbols:
        lb = last_bars.get(sym)
        if lb is None:
            no_bars.append(sym)
        elif lb < expected:
            behind.append({"symbol": sym, "last_bar": str(lb),
                           "sessions_behind": max(1, sessions_behind(lb, expected, recent_sessions))
                           if recent_sessions else None})
    behind.sort(key=lambda r: (-(r["sessions_behind"] or 0), r["symbol"]))
    stale = {"count": len(behind), "pct": round(100.0 * len(behind) / max(len(symbols), 1), 2),
             "expected_last_session": str(expected), "no_bars": no_bars[:SAMPLE * 3],
             "worst": behind[:SAMPLE]}

    # Market-wide absence vs the CALENDAR over the recent window: a calendar
    # session on which almost no symbol has a bar. (The calendar is trusted
    # only for the recent past, where trading_holidays is loaded.)
    from market_calendar import is_trading_holiday
    mw = []
    with conn.cursor() as cur:
        cur.execute("""select trade_date, count(*) from ohlcv_daily
                       where symbol = any(%(s)s) and trade_date >= %(d)s
                       group by trade_date""",
                    {"s": symbols, "d": expected - timedelta(days=30)})
        present = {r[0]: r[1] for r in cur.fetchall()}
    d = expected - timedelta(days=30)
    while d <= expected:
        if d.weekday() < 5 and not is_trading_holiday(conn, d):
            if present.get(d, 0) < 0.2 * len(symbols):
                mw.append(str(d))
        d += timedelta(days=1)

    with conn.cursor() as cur:
        cur.execute(GAPS_SQL, {"syms": symbols, "start": start_gaps, "cov": SESSION_COVERAGE})
        gap_rows = cur.fetchall()
        cur.execute(SESSIONS_SQL, {"syms": symbols, "start": start_gaps, "cov": SESSION_COVERAGE})
        n_sessions = cur.fetchone()[0]
    gaps = {"window_start": str(start_gaps), "sessions_derived_from_data": int(n_sessions),
            "symbols_with_gaps": len(gap_rows), "missing_bars": int(sum(r[1] for r in gap_rows)),
            "worst": [{"symbol": r[0], "missing": int(r[1]), "sample_dates": [str(x) for x in r[2][:5]]}
                      for r in gap_rows[:SAMPLE]],
            "market_wide_missing_recent": mw}

    start_ticks = expected - timedelta(days=int(BAD_TICK_WINDOW_SESSIONS * 7 / 5) + 5)
    with conn.cursor() as cur:
        cur.execute(BAD_TICKS_SQL, {"syms": symbols, "start": start_ticks})
        bars = cur.fetchall()
    by_type: dict[str, int] = {}
    examples, latest, n_bad, zero_vol = [], 0, 0, 0
    for sym, td, problems, zv in bars:
        if zv:
            zero_vol += 1
        if not problems:
            continue
        n_bad += 1
        if td == expected:
            latest += 1
        for p in problems:
            by_type[p] = by_type.get(p, 0) + 1
        if len(examples) < SAMPLE:
            examples.append({"symbol": sym, "date": str(td), "problems": list(problems)})
    bad = {"window_sessions": BAD_TICK_WINDOW_SESSIONS, "bars_checked": len(bars),
           "bars": n_bad, "latest_session": latest, "by_type": by_type,
           "zero_volume_bars_informational": zero_vol, "examples": examples}

    with conn.cursor() as cur:
        cur.execute(FUNDAMENTALS_SQL, {"syms": symbols})
        frows = cur.fetchall()
    cutoff = expected - timedelta(days=FUNDAMENTALS_STALE_DAYS)
    none = [r[0] for r in frows if r[1] is None]
    old = [(r[0], r[1]) for r in frows if r[1] is not None and r[1] < cutoff]
    fund = {"no_data": len(none), "stale": len(old), "cutoff_period_end": str(cutoff),
            "no_data_symbols": none[:SAMPLE * 2],
            "stale_examples": [{"symbol": s, "latest_period": str(p)} for s, p in old[:SAMPLE]]}

    report = {"as_of": str(expected), "universe": len(symbols), "stale": stale, "gaps": gaps,
              "bad_ticks": bad, "fundamentals": fund,
              "policy": {"provisional": True, "stale_critical_pct": STALE_CRITICAL_PCT,
                         "note": "thresholds are starting values; recalibrate from ~2 weeks of reports"}}
    report.update(classify(report))
    return report
