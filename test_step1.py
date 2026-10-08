"""
Step 1 tests. Run against a REAL PostgreSQL carrying the engine's actual
schema.sql + migrations (not mocks), because the things under test are SQL
semantics: which symbols a query returns, what a staleness check counts.

    export TEST_PG_DSN="host=127.0.0.1 port=55432 user=postgres dbname=eng \
                        options='-c search_path=swing,public'"
    python test_step1.py

Every change in Step 1 has a section here. Synthetic data only: these prove
the CODE does what it says. They do not measure the live database.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from unittest import mock

import numpy as np
import pandas as pd
import psycopg

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DSN = os.environ.get(
    "TEST_PG_DSN",
    "host=127.0.0.1 port=55432 user=postgres dbname=eng options='-c search_path=swing,public'")
PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


_OPEN: list = []


def fresh():
    # Close earlier connections first: a connection left with an open read
    # transaction blocks TRUNCATE (ACCESS EXCLUSIVE) forever.
    while _OPEN:
        try:
            _OPEN.pop().close()
        except Exception:
            pass
    conn = psycopg.connect(DSN)
    _OPEN.append(conn)
    with conn.cursor() as cur:
        cur.execute("truncate symbols, surveillance, engine_settings, ohlcv_daily, "
                    "trading_holidays, fundamentals_quarterly, ingestion_runs, shareholding cascade")
    conn.commit()
    return conn


def add_symbol(conn, sym, active=True, in_index=True, series="EQ", name=None, industry=None):
    with conn.cursor() as cur:
        cur.execute("insert into symbols (symbol, is_active, in_nifty500, series, company_name, industry) "
                    "values (%s,%s,%s,%s,%s,%s)", (sym, active, in_index, series, name or sym, industry))
    conn.commit()


# =====================================================================
print("A. 1.1 / 1.2  Scan universe (real Postgres)")
import universe as U

conn = fresh()
for sym, kw in {
    "AAA": {}, "BBB": {},
    "CCC": {"in_index": False},                      # left the index: 27-name drift
    "DDD": {"active": False},                        # delisted
    "NIFTY50": {"series": "INDEX", "in_index": True},
    "FFF": {"in_index": None},                       # never classified
    "BIRET": {}, "EMBASSY": {}, "BAGMANE": {},       # REITs
}.items():
    add_symbol(conn, sym, **kw)
with conn.cursor() as cur:
    cur.execute("insert into surveillance (symbol, as_of, list_type) values ('BBB', current_date, 'ASM')")
    cur.execute("insert into surveillance (symbol, as_of, list_type) values ('AAA', current_date - 30, 'ASM')")
conn.commit()

os.environ.pop("SCAN_EXCLUDE_SYMBOLS", None)
u = U.load_scan_universe(conn)
check("scans only active + in_nifty500, excluding REITs", u.symbols == ["AAA", "BBB"], str(u.symbols))
check("index exit (in_nifty500=false) is NOT scanned", "CCC" not in u.symbols)
check("symbol never classified (in_nifty500 NULL) is NOT scanned", "FFF" not in u.symbols)
check("delisted is NOT scanned", "DDD" not in u.symbols)
check("INDEX series is NOT scanned", "NIFTY50" not in u.symbols)
check("all three REITs excluded, BAGMANE included in the list",
      not ({"BIRET", "EMBASSY", "BAGMANE"} & set(u.symbols)))
check("surveillance flag: recent entry flagged, 30-day-old entry not",
      u.flagged == {"BBB"}, str(u.flagged))
d = u.detail
check("breakdown is exact: active=7 in_index=5 dropped_not_in_index=2 excluded=3 scanned=2",
      (d["active"], d["in_nifty500"], d["dropped_not_in_index"],
       d["dropped_excluded_symbols"], d["scanned"]) == (7, 5, 2, 3, 2), str(d))
check("breakdown is JSON-serialisable (goes into last_scan_summary)", json.dumps(d) is not None)

os.environ["SCAN_EXCLUDE_SYMBOLS"] = ""
check("SCAN_EXCLUDE_SYMBOLS='' -> no exclusions, REITs scanned again",
      U.load_scan_universe(conn).symbols == ["AAA", "BAGMANE", "BBB", "BIRET", "EMBASSY"])
os.environ["SCAN_EXCLUDE_SYMBOLS"] = " aaa , Bbb ,, "
check("env list is normalised (case, spaces, blanks)", U.excluded_symbols() == ["AAA", "BBB"])
os.environ.pop("SCAN_EXCLUDE_SYMBOLS")

add_symbol(conn, "ZZINVIT", name="Some Infrastructure Investment Trust")
add_symbol(conn, "LISTED", name="Embassy Realty Pvt")                   # not a trust
look = U.trust_lookalikes(conn)
check("a new trust-like listing is SURFACED for a human, not silently admitted or excluded",
      [x["symbol"] for x in look] == ["ZZINVIT"], str(look))
check("...and it IS in the scan universe until a human adds it to the list",
      "ZZINVIT" in U.load_scan_universe(conn).symbols)

# run_scan refuses a degraded universe, loudly, before writing anything
import scan
from universe import ScanUniverse
class _C:
    def __init__(s): s.sql = []
    def cursor(s): return s
    def __enter__(s): return s
    def __exit__(s, *a): pass
    def execute(s, q, p=None): s.sql.append(" ".join(q.split())[:40])
    def fetchone(s): return None
    def fetchall(s): return []
    def commit(s): s.sql.append("COMMIT")
    def rollback(s): pass
    def close(s): pass
c = _C()
tiny = ScanUniverse(symbols=["A"] * 5, flagged=set(), detail={"scanned": 5})
with mock.patch.object(scan, "connect", return_value=c), \
     mock.patch.object(scan, "kill_switch_active", return_value=False), \
     mock.patch.object(scan, "is_trading_holiday", return_value=False), \
     mock.patch.object(scan, "is_muhurat", return_value=False), \
     mock.patch.object(scan, "ensure_bars_current", return_value={"refreshed": False}), \
     mock.patch.object(scan, "load_scan_universe", return_value=tiny):
    try:
        scan.run_scan(mode="postclose")
        raised = None
    except scan.ScanAborted as e:
        raised = str(e)
check("run_scan raises ScanAborted on a 5-symbol universe, naming the breakdown",
      raised is not None and "minimum" in raised and "scanned" in raised, str(raised))
check("...and wrote nothing before refusing", not any(x.startswith(("insert", "delete", "COMMIT")) for x in c.sql), str(c.sql))


# =====================================================================
print("B. Kill switch: shared read + both run_scan checks (real Postgres)")
import ingest

conn = fresh()
check("no kill_switch row -> not killed", ingest.kill_switch_active(conn) is False)
with conn.cursor() as cur:
    cur.execute("insert into engine_settings (key, value) values ('kill_switch', %s::jsonb)", (json.dumps({"active": True}),))
conn.commit()
check("active=true -> killed", ingest.kill_switch_active(conn) is True)
with conn.cursor() as cur:
    cur.execute("update engine_settings set value = %s::jsonb where key='kill_switch'", (json.dumps({"active": False}),))
conn.commit()
check("active=false -> not killed", ingest.kill_switch_active(conn) is False)
bad = psycopg.connect(DSN); bad.close()
check("unreadable flag FAILS OPEN (False) and does not raise", ingest.kill_switch_active(bad) is False)
brk = psycopg.connect(DSN)
with brk.cursor() as cur:
    cur.execute("alter table engine_settings rename to engine_settings_x") if False else None
with brk.cursor() as cur:
    try:
        cur.execute("select * from no_such_table")
    except Exception:
        pass
check("after a failed query the connection is rolled back and usable again",
      (ingest.kill_switch_active(brk) in (True, False)) and brk.execute("select 1").fetchone() == (1,))

import scheduler
for want, label in ((True, "killed"), (False, "not killed")):
    with conn.cursor() as cur:
        cur.execute("update engine_settings set value=%s::jsonb where key='kill_switch'", (json.dumps({"active": want}),))
    conn.commit()
    with mock.patch.object(ingest, "connect", side_effect=lambda: psycopg.connect(DSN)):
        check(f"scheduler._kill_switch_active() reads the same flag ({label})",
              scheduler._kill_switch_active() is want)
with mock.patch.object(ingest, "connect", side_effect=Exception("db down")):
    check("scheduler._kill_switch_active() survives connect() failure -> False",
          scheduler._kill_switch_active() is False)

# run_scan, first check: stopped engine does ONE read and returns
with conn.cursor() as cur:
    cur.execute("update engine_settings set value=%s::jsonb where key='kill_switch'", (json.dumps({"active": True}),))
conn.commit()
class Spy:
    def __init__(s, real): s.real, s.sql = real, []
    def cursor(s):
        outer = s
        real_cur = s.real.cursor()
        class CC:
            def __enter__(c): c.r = real_cur.__enter__(); return c
            def __exit__(c, *a): return real_cur.__exit__(*a)
            def execute(c, q, p=None): outer.sql.append(" ".join(str(q).split())[:90]); return c.r.execute(q, p)
            def __getattr__(c, n): return getattr(c.r, n)
        return CC()
    def commit(s): s.sql.append("COMMIT"); s.real.commit()
    def rollback(s): s.real.rollback()
    def close(s): pass
for mode in ("postclose", "intraday", "premarket"):
    spy = Spy(conn)
    with mock.patch.object(scan, "connect", return_value=spy):
        r = scan.run_scan(mode=mode)
    check(f"run_scan({mode}) while stopped -> skipped_kill_switch after exactly one statement",
          r["status"] == "skipped_kill_switch" and len(spy.sql) == 1 and "kill_switch" in spy.sql[0], str((r.get("status"), spy.sql)))

# run_scan, second check: fired mid-scan -> nothing from the publish phase is written
def make_nifty(n=320):
    close = np.linspace(20000, 24000, n)
    return pd.DataFrame({"trade_date": pd.bdate_range("2025-06-02", periods=n), "open": close,
                         "high": close * 1.005, "low": close * 0.995, "close": close, "volume": np.zeros(n)})
def run_scan_with_kill(kill_on_read):
    state = {"reads": 0, "log": []}
    def fake_kill(c):
        state["reads"] += 1
        return kill_on_read(state["reads"])
    class FC:
        rowcount = 0
        def __init__(s, outer): s.o = outer
        def __enter__(s): return s
        def __exit__(s, *a): pass
        def execute(s, q, p=None): s.o.log.append(" ".join(q.split())); s.last = " ".join(q.split())
        def executemany(s, q, rows): s.o.log.append("MANY " + " ".join(q.split())); s.last = "many"
        def fetchone(s): return (None,)
        def fetchall(s): return []
    class FConn:
        def __init__(s): s.log = state["log"]
        def cursor(s): return FC(s)
        def commit(s): s.log.append("COMMIT")
        def rollback(s): pass
        def close(s): pass
    regime = {"state": "neutral", "nifty_close": 24000.0, "nifty_vs_20dma": 1.0, "nifty_vs_50dma": 1.0,
              "breadth_above_50dma": 30.0, "vix": 14.0, "vix_10d_change": 0.0, "distribution_days": 0, "notes": {}}
    nifty = make_nifty()
    uni = ScanUniverse(symbols=[f"S{i}" for i in range(5)], flagged=set(), detail={"scanned": 5})
    with mock.patch.object(scan, "connect", return_value=FConn()), \
         mock.patch.object(scan, "kill_switch_active", side_effect=fake_kill), \
         mock.patch.object(scan, "is_trading_holiday", return_value=False), \
         mock.patch.object(scan, "is_muhurat", return_value=False), \
         mock.patch.object(scan, "ensure_bars_current", return_value={"refreshed": False, "reason": "t"}), \
         mock.patch.object(scan, "transition_readiness", return_value={"enabled": False, "reason": "t"}), \
         mock.patch.object(scan, "load_snapshots", return_value={}), \
         mock.patch.object(scan, "load_scan_universe", return_value=uni), \
         mock.patch.object(scan, "_load_symbol", side_effect=lambda c, sym: nifty if sym == "NIFTY50" else None), \
         mock.patch.object(scan, "_breadth", return_value=30.0), \
         mock.patch.object(scan, "evaluate_regime", return_value=regime), \
         mock.patch.object(scan, "calendar_health", return_value={}), \
         mock.patch.object(scan, "UNIVERSE_MIN_SYMBOLS", 1):
        res = scan.run_scan(mode="postclose")
    return res, state
PUBLISH = ("MANY insert into gate_log", "insert into signals", "delete from signals", "update signals")
res, st = run_scan_with_kill(lambda n: n >= 2)
pub = [x for x in st["log"] if x.startswith(PUBLISH)]
check("kill fired MID-scan -> aborted_kill_switch, publish phase wrote nothing",
      res["status"] == "aborted_kill_switch" and not pub, str((res.get("status"), pub)))
res, st = run_scan_with_kill(lambda n: False)
pub = [x for x in st["log"] if x.startswith(PUBLISH)]
check("control: never killed -> publish phase DOES write (so the abort test is not vacuous)", len(pub) >= 3, str(len(pub)))
check("universe breakdown (active / dropped / excluded / scanned) reaches the scan summary",
      res.get("universe_detail", {}).get("scanned") == 5 and res.get("universe") == 5, str({k: res.get(k) for k in ("universe", "universe_detail")}))

# =====================================================================
print("C. 1.3  sync_universe monthly slot + 1.4 fundamentals slot registered")
from apscheduler.schedulers.background import BackgroundScheduler
with mock.patch.object(scheduler, "SCHEDULER_ENABLED", True), \
     mock.patch.object(scheduler, "_kill_switch_active", return_value=False), \
     mock.patch.object(scheduler, "_scheduler", None):
    sch = scheduler.start()
    try:
        jobs = {j.id: j for j in sch.get_jobs()}
    finally:
        sch.shutdown(wait=False)
        scheduler._scheduler = None
check("all slots registered: premarket, intraday, postclose, sync_universe, sync_holidays, refresh_fundamentals",
      {"premarket", "intraday", "postclose", "sync_universe", "sync_holidays", "refresh_fundamentals"} <= set(jobs),
      str(sorted(jobs)))
su = jobs["sync_universe"].trigger.get_next_fire_time(None, datetime(2026, 10, 8, tzinfo=scheduler.IST))
check("sync_universe next fire is the FIRST Saturday at 06:00 IST (7 Nov 2026)",
      (su.year, su.month, su.day, su.hour, su.minute, su.weekday()) == (2026, 11, 7, 6, 0, 5), str(su))
trig = None
with mock.patch.object(scheduler, "SCHEDULER_ENABLED", True), \
     mock.patch.object(scheduler, "_kill_switch_active", return_value=False), \
     mock.patch.object(scheduler, "_scheduler", None):
    sch = scheduler.start()
    trig = {j.id: j.trigger for j in sch.get_jobs()}["refresh_fundamentals"]
    sch.shutdown(wait=False); scheduler._scheduler = None
prev, now, months, dows, hours = None, datetime(2026, 1, 1, tzinfo=scheduler.IST), set(), set(), set()
for _ in range(60):
    nxt = trig.get_next_fire_time(prev, now)
    if nxt is None or nxt.year > 2026: break
    months.add(nxt.month); dows.add(nxt.weekday()); hours.add((nxt.hour, nxt.minute))
    prev, now = nxt, nxt + timedelta(seconds=1)
check("refresh_fundamentals fires only in results-season months (1,2,4,5,7,8,10,11)",
      months == {1, 2, 4, 5, 7, 8, 10, 11}, str(sorted(months)))
check("...only on Fridays at 19:30 IST (an hour after postclose, when a token is valid)", dows == {4} and hours == {(19, 30)}, str((dows, hours)))
nov = trig.get_next_fire_time(None, datetime(2026, 10, 8, tzinfo=scheduler.IST))
check("...first run after today is Friday 9 Oct 2026 19:30 (Q2 season month; flag in the report)",
      (nov.month, nov.day, nov.weekday(), nov.hour, nov.minute) == (10, 9, 4, 19, 30), str(nov))

# the failure rule is pure
check("an Upstox sync that REPORTS an error (e.g. no ISINs stored) is a problem even without raising",
      "no ISINs" in (scheduler._fundamentals_problem("s", {"written": 0, "error": "no ISINs stored"}, 500) or ""))
fp = scheduler._fundamentals_problem
check("a sync that wrote nothing is a problem", fp("s", {"written": 0, "failed": 0}, 500) is not None)
check("a healthy sync is not", fp("s", {"written": 900, "failed": 10}, 500) is None)
check("30% of symbols failing is a problem even though rows were written",
      "150/500" in (fp("s", {"written": 900, "failed": 150}, 500) or ""))
check("exactly at the threshold (20%) is tolerated, above it is not",
      fp("s", {"written": 9, "failed": 100}, 500) is None and fp("s", {"written": 9, "failed": 101}, 500) is not None)

# the job: both syncs always run; failures raised together; run log written
import fundamentals as F
conn = fresh()
for i in range(10):
    add_symbol(conn, f"X{i}")
dsn_conn = lambda: psycopg.connect(DSN)

# --- NSE path (UPSTOX_FUNDAMENTALS_ENABLED=false)
calls = []
def good_sh(c, symbols=None): calls.append("sh"); return {"written": 40, "failed": 0, "empty": 0}
def bad_q(c, symbols=None): calls.append("q"); raise RuntimeError("NSE blocked")
with mock.patch.object(F, "UPSTOX_FUNDAMENTALS_ENABLED", False), \
     mock.patch.object(F, "sync_shareholding", good_sh), mock.patch.object(F, "sync_quarterly_results", bad_q), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    try:
        scheduler._refresh_fundamentals_job(); raised = None
    except RuntimeError as e:
        raised = str(e)
check("flag=false -> NSE path; one sync failing does not stop the other and is RAISED, not swallowed",
      calls == ["sh", "q"] and raised and "NSE blocked" in raised and "degraded" in raised, str((calls, raised)))
with psycopg.connect(DSN) as c2:
    runs = dict(c2.execute("select job, status from ingestion_runs").fetchall())
check("both outcomes persisted to ingestion_runs (visible in the DB, not only stderr)",
      runs == {"sync_shareholding": "success", "sync_quarterly_results": "failed"}, str(runs))

# --- Upstox path (UPSTOX_FUNDAMENTALS_ENABLED=true, the live setting)
import upstox_client
seen = []
def up_sh(c, client, symbols=None): seen.append("sh"); return {"written": 300, "failed": 1, "empty": 1, "symbols": 10}
def up_q(c, client, symbols=None): seen.append("q"); return {"written": 0, "failed": 0, "empty": 0, "error": "no ISINs stored — run sync_universe first"}
class _TS:
    def __init__(s, tok): s.tok = tok
    def valid_token(s): return s.tok
with mock.patch.object(F, "UPSTOX_FUNDAMENTALS_ENABLED", True), \
     mock.patch.object(F, "sync_shareholding_upstox", up_sh), mock.patch.object(F, "sync_quarterly_results_upstox", up_q), \
     mock.patch.object(upstox_client, "TokenStore", lambda: _TS("tok")), \
     mock.patch.object(upstox_client, "UpstoxClient", lambda: object()), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    try:
        scheduler._refresh_fundamentals_job(); raised = None
    except RuntimeError as e:
        raised = str(e)
check("flag=true -> Upstox path; a sync that reports an error is surfaced as a failure",
      seen == ["sh", "q"] and raised and "no ISINs stored" in raised, str((seen, raised)))
with mock.patch.object(F, "UPSTOX_FUNDAMENTALS_ENABLED", True), \
     mock.patch.object(upstox_client, "TokenStore", lambda: _TS(None)), \
     mock.patch.object(F, "sync_shareholding_upstox", up_sh), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    seen.clear()
    try:
        scheduler._refresh_fundamentals_job(); raised = None
    except RuntimeError as e:
        raised = str(e)
check("flag=true and no valid token -> raises token_invalid BEFORE touching anything (never falls back to NSE)",
      raised == "token_invalid" and seen == [], str((raised, seen)))
def up_q_ok(c, client, symbols=None): return {"written": 80, "failed": 1, "empty": 1, "symbols": 10}
with mock.patch.object(F, "UPSTOX_FUNDAMENTALS_ENABLED", True), \
     mock.patch.object(F, "sync_shareholding_upstox", up_sh), mock.patch.object(F, "sync_quarterly_results_upstox", up_q_ok), \
     mock.patch.object(upstox_client, "TokenStore", lambda: _TS("tok")), \
     mock.patch.object(upstox_client, "UpstoxClient", lambda: object()), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    out = scheduler._refresh_fundamentals_job()
check("both healthy -> returns a summary naming the source",
      out["source"] == "upstox" and out["sync_shareholding_upstox"]["written"] == 300 and out["symbols"] == 10, str(out))
# _guarded turns the raise into a recorded failure
scheduler._last_runs.clear()
with mock.patch.object(ingest, "connect", side_effect=lambda: psycopg.connect(DSN)):
    scheduler._guarded("refresh_fundamentals", lambda: (_ for _ in ()).throw(RuntimeError("fundamentals_refresh_degraded: x")))
check("_guarded records the slot as failed with the reason", scheduler._last_runs["refresh_fundamentals"]["status"] == "failed")


# =====================================================================
print("D. 1.6  Data quality checks (real Postgres, defects injected at known places)")
import data_quality as DQ

EXPECTED = date(2026, 10, 7)
HOLIDAYS = [date(2026, 5, 1), date(2026, 8, 26)]                    # weekdays, full closures


def build_dq(conn, mutate=None, symbols=40):
    """40 symbols, every non-holiday weekday 1 May -> 7 Oct 2026, then defects.
    Row layout: [open, high, low, close, volume, adj_factor]. adj_* columns are
    GENERATED (raw * factor) in the real schema, so they cannot be inserted."""
    with conn.cursor() as cur:
        for h in HOLIDAYS:
            cur.execute("insert into trading_holidays (trade_date, year, description, is_muhurat) "
                        "values (%s,%s,'closed',false)", (h, h.year))
    days = [d.date() for d in pd.bdate_range("2026-05-01", EXPECTED) if d.date() not in HOLIDAYS]
    syms = [f"S{i:02d}" for i in range(symbols)]
    rows = {}
    for s in syms:
        add_symbol(conn, s)
        for d in days:
            rows[(s, d)] = [100.0, 102.0, 99.0, 101.0, 1000, 1.0]
    if mutate:
        mutate(rows, syms, days)
    with conn.cursor() as cur:
        cur.executemany(
            "insert into ohlcv_daily (symbol, trade_date, open, high, low, close, volume, adj_factor) "
            "values (%s,%s,%s,%s,%s,%s,%s,%s)",
            [(s, d, *v) for (s, d), v in rows.items()])
        for s in syms:
            cur.execute("insert into fundamentals_quarterly (symbol, period_end) values (%s, %s)",
                        (s, date(2026, 6, 30)))
    conn.commit()
    return syms


def defects(rows, syms, days):
    last = EXPECTED
    for d in days:                                       # S01 stale: stops 5 Oct (2 sessions behind)
        if d > date(2026, 10, 5): rows.pop(("S01", d))
    rows.pop(("S02", date(2026, 7, 14))); rows.pop(("S02", date(2026, 7, 15)))     # S02 interior gap
    rows[("S03", last)][3] = 105.0                                                   # S03 close>high, latest
    r = rows[("S04", date(2026, 6, 10))]; r[1], r[2] = 98.0, 103.0                   # S04 high<low (hist)
    for d in days: rows.pop(("S05", d))                                              # S05 no bars at all
    rows[("S06", date(2026, 6, 11))][4] = -5                                         # S06 negative volume
    rows[("S07", date(2026, 6, 12))][0] = 0.0                                        # S07 open = 0
    rows[("S08", date(2026, 6, 15))][5] = 0.5                                        # S08 legit split factor: NOT a defect
    rows[("S09", date(2026, 6, 15))][5] = 0.0                                        # S09 adj_factor = 0: zeroes every adj price
    for d in days:                                                                   # S10 late listing
        if d < date(2026, 8, 3): rows.pop(("S10", d))
    rows[("S11", date(2026, 6, 16))][4] = 0                                          # S11 zero volume (info)


conn = fresh()
syms = build_dq(conn, defects)
with conn.cursor() as cur:
    cur.execute("delete from fundamentals_quarterly where symbol = 'S12'")           # S12: none
    cur.execute("update fundamentals_quarterly set period_end = %s where symbol = 'S13'", (date(2025, 12, 31),))
conn.commit()
rep = DQ.run_checks(conn, symbols=syms, as_of=EXPECTED)

st = rep["stale"]
check("stale: exactly S01 is behind (S05 has NO bars and is reported separately, not as stale)",
      st["count"] == 1 and st["worst"][0]["symbol"] == "S01" and st["no_bars"] == ["S05"], str(st))
check("stale: counted in SESSIONS (2 behind), not calendar days", st["worst"][0]["sessions_behind"] == 2, str(st["worst"]))
check("stale: percentage is of the universe", st["pct"] == round(100 * 1 / 40, 2))

gp = rep["gaps"]
check("gaps: only S02 has interior gaps (S01's missing tail, S10's late listing, S05's emptiness are NOT gaps)",
      gp["symbols_with_gaps"] == 1 and gp["worst"][0]["symbol"] == "S02", str(gp["worst"]))
check("gaps: S02 is missing exactly 2 bars on the right dates",
      gp["worst"][0]["missing"] == 2 and gp["worst"][0]["sample_dates"] == ["2026-07-14", "2026-07-15"], str(gp["worst"][0]))
check("gaps: the market holidays (1 May, 26 Aug) are NOT flagged for anyone (sessions come from the data)",
      gp["missing_bars"] == 2, str(gp["missing_bars"]))
check("gaps: no market-wide missing session in a clean calendar", gp["market_wide_missing_recent"] == [])

bt = rep["bad_ticks"]
check("bad ticks: 5 bars flagged (S03, S04, S06, S07, S09); S08 (a legitimate 0.5 split factor) is not",
      bt["bars"] == 5 and {e["symbol"] for e in bt["examples"]} == {"S03", "S04", "S06", "S07", "S09"}, str(bt["examples"]))
check("bad ticks: latest-session count is 1 (S03) -- the one that can corrupt today's signal", bt["latest_session"] == 1)
check("bad ticks: types identified",
      {"close_outside_range", "high_below_low", "negative_volume", "non_positive_price", "bad_adj_factor"}
      <= set(bt["by_type"]), str(bt["by_type"]))
check("bad ticks: zero volume is informational, not a violation",
      bt["zero_volume_bars_informational"] >= 1 and "S11" not in {e["symbol"] for e in bt["examples"]})

fu = rep["fundamentals"]
check("fundamentals: per-symbol view (S12 none, S13 stale) even though the aggregate latest period is fresh",
      fu["no_data"] == 1 and fu["no_data_symbols"] == ["S12"] and fu["stale"] == 1
      and fu["stale_examples"][0]["symbol"] == "S13", str(fu))
check("severity is CRITICAL (a bad tick sits in the latest session) with the reason stated",
      rep["severity"] == "critical" and any("latest session" in x for x in rep["critical"]), str(rep["critical"]))
check("report is JSON-serialisable (goes to engine_settings)", json.dumps(rep) is not None)

# clean data -> ok
conn = fresh()
syms = build_dq(conn)
clean = DQ.run_checks(conn, symbols=syms, as_of=EXPECTED)
check("clean data -> severity ok, no reasons",
      clean["severity"] == "ok" and not clean["critical"] and not clean["warnings"], str((clean["severity"], clean["critical"], clean["warnings"])))

# market-wide missing session vs the calendar (feed outage on a day that SHOULD have a session)
def drop_day(rows, syms, days):
    for s in syms: rows.pop((s, date(2026, 10, 1)))
conn = fresh()
syms = build_dq(conn, drop_day)
mw = DQ.run_checks(conn, symbols=syms, as_of=EXPECTED)
check("a whole session missing for the market is CRITICAL and named by date",
      mw["gaps"]["market_wide_missing_recent"] == ["2026-10-01"] and mw["severity"] == "critical", str(mw["gaps"]["market_wide_missing_recent"]))
check("...and it is not double-reported as a per-symbol gap (the date is not a data-derived session)",
      mw["gaps"]["symbols_with_gaps"] == 0, str(mw["gaps"]["symbols_with_gaps"]))

# mass staleness -> critical by percentage
def mass_stale(rows, syms, days):
    for s in syms[:3]:
        for d in days:
            if d > date(2026, 10, 2): rows.pop((s, d))
conn = fresh()
syms = build_dq(conn, mass_stale)
ms = DQ.run_checks(conn, symbols=syms, as_of=EXPECTED)
check("3 of 40 symbols stale (7.5% > 3%) is CRITICAL: feed outage, not noise",
      ms["stale"]["count"] == 3 and ms["severity"] == "critical" and any("behind" in x for x in ms["critical"]), str(ms["critical"]))
one = {"stale": {"count": 1, "pct": 2.5, "no_bars": []}, "gaps": {"symbols_with_gaps": 0, "missing_bars": 0, "market_wide_missing_recent": []},
       "bad_ticks": {"latest_session": 0, "bars": 0, "window_sessions": 520}, "fundamentals": {"no_data": 0, "stale": 0}}
check("1 stale symbol (2.5%, below the 3% line) is a WARNING, not critical", DQ.classify(one)["severity"] == "warn")


# =====================================================================
print("E. 1.5  Intraday availability probe")
import data_probe as DP

check("URL uses the minutes unit and the same shape historical_daily() uses",
      DP.intraday_path("NSE_EQ|INE002A01018", 5, date(2026, 9, 15))
      == "/v3/historical-candle/NSE_EQ|INE002A01018/minutes/5/2026-09-15/2026-09-15")


def candles(n, start_h=9, start_m=15, step=5):
    out, h, m = [], start_h, start_m
    for _ in range(n):
        out.append([f"2026-09-15T{h:02d}:{m:02d}:00+05:30", 1, 2, 0.5, 1.5, 100, 0])
        m += step
        h, m = h + m // 60, m % 60
    return out

full = DP.summarise_candles({"status": "success", "data": {"candles": candles(75)}}, 5)
check("a full 5-min session (75 bars) is OK with coverage 1.0, 09:15 -> 15:25",
      full["ok"] and full["coverage"] == 1.0 and (full["first"], full["last"]) == ("09:15", "15:25"), str(full))
part = DP.summarise_candles({"status": "success", "data": {"candles": candles(30)}}, 5)
check("a partial session (30 of 75) is NOT ok", not part["ok"] and part["coverage"] == 0.4, str(part))
check("an API error payload is NOT ok and carries the reason",
      (lambda c: not c["ok"] and "UDAPI" in c["error"])(
          DP.summarise_candles({"status": "error", "errors": [{"errorCode": "UDAPI100011", "message": "range"}]}, 5)))
check("success with no candles is NOT ok", not DP.summarise_candles({"status": "success", "data": {"candles": []}}, 5)["ok"])
check("a non-dict response does not crash", not DP.summarise_candles(None, 5)["ok"])
check("1-minute session expects 375 bars", DP.summarise_candles({"status": "success", "data": {"candles": []}}, 1)["expected"] == 375)

ok, bad_ = {"ok": True}, {"ok": False}
d = [date(2026, 10, 7), date(2026, 9, 7), date(2026, 4, 7), date(2025, 10, 7), date(2024, 10, 7), date(2023, 10, 9)]
v = DP.verdict({5: list(zip(d, [ok, ok, ok, ok, ok, ok]))})
check("fully available back 3y -> usable for the 3-year backtest", v["5"]["usable_for_3y_backtest"] and v["5"]["first_probed_date_that_failed"] is None, str(v))
v = DP.verdict({5: list(zip(d, [ok, ok, ok, bad_, bad_, bad_]))})
check("available to Apr 2026 only -> reports the limit and the first failing date",
      v["5"]["full_sessions_back_to"] == "2026-04-07" and v["5"]["first_probed_date_that_failed"] == "2025-10-07"
      and not v["5"]["usable_for_3y_backtest"], str(v))
v = DP.verdict({5: list(zip(d, [ok, bad_, ok, ok, ok, ok]))})
check("a hole in the middle is not hidden by older dates that happen to succeed",
      v["5"]["full_sessions_back_to"] == "2026-10-07" and not v["5"]["usable_for_3y_backtest"], str(v))

conn = fresh()
with conn.cursor() as cur:
    cur.execute("insert into trading_holidays (trade_date, year, description, is_muhurat) values ('2026-10-02', 2026, 'Gandhi Jayanti', false)")
conn.commit()
picked = DP.pick_probe_dates(conn, date(2026, 10, 10), (1, 7, 8))
check("probe dates are weekdays and never a holiday (Sat 3 Oct -> Fri 2 Oct holiday -> Thu 1 Oct), de-duplicated",
      picked == [date(2026, 10, 9), date(2026, 10, 1)], str(picked))

with conn.cursor() as cur:
    cur.execute("insert into symbols (symbol, is_active, in_nifty500, upstox_instrument_key) values ('RELIANCE', true, true, 'NSE_EQ|INE002A01018')")
conn.commit()
class FakeClient:
    def __init__(s, cutoff): s.cutoff, s.paths = cutoff, []
    def _get(s, path):
        s.paths.append(path)
        d_ = date.fromisoformat(path.split("/")[-1])
        if d_ >= s.cutoff:
            return {"status": "success", "data": {"candles": candles(75)}}
        return {"status": "error", "errors": [{"message": "data not available"}]}
fc = FakeClient(date(2025, 1, 1))
res = DP.probe_intraday(fc, conn, "RELIANCE", (5,), today=date(2026, 10, 8), lookbacks=(1, 30, 365, 730))
check("one API call per (date, interval)", len(fc.paths) == 4, str(len(fc.paths)))
check("probe reports availability back to the oldest good date and the first failure",
      res["verdict"]["5"]["full_sessions_back_to"] == "2025-10-08" and res["verdict"]["5"]["first_probed_date_that_failed"] == "2024-10-08", str(res["verdict"]))
check("probe states plainly that nothing is stored", res["stored_intraday_history"] is False)
class DeadClient:
    def _get(s, path):
        class TokenExpired(Exception): pass
        raise TokenExpired("expired")
try:
    DP.probe_intraday(DeadClient(), conn, "RELIANCE", (5,), today=date(2026, 10, 8), lookbacks=(1,)); dead = False
except Exception as e:
    dead = type(e).__name__ == "TokenExpired"
check("an expired token is RAISED, not recorded as 'no data' (would read as a false negative)", dead)
check("unknown symbol -> clear error, no API calls", "error" in DP.probe_intraday(FakeClient(date(2025, 1, 1)), conn, "NOPE", (5,), today=date(2026, 10, 8)))

# =====================================================================
print("F. Scheduled data-quality job, holiday guards, and routes (real Postgres + FastAPI)")
EXPECTED_PATCH = mock.patch("market_calendar.expected_last_session", return_value=EXPECTED)
dsn_conn = lambda: psycopg.connect(DSN)

conn = fresh(); build_dq(conn)
with EXPECTED_PATCH, mock.patch.object(scheduler, "_skip_today", return_value=None), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    out = scheduler._data_quality_job()
with psycopg.connect(DSN) as c2:
    stored = c2.execute("select value from engine_settings where key='data_quality_latest'").fetchone()
    runs = c2.execute("select job, status from ingestion_runs").fetchall()
check("clean data: job returns ok, stores the report, logs one success row",
      out["severity"] == "ok" and stored and stored[0]["severity"] == "ok" and runs == [("data_quality", "success")], str((out, runs)))

conn = fresh(); build_dq(conn, defects)
scheduler._last_runs.clear()
with EXPECTED_PATCH, mock.patch.object(scheduler, "_skip_today", return_value=None), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    scheduler._guarded("data_quality", scheduler._data_quality_job)
with psycopg.connect(DSN) as c2:
    stored = c2.execute("select value from engine_settings where key='data_quality_latest'").fetchone()
    runs = c2.execute("select job, status from ingestion_runs").fetchall()
check("critical finding: the slot is recorded FAILED with the reason",
      scheduler._last_runs["data_quality"]["status"] == "failed" and "data_quality_critical" in scheduler._last_runs["data_quality"]["detail"]["error"])
check("...the full report was stored BEFORE the raise, so the detail is readable", stored and stored[0]["severity"] == "critical")
check("...and the failure is logged exactly once (by _guarded, not twice)", runs == [("data_quality", "failed")], str(runs))

skip = {"as_of": "2026-10-20", "mode": "x", "status": "skipped_holiday", "reason": "Dussehra"}
with mock.patch.object(scheduler, "_skip_today", return_value=skip):
    check("data_quality skips on a market holiday", scheduler._data_quality_job() == skip)
    check("refresh_fundamentals skips on a market holiday BEFORE the token check (no false token_invalid)",
          scheduler._refresh_fundamentals_job() == skip)

# ---- independence from postclose, with the REAL calendar (no EXPECTED_PATCH):
# the 19:00 check must notice that postclose did not deliver today's bars.
import data_quality as DQ, market_calendar as MC
THU_1900_IST = datetime(2026, 10, 8, 19, 0, tzinfo=scheduler.IST)       # Thu 8 Oct 19:00 IST
conn = fresh(); build_dq(conn)                                          # bars run to Wed 7 Oct
with conn.cursor() as cur:
    cur.execute("select count(*) from trading_holidays where trade_date = '2026-10-08'")
    assert cur.fetchone()[0] == 0
def _boom(): raise RuntimeError("upstox down")
scheduler._last_runs.clear()
with mock.patch.object(DQ, "_now", return_value=THU_1900_IST) as now_spy, mock.patch.object(scheduler, "_skip_today", return_value=None), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    scheduler._guarded("postclose", _boom)
    scheduler._guarded("data_quality", scheduler._data_quality_job)
check("the check's session expectation is driven by the IST clock (_now), not the container's local time", now_spy.called)
with psycopg.connect(DSN) as c2:
    stored = c2.execute("select value from engine_settings where key='data_quality_latest'").fetchone()
check("postclose FAILED, yet the 19:00 data-quality slot still ran (independent slots)",
      scheduler._last_runs["postclose"]["status"] == "failed" and "data_quality" in scheduler._last_runs)
check("...and it reported the missing bars: every symbol one session behind, severity critical, slot marked failed",
      scheduler._last_runs["data_quality"]["status"] == "failed" and stored and stored[0]["severity"] == "critical"
      and stored[0]["stale"]["count"] == 40 and stored[0]["stale"]["expected_last_session"] == "2026-10-08", str(stored and stored[0]["stale"]))
with conn.cursor() as cur:
    cur.execute("insert into ohlcv_daily (symbol, trade_date, open, high, low, close, volume) "
                "select symbol, '2026-10-08', 100, 102, 99, 101, 1000 from symbols where in_nifty500")
conn.commit()
with mock.patch.object(DQ, "_now", return_value=THU_1900_IST), mock.patch.object(scheduler, "_skip_today", return_value=None), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    ok_out = scheduler._data_quality_job()
check("postclose DID deliver today's bars -> the same slot reports ok", ok_out["severity"] == "ok", str(ok_out))
check("the clock is IST: market_calendar given a UTC-naive 13:30 would expect YESTERDAY (the miss this guards against); given IST it expects today",
      MC.expected_last_session(conn, now=datetime(2026, 10, 8, 13, 30)) == date(2026, 10, 7)
      and MC.expected_last_session(conn, now=THU_1900_IST) == date(2026, 10, 8))
check("data_quality._now() is timezone-aware IST", DQ._now().tzinfo is not None and DQ._now().utcoffset() == timedelta(hours=5, minutes=30))

with mock.patch.object(scheduler, "SCHEDULER_ENABLED", True), mock.patch.object(scheduler, "_kill_switch_active", return_value=False), \
     mock.patch.object(scheduler, "_scheduler", None):
    sch = scheduler.start(); js = {j.id: j for j in sch.get_jobs()}; sch.shutdown(wait=False); scheduler._scheduler = None
dq = js["data_quality"].trigger.get_next_fire_time(None, datetime(2026, 10, 8, 20, 0, tzinfo=scheduler.IST))
check("data_quality is scheduled weekdays 19:00 IST (next after Thu 20:00 is Fri 9 Oct 19:00)",
      (dq.day, dq.hour, dq.minute, dq.weekday()) == (9, 19, 0, 4), str(dq))

# ---- routes through the real app
os.environ["INTERNAL_API_KEY"] = "k-test"
import server
from fastapi.testclient import TestClient
cl, H = TestClient(server.app), {"x-internal-key": "k-test"}

conn = fresh()
for sym, kw in {"AAA": {}, "BBB": {}, "CCC": {"in_index": False}, "DDD": {"in_index": False}, "BIRET": {}}.items():
    add_symbol(conn, sym, **kw)
add_symbol(conn, "TRUSTX", name="Some Real Estate Investment Trust")
with conn.cursor() as cur:
    cur.execute("insert into fundamentals_quarterly (symbol, period_end) values ('AAA', '2026-06-30')")
    for s_ in ("AAA", "BBB", "CCC", "DDD", "BIRET", "TRUSTX"):
        for i in range(60):
            cur.execute("insert into ohlcv_daily (symbol, trade_date, open, high, low, close, volume) "
                        "values (%s, %s, 100, 102, 99, %s, 1000)", (s_, date(2026, 7, 1) + timedelta(days=i), 100 + (i if s_ in ("AAA", "BBB", "TRUSTX") else -i)))
conn.commit()
with mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    check("no key -> 401", cl.get("/jobs/universe-report").status_code == 401)
    r = cl.get("/jobs/universe-report", headers=H).json()
check("universe-report: before=6 (all active), after=3 (AAA, BBB, TRUSTX), delta -3",
      (r["before"]["universe"], r["after"]["universe"], r["delta"]["universe"]) == (6, 3, -3), str(r["delta"]))
check("universe-report: names the removed symbols (index exits + REIT)", r["removed_symbols"] == ["BIRET", "CCC", "DDD"], str(r["removed_symbols"]))
check("universe-report: breadth is recomputed over each set (exits trend down, so removing them RAISES breadth)",
      r["before"]["breadth_above_50dma"] < r["after"]["breadth_above_50dma"] and r["delta"]["breadth_pts"] > 0, str((r["before"], r["after"]["breadth_above_50dma"])))
check("universe-report: no-fundamentals names before/after, and which the filter removed",
      r["no_fundamentals_before"] == ["BBB", "BIRET", "CCC", "DDD", "TRUSTX"] and r["no_fundamentals_after"] == ["BBB", "TRUSTX"]
      and r["no_fundamentals_removed_by_filter"] == ["BIRET", "CCC", "DDD"], str(r["no_fundamentals_removed_by_filter"]))
check("universe-report: surfaces the trust-like listing that is NOT excluded",
      [x["symbol"] for x in r["trust_lookalikes_not_excluded"]] == ["TRUSTX"])

conn = fresh(); build_dq(conn)
with mock.patch.object(ingest, "connect", side_effect=dsn_conn), EXPECTED_PATCH:
    g0 = cl.get("/jobs/data-quality", headers=H).json()
    p = cl.post("/jobs/data-quality", headers=H)
    g1 = cl.get("/jobs/data-quality", headers=H).json()
check("data-quality: GET before any run says so; POST runs and returns the report; GET then returns it",
      g0["report"] is None and p.status_code == 200 and p.json()["severity"] == "ok" and g1["report"]["severity"] == "ok", str((g0, p.status_code)))

import upstox_client
class _TS2:
    def __init__(s, tok): s.tok = tok
    def valid_token(s): return s.tok
with mock.patch.object(upstox_client, "TokenStore", lambda: _TS2(None)):
    check("intraday-probe without a token -> 409 with a clear message", cl.get("/jobs/intraday-probe", headers=H).status_code == 409)
with conn.cursor() as cur:
    cur.execute("insert into symbols (symbol, is_active, in_nifty500, upstox_instrument_key) values ('RELIANCE', true, true, 'NSE_EQ|INE002A01018')")
conn.commit()
with mock.patch.object(upstox_client, "TokenStore", lambda: _TS2("t")), mock.patch.object(upstox_client, "UpstoxClient", lambda: FakeClient(date(2025, 1, 1))), \
     mock.patch.object(ingest, "connect", side_effect=dsn_conn):
    pr = cl.get("/jobs/intraday-probe?symbol=reliance&intervals=5,15", headers=H)
    bad_arg = cl.get("/jobs/intraday-probe?intervals=five", headers=H)
check("intraday-probe: returns the availability verdict per interval", pr.status_code == 200 and set(pr.json()["verdict"]) == {"5", "15"}, pr.text[:160])
check("intraday-probe: a malformed intervals argument is a 400, not a 500", bad_arg.status_code == 400)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
