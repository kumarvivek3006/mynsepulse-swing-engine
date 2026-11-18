"""Provisional rows' fate is logged by name (owner, 18 Nov). Needs the test Postgres. python test_publish_log.py"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date

import psycopg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_publish as PUB

DSN = os.environ.get("TEST_PG_DSN", "host=127.0.0.1 port=55432 user=postgres dbname=eng options='-c search_path=swing,public'")
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


class H(logging.Handler):
    def __init__(self):
        super().__init__(); self.lines = []

    def emit(self, r):
        self.lines.append(r.getMessage())


h = H()
logging.getLogger("engine2.publish").addHandler(h)
logging.getLogger("engine2.publish").setLevel(logging.INFO)

conn = psycopg.connect(DSN)
with conn.cursor() as cur:
    cur.execute("truncate symbols, signals, market_regime, engine_settings cascade")
    for s in ("AAA", "BBB", "CCC"):
        cur.execute("insert into symbols (symbol, is_active) values (%s, true)", (s,))
        cur.execute("insert into ohlcv_daily (symbol, trade_date, open, high, low, close, volume) values (%s, '2026-11-09', 100,101,99,100,1000)", (s,))
conn.commit()


def pick(sym, det="gap_and_go", score=70.0):
    return {"symbol": sym, "detector": det, "family": "momentum", "score": score, "quality": 70, "signal_date": "2026-11-09",
            "plan": {"entry_ref": 100.0, "stop": 95.0, "t1": 110.0, "t2": 115.0, "shares": 10, "risk_pct": 5.0, "trail": "x", "time_stop": "20"}}


out = {"as_of": "2026-11-09", "regime": {"state": "neutral", "score": 55, "size_mult": 1.0, "max_positions": 6}, "market": {}, "picks": [pick("AAA"), pick("BBB")]}
d = date(2026, 11, 9)
out["provisional"] = True
r1 = PUB.publish(conn, out, "intraday", today=d)
check("intraday publish wrote two provisional rows", r1["inserted"] == 2, r1)
h.lines.clear()
out2 = {**out, "picks": [pick("AAA"), pick("CCC")]}
out2["provisional"] = False
r2 = PUB.publish(conn, out2, "postclose", today=d)
check("BBB (provisional, not re-picked) withdrawn by name", r2["withdrawn_symbols"] == ["BBB:gap_and_go"], r2.get("withdrawn_symbols"))
check("AAA (provisional, re-picked) confirmed by name", r2["confirmed_symbols"] == ["AAA:gap_and_go"], r2.get("confirmed_symbols"))
line = [l for l in h.lines if "provisional rows settled" in l]
check("one settled line, naming both", len(line) == 1 and "BBB:gap_and_go" in line[0] and "AAA:gap_and_go" in line[0], line)
h.lines.clear()
PUB.publish(conn, {**out, "picks": [pick("AAA")]}, "intraday", today=d)
check("no settled line on an intraday (provisional) publish", not [l for l in h.lines if "settled" in l])
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
