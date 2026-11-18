"""Tests for the 15-minute sample probe. Needs the test Postgres (see test_step1.py). python test_probe_sample.py"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta

import psycopg

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_probe30 as P

DSN = os.environ.get("TEST_PG_DSN", "host=127.0.0.1 port=55432 user=postgres dbname=eng options='-c search_path=swing,public'")
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


conn = psycopg.connect(DSN)
with conn.cursor() as cur:
    pass  # trading_holidays comes from the 14 Sep migration (see the test DB setup)
    cur.execute("truncate symbols, ohlcv_daily, engine_settings, trading_holidays cascade")
    for i in range(40):
        sym = f"S{i:02d}"
        listing = date(2025, 6, 1) if i == 39 else None
        cur.execute("insert into symbols (symbol, upstox_instrument_key, is_active, listing_date) values (%s,%s,true,%s)", (sym, f"NSE_EQ|K{i:02d}", listing))
        for d in range(15):
            cur.execute("insert into ohlcv_daily (symbol, trade_date, open, high, low, close, volume) values (%s,%s,10,10,10,10,%s)",
                        (sym, date.today() - timedelta(days=d), (40 - i) * 1000))
conn.commit()

print("sample selection")
s = P.pick_sample(conn, 15)
check("15 symbols, most and least liquid both included", len(s) == 15 and s[0]["symbol"] == "S00" and s[-1]["symbol"] == "S39", [x["symbol"] for x in s])
check("ranks ascend", [x["liquidity_rank"] for x in s] == sorted(x["liquidity_rank"] for x in s))


class Client:
    def __init__(self, span_ok=30, fail_old_for=()):
        self.calls, self.span_ok, self.fail_old_for = [], span_ok, fail_old_for

    def _get(self, path):
        self.calls.append(path)
        parts = path.split("/")
        frm, to = parts[-1], parts[-2]
        days = (date.fromisoformat(to) - date.fromisoformat(frm)).days + 1 if to >= frm else (date.fromisoformat(frm) - date.fromisoformat(to)).days + 1
        key = parts[3]
        if days > 1:
            if days > self.span_ok:
                return {"status": "error", "errors": [{"message": "range too long"}]}
            candles = [[f"{(date.fromisoformat(to) - timedelta(days=k)).isoformat()}T09:15:00+05:30"] + [1] * 5 for k in range(min(days, 20))]
            return {"status": "success", "data": {"candles": candles}}
        if key in self.fail_old_for and date.fromisoformat(frm) < date.today() - timedelta(days=400):
            return {"status": "success", "data": {"candles": []}}
        return {"status": "success", "data": {"candles": [[f"{frm}T09:{15 + k}:00+05:30", 1, 1, 1, 1, 1] for k in range(25)]}}


print("run with a fake client")
c = Client(span_ok=30, fail_old_for=("NSE_EQ|K20",))
out = P.run_sample(conn=conn, client=c)
check("ran and done", out["ran"] and out["state"] == "done", out.get("error"))
check("coverage counted; one symbol short on old dates", out["coverage"]["symbols"] == 15 and out["coverage"]["symbols_fully_covered"] <= 14 and 0 < out["coverage"]["share_full"] < 1, out["coverage"])
check("a symbol listed after a date is not penalised", any(c_["status"] == "not_listed_yet" for sm in out["sample"] for c_ in sm["cells"]))
check("30-day span accepted, 90-day refused -> request span 30", out["backfill_estimate"]["request_span_days"] == 30 and out["backfill_estimate"]["requests_per_symbol"] == 49, out["backfill_estimate"])
check("estimate carries limits and storage rows", {"limit_per_day", "days_at_daily_limit", "rows_if_stored", "requests_total"} <= set(out["backfill_estimate"]))
out2 = P.run_sample(conn=conn, client=Client(span_ok=1))
check("no span accepted -> one request per day, flagged by the estimate", out2["backfill_estimate"]["request_span_days"] == 1 and out2["backfill_estimate"]["requests_total"] == 497 * 1460)
txt = P.render_sample(out)
check("renders", "coverage:" in txt and "backfill estimate" in txt and "span tests" in txt)
with conn.cursor() as cur:
    cur.execute("select count(*) from engine_settings where key = %s", (P.SAMPLE_KEY,))
    check("stored under its key", cur.fetchone()[0] == 1)
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
