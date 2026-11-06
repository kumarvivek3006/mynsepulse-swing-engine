"""backfill_prices names the symbol whose history fetch the broker rejects, and returns exactly what it did before."""
from __future__ import annotations
import logging, os, sys
from datetime import date
from unittest import mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/eng")
import ingest, psycopg
from upstox_client import UpstoxClient

PASS = FAIL = 0
def check(n, c, d=""):
    global PASS, FAIL
    PASS += c; FAIL += (not c); print(("  ok    " if c else "  FAIL  ") + n, d if not c else "")

conn = ingest.connect()
with conn.cursor() as cur:
    cur.execute("truncate symbols cascade")
    for s, k in (("GOOD", "NSE_EQ|INE000000001"), ("BADK", "NSE_EQ|INE133A01011")):
        cur.execute("insert into symbols (symbol, upstox_instrument_key, is_active, series) values (%s,%s,true,'EQ')", (s, k))
        cur.execute("insert into ohlcv_daily (symbol, trade_date, open, high, low, close, volume) values (%s,'2026-10-21',10,11,9,10,1000)", (s,))
conn.commit()

client = UpstoxClient.__new__(UpstoxClient)
def fake_get(path):
    if "INE133A01011" in path:
        return {"status": "error", "errors": [{"errorCode": "UDAPI100011", "message": "Invalid Instrument key"}]}
    return {"status": "success", "data": {"candles": [["2026-10-22T00:00:00+05:30", 10, 11, 9, 10.5, 2000, 0]]}}
records = []
class H(logging.Handler):
    def emit(self, r): records.append((r.levelname, r.getMessage()))
for n in ("ingest", "upstox_client"):
    logging.getLogger(n).addHandler(H()); logging.getLogger(n).setLevel(logging.INFO)

with mock.patch.object(client, "_get", side_effect=fake_get), mock.patch.object(ingest, "date") as d:
    d.today.return_value = date(2026, 10, 22)
    total = ingest.backfill_prices(conn, client)
msgs = [m for _, m in records]
check("the good symbol still backfills (1 bar)", total == 1, str(total))
check("the per-call error line is unchanged apart from naming the symbol",
      any(m.startswith("history failed NSE_EQ|INE133A01011 (BADK) 2026-10-22") for m in msgs), str(msgs))
w = [m for lv, m in records if lv == "WARNING" and "rejected by the history endpoint" in m]
check("one summary line names the symbol, key, code and last stored bar",
      len(w) == 1 and "BADK NSE_EQ|INE133A01011 UDAPI100011 last_stored_bar=2026-10-21" in w[0], str(w))
print(f"\n{PASS} passed, {FAIL} failed"); sys.exit(1 if FAIL else 0)
