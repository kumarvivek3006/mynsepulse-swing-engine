"""
Signal verification tests, against a REAL PostgreSQL with the engine's schema.sql + migrations (like test_step1.py).

    export TEST_PG_DSN="host=127.0.0.1 port=55432 user=postgres dbname=eng options='-c search_path=swing,public'"
    DATABASE_URL=postgresql://postgres@127.0.0.1:55432/eng INTERNAL_API_KEY=k python test_verification.py
"""
from __future__ import annotations

import logging
import os
import sys
import uuid

import psycopg

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/eng")
os.environ.setdefault("INTERNAL_API_KEY", "k")
DSN = os.environ.get("TEST_PG_DSN", "host=127.0.0.1 port=55432 user=postgres dbname=eng options='-c search_path=swing,public'")
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


db = psycopg.connect(DSN, autocommit=True)
cur = db.cursor()
cur.execute("truncate symbols, engine_settings cascade")
for col in ("verification_outcome", "verification_reason", "verification_note", "verification_at"):
    cur.execute(f"alter table signals drop column if exists {col}")          # start from the pre-feature schema
cur.execute("drop index if exists signals_unreviewed_idx")
for s in ("AAA", "BBB", "CCC", "DDD"):
    cur.execute("insert into symbols (symbol, company_name) values (%s, %s)", (s, s + " Ltd"))


def sig(symbol, pattern, as_of, status="pending", hours_old=100, withdrawn=False, score=70):
    sid = str(uuid.uuid4())
    cur.execute("""insert into signals (id, symbol, generated_at, as_of_date, setup_type, pattern, entry_trigger, stop_loss, t1,
                   r_multiple_t1, score_total, score_breakdown, status, notes)
                   values (%s,%s, now() - make_interval(hours => %s), %s, 'breakout', %s, 100, 95, 110, 2, %s, '{}'::jsonb, %s, %s::jsonb)""",
                (sid, symbol, hours_old, as_of, pattern, score, status, '{"withdrawn": "x"}' if withdrawn else '{}'))
    return sid


print("verification schema")
a = sig("AAA", "pocket_pivot", "2026-11-02")                       # unreviewed, old
b = sig("BBB", "episodic_pivot", "2026-11-02")                     # will be skipped
c = sig("CCC", "pullback_ema", "2026-11-02", status="triggered")   # an existing taken trade (has a fill) - must be back-filled
d = sig("DDD", "pocket_pivot", "2026-11-05", hours_old=5)          # recent: not yet 48 h
w = sig("AAA", "gap_and_go", "2026-11-02", status="expired", withdrawn=True)   # withdrawn provisional: never counted
old = sig("BBB", "vcp", "2026-09-01")                              # before the cutover: never counted
cur.execute("insert into signal_outcomes (signal_id, entry_date, entry_price) values (%s, '2026-11-03', 101)", (c,))

import verification as V
check("ensure_schema applies on a pre-feature schema", V.ensure_schema() is True)
check("ensure_schema is idempotent", V.ensure_schema() is True)
cur.execute("select verification_outcome from signals where id = %s", (c,))
check("existing trade with a fill back-filled to 'taken'", cur.fetchone()[0] == "taken")
cur.execute("select count(*) from signals where verification_outcome is not null")
check("nothing else back-filled", cur.fetchone()[0] == 1)

from fastapi.testclient import TestClient
import server
cl = TestClient(server.app)
H = {"x-internal-key": "k"}
r = cl.post("/signals/verify", json={"signal_id": b, "outcome": "skipped"})
check("verify without the key is refused", r.status_code == 401)
r = cl.post("/signals/verify", headers=H, json={"signal_id": b, "outcome": "skipped"})
check("skipped without a reason is a 400", r.status_code == 400, r.text)
r = cl.post("/signals/verify", headers=H, json={"signal_id": b, "outcome": "skipped", "reason": "made_up"})
check("skipped with an unknown reason is a 400", r.status_code == 400)
r = cl.post("/signals/verify", headers=H, json={"signal_id": b, "outcome": "taken", "reason": "bad_rr"})
check("reason with outcome taken is a 400", r.status_code == 400)
r = cl.post("/signals/verify", headers=H, json={"signal_id": b, "outcome": "maybe"})
check("unknown outcome is a 400", r.status_code == 400)
r = cl.post("/signals/verify", headers=H, json={"signal_id": str(uuid.uuid4()), "outcome": "taken"})
check("unknown signal is a 404", r.status_code == 404)
r = cl.post("/signals/verify", headers=H, json={"signal_id": c, "outcome": "skipped", "reason": "bad_rr"})
check("cannot skip a signal with an open trade (409)", r.status_code == 409, r.text)
r = cl.post("/signals/verify", headers=H, json={"signal_id": b, "outcome": "skipped", "reason": "pattern_absent", "note": "no base"})
check("skip with reason + note is accepted", r.status_code == 200 and r.json()["verification_reason"] == "pattern_absent", r.text)
cur.execute("select verification_outcome, verification_reason, verification_note, verification_at is not null from signals where id = %s", (b,))
check("stored", cur.fetchone() == ("skipped", "pattern_absent", "no base", True))

print("reconciliation + status")
rec = cl.get("/jobs/verification-status", headers=H).json()
ids = {i["id"] for i in rec["reconciliation"]["items"]}
check("status endpoint works", rec["schema_ready"] is True and "by_detector" in rec, str(rec)[:200])
check("only the old unreviewed confirmed signal is listed (not recent, withdrawn, pre-cutover, taken or skipped)", ids == {a}, str(ids))
check("count matches and is not healthy", rec["reconciliation"]["count"] == 1 and rec["reconciliation"]["healthy"] is False)
bd = {x["detector"]: x for x in rec["by_detector"]}
check("per-detector tallies", bd["pullback_ema"]["taken"] == 1 and bd["episodic_pivot"]["skipped"] == 1 and bd["pocket_pivot"]["unreviewed"] == 2, str(bd))
check("withdrawn and pre-cutover rows are not in the tallies", "gap_and_go" not in bd and "vcp" not in bd)
check("skip reasons clustered by detector", rec["skip_reasons"] == [{"detector": "episodic_pivot", "reason": "pattern_absent", "count": 1}], str(rec["skip_reasons"]))
r = cl.post("/signals/verify", headers=H, json={"signal_id": a, "outcome": "taken"})
rec = cl.get("/jobs/verification-status", headers=H).json()
check("after reviewing it the reconciliation is healthy (0)", rec["reconciliation"]["count"] == 0 and rec["reconciliation"]["healthy"])
r = cl.post("/signals/verify", headers=H, json={"signal_id": a, "outcome": None})
cur.execute("select verification_outcome, verification_at from signals where id = %s", (a,))
check("a mark can be cleared", cur.fetchone() == (None, None))
check("the daily slot function returns the count", V.run_reconciliation()["count"] == 1)

print("existing endpoints carry the fields")
r = cl.get("/daily-log", headers=H, params={"day": "2026-11-02"})
check("daily-log still answers", r.status_code == 200, r.text[:200])
rows = {s["symbol"] + s["pattern"]: s for s in r.json()["signals"]}
check("daily-log rows carry verification fields", rows["BBBepisodic_pivot"]["verification_outcome"] == "skipped"
      and rows["BBBepisodic_pivot"]["verification_reason"] == "pattern_absent", str(rows.get("BBBepisodic_pivot")))
r = cl.get("/signals", headers=H)
check("/signals still answers", r.status_code == 200, r.text[:200])

print("take sets the outcome; verification trouble never blocks a take")
e = sig("DDD", "breakout_retest", "2026-11-05", hours_old=1)
r = cl.post("/signals/take", headers=H, json={"signal_id": e, "entry_price": 101.5})
cur.execute("select verification_outcome, status from signals where id = %s", (e,))
check("/signals/take also marks taken", r.status_code == 200 and cur.fetchone() == ("taken", "triggered"), r.text)
cur.execute("select entry_price from signal_outcomes where signal_id = %s", (e,))
check("fill price recorded as before", float(cur.fetchone()[0]) == 101.5)
f = sig("DDD", "flat_base", "2026-11-05", hours_old=1)
from unittest import mock
with mock.patch.object(V, "mark_taken", side_effect=RuntimeError("boom")):
    r = cl.post("/signals/take", headers=H, json={"signal_id": f, "entry_price": 50})
cur.execute("select status from signals where id = %s", (f,))
check("a verification failure does not block the take", r.status_code == 200 and cur.fetchone()[0] == "triggered", r.text)

print("schema missing -> degrade loudly, not silently")
for col in ("verification_outcome", "verification_reason", "verification_note", "verification_at"):
    cur.execute(f"alter table signals drop column if exists {col}")
V._state.update(ready=False, checked=0.0, error="simulated")
r = cl.post("/signals/verify", headers=H, json={"signal_id": a, "outcome": "taken"})
check("verify answers 503 with the reason", r.status_code == 503 and "not available" in r.text, r.text)
r = cl.get("/signals", headers=H)
check("/signals is unaffected without the columns", r.status_code == 200)
r = cl.get("/jobs/verification-status", headers=H)
check("status says schema_ready false", r.json()["schema_ready"] is False)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
