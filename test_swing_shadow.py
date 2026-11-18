"""oversold_dip as a SHADOW-ONLY live detector (owner, 18 Nov). Needs the test Postgres. python test_swing_shadow.py"""
from __future__ import annotations
import logging, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/eng")
os.environ.setdefault("INTERNAL_API_KEY", "k")
logging.disable(logging.CRITICAL)
os.environ["E2_PUBLISH_DETECTORS"] = ""            # worst case: no whitelist, no shadow list
os.environ["E2_SHADOW_DETECTORS"] = ""
import ingest, e2_live, e2_volmatch as VM, e2_swing_shadow as SW, e2_detectors
from e2_features import compute_features
from e2_synthmarket import make_market
PASS = FAIL = 0
def check(n, c, d=""):
    global PASS, FAIL
    PASS += bool(c); FAIL += (not c); print(("  ok    " if c else "  FAIL  ") + n, d if not c else "")

print("registry isolation")
check("oversold_dip is NOT in the position engine's registry", "oversold_dip" not in e2_detectors.DETECTORS)
check("is_shadow is True with no whitelist and no shadow list", e2_live.is_shadow("oversold_dip") and not e2_live.is_shadow("gap_and_go"))

print("definition equals the entry test's")
data = make_market(n_stocks=40, n_days=1100, seed=11)
tot = 0
same = True
for sym, df in sorted(data.frames.items()):
    b = compute_features(sym, df)
    if b is None:
        continue
    a = [c.idx for c in SW.oversold_dip(b, {}, 205, b.n)]
    v = VM.oversold_signals(b, 205)["oversold_dip"]
    tot += len(a)
    same &= (a == v)
check("same signal bars as e2_volmatch.oversold_signals on every symbol", same and tot > 0, (tot, same))
b0 = compute_features(sorted(data.frames)[0], data.frames[sorted(data.frames)[0]])
c0 = [c for s_ in sorted(data.frames) for c in SW.oversold_dip(compute_features(s_, data.frames[s_]), {}, 205, 10**6)][:1]
check("candidate is a long reversal-family candidate with the day's low as the structural stop", c0 and c0[0].family == "reversal" and c0[0].stop < c0[0].pivot and "shadow_only" in c0[0].meta)

print("live scan: rows, never selected, flagged in the forward log")
class Stub:                                                        # fires on the last bar of every symbol, so the scan path is exercised
    @staticmethod
    def fn(b, ctx, lo, hi):
        from e2_core import Candidate
        i = b.n - 1
        return [Candidate(b.symbol, "oversold_dip", i, b.date[i], float(b.c[i]), float(b.l[i]) , 50.0, "reversal", {"shadow_only": "x"})] if lo <= i < hi else []
orig = dict(SW.SWING_DETECTORS)
SW.SWING_DETECTORS.clear(); SW.SWING_DETECTORS["oversold_dip"] = Stub.fn
conn = ingest.connect()
with conn.cursor() as cur:
    cur.execute("delete from engine_settings where key in ('engine2_shadow_log','engine2_shadow_latest')")
conn.commit()
out = e2_live.scan(conn, "postclose", data=make_market(n_stocks=40, n_days=700, seed=3))
sw = [r for r in out["all_candidates"] if r["detector"] == "oversold_dip"]
check("swing rows are scored and ranked", len(sw) > 0 and all(r["score"] is not None for r in sw), len(sw))
check("none selected, why_not shadow_detector", all(r["selected"] is False and r["why_not"] == "shadow_detector" for r in sw))
check("none among the picks", all(p["detector"] != "oversold_dip" for p in out["picks"]))
check("they rank after every publishable row", all(r["rank"] > max([x["rank"] for x in out["all_candidates"] if x["detector"] != "oversold_dip" and not x.get("shadow")] or [0]) for r in sw))
flog, _ = e2_live._get(conn, "engine2_shadow_log")
fl = [e for e in flog if e["d"] == "oversold_dip"]
check("forward-log entries carry x=swing_shadow and sel False", fl and all(e.get("x") == "swing_shadow" and e["sel"] is False for e in fl))
fs = e2_live.forward_summary([dict(e, r=0.5) for e in fl] + [{"s": "A", "d": "gap_and_go", "dt": "2026-01-01", "r": 1.0, "sel": True}])
check("position engine's forward summary excludes them; own labelled bucket", fs["closed"] == 1 and "oversold_dip" not in fs["by_detector"] and fs["swing_shadow"]["closed"] == len(fl) and "swing exits" in fs["swing_shadow"]["note"])
SW.SWING_DETECTORS.clear(); SW.SWING_DETECTORS.update(orig)

print("switch off")
SW.ENABLED = False
check("E2_SWING_SHADOW=0 -> nothing", SW.detect(b0, 0) == [])
SW.ENABLED = True

print("forward-return test picks it up")
import e2_fwd
out2 = {}
e2_fwd.compute(make_market(n_stocks=30, n_days=1000, seed=5), out2)
check("e2_fwd reports oversold_dip next to the control at every horizon (when it fires)", "ALL_BARS (control)" in out2["detectors"] and (("oversold_dip" in out2["detectors"]) or tot == 0))
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
