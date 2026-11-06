"""Tests for the one-detector forward-return diagnosis. python test_fwddiag.py"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_fwddiag as FD
from e2_synthmarket import make_market

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


print("analysis on a planted effect")
import pandas as pd
rng = np.random.default_rng(3)
n = 1200
x = rng.normal(size=n)
sig = np.datetime64("2024-01-01") + (np.arange(n) // 6) * np.timedelta64(7, "D")
y20 = 1.0 * x + rng.normal(scale=2.0, size=n)
df = pd.DataFrame({"sig": sig.astype("datetime64[ns]"), "symbol": "X", "idx": 0, "adj10": y20 * 0.5, "adj20": y20, "good": x, "noise": rng.normal(size=n)})
ctrl = {10: (rng.normal(size=500), np.array([np.datetime64("2024-06-01")] * 500)), 20: (rng.normal(size=500), np.array([np.datetime64("2024-06-01")] * 500))}
res = FD.analyse(df, ctrl, np.datetime64("2025-01-01"))
byf = {r["feature"]: r for r in res["features"]}
check("the planted feature is monotone, significant and sign-stable", byf["good"]["h20"]["interval_excludes_zero"] and byf["good"]["h20"]["same_sign_both_halves"] and byf["good"]["h20"]["rho"] > 0.3, str(byf["good"]["h20"]))
check("the noise feature is not flagged", not (byf["noise"]["h20"]["interval_excludes_zero"] and byf["noise"]["h20"]["same_sign_both_halves"]))
chk = {c["feature"]: c for c in res["held_out_checks"]}
check("held-out check exists for the planted feature only", set(chk) == {"good"}, str(list(chk)))
check("kept observations beat the dropped ones on the held-out half", chk["good"]["second_half_kept"]["mean20"] > chk["good"]["second_half_dropped"]["mean20"])
check("cut is the FIRST-half median", abs(float(chk["good"]["rule"].split()[1]) - float(np.median(x[sig < np.datetime64("2025-01-01")]))) < 0.01, chk["good"]["rule"])

print("end to end on synthetic data")
out = {}
d = make_market(n_stocks=25, n_days=950, seed=5)
FD.compute(d, "pullback_ema", out)
check("candidates found and overall reported", out.get("n", 0) > 30 and "20" in out["overall"], str({k: out.get(k) for k in ("n", "error")}))
check("detector-recorded meta features included (depth, bars down)", {"meta_depth_pct", "meta_bars_down"} <= {r["feature"] for r in out["features"]})
check("renders and serialises", bool(FD.render_markdown({**out, "state": "done"})) and bool(json.dumps(out, default=str)))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
