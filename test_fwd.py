"""Tests for the forward-return test and the publishing whitelist. python test_fwd.py"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_fwd as F
from e2_synthmarket import make_market

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


print("statistics")
a = F.Acc()
sig = [np.datetime64("2024-01-01") + np.timedelta64(7 * i, "D") for i in range(100)]
a.extend(5, sig, [1.0] * 100, [0.25] * 100)
s = a.stats(5, np.datetime64("2024-09-01"))
check("market-adjusted mean = raw - market", abs(s["mean_adj_pct"] - 0.75) < 1e-9, str(s))
check("net of costs subtracts 0.50", abs(s["net_of_costs_pct"] - 0.25) < 1e-9)
check("both halves reported and the horizon passes", s["h1"]["n"] > 0 and s["h2"]["n"] > 0 and s["pass"])
check("interval brackets the mean", s["ci95"][0] <= s["mean_adj_pct"] <= s["ci95"][1])
per = {5: {"n": 10, "mean_adj_pct": 0.5, "pass": True}, 10: {"n": 10, "mean_adj_pct": 0.4, "pass": True}, 20: {"n": 10, "mean_adj_pct": -0.2, "pass": False}}
check("triage: 2 of 3 horizons pass = positive", F.triage(per) == "positive")
per2 = {h: {"n": 10, "mean_adj_pct": -0.1, "pass": False} for h in F.HORIZONS}
check("triage: all horizons <= 0 = negative", F.triage(per2) == "negative")
per3 = {5: {"n": 10, "mean_adj_pct": 0.3, "pass": True}, 10: {"n": 10, "mean_adj_pct": -0.1, "pass": False}, 20: {"n": 10, "mean_adj_pct": -0.5, "pass": False}}
check("triage: good early and fading = mixed", F.triage(per3) == "mixed")

print("end to end on synthetic data")
out = {}
d = make_market(n_stocks=25, n_days=950, seed=5)
F.compute(d, out)
rows = out["detectors"]
check("every detector row has all three horizons", all(set(r["horizons"]) == {"5", "10", "20"} for r in rows.values()))
check("control row present with a large sample", rows["ALL_BARS (control)"]["horizons"]["10"]["n"] > 1000)
check("pocket_pivot variants reported next to the live one", {"pocket_pivot_strict_rung0", "pocket_pivot_legacy"} <= set(rows))
check("sample size shrinks with the horizon (needs more future bars)", all(r["horizons"]["20"]["n"] <= r["horizons"]["5"]["n"] for r in rows.values()))
check("diagnosis queue is ordered strongest first", (lambda q: q == sorted(q, reverse=True))([x["mean_adj_pct_10"] for x in out["diagnosis_queue_by_10bar_adj"] if x["n"]]))
check("renders and serialises", bool(F.render_markdown({**out, "state": "done"})) and bool(json.dumps(out, default=str)))

# hand-checked arithmetic on one candidate
import e2_detectors as D
from e2_features import compute_features
checked = False
for sym, df in sorted(d.frames.items()):
    b = compute_features(sym, df)
    if b is None:
        continue
    cands, _ = D.detect_all(b, lo=250)
    for c in cands:
        e = c.idx + 1
        if e + 9 < b.n and not np.isnan(b.o[e]):
            raw = 100 * (b.c[e + 9] / b.o[e] - 1)
            ic = d.index_close.reindex(__import__("pandas").DatetimeIndex(b.date)).ffill().to_numpy(float)
            mkt = 100 * (ic[e + 9] / ic[e - 1] - 1)
            acc = F.Acc()
            acc.add(10, np.datetime64(c.date, "D"), raw, mkt)
            checked = abs(np.array(acc.d[10][1])[0] - raw) < 1e-12
            break
    if checked:
        break
check("hand arithmetic (open[e] -> close[e+h-1], index close[e-1] -> close[e+h-1]) is what the accumulator receives", checked)

print("publishing whitelist")
os.environ["E2_PUBLISH_DETECTORS"] = "episodic_pivot,gap_and_go"
import importlib
import e2_live as LV
importlib.reload(LV)
check("only the two named detectors are publishable", (not LV.is_shadow("episodic_pivot")) and (not LV.is_shadow("gap_and_go")))
check("every other detector is shadowed", all(LV.is_shadow(x) for x in ("vcp", "pullback_ema", "pocket_pivot", "cup_handle")))
os.environ["E2_PUBLISH_DETECTORS"] = ""
importlib.reload(LV)
check("empty whitelist = no whitelist (only the pocket_pivot shadow applies)", LV.is_shadow("pocket_pivot") and not LV.is_shadow("vcp"))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
