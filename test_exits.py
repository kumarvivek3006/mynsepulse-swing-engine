"""
Tests for the exit-redesign study: the parameterised simulator reproduces the live exits exactly, each variant does what its
definition says on hand-made price paths, the shadow-detector and pocket_pivot rung mechanics, and the study end to end on synthetic data.
    python test_exits.py
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_det_momentum as DM
import e2_exits as EX
import e2_manage as MG
from e2_core import Candidate
from e2_detectors import detect_all
from e2_features import compute_features
from e2_synthmarket import make_market

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


print("BASE spec reproduces the live simulator exactly")
data = make_market(n_stocks=30, n_days=900, seed=11)
same = tot = 0
bad = []
for s, df in sorted(data.frames.items())[:30]:
    b = compute_features(s, df)
    if b is None:
        continue
    cands, _ = detect_all(b, lo=200)
    for c in cands:
        ev = data.events.get(s)
        a, x = MG.simulate(b, c, ev), EX.simulate_x(b, c, ev, EX.BASE)
        tot += 1
        ok = (a.skipped == x.skipped) and (a.skipped is not None or (abs(a.r - x.r) < 1e-12 and a.exits == x.exits and a.bars == x.bars
                                                                     and a.reason == x.reason and abs(a.mfe_r - x.mfe_r) < 1e-12))
        same += ok
        if not ok and len(bad) < 3:
            bad.append((s, c.detector, c.idx, a.reason, x.reason, a.r, x.r))
check(f"identical on all {tot} synthetic candidates", tot > 100 and same == tot, f"{same}/{tot} {bad}")


# ---- hand-made paths ---------------------------------------------------------------------------------------------
def mk(path, n_hist=260):
    """flat history at 100 (ATR ~2), then `path` = list of (o,h,l,c) bars"""
    rows = []
    d0 = pd.Timestamp("2025-01-01")
    for i in range(n_hist):
        rows.append((100.0, 101.0, 99.0, 100.0, 1e6))
    for o, h, l, c in path:
        rows.append((o, h, l, c, 1e6))
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "trade_date", pd.bdate_range(d0, periods=len(rows)))
    return df


def run(path, X, struct_stop=95.0):
    df = mk(path)
    b = compute_features("T", df)
    i = 259
    c = Candidate("T", "x", i, b.date[i], 101.0, struct_stop, 50.0, "f")
    return b, EX.simulate_x(b, c, None, X)


bar = lambda o, h, l, c: (o, h, l, c)       # noqa: E731
flat = lambda n, px=100.0: [bar(px, px + 1, px - 1, px)] * n      # noqa: E731

try:
    b, tr = run([bar(100, 100.5, 99.5, 100)] + flat(3), EX.SPECS["A"])
    atr = float(b.atr14[259])
    check("A: initial stop is entry - 2.0 ATR", abs((tr.entry - tr.stop0) - 2.0 * atr) < 1e-6, f"{tr.entry - tr.stop0} vs {2 * atr}")
    b, tr = run([bar(100, 100.5, 99.5, 100)] + flat(3), EX.SPECS["B"])
    check("B: initial stop is entry - 1.5 ATR", abs((tr.entry - tr.stop0) - 1.5 * float(b.atr14[259])) < 1e-6)
    b, tr = run([bar(100, 100.5, 99.5, 100)] + flat(3), EX.SPECS["D"], struct_stop=90.0)
    check("D: stop is 5% below the structural level", abs(tr.stop0 - 90.0 * 0.95) < 0.5 or tr.floored, f"{tr.stop0}")
except Exception as e:                                                         # noqa: BLE001
    check("path setup works", False, repr(e))

# B: partial at +1R (50%), then stop to break-even, then the remainder exits at the break-even stop
b, tr0 = run([bar(100, 100.5, 99.5, 100)] + flat(1), EX.SPECS["B"])
R = tr0.entry - tr0.stop0
up = tr0.entry + 1.05 * R
b, tr = run([bar(100, 100.5, 99.5, 100), bar(100, up, 99.8, up - 0.2)] + [bar(up - 0.2, up, up - 5 * R, up - 5 * R)], EX.SPECS["B"])
check("B: 50% is sold at +1R", tr.exits[0][3] == "T1" and abs(tr.exits[0][2] - 0.5) < 1e-9, str(tr.exits))
check("B: the rest leaves at break-even (not at the initial stop)", tr.exits[-1][3] in ("breakeven_stop", "stop", "stop_gap") and tr.exits[-1][1] >= tr.entry * 0.99 and tr.r > -0.5, f"{tr.exits} r={tr.r}")

# C: chandelier ratchets: a run-up then a pullback of 2 ATR from the high exits in profit, no partials
rise = [bar(100 + 2 * k, 101 + 2 * k + 1, 99 + 2 * k, 100 + 2 * k + 1) for k in range(1, 9)]
b, tr = run([bar(100, 100.5, 99.5, 100)] + rise + [bar(116, 116, 90, 90)], EX.SPECS["C"])
check("C: no partials, exits via the trailing stop in profit", all(x[3] != "T1" for x in tr.exits) and tr.r > 0 and tr.reason in ("stop", "stop_gap", "trail_stop", "breakeven_stop"), f"{tr.exits} r={tr.r}")
check("C: loss_stop is False when the trail exits above entry", getattr(tr, "loss_stop", True) is False)

# A: break-even after +1.5R (before any partial): touches 1.5R, falls back to entry -> exits flat, no T1
b, tr0 = run([bar(100, 100.5, 99.5, 100)] + flat(1), EX.SPECS["A"])
R = tr0.entry - tr0.stop0
hi = tr0.entry + 1.6 * R
b, tr = run([bar(100, 100.5, 99.5, 100), bar(100, hi, 99.9, hi - 0.1), bar(hi - 0.1, hi - 0.1, tr0.entry - 3 * R, tr0.entry - 3 * R)], EX.SPECS["A"])
check("A: stop moves to break-even after +1.5R, no partial taken", all(x[3] != "T1" for x in tr.exits) and tr.exits[-1][1] >= tr.entry * 0.99 and not getattr(tr, "loss_stop", False), f"{tr.exits}")

# time stop: flat path for 15 bars -> A/B exit at bar 16's open; hard vs conditional differ for B when the partial HAS filled
b, tr = run([bar(100, 100.5, 99.5, 100)] + flat(25), EX.SPECS["B"])
check("B: time stop at bar 15 exits the next open", tr.reason == "time_stop" and tr.bars == 16, f"{tr.reason} {tr.bars}")
b, tr = run([bar(100, 100.5, 99.5, 100)] + flat(30), EX.SPECS["A"])
check("A: time stop at bar 15 (below +0.5R)", tr.reason == "time_stop" and tr.bars == 16, f"{tr.reason} {tr.bars}")

print("pocket_pivot rungs and shadow detectors")
bars = {}
for s, df in sorted(data.frames.items()):
    bb = compute_features(s, df)
    if bb is not None:
        bars[s] = bb
counts = {r: sum(len(DM.pocket_pivot(bb, {}, 200, bb.n, strict=True, rung=r)) for bb in bars.values()) for r in (0, 1, 2, 3, 4, 5)}
check("each rung is a subset of the one before (counts non-increasing)", all(counts[r] >= counts[r + 1] for r in range(5)) and counts[0] > 0, str(counts))
ok = True
for bb in bars.values():
    for c in DM.pocket_pivot(bb, {}, 200, bb.n, strict=True, rung=3):
        i = c.idx
        ok &= bool(c.meta["vol_vs_down"] >= 1.5 and bb.relvol[i] >= 2.0 and bb.close_pos[i] >= 0.75)
check("rung 3 hits satisfy every clause", ok)
check("default rung is 3", DM.PP_RUNG == 3)
ok5, any5 = True, 0
for bb in bars.values():
    for c in DM.pocket_pivot(bb, {}, 200, bb.n, strict=True, rung=5):
        any5 += 1
        ok5 &= bool((1 - bb.c[c.idx] / bb.high52[c.idx]) * 100 <= 2.0 and c.meta["near_52w_high_2pct"] and (bb.c[c.idx] / bb.ema20[c.idx] - 1) * 100 <= 5.0)
check("rung 5 hits are within 2% of the 52-week high (and satisfy rung 4)", ok5, str(any5))
flags = [c.meta["near_52w_high_2pct"] for bb in bars.values() for c in DM.pocket_pivot(bb, {}, 200, bb.n, strict=True, rung=3)]
check("every live (rung 3) candidate carries the near_52w_high_2pct flag", len(flags) > 0 and all(isinstance(f, bool) for f in flags))
import e2_live as LV
check("pocket_pivot is a shadow detector by default", "pocket_pivot" in LV.SHADOW_DETECTORS)

print("the study end to end on synthetic data")
out = {}
EX.compute(make_market(n_stocks=25, n_days=950, seed=5), out)
v = out["variants"]
check("all variants reported", set(v) == set(EX.ALL), str(list(v)))
check("candidate count is the same for every variant", len({x["candidates"] for x in v.values()}) == 1)
check("filled counts are close across variants (exits only; skips depend on the stop only at the open)", max(x["filled"] for x in v.values()) - min(x["filled"] for x in v.values()) <= 0.05 * max(x["filled"] for x in v.values()), str({k: x["filled"] for k, x in v.items()}))
check("rule flag matches the numbers", all(x["positive_both_halves_and_full"] == (x["net_r"] > 0 and x["first_half"]["net_r"] > 0 and x["second_half"]["net_r"] > 0) for x in v.values()))
check("stop-out rates are percentages", all(0 <= x["stop_out_pct"] <= 100 and 0 <= x["early_stop_out_pct"] <= x["stop_out_pct"] for x in v.values()))
check("report is JSON-serialisable and renders", bool(EX.render_markdown({**out, "state": "done"})) and bool(__import__("json").dumps(out, default=str)))
check("paired deltas exist for every non-base variant", set(out["paired_vs_base"]) == set(EX.ALL) - {"BASE"})


print("horizon rulebooks (e2_exits_wl)")
import e2_exits_wl as XW
b, t0 = run([bar(100, 100.5, 99.5, 100)] + flat(1), XW.H5["S3"])
R = t0.entry - t0.stop0
b, tr = run([bar(100, 100.5, 99.5, 100), bar(100, t0.entry + 1.6 * R, 99.9, t0.entry + 1.2 * R)] + flat(10), XW.H5["S3"])
check("S3: the whole position is sold at +1.5R and the trade ends there", len(tr.exits) == 1 and abs(tr.exits[0][2] - 1.0) < 1e-9 and tr.reason == "T1" and tr.r > 1.2, f"{tr.exits} {tr.reason} {tr.r}")
b, tr = run([bar(100, 100.5, 99.5, 100)] + flat(12), XW.H5["S1"])
check("S1: hard exit at the next open after bar 5", tr.reason == "time_stop" and tr.bars == 6, f"{tr.reason} {tr.bars}")
b, tr = run([bar(100, 100.5, 99.5, 100)] + flat(30), XW.H20["L1"])
check("L1: no time stop before bar 20, hard exit after it", tr.reason == "time_stop" and tr.bars == 21, f"{tr.reason} {tr.bars}")
b, tr0 = run([bar(100, 100.5, 99.5, 100)] + flat(1), XW.H20["L1"])
R = tr0.entry - tr0.stop0
b, tr = run([bar(100, 100.5, 99.5, 100), bar(100, tr0.entry + 3.1 * R, 99.9, tr0.entry + 2.5 * R)] + flat(25), XW.H20["L1"])
check("L1: 50% sold at +3R, then the rest runs to the time exit", tr.exits[0][3] == "T1" and abs(tr.exits[0][2] - 0.5) < 1e-9, str(tr.exits))
b, t0 = run([bar(100, 100.5, 99.5, 100)] + flat(1), XW.H5["S1"])
R = t0.entry - t0.stop0
b, tr = run([bar(100, 100.5, 99.5, 100), bar(100, t0.entry + 0.9 * R, 99.9, t0.entry + 0.8 * R), bar(t0.entry + 0.8 * R, t0.entry + 0.8 * R, t0.entry - 0.5 * R, t0.entry - 0.5 * R)] + flat(5), XW.H5["S1"])
check("trail 'none': a run-up then a pullback does not exit at break-even or a trailed level (the initial stop is not touched)", tr.reason == "time_stop" and tr.r < 0.2, f"{tr.reason} {tr.exits}")

print("the whitelist study end to end on synthetic data")
XW.WHITELIST = ("pullback_ema", "pocket_pivot")
out2 = {}
XW.compute(make_market(n_stocks=25, n_days=950, seed=5), out2)
check("every rulebook reported for each detector and both together", all(set(out2["by_detector"][d]) == set(XW.ALL) for d in ("pullback_ema", "pocket_pivot", "BOTH")))
check("rules evaluated", "passing" in out2["part1_rule"] and "mode_5_session_passing" in out2["part2_rule"])
check("candidate counts identical across rulebooks", all(len({v["candidates"] for v in out2["by_detector"][d].values()}) == 1 for d in out2["by_detector"]))
check("renders and serialises", bool(XW.render_markdown({**out2, "state": "done"})) and bool(__import__("json").dumps(out2, default=str)))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
