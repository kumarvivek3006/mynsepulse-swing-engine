"""Tests for the gap_and_hold conditional check. python test_swingcheck.py"""
from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_swingcheck as SC
from e2_synthmarket import make_market

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


print("owner's decision rule")
check("below +0.10% -> drop gap_and_hold, build episodic_pivot_swing", SC.verdict(0.09).startswith("drop") and SC.verdict(-1.0).startswith("drop"))
check("+0.10 .. +0.30 -> shelve", SC.verdict(0.10).startswith("shelve") and SC.verdict(0.29).startswith("shelve"))
check(">= +0.30 -> build gap_and_hold", SC.verdict(0.30).startswith("build") and SC.verdict(2.0).startswith("build"))
check("no data", SC.verdict(None) == "no_data")

print("hold test (gap_and_go's own: no low under the pre-gap close, no close under 97% of the gap-day open)")
# gap day g = 1: previous close 100, open 104; sessions 2 and 3 are the two hold sessions
def bars(lows, closes):
    return NS(n=4, c_prev=np.array([0, 100.0, 0, 0]), o=np.array([100.0, 104.0, 105, 105]), l=np.array([0, 103.0, *lows]), c=np.array([100.0, 106.0, *closes]))
check("holds: lows above the pre-gap close, closes above 97% of the open", SC.held(bars([102.0, 103.0], [105.0, 106.0]), 1))
check("a low under the pre-gap close = gap filled", not SC.held(bars([99.9, 103.0], [105.0, 106.0]), 1))
check("a close under 97% of the gap-day open (100.88) = faded", not SC.held(bars([102.0, 103.0], [100.0, 106.0]), 1))
check("not enough sessions after the gap = not held", not SC.held(NS(n=3, c_prev=np.array([0, 100.0, 0]), o=np.array([100.0, 104.0, 105]), l=np.array([0, 103.0, 102.0]), c=np.array([100.0, 106.0, 105.0])), 1))

print("end to end on synthetic data")
out = {}
d = make_market(n_stocks=25, n_days=950, seed=5)
SC.compute(d, out)
pops = out["populations"]
check("every population and the control reported at 1/3/5/10", set(SC.POPS) | {"ALL_BARS (control)"} == set(pops) and all(set(r["horizons"]) == {"1", "3", "5", "10"} for r in pops.values()))
check("control has a large sample", pops["ALL_BARS (control)"]["horizons"]["3"]["n"] > 1000)
n_gap, n_hold = pops["gap3_day1"]["horizons"]["3"]["n"], pops["gap_and_hold"]["horizons"]["3"]["n"]
check("gap_and_hold is a subset of the gap days", n_gap >= n_hold, f"{n_gap} {n_hold}")
check("ep_held_day3 is a subset of gap_and_hold", pops["ep_held_day3"]["horizons"]["3"]["n"] <= n_hold)
check("decision carries the rule, the halves and the verdict", {"net_of_costs_pct", "halves_adj_pct", "verdict", "n"} <= set(out["decision"]))
check("episodic_pivot day-1 profile present", "episodic_pivot_day1" in out and "n" in out["episodic_pivot_day1"])
md = SC.render_markdown({**out, "state": "done"})
check("renders and serialises", "gap_and_hold" in md and bool(json.dumps(out, default=str)))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
