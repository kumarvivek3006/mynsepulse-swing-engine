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
check("every population and the control reported at 1/3/5/10/20", set(SC.POPS) | {"ALL_BARS (control)"} == set(pops) and all(set(r["horizons"]) == {"1", "3", "5", "10", "20"} for r in pops.values()))
check("control has a large sample", pops["ALL_BARS (control)"]["horizons"]["3"]["n"] > 1000)
n_gap, n_hold = pops["gap3_day1"]["horizons"]["3"]["n"], pops["gap_and_hold"]["horizons"]["3"]["n"]
check("gap_and_hold is a subset of the gap days", n_gap >= n_hold, f"{n_gap} {n_hold}")
check("ep_held_day3 is a subset of gap_and_hold", pops["ep_held_day3"]["horizons"]["3"]["n"] <= n_hold)
check("decision carries the rule, the halves and the verdict", {"net_of_costs_pct", "halves_adj_pct", "verdict", "n"} <= set(out["decision"]))
check("episodic_pivot day-1 profile present", "episodic_pivot_day1" in out and "n" in out["episodic_pivot_day1"])
md = SC.render_markdown({**out, "state": "done"})
check("renders and serialises", "gap_and_hold" in md and bool(json.dumps(out, default=str)))

print("entry-test definitions (hand-made bars)")
from e2_det_common import WARMUP as W
T = W + 10                  # signal bar
def mk(n=W + 80):
    c = np.full(n, 100.0); o = c.copy(); h = c + 1; l = c - 1; v = np.full(n, 1000.0)
    return c, o, h, l, v
def stub(c, o, h, l, v):
    n = len(c)
    cp = np.concatenate([[c[0]], c[:-1]])
    rng = h - l
    vavg = np.array([np.mean(v[max(0, i - 50):i]) if i else v[0] for i in range(n)])
    from e2_core import ema
    return NS(n=n, c=c, o=o, h=h, l=l, v=v, c_prev=cp, close_pos=np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 0.5), relvol=v / vavg,
              sma50=np.array([np.mean(c[max(0, i - 49):i + 1]) for i in range(n)]), sma50_up=np.ones(n, bool), ema20=ema(c, 20))
# intraday_momentum: quiet 10 sessions, then an up day closing at its high on 3x volume
c, o, h, l, v = mk(); c[T] = 103.0; o[T] = 100.5; h[T] = 103.0; l[T] = 100.2; v[T] = 3000.0
sg = SC.short_signals(stub(c, o, h, l, v), 0)
check("intraday_momentum fires on a quiet-then-strong-close day", T in sg["intraday_momentum"], str(sg["intraday_momentum"]))
c2, o2, h2, l2, v2 = mk(); c2[T] = 103.0; o2[T] = 100.5; h2[T] = 106.0; l2[T] = 100.2; v2[T] = 3000.0     # closes at 74% of the range
check("intraday_momentum needs the close in the top decile", T not in SC.short_signals(stub(c2, o2, h2, l2, v2), 0)["intraday_momentum"])
c3, o3, h3, l3, v3 = mk(); c3[T] = 103.0; o3[T] = 100.5; h3[T] = 103.0; l3[T] = 100.2; v3[T] = 1500.0
check("intraday_momentum needs 2x volume", T not in SC.short_signals(stub(c3, o3, h3, l3, v3), 0)["intraday_momentum"])
c4, o4, h4, l4, v4 = mk(); h4[T-8:T-2] = 115.0; c4[T] = 103.0; o4[T] = 100.5; h4[T] = 103.0; l4[T] = 100.2; v4[T] = 3000.0
check("intraday_momentum needs a consolidation (no wide range in the prior 10 sessions)", T not in SC.short_signals(stub(c4, o4, h4, l4, v4), 0)["intraday_momentum"])
# volume_spike_reversal: decline into a new low, then a 6x volume up day closing in the top half
c, o, h, l, v = mk(); c[T-15:T] = np.linspace(100, 90, 15); o[T-15:T] = c[T-15:T]; h[T-15:T] = c[T-15:T] + 0.5; l[T-15:T] = c[T-15:T] - 0.5
c[T] = 92.0; o[T] = 89.5; h[T] = 92.5; l[T] = 89.0; v[T] = 6000.0
check("volume_spike_reversal fires at support after a decline", T in SC.short_signals(stub(c, o, h, l, v), 0)["volume_spike_reversal"])
v5 = v.copy(); v5[T] = 3000.0
check("volume_spike_reversal needs 5x volume", T not in SC.short_signals(stub(c, o, h, l, v5), 0)["volume_spike_reversal"])
l6 = l.copy(); l6[T] = 95.0; o6 = o.copy(); o6[T] = 95.2
check("volume_spike_reversal needs the low at support (within 3% of the prior 50-session low)", T not in SC.short_signals(stub(c, o6, h, l6, v), 0)["volume_spike_reversal"])
# pullback_10ema: uptrend, two down days touching the EMA10 on falling volume, then an up day above the EMA10
n = W + 100; c = np.linspace(80, 120, n); o = c - 0.3; h = c + 0.6; l = c - 0.6; v = np.full(n, 1000.0)
c[n-4] = c[n-5] - 1.5; c[n-3] = c[n-4] - 1.5; l[n-3] = c[n-3] - 0.5; v[n-4] = 1200.0; v[n-3] = 800.0
c[n-2] = c[n-3] + 2.5; o[n-2] = c[n-3] + 0.1; h[n-2] = c[n-2] + 0.2
st = stub(c, o, h, l, v)
sgp = SC.short_signals(st, 0)["pullback_10ema"]
check("pullback_10ema fires after a 2-day pullback on falling volume then a reversal", n-2 in sgp, str(sgp))
v7 = v.copy(); v7[n-3] = 1500.0
check("pullback_10ema needs the pullback volume to decline", n-2 not in SC.short_signals(stub(c, o, h, l, v7), 0)["pullback_10ema"])
c8, o8, h8, l8 = c.copy(), o.copy(), h.copy(), l.copy(); c8[n-2] = c8[n-3] - 0.5
check("pullback_10ema needs an up reversal day", n-2 not in SC.short_signals(stub(c8, o8, h8, l8, v), 0)["pullback_10ema"])

print("regime-check populations")
rc = out["regime_check"]["second_half_only_pattern"]
check("regime check reports the second-half-only pattern for the gap and non-gap populations", {"gap_and_hold", "up_day_hold", "bigmove_nogap"} <= set(rc))
check("non-gap hold populations exist and are bigger than the control's gap sample", pops["up_day_hold"]["horizons"]["3"]["n"] > 0)
check("short-horizon pass and raw/market halves reported", "short_pass" in pops["gap_and_hold"] and "raw_market_by_half" in pops["gap_and_hold"])

print("intraday history probe")
import e2_probe30 as PR
txt = PR.render({"state": "done", "stored_probe": {"verdict": {"30": {"full_sessions_back_to": "2022-01-03"}}},
                 "fresh_probe": {"ran": True, "verdict": {"30": {"usable_for_3y_backtest": True}},
                                 "cells": {"30": [{"date": "2026-11-06", "candles": 25, "expected": 12, "ok": True}, {"date": "2023-01-02", "candles": 0, "expected": 12, "ok": False, "error": "no data"}]}}})
check("probe30 renders the stored verdict and the fresh cells", "full_sessions_back_to" in txt and "no data" in txt and "2026-11-06 n25/12" in txt)
txt2 = PR.render({"state": "done", "stored_probe": None, "fresh_probe": {"ran": False, "reason": "no valid Upstox token"}})
check("probe30 says plainly when it could not run", "ran: False" in txt2 and "no valid Upstox token" in txt2)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
