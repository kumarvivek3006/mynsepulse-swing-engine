"""Tests for the volatility-matched control and the oversold entry test. python test_volmatch.py"""
from __future__ import annotations

import json
import logging
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_volmatch as VM
from e2_synthmarket import make_market

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ok    {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}  {detail}")


rng = np.random.default_rng(3)
D0 = np.datetime64("2024-01-01", "D")
MID = np.datetime64("2025-05-02", "D")


def make(n, atr, drift_per_atr, edge=0.0, seed=1):
    r = np.random.default_rng(seed)
    sig = D0 + r.integers(0, 700, n).astype("timedelta64[D]")
    a = atr(r, n)
    adj = drift_per_atr * a + r.normal(0, 2.0, n) + edge
    return {"sig": sig, "adj": adj, "atr": a}


print("matching removes a pure volatility effect")
ctl = make(40000, lambda r, n: r.uniform(1.5, 6.0, n), 0.5, seed=1)                 # higher volatility -> higher return
pop_hv = make(2000, lambda r, n: r.uniform(4.5, 6.0, n), 0.5, seed=2)                # only high-vol stocks, NO detector edge
m = VM.analyse(pop_hv, ctl, MID, 0.2)
check("raw excess is clearly positive (the unmatched control is fooled)", (pop_hv["adj"].mean() - ctl["adj"].mean()) > 0.5)
check("matched excess is near zero", abs(m["matched_excess_pct"]) < 0.25, m["matched_excess_pct"])
check("pure volatility effect does not pass", m["passes"] is False)
check("ATR mix of the signals is close to their matched control", abs(m["atr_mean_pop"] - m["atr_mean_matched_ctl"]) < 0.6, (m["atr_mean_pop"], m["atr_mean_matched_ctl"]))

print("a real edge survives the match")
pop_edge = make(2000, lambda r, n: r.uniform(4.5, 6.0, n), 0.5, edge=1.5, seed=4)
m2 = VM.analyse(pop_edge, ctl, MID, 0.2)
check("matched excess close to the true +1.5", 1.1 < m2["matched_excess_pct"] < 1.9, m2["matched_excess_pct"])
check("passes with both halves above zero and an interval clear of zero", m2["passes"] and m2["above_zero_both_halves"] and m2["ci95"][0] > 0)

print("an edge in only one half fails the both-halves rule")
pop_one = make(2000, lambda r, n: r.uniform(4.5, 6.0, n), 0.5, seed=5)
pop_one["adj"] = pop_one["adj"] + np.where(pop_one["sig"] >= MID, 3.0, 0.0)
m3 = VM.analyse(pop_one, ctl, MID, 0.2)
check("second-half-only edge is flagged lopsided", m3["lopsided"] is True and m3["halves_matched_excess"][0] < 0.5 and m3["halves_matched_excess"][1] > 2.0, m3)
check("a balanced edge is not lopsided", m2["lopsided"] is False, m2["halves_matched_excess"])

print("the control's width sets the bar")
pop_small = make(2000, lambda r, n: r.uniform(4.5, 6.0, n), 0.5, edge=0.3, seed=6)
check("a +0.3 edge fails a bar of 1.0 (half of a width of 2.0)", VM.analyse(pop_small, ctl, MID, 2.0)["passes"] is False)

print("bins and edges")
e = VM.edges_for(np.arange(1000.0), 5)
check("5 bins have 4 edges and cover everything", len(e) == 4 and set(VM.bin_of(np.array([0.0, 999.0]), e)) == {0, 4})

print("oversold definitions (hand-made bars)")
from e2_det_common import WARMUP as W


def bars(n, close, rsi, sma200=None):
    c = np.asarray(close, float)
    return NS(n=n, c=c, o=c * 1.0, h=c * 1.01, l=c * 0.99, rsi14=np.asarray(rsi, float), sma200=np.full(n, 50.0) if sma200 is None else np.asarray(sma200, float))


n = W + 250
close = np.full(n, 100.0)
rsi = np.full(n, 50.0)
t = W + 130
close[t - 5:t] = [100, 99, 97, 94, 91]; close[t] = 90.0                               # 10% down over 5 sessions
rsi[t] = 25.0
b = bars(n, close, rsi)
s = VM.oversold_signals(b, W)
check("oversold in trend fires once on the oversold day", s["oversold_dip"] == [t], s["oversold_dip"])
check("the no-trend variant fires too", s["oversold_dip_any"] == [t])
b2 = bars(n, close, rsi, sma200=np.full(n, 200.0))                                    # close under the 200-day average
s2 = VM.oversold_signals(b2, W)
check("under the 200-day average: only the any-variant fires", s2["oversold_dip"] == [] and s2["oversold_dip_any"] == [t])
rsi2 = rsi.copy(); rsi2[t] = 40.0
check("RSI above 30 does not fire", VM.oversold_signals(bars(n, close, rsi2), W)["oversold_dip"] == [])
close3 = close.copy(); close3[t - 5:t] = [100, 100, 99.5, 99.0, 98.5]; close3[t] = 97.0
check("a drop under 8% does not fire", VM.oversold_signals(bars(n, close3, rsi), W)["oversold_dip"] == [])
close4 = close.copy(); close4[t + 1] = 92.0                                           # up day through the oversold day's high
b4 = bars(n, close4, rsi)
b4.o[t + 1] = 90.5
s4 = VM.oversold_signals(b4, W)
check("reversal: up day closing above the previous high", t + 1 in s4["oversold_reversal"], s4["oversold_reversal"])
b5 = bars(n, close4, rsi); b5.o[t + 1] = 93.0                                          # a down day (open above close)
check("a down day is not a reversal", t + 1 not in VM.oversold_signals(b5, W)["oversold_reversal"])
rsi6 = rsi.copy(); rsi6[t:t + 3] = 25.0
close6 = close.copy(); close6[t - 5:t + 3] = [100, 99, 97, 94, 91, 90, 89, 88]
check("one signal per episode (cooldown)", len(VM.oversold_signals(bars(n, close6, rsi6), W)["oversold_dip"]) == 1)

print("end to end on synthetic data")
for kind in ("volmatch", "oversold"):
    out = {}
    d = make_market(n_stocks=25, n_days=950, seed=5)
    VM.compute(d, out, kind)
    cfg = VM.KINDS[kind]
    check(f"{kind}: every population reported at every horizon", set(out["populations"]) == set(cfg["pops"]) and all(set(r) == {str(h) for h in cfg["H"]} for r in out["populations"].values()))
    check(f"{kind}: control has a large sample", all(out["control"][str(h)]["n"] > 500 for h in cfg["H"]))
    check(f"{kind}: decision and rule present", bool(out.get("decision")) and "rule" in out)
    md = VM.render_markdown({**out, "state": "done"})
    check(f"{kind}: renders and serialises", bool(md) and bool(json.dumps(out, default=str)))
    if kind == "volmatch":
        gh = out["populations"]["gap_and_hold"]["10"].get("n", 0)
        nc = out["populations"]["gap_hold_not_confirmed"]["10"].get("n", 0)
        check("gap_hold_not_confirmed is a subset of gap_and_hold", nc <= gh, (nc, gh))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
