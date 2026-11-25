"""
Tests for the Tier 1 replacement components and the backtest study (26 Nov). Synthetic data only: these prove the CODE does what it says, not that any
component has an edge.
    python test_t1study.py
"""
from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_scorer as SC
import e2_t1study as S
import e2_tier1 as T1
from e2_features import compute_features
from e2_synthmarket import make_market

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


print("tiers")
check("relvol tiers", [T1.relvol_tier(x) for x in (3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.5)] == [1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.0])
check("relvol unknown is neutral", T1.relvol_tier(float("nan")) == 0.5)
check("sector tiers (n=11)", [T1.sector_fraction_t1(r, 11) for r in (1, 3, 4, 8, 9, 11)] == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
check("sector unknown / too few sectors neutral", T1.sector_fraction_t1(float("nan"), 11) == 0.5 and T1.sector_fraction_t1(1, 4) == 0.5)
check("delivery tiers", [T1.delivery_fraction(x) for x in (60, 50.1, 50, 40, 35, 30, 25, 24.9, 10)] == [1.0, 1.0, 0.75, 0.75, 0.75, 0.5, 0.5, 0.0, 0.0])
check("delivery missing is neutral", T1.delivery_fraction(None) == 0.5 and T1.delivery_fraction(float("nan")) == 0.5)

print("earnings window")
ev = np.array(["2025-03-10", "2025-06-12"], dtype="datetime64[D]")
d = lambda s: np.datetime64(s, "D")          # noqa: E731
check("inside +-5 days", T1.earnings_near(ev, d("2025-03-05")) and T1.earnings_near(ev, d("2025-03-15")) and T1.earnings_near(ev, d("2025-03-10")))
check("outside", not T1.earnings_near(ev, d("2025-03-04")) and not T1.earnings_near(ev, d("2025-03-16")) and not T1.earnings_near(ev, d("2025-04-20")))
check("no calendar", not T1.earnings_near(None, d("2025-03-10")) and not T1.earnings_near(np.array([], dtype="datetime64[D]"), d("2025-03-10")))

print("delivery lookup")
dd = {"AAA": (np.array(["2025-03-03", "2025-03-04", "2025-03-06"], dtype="datetime64[D]"), np.array([40.0, 55.0, 20.0]))}
check("lag0 / lag1 on a present day", S.delivery_at(dd, "AAA", d("2025-03-04")) == (55.0, 40.0))
check("a missing day has no lag0 but has a lag1", S.delivery_at(dd, "AAA", d("2025-03-05")) == (None, 55.0))
check("first day has no lag1; unknown symbol has neither", S.delivery_at(dd, "AAA", d("2025-03-03")) == (40.0, None) and S.delivery_at(dd, "ZZZ", d("2025-03-03")) == (None, None))

print("components on synthetic bars")
data = make_market(n_stocks=12, n_days=700, seed=5)
sym, df = sorted(data.frames.items())[0]
b = compute_features(sym, df)
i = b.n - 5
cand = SimpleNamespace(detector="pocket_pivot", meta={}, idx=i)
frac_old = {"trend": 0.5, "rs": 0.6, "volume": SC.volume_fraction(b, i), "base": 0.5, "catalyst": 0.0, "sector": 0.5, "regime": 0.5, "_catalyst_source": "proxy", "_rs_missing": False}
snap = dict(frac_old)
new = T1.t1_components(frac_old, b, cand, 2.0, 11, 62.0, ev)
check("old fractions are not modified", frac_old == snap)
check("delivery added, volume/sector/catalyst replaced, others untouched",
      new["delivery"] == 1.0 and new["sector"] == 1.0 and new["trend"] == 0.5 and new["rs"] == 0.6 and new["base"] == 0.5 and new["regime"] == 0.5)
check("all fractions within 0-1", all(0.0 <= new[c] <= 1.0 for c in T1.NEW_COMPONENTS))
check("volume t1 shares the 60% non-relvol part with the old one",
      abs((SC.volume_fraction(b, i) - 0.4 * min(max(b.relvol[i] / 3.0, 0), 1)) - (T1.volume_fraction_t1(b, i) - 0.4 * T1.relvol_tier(b.relvol[i]))) < 1e-9)
cat_none, _ = T1.catalyst_fraction_t1(b, cand, None)
cat_old, _ = SC.catalyst_fraction(b, cand)
check("catalyst equals the old one without a calendar", cat_none == cat_old)
b2 = SimpleNamespace(**{**b.__dict__, "date": b.date})
ev_near = np.array([b.date[i] + np.timedelta64(3, "D")], dtype="datetime64[D]")
check("earnings inside the window lifts catalyst to 1.0 and never lowers it", T1.catalyst_fraction_t1(b, cand, ev_near)[0] == 1.0 and T1.catalyst_fraction_t1(b, cand, ev_near)[0] >= cat_old)

print("study on synthetic records (a planted signal in delivery and nothing else)")
rng = np.random.default_rng(3)
recs = []
dates = np.arange(np.datetime64("2023-01-02"), np.datetime64("2023-01-02") + np.timedelta64(900, "D"))
for k in range(2400):
    dl = rng.uniform(10, 70)
    fo = {c: float(rng.uniform(0, 1)) for c in SC.COMPONENTS}
    fn = dict(fo)
    fn["delivery"] = T1.delivery_fraction(dl)
    r = (fn["delivery"] - 0.5) * 2.0 + rng.normal(0, 1.0)            # R depends on delivery only
    recs.append({"det": "pocket_pivot" if k % 2 else "gap_and_go", "family": "breakout" if k % 2 else "momentum", "date": dates[k % 900], "r": r, "old": fo, "new": fn,
                 "deliv_lag1": fn["delivery"], "deliv_missing": False, "earn": False, "sector_known": True, "d1": float(rng.normal(0, 1)),
                 **{f"f{h}": float(rng.normal(0, 3)) for h in S.HORIZONS}})
out = S.study(recs, np.datetime64("2024-05-20"))
check("study runs and reports coverage", out["coverage"]["delivery_present_pct"] == 100.0 and out["n_train"] > 200 and out["n_test"] > 100)
w_new = out["weights"]["new_pooled"]
check("delivery earns nearly all the weight, the noise components none", w_new["delivery"] > 90.0 and all(w_new[c] == 0.0 for c in SC.COMPONENTS), str(out["weights"]))
check("new composite beats the old one on the planted signal", out["composite_test"]["new_pooled"] > (out["composite_test"]["old_pooled"] or 0) + 0.1, str(out["composite_test"]))
check("bootstrap interval on the difference excludes zero", out["composite_test"]["diff_ci95"][0] > 0, str(out["composite_test"]))
check("per-detector and day-1 tables present", set(out["per_detector"]) == {"pocket_pivot", "gap_and_go"} and "top20_new_mean_d1_pct" in out["day1_breakout_pooled"])

print("study on pure noise: no component earns weight, no false gate pass")
for r in recs:
    r["r"] = float(rng.normal(0, 1))
    r["new"]["delivery"] = float(rng.choice([0.0, 0.5, 0.75, 1.0]))
out2 = S.study(recs, np.datetime64("2024-05-20"))
check("noise: gate not passed", out2["composite_test"]["gate_new_gt_0_25"] is False)

print("small sample is refused, not guessed")
check("too few trades -> error key", "error" in S.study(recs[:50], np.datetime64("2024-05-20")))

print("end to end on a synthetic market (per_symbol -> enrich -> study), with synthetic delivery and no calendar")
import time
t0 = time.time()
mk = make_market(n_stocks=40, n_days=1000, seed=21)
deliv = {}
for sy, df_ in mk.frames.items():
    deliv[sy] = (df_["trade_date"].to_numpy().astype("datetime64[D]"), np.random.default_rng(abs(hash(sy)) % 1000).uniform(15, 70, len(df_)))
res = {}
S.compute(mk, deliv, res)
check("end to end: trades enriched and a verdict produced", res.get("n_trades", 0) > 100 and "composite_test" in res and "per_detector" in res, str({k: res.get(k) for k in ("n_trades", "error")}))
check("end to end: delivery coverage is reported", res["coverage"]["delivery_present_pct"] is not None and res["coverage"]["delivery_present_pct"] > 90, str(res.get("coverage")))
print("   (%d trades, %.0fs)" % (res.get("n_trades", 0), time.time() - t0))

print("\n%d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
