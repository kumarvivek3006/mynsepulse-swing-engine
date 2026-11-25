"""
Tests for the inverse-delivery study (e2_t1inv) and the 52-week-high standalone study (e2_nearhigh), 26 Nov. Synthetic data only: these prove the CODE
reads a planted effect, rejects noise and a volatility confound, and runs end to end; they say nothing about the real market.
    python test_t1inv_nh.py
"""
from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_nearhigh as NH
import e2_scorer as SC
import e2_t1inv as INV
import e2_t1study as S
import e2_tier1 as T1
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


MID = np.datetime64("2024-05-20")
# ------------------------------------------------------------------------------------------------ 52-week-high filter
print("near-high filter: statistics")
rng = np.random.default_rng(5)
days = np.arange(np.datetime64("2023-01-02"), np.datetime64("2023-01-02") + np.timedelta64(900, "D"))
days = days[(days.astype("datetime64[D]").astype(int) + 4) % 7 < 5]
N = 40000
sig = days[rng.integers(0, len(days), N)]
atr = rng.uniform(1.0, 5.0, N)
dist = -np.abs(rng.normal(0, 14, N))


def uni_of(adj):
    return {"sig": sig, "adj": adj, "atr": atr, "dist": dist}


near = dist >= -10
planted = uni_of(rng.normal(0, 8, N) + 3.0 * near)
s = NH.filter_stats(planted, near, MID)
check("planted +3pp near-high edge: difference > 1pp and the interval excludes zero", s["raw_difference_pp"] > 1.0 and s["raw_difference_ci95_week_block"][0] > 0, str(s))
check("planted edge: positive in both halves, strict gate passes, survives ATR matching", s["gate_owner_strict"] and s["gate_survives_atr_matching"], str(s))

noise = uni_of(rng.normal(0, 8, N))
s = NH.filter_stats(noise, near, MID)
check("pure noise: no gate", not s["gate_difference_gt_1pp"] and not s["gate_owner_strict"], str(s))
check("pure noise: the interval spans zero", s["raw_difference_ci95_week_block"][0] < 0 < s["raw_difference_ci95_week_block"][1], str(s))

# volatility confound: near-high rows are the high-ATR ones, and return rises with ATR for everybody
atr_c = np.where(near, rng.uniform(3.5, 6.0, N), rng.uniform(1.0, 3.5, N))
conf = {"sig": sig, "adj": rng.normal(0, 6, N) + 2.0 * (atr_c - 3.0), "atr": atr_c, "dist": dist}
s = NH.filter_stats(conf, near, MID)
check("volatility confound: the RAW gate passes", s["gate_difference_gt_1pp"], str(s))
check("volatility confound: ATR matching removes it", not s["gate_survives_atr_matching"] and abs(s["atr_matched_excess_pp"]) < 0.8, str(s))

# bull-phase confound: the market drifts, near-high rows cluster in the up half
up = sig >= MID
near_b = (dist >= -10) & (rng.random(N) < np.where(up, 1.0, 0.35))
drift = np.where(up, 4.0, -2.0)
bull = uni_of(rng.normal(0, 6, N) + drift)
s = NH.filter_stats(bull, near_b, MID)
check("bull-phase confound: raw difference positive, same-day excess near zero", s["raw_difference_pp"] > 1.0 and abs(s["same_day_excess_pp"]) < 0.8, str(s))

print("near-high filter: detector against the filtered universe")
det_n = 1500
dsig = days[rng.integers(0, len(days), det_n)]
ddist = -np.abs(rng.normal(0, 10, det_n))
dd = {"sig": dsig, "adj": rng.normal(0, 8, det_n) + 3.0 * (ddist >= -10), "atr": rng.uniform(1, 5, det_n), "dist": ddist}
r = NH.detector_vs_filter(dd, planted, 10, MID)
check("detector that is only the screen adds ~nothing beyond the near-high universe", abs(r["raw_excess_vs_near_universe_pp"]) < 1.5 and r["n_near"] > 100, str(r))
dd["adj"] = rng.normal(0, 8, det_n) + 3.0 * (ddist >= -10) + 4.0
r = NH.detector_vs_filter(dd, planted, 10, MID)
check("detector with its own edge shows it", r["raw_excess_vs_near_universe_pp"] > 2.5, str(r))
check("small set returns n only", "mean_adj_pct" not in NH.detector_vs_filter({k: v[:10] for k, v in dd.items()}, planted, 10, MID))

# ------------------------------------------------------------------------------------------------ inverse delivery
print("inverse delivery: planted inverse, noise, flip, confound")


def make_recs(kind, n=3600, seed=9):
    rg = np.random.default_rng(seed)
    d0 = np.arange(np.datetime64("2023-01-02"), np.datetime64("2023-01-02") + np.timedelta64(900, "D"))
    recs = []
    for k in range(n):
        dt = d0[rg.integers(0, 900)]
        atr_ = rg.uniform(1, 6)
        dl = float(np.clip(rg.normal(45 - 5 * (atr_ - 3), 14), 5, 90)) if kind == "confound" else rg.uniform(10, 70)
        fo = {c: float(rg.uniform(0, 1)) for c in SC.COMPONENTS}
        fn = dict(fo)
        fn["delivery"] = T1.delivery_fraction(dl)
        if kind == "inverse":
            r = (0.5 - fn["delivery"]) * 2.0 + rg.normal(0, 1)
        elif kind == "flip":
            sgn = 1.0 if dt < MID else -1.0                          # inverse in train, the opposite in test
            r = sgn * (0.5 - fn["delivery"]) * 2.0 + rg.normal(0, 1)
        elif kind == "confound":
            r = 0.9 * (atr_ - 3.5) + rg.normal(0, 1)                  # R depends on volatility only; delivery is a proxy for it
        else:
            r = rg.normal(0, 1)
        recs.append({"det": "pocket_pivot" if k % 2 else "gap_and_go", "family": "breakout", "date": dt, "r": float(r), "old": fo, "new": fn,
                     "deliv_lag1": fn["delivery"], "deliv_missing": False, "deliv_pct": dl, "deliv_pct1": dl, "earn": False, "sector_known": True, "d1": 0.0,
                     "atrp": atr_, "dist52": -abs(float(rg.normal(0, 10))), "week": int((dt - np.datetime64("2000-01-03", "D")).astype(int) // 7), "sym": f"S{k % 80}",
                     **{f"f{h}": 0.0 for h in S.HORIZONS}})
    return recs


o = INV.study(make_recs("inverse"), MID)
check("planted inverse: delivery_inv earns the weight and delivery alone has the same sign in both halves", o["weights"]["C"]["dinv"] > 50 and o["delivery_alone_inverse"]["tier_lag0"]["same_sign"], str(o["weights"]))
check("planted inverse: paired gate passes (C vs B), interval excludes zero", o["gate"]["paired_improves_on_no_delivery_composite"]["C vs B"] and o["gate"]["pass_paired"], str(o["paired_differences_test"]))
check("planted inverse: literal gate passes", o["gate"]["pass_literal"], str(o["composites"]))
check("per-detector table and sign consistency present", set(o["per_detector"]) == {"pocket_pivot", "gap_and_go"} and o["per_detector_sign_consistency"]["same_sign_train_and_test"] == 2)
check("week-block interval is wider than the trade-level one", (o["composites"]["C"]["test_ci95_week_block"][1] - o["composites"]["C"]["test_ci95_week_block"][0])
      >= 0.8 * (o["composites"]["C"]["test_ci95_trade_level"][1] - o["composites"]["C"]["test_ci95_trade_level"][0]))

o = INV.study(make_recs("noise"), MID)
check("noise: no pass on either reading", not o["gate"]["pass_literal"] and not o["gate"]["pass_paired"], str(o["gate"]))

o = INV.study(make_recs("flip"), MID)
check("flip: delivery does not keep its sign on the test half", not o["delivery_alone_inverse"]["tier_lag0"]["same_sign"] or o["delivery_alone_inverse"]["tier_lag0"]["inverse_rho_test"] < 0.05, str(o["delivery_alone_inverse"]))
check("flip: the paired gate is not passed", not o["gate"]["pass_paired"], str(o["gate"]))

o = INV.study(make_recs("confound"), MID)
c = o["confounds_test"]
check("confound: delivery looks inverse-ranked on its own", o["delivery_alone_inverse"]["tier_lag0"]["inverse_rho_test"] > 0.1, str(o["delivery_alone_inverse"]))
check("confound: the partial correlation controlling for ATR collapses", abs(c["partial_inverse_delivery_R_control_atr"]) < 0.05, str(c))

print("inverse delivery: partial Spearman and small samples")
x = rng.normal(size=500); z = rng.normal(size=500)
check("partial removes a pure proxy", abs(INV.partial_spearman(x + z, 2 * z + rng.normal(0, 0.3, 500), [z])) < 0.2)
check("too few trades -> error, not a guess", "error" in INV.study(make_recs("noise")[:60], MID))

# ------------------------------------------------------------------------------------------------ end to end
print("end to end on a synthetic market")
t0 = time.time()
mk = make_market(n_stocks=40, n_days=1000, seed=21)
res = {}
NH.compute(mk, res)
f20 = res["universe_filter"]["20"]
check("nearhigh end to end: bands reported at 20 sessions", all(f"within_{b}pct" in f20 for b in NH.BANDS) and f20["within_10pct"].get("n", 0) > 100, str(list(f20)))
check("nearhigh end to end: decision block and detector portfolio present", "pass_strict_any_band" in res["decision"] and len(res["detector_portfolio_h20"]) >= 1, str(res.get("decision")))
check("nearhigh end to end: bands are nested (within 5 <= 10 <= 15 <= 20)", f20["within_5pct"]["n"] <= f20["within_10pct"]["n"] <= f20["within_15pct"]["n"] <= f20["within_20pct"]["n"])
check("nearhigh end to end: renders", "E2NH" in NH.render(res) and len(NH.render(res)) > 500)
print("   nearhigh %.0fs" % (time.time() - t0))
t0 = time.time()
deliv = {}
for sy, df_ in mk.frames.items():
    deliv[sy] = (df_["trade_date"].to_numpy().astype("datetime64[D]"), np.random.default_rng(abs(hash(sy)) % 1000).uniform(15, 70, len(df_)))
res2 = {}
INV.compute(mk, deliv, res2)
check("t1inv end to end: composites, paired differences and gate produced", res2.get("n_trades", 0) > 100 and "composites" in res2 and "gate" in res2, str({k: res2.get(k) for k in ("n_trades", "error")}))
check("t1inv end to end: noise delivery does not pass", not res2["gate"]["pass_paired"], str(res2.get("gate")))
check("t1inv end to end: renders", "E2T1INV" in INV.render(res2))
print("   t1inv %.0fs" % (time.time() - t0))

print("\n%d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
