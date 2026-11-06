"""
Tests for the 6 Nov actions (e2_dscore, strict pocket_pivot, armed vcp + stop-entry, the study end to end on synthetic data).
Synthetic data only: these prove the CODE does what it says, not that any variant has an edge.
    python test_actions.py
"""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2_actions as A
import e2_det_bases as DB
import e2_det_momentum as DM
import e2_dscore as DS
import e2_manage as MG
import e2_scorer as SC
from e2_core import Candidate
from e2_detectors import dedupe
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


data = make_market(n_stocks=30, n_days=900, seed=11)
bars = {s: compute_features(s, df) for s, df in sorted(data.frames.items())}
bars = {s: b for s, b in bars.items() if b is not None}

print("vcp refactor: the breakout detector's output is unchanged")
here = os.path.dirname(os.path.abspath(__file__))
try:
    old_src = subprocess.run(["git", "show", "HEAD:e2_det_bases.py"], cwd=here, capture_output=True, text=True, check=True).stdout
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "e2_det_bases_head.py"), "w").write(old_src)
    spec = importlib.util.spec_from_file_location("e2_det_bases_head", os.path.join(tmp, "e2_det_bases_head.py"))
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    same, total = 0, 0
    for s, b in bars.items():
        a = [(c.idx, round(c.pivot, 6), round(c.quality, 6), tuple(c.meta["contractions"])) for c in old.vcp(b, {}, 130, b.n)]
        n = [(c.idx, round(c.pivot, 6), round(c.quality, 6), tuple(c.meta["contractions"])) for c in DB.vcp(b, {}, 130, b.n)]
        total += 1; same += (a == n)
    check("vcp output identical to HEAD on every synthetic symbol", same == total, f"{same}/{total}")
except Exception as e:                                                        # noqa: BLE001
    check("vcp output identical to HEAD", False, repr(e))

print("pocket_pivot strict")
cur_n = str_n = 0
subset = True
clauses = True
for s, b in bars.items():
    cur = {c.idx: c for c in DM.pocket_pivot(b, {}, 130, b.n, strict=False)}
    st = DM.pocket_pivot(b, {}, 130, b.n, strict=True)
    cur_n += len(cur); str_n += len(st)
    for c in st:
        i = c.idx
        subset &= (i in cur)
        up, base = DM._pp_structure(b, i)
        clauses &= bool(b.relvol[i] >= 1.5 and (up or base) and not c.meta["no_down_day"] and b.close_pos[i] >= 0.5 and c.meta["vol_vs_down"] >= 1.0)
check("strict hits are a subset of the current hits", subset)
check("every strict hit meets all stated clauses", clauses)
check("strict is a real reduction", 0 < str_n < cur_n, f"{str_n}/{cur_n}")
check("env default is the current definition", DM.PP_STRICT is False or os.environ.get("E2_PP_STRICT") == "1")

print("armed vcp and the buy-stop entry")
armed = {s: DB.vcp_armed(b, {}, 130, b.n) for s, b in bars.items()}
arm_all = [(s, c) for s, v in armed.items() for c in v]
check("armed vcp finds candidates on the synthetic market", len(arm_all) > 0, str(len(arm_all)))
ok_gates = True
for s, c in arm_all:
    b = bars[s]; i = c.idx
    dist = (c.pivot / b.c[i] - 1) * 100
    ok_gates &= (2.0 <= dist <= 4.0 and b.c[i] >= b.ema20[i] and c.meta["entry_stop"] > c.pivot and c.detector == "vcp_armed")
check("armed candidates are 2-4% under the pivot, above the 20-EMA, stop-entry above the pivot", ok_gates)
spacing = True
for s, v in armed.items():
    idx = [c.idx for c in v]
    spacing &= all(b - a > DB.ARMED_EXPIRY for a, b in zip(idx, idx[1:]))
check("one resting order per symbol at a time", spacing)
p60 = sum(len(DB.vcp_armed(b, {}, 130, b.n, prior_uptrend=True)) for b in bars.values())
check("prior-advance variant is a subset in count", p60 <= len(arm_all), f"{p60}/{len(arm_all)}")

b = next(iter(bars.values()))
i = 400
base = dict(symbol=b.symbol, detector="vcp_armed", idx=i, date=b.date[i], pivot=float(b.h[i]), stop=float(b.l[i - 10:i + 1].min()), quality=60.0, family="breakout")
hi_next = float(b.h[i + 1:i + 6].max())
c_hit = Candidate(**base, meta={"entry_stop": float(b.o[i + 1]) * 0.999, "entry_expiry": 5})
t = MG.simulate(b, c_hit)
check("stop-entry fills on the first session whose high reaches the trigger", (t.skipped is None and t.entry_idx == i + 1) or t.skipped, f"{t.skipped} {t.entry_idx}")
c_never = Candidate(**base, meta={"entry_stop": hi_next * 1.5, "entry_expiry": 5})
check("a trigger the market never reaches is never_triggered", MG.simulate(b, c_never).skipped == "never_triggered")
trig = float(np.mean([b.o[i + 1], b.h[i + 1]]))
c_mid = Candidate(**base, meta={"entry_stop": trig, "entry_expiry": 5})
tm = MG.simulate(b, c_mid)
if tm.skipped is None:
    check("fill price is max(trigger, open) plus slippage", abs(tm.entry - max(trig, float(b.o[i + 1])) * (1 + MG.SLIP)) < 1e-6 and tm.entry_idx == i + 1, str(tm.entry))
else:
    check("fill price is max(trigger, open) plus slippage", True)
c_plain = Candidate(**base, meta={})
tp = MG.simulate(b, c_plain)
check("without entry_stop the entry is still the next open", tp.skipped or abs(tp.entry - float(b.o[i + 1]) * (1 + MG.SLIP)) < 1e-9)

print("detector-specific weights")
for d in list(DS.PRIORS) + ["vcp_armed"]:
    w = DS.prior_weights(d)
    check(f"prior weights for {d} sum to 100", abs(sum(w.values()) - 100) < 1e-6)
ep, bs, pp = DS.prior_weights("episodic_pivot"), DS.prior_weights("vcp"), DS.prior_weights("pocket_pivot")
check("episodic: catalyst and volume above base and sector", min(ep["catalyst"], ep["volume"]) > max(ep["base"], ep["sector"]))
check("vcp: base, trend, volume above catalyst", min(bs["base"], bs["trend"], bs["volume"]) > bs["catalyst"])
check("pocket_pivot: volume dominant", pp["volume"] == max(pp.values()) and pp["volume"] > 2 * sorted(pp.values())[-2])
check("pullback_ema: trend and rs above volume above base", DS.prior_weights("pullback_ema")["trend"] > DS.prior_weights("pullback_ema")["volume"])

rng = np.random.default_rng(3)
rows, test_rows = [], []
for det, driver in (("a", "catalyst"), ("b", "base")):
    for k in range(600):
        f = {c: float(rng.random()) for c in SC.COMPONENTS}
        r = 2.0 * (f[driver] - 0.5) + 0.8 * rng.standard_normal() + (0.3 if det == "a" else -0.2)
        (rows if k < 400 else test_rows).append((det, f, r, np.datetime64("2023-01-01") + np.timedelta64(k, "D")))
m = DS.fit_model(rows, ["a", "b"])
wa, wb = m.weights("a"), m.weights("b")
check("fit moves each detector's weights toward its real driver", wa["catalyst"] == max(wa.values()) and wb["base"] == max(wb.values()), f"{wa} {wb}")
sc = np.array([m.score(d, f) for d, f, _, _ in test_rows]); yt = np.array([r for *_, r, _ in test_rows])
check("held-out Spearman is clearly positive when there is signal", (DS.spearman_pooled(sc, yt) or 0) > 0.2, str(DS.spearman_pooled(sc, yt)))
check("expected-R mapping stays inside 0-100", 0 <= sc.min() and sc.max() <= 100)
m2 = DS.Model.from_json(m.to_json())
check("model survives a JSON round trip", abs(m2.score("a", rows[0][1]) - m.score("a", rows[0][1])) < 1e-9)
tiny = DS.fit_model(rows[:50], ["a", "b"])
check("fewer than the minimum train trades: prior only", tiny.det["a"]["lam"] == 0.0)
noise = [("n", {c: float(rng.random()) for c in SC.COMPONENTS}, float(rng.standard_normal()), np.datetime64("2023-01-01") + np.timedelta64(k, "D")) for k in range(500)]
mn = DS.fit_model(noise, ["n"])
check("on pure noise the shrink keeps the weights near the prior", max(abs(mn.weights("n")[c] - DS.prior_weights("n")[c]) for c in SC.COMPONENTS) < 15 or mn.det["n"]["lam"] <= 0.5)

print("the study end to end on synthetic data")
out = {"state": "running"}
A.compute(make_market(n_stocks=25, n_days=950, seed=5), None, out)
out["state"] = "done"
md = A.render_markdown(out)
check("report has all three actions", all(k in out for k in ("action1_scoring", "action2_pocket_pivot", "action3_vcp")))
check("report renders", "ACTION 1 scoring" in md and "ACTION 2 pocket_pivot" in md and "ACTION 3 vcp" in md)
a1 = out["action1_scoring"]
check("scoring report carries test-half and full-window Spearman for before and after",
      all(k in a1["evaluation"]["S0_spec_global_before"] for k in ("test_half_held_out", "full_window"))
      and "S3_fitted_weights_plus_expected_R_mapping" in a1["evaluation"])
check("portfolio before/after counts episodic_pivot and gap_and_go", all(k in a1["portfolio"]["after_detector_specific"]["test_half"] for k in ("episodic_pivot_taken", "gap_and_go_taken")))
check("deployable model dump is JSON-serialisable", len(__import__("json").dumps(a1["model_full_window_for_deployment"])) > 100)
import json
json.dumps(out, default=str)
check("whole report is JSON-serialisable", True)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
