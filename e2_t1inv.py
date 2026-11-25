"""
e2_t1inv.py — INVERSE-DELIVERY TEST (owner's order of 26 Nov 03:22 IST, finding 1). Measurement only: nothing live reads this, nothing is published or gated.

Question: delivery % ranked the trades' realised R NEGATIVELY on the train half (Spearman -0.134). Is that a real, inverted component, or a train-half accident?

Method (declared before the run):
  component   delivery_inv = 1 - delivery fraction (the tier fraction of e2_tier1; missing stays neutral 0.5). A negative weight on delivery is a positive weight on delivery_inv.
              Variants: tier lag 0 (signal-session delivery), tier lag 1 (previous session, the live-feasible one), continuous lag 0 (1 - pct/100).
  refit       the pooled rule already in force (keep a component if TRAIN Spearman with R >= 0.05, weight proportional) on the train half only; scored on the untouched test half.
  composites  A   old components, old pooled weights                       (what is in force)
              A+  old components + delivery_inv                             (isolates the delivery effect)
              B   Tier 1 replacement components (no delivery)               (the Tier 1 result)
              C   B + delivery_inv, tier lag 0  (C1: lag 1, Cc: continuous)
  intervals   95%: (a) trade-level bootstrap (as in the Tier 1 report) and (b) WEEK-BLOCK bootstrap (trades in one week share a market; (a) is too narrow).
              Paired differences (A+ - A, C - B) are bootstrapped on the same resamples: that is the meaningful test, because the baseline composite is already about 0.20.
  gate        the owner's literal gate: composite test-half Spearman > 0.20 AND its interval excludes zero. Also reported: the paired gate, which needs the
              difference to the no-delivery composite to be > 0 with an interval that excludes zero.
  flip check  delivery alone on the test half (lag 0, lag 1, continuous) with intervals; a sign flip or an interval spanning zero means the train finding was spurious.
  confounds   delivery % tracks the stock type (large, low-volatility names have high delivery). Spearman of delivery with ATR% and with distance from the 52-week high,
              and the PARTIAL Spearman of delivery_inv with R after removing the rank effect of ATR% and of the distance from the high.
  per detector  top-quintile mean R by score A / B / C, per-detector delivery rho on train and test, and how many detectors keep the sign.
Caveat: the universe is today's constituents (survivorship), and the train half was used to choose the sign, so only the test half counts.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

import e2_backtest as BT
import e2_scorer as SC
import e2_t1study as S
import e2_tier1 as T1

log = logging.getLogger("engine2.t1inv")
KEY = "e2_t1inv"
LOG_PREFIX = "E2T1INV|"
BOOT_REPS = 600
OLD_C, NEW7 = SC.COMPONENTS, [c for c in T1.NEW_COMPONENTS if c != "delivery"]
_f, spearman = S._f, S.spearman


def _cont_inv(pct) -> float:
    return 0.5 if pct is None or pct != pct else float(1.0 - min(max(pct, 0.0), 100.0) / 100.0)


def augment(recs: list[dict]) -> None:
    """Adds the variant component dicts to every record in place."""
    for r in recs:
        o, n = r["old"], r["new"]
        inv0, inv1 = 1.0 - n["delivery"], 1.0 - r["deliv_lag1"]
        r["vA"] = dict(o)
        r["vAp"] = {**o, "dinv": inv0}
        r["vB"] = {c: n[c] for c in NEW7}
        r["vC"] = {**r["vB"], "dinv": inv0}
        r["vC1"] = {**r["vB"], "dinv": inv1}
        r["vCc"] = {**r["vB"], "dinv": _cont_inv(r.get("deliv_pct"))}
        r["dinv0"], r["dinv1"], r["dinvc"] = inv0, inv1, _cont_inv(r.get("deliv_pct"))


def _fit(train, key):
    comps = list(train[0][key])
    rho, w = S.pooled_weights(train, comps, key)
    return rho, w


def _scores(rows, key, w):
    if sum(w.values()) <= 0:                      # no component qualifies on the train half: the spec composite, as in the Tier 1 study
        return np.array([SC.total(r["old"]) for r in rows])
    return np.array([S.score_of(r, key, w) for r in rows])


def week_boot(rows, ys, score_map: dict, pairs: list, reps=BOOT_REPS, seed=23, by_week=True):
    """95% intervals for each score's Spearman and for each (a, b) pair difference, resampling WEEKS (or single trades when by_week is False)."""
    n = len(rows)
    if n < 100:
        return None
    rng = np.random.default_rng(seed)
    if by_week:
        wk = np.array([r["week"] for r in rows])
        u, inv = np.unique(wk, return_inverse=True)
        groups = [np.flatnonzero(inv == k) for k in range(len(u))]
    res = {k: [] for k in score_map}
    dres = {p: [] for p in pairs}
    for _ in range(reps):
        if by_week:
            pick = rng.integers(0, len(groups), len(groups))
            ix = np.concatenate([groups[k] for k in pick])
        else:
            ix = rng.integers(0, n, n)
        v = {k: spearman(sc[ix], ys[ix]) for k, sc in score_map.items()}
        for k, x in v.items():
            res[k].append(x)
        for a, b in pairs:
            dres[(a, b)].append(v[a] - v[b])
    ci = lambda xs: [_f(np.nanpercentile(xs, 2.5)), _f(np.nanpercentile(xs, 97.5))]            # noqa: E731
    return {"ci95": {k: ci(x) for k, x in res.items()}, "diff_ci95": {f"{a}-{b}": ci(dres[(a, b)]) for a, b in pairs}}


def _ranks(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def partial_spearman(x, y, controls: list) -> float:
    """Spearman of x with y after removing the linear effect (on ranks) of the controls; rows with any NaN are dropped."""
    cols = [np.asarray(c, float) for c in controls]
    m = ~(np.isnan(x) | np.isnan(y) | np.any([np.isnan(c) for c in cols], axis=0))
    if m.sum() < 50:
        return float("nan")
    rx, ry = _ranks(x[m]), _ranks(y[m])
    Z = np.column_stack([np.ones(m.sum())] + [_ranks(c[m]) for c in cols])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def study(recs: list[dict], mid) -> dict:
    split = np.datetime64(mid, "D")
    augment(recs)
    train = [r for r in recs if r["date"] < split]
    test = [r for r in recs if r["date"] >= split]
    out: dict = {"n_trades": len(recs), "n_train": len(train), "n_test": len(test), "split": str(split)}
    out["coverage"] = {"delivery_present_pct_train": _f(100 * np.mean([not r["deliv_missing"] for r in train])) if train else None,
                       "delivery_present_pct_test": _f(100 * np.mean([not r["deliv_missing"] for r in test])) if test else None}
    if len(train) < 200 or len(test) < 100:
        out["error"] = "too few trades for a train/test comparison"
        return out
    ytr = np.array([r["r"] for r in train], float)
    yte = np.array([r["r"] for r in test], float)

    # --- flip check: delivery alone, train vs test (+ intervals on the test half)
    solo = {}
    for name, key in (("tier_lag0", "dinv0"), ("tier_lag1", "dinv1"), ("continuous_lag0", "dinvc")):
        a = np.array([r[key] for r in train], float)
        b = np.array([r[key] for r in test], float)
        bt = week_boot(test, yte, {"x": b}, [], reps=BOOT_REPS) or {}
        bt_tr = week_boot(train, ytr, {"x": a}, [], reps=BOOT_REPS) or {}
        rt, re_ = spearman(a, ytr), spearman(b, yte)
        solo[name] = {"inverse_rho_train": _f(rt), "inverse_rho_test": _f(re_), "train_ci95_week": (bt_tr.get("ci95") or {}).get("x"), "test_ci95_week": (bt.get("ci95") or {}).get("x"),
                      "same_sign": bool(rt == rt and re_ == re_ and rt * re_ > 0),
                      "test_interval_excludes_zero": bool(bt and bt["ci95"]["x"][0] is not None and (bt["ci95"]["x"][0] > 0 or bt["ci95"]["x"][1] < 0))}
    out["delivery_alone_inverse"] = solo
    out["delivery_alone_note"] = "inverse_rho = Spearman of (1 - delivery fraction) with R; positive means LOW delivery ranks higher R. Train was the half the sign was read from."

    # --- composites
    fit = {}
    for label, key in (("A", "vA"), ("A+", "vAp"), ("B", "vB"), ("C", "vC"), ("C1", "vC1"), ("Cc", "vCc")):
        rho, w = _fit(train, key)
        fit[label] = (key, rho, w)
    out["weights"] = {k: {c: _f(v, 1) for c, v in w.items()} for k, (_, _, w) in fit.items()}
    out["train_spearman_components"] = {k: {c: _f(v) for c, v in rho.items()} for k, (_, rho, _) in fit.items() if k in ("A+", "C")}
    te = {k: _scores(test, key, w) for k, (key, _, w) in fit.items()}
    tr = {k: _scores(train, key, w) for k, (key, _, w) in fit.items()}
    te_ok = {k: v for k, v in te.items() if not np.isnan(v).all()}
    pairs = [p for p in (("A+", "A"), ("C", "B"), ("C", "A"), ("C1", "B"), ("Cc", "B")) if p[0] in te_ok and p[1] in te_ok]
    wb = week_boot(test, yte, te_ok, pairs, by_week=True) or {}
    tb = week_boot(test, yte, te_ok, pairs, by_week=False, seed=29) or {}
    comp = {}
    for k in te_ok:
        comp[k] = {"train_spearman": _f(spearman(tr[k], ytr)), "test_spearman": _f(spearman(te_ok[k], yte)),
                   "test_ci95_week_block": (wb.get("ci95") or {}).get(k), "test_ci95_trade_level": (tb.get("ci95") or {}).get(k)}
    out["composites"] = comp
    out["paired_differences_test"] = {f"{a}-{b}": {"diff": _f(spearman(te_ok[a], yte) - spearman(te_ok[b], yte)), "ci95_week_block": (wb.get("diff_ci95") or {}).get(f"{a}-{b}"),
                                                    "ci95_trade_level": (tb.get("diff_ci95") or {}).get(f"{a}-{b}")} for a, b in pairs}

    # --- decision gate
    def lit(k):
        c = comp.get(k)
        return bool(c and c["test_spearman"] is not None and c["test_spearman"] > 0.20 and c["test_ci95_week_block"] and c["test_ci95_week_block"][0] is not None
                    and c["test_ci95_week_block"][0] > 0)
    def paired(a, b):
        d = out["paired_differences_test"].get(f"{a}-{b}")
        return bool(d and d["diff"] is not None and d["diff"] > 0 and d["ci95_week_block"] and d["ci95_week_block"][0] is not None and d["ci95_week_block"][0] > 0)
    out["gate"] = {"literal_composite_gt_0_20_interval_excludes_zero": {k: lit(k) for k in ("A+", "C", "C1", "Cc") if k in comp},
                   "paired_improves_on_no_delivery_composite": {"A+ vs A": paired("A+", "A"), "C vs B": paired("C", "B"), "C1 vs B": paired("C1", "B"), "Cc vs B": paired("Cc", "B")},
                   "baseline_already_above_0_20": bool(comp.get("A", {}).get("test_spearman") is not None and comp["A"]["test_spearman"] > 0.20),
                   "delivery_alone_same_sign_both_halves": solo["tier_lag0"]["same_sign"],
                   "note": "the literal gate is met by a baseline that already sits at about 0.20, so the paired gate is the meaningful one; lag 1 is the variant a live scan could use"}
    out["gate"]["pass_literal"] = bool(any(out["gate"]["literal_composite_gt_0_20_interval_excludes_zero"].values()))
    out["gate"]["pass_paired"] = bool(any(out["gate"]["paired_improves_on_no_delivery_composite"].values()))

    # --- confounds (test half)
    atr = np.array([r["atrp"] for r in test], float)
    d52 = np.array([r["dist52"] for r in test], float)
    di = np.array([r["dinv0"] for r in test], float)
    dl = np.array([r["dinv1"] for r in test], float)
    mis = np.array([r["deliv_missing"] for r in test])
    ok = ~mis
    out["confounds_test"] = {"spearman_delivery_inv_with_atr_pct": _f(spearman(di[ok], atr[ok])), "spearman_delivery_inv_with_dist_from_52w_high": _f(spearman(di[ok], d52[ok])),
                             "spearman_atr_pct_with_R": _f(spearman(atr, yte)), "spearman_dist52_with_R": _f(spearman(d52, yte)),
                             "partial_inverse_delivery_R_control_atr": _f(partial_spearman(di[ok], yte[ok], [atr[ok]])),
                             "partial_inverse_delivery_R_control_atr_dist52": _f(partial_spearman(di[ok], yte[ok], [atr[ok], d52[ok]])),
                             "partial_inverse_delivery_lag1_control_atr_dist52": _f(partial_spearman(dl, yte, [atr, d52])),
                             "note": "if the partial values collapse toward zero, low delivery was a proxy for high-volatility / far-from-high stocks, not a component of its own"}
    terc = np.nanpercentile(atr, [33.3, 66.7])
    by = {}
    for name, sel in (("low_atr", atr <= terc[0]), ("mid_atr", (atr > terc[0]) & (atr <= terc[1])), ("high_atr", atr > terc[1])):
        m = sel & ok
        by[name] = {"n": int(m.sum()), "inverse_rho": _f(spearman(di[m], yte[m])) if m.sum() >= 30 else None}
    out["confounds_test"]["inverse_rho_within_atr_terciles"] = by

    # --- per detector
    per = {}
    k_same = 0
    k_all = 0
    wA, wB, wC = fit["A"][2], fit["B"][2], fit["C"][2]
    for d in sorted({r["det"] for r in test}):
        ts = [r for r in test if r["det"] == d]
        trn = [r for r in train if r["det"] == d]
        if len(ts) < 30 or len(trn) < 30:
            continue
        yo = np.array([r["r"] for r in ts], float)
        row = {"n_test": len(ts), "n_train": len(trn), "mean_R_all": _f(yo.mean()),
               "inverse_rho_train": _f(spearman([r["dinv0"] for r in trn], [r["r"] for r in trn])), "inverse_rho_test": _f(spearman([r["dinv0"] for r in ts], yo))}
        if row["inverse_rho_train"] is not None and row["inverse_rho_test"] is not None:
            k_all += 1
            k_same += int(row["inverse_rho_train"] * row["inverse_rho_test"] > 0)
        for lab, key, w in (("A", "vA", wA), ("B", "vB", wB), ("C", "vC", wC)):
            sc = _scores(ts, key, w)
            if np.isnan(sc).all():
                continue
            top = S._top(ts, sc)
            row[f"rho_{lab}"] = _f(spearman(sc, yo))
            row[f"top20_R_{lab}"] = S._mean(top, "r")
        per[d] = row
    out["per_detector"] = per
    out["per_detector_sign_consistency"] = {"detectors_compared": k_all, "same_sign_train_and_test": k_same}
    return out


def render(d: dict) -> str:
    L = ["# E2T1INV inverse-delivery test (measurement only; nothing live reads this)", json.dumps(d.get("window"), default=str),
         f"trades {d.get('n_trades')} (train {d.get('n_train')}, test {d.get('n_test')}), split {d.get('split')}"]
    for k in ("coverage", "delivery_alone_inverse", "weights", "train_spearman_components", "composites", "paired_differences_test", "gate", "confounds_test",
              "per_detector_sign_consistency"):
        L.append(f"## {k}")
        L.append(json.dumps(d.get(k), default=str))
    L.append("## per detector (test half; top20 = top-quintile picks by that score; rho_* Spearman of the score with R)")
    for det, row in (d.get("per_detector") or {}).items():
        L.append(f"{det} " + json.dumps(row, default=str))
    if d.get("error"):
        L.append(f"error: {d['error']}")
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for i in range(0, max(len(line), 1), 1400):
            log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
            n += 1
    return n


def compute(data, delivery: dict, out: dict, prog=None) -> dict:
    recs, mid = S.collect_recs(data, delivery, out, prog)
    out.update(study(recs, mid))
    return out


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running",
           "note": "Measurement only. The live scan is unchanged by this run."}
    try:
        data = BT.load_from_db(conn)
        start = min(df["trade_date"].iloc[0] for df in data.frames.values() if len(df))
        delivery = S.load_delivery(conn, list(data.frames), start)
        out["delivery_symbols"] = len(delivery)
        compute(data, delivery, out, lambda m: log.info("t1inv: %s", m))
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("inverse delivery study failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        conn.rollback()
        log.exception("could not store the inverse delivery study")
    try:
        log_markdown(render(out))
    finally:
        if own:
            conn.close()
    return out
