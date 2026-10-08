"""
e2_scorer.py — the 0-100 composite. It RANKS candidates; it rejects none. The portfolio
layer takes the top N.

  trend quality        0-20   MA stack, slopes, position vs 20EMA, nearness to the 52-week high
  relative strength    0-20   percentile of a blended 3/6/12-month return among the universe
  volume confirmation  0-15   breakout relative volume, OBV trend, prior dry-up, up/down volume
  base quality         0-20   the detector's own shape score
  catalyst strength    0-10   confirmed (news/results) or proxy (gap + volume in the last 10 bars)
  sector alignment     0-10   3-month RS rank of the stock's sector (top 3 boosted, bottom 3 penalised)
  regime fit           0-5    how well the setup family suits the current regime

Weights start at the spec's point values. `calibrate` re-weights them on the TRAIN half only
(non-negative ridge of realised R on the seven component fractions, shrunk 50/50 toward the
spec weights and bounded to 0.5x-2x of them) and the result is kept only if it ranks the
TEST half at least as well as the spec weights; otherwise the spec weights stay and the
report says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from e2_core import Candidate, clamp
from e2_features import Bars
from e2_regime import REGIME_FIT

COMPONENTS = ["trend", "rs", "volume", "base", "catalyst", "sector", "regime"]
SPEC_WEIGHTS = {"trend": 20.0, "rs": 20.0, "volume": 15.0, "base": 20.0, "catalyst": 10.0, "sector": 10.0, "regime": 5.0}


def _f(x, default=0.0):
    return default if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def trend_fraction(b: Bars, i: int) -> float:
    c = b.c[i]
    f = 0.0
    f += 0.15 * (c > b.sma50[i])
    f += 0.10 * (b.sma50[i] > b.sma150[i])
    f += 0.10 * (b.sma150[i] > b.sma200[i])
    f += 0.15 * bool(b.sma200_up[i])
    f += 0.10 * bool(b.sma50_up[i])
    f += 0.10 * (c > b.ema20[i])
    f += 0.10 * (b.ema8[i] > b.ema20[i])
    h52 = b.high52[i]
    if not np.isnan(h52) and h52 > 0:
        f += 0.20 * clamp(1.0 - (1.0 - c / h52) / 0.30, 0, 1)
    return float(min(f, 1.0))


def volume_fraction(b: Bars, i: int) -> float:
    f = 0.4 * clamp(_f(b.relvol[i]) / 3.0, 0, 1)
    f += 0.2 * (b.obv_trend[i] > 0)
    f += 0.2 * bool(np.nan_to_num(b.vol_dry5[max(0, i - 10):i].astype(float)).max() > 0) if i > 0 else 0.0
    lo = max(1, i - 49)
    up = b.v[lo:i + 1][b.c[lo:i + 1] > b.c[lo - 1:i]].sum()
    dn = b.v[lo:i + 1][b.c[lo:i + 1] < b.c[lo - 1:i]].sum()
    if dn > 0:
        f += 0.2 * clamp((up / dn - 0.8) / 0.8, 0, 1)
    elif up > 0:
        f += 0.2
    return float(min(f, 1.0))


def catalyst_fraction(b: Bars, cand: Candidate, override: float | None = None) -> tuple[float, str]:
    """override: a 0-1 value from real news/results data (live shadow scan). Without it, a
    price/volume proxy: a gap >= 3% on >= 3x volume in the last 10 bars."""
    if override is not None:
        return float(clamp(override, 0, 1)), "confirmed"
    if cand.detector == "episodic_pivot":
        if cand.meta.get("catalyst") == "confirmed":
            return 1.0, "confirmed"
        return float(0.6 * clamp(_f(cand.meta.get("gap_pct")) / 10.0, 0, 1)), "proxy"
    i = cand.idx
    w = slice(max(0, i - 9), i + 1)
    hit = (np.nan_to_num(b.gap_pct[w]) >= 3.0) & (np.nan_to_num(b.relvol[w]) >= 3.0)
    return (0.5 if hit.any() else 0.0), "proxy"


def components(b: Bars, cand: Candidate, rs_pct: float | None, sector_pts: float, regime_state: str,
               catalyst: float | None = None) -> dict:
    i = cand.idx
    cat, cat_src = catalyst_fraction(b, cand, catalyst)
    frac = {
        "trend": trend_fraction(b, i),
        "rs": 0.5 if rs_pct is None or np.isnan(rs_pct) else float(rs_pct) / 100.0,
        "volume": volume_fraction(b, i),
        "base": float(cand.quality) / 100.0,
        "catalyst": cat,
        "sector": float(sector_pts) / 10.0,
        "regime": REGIME_FIT.get(regime_state, REGIME_FIT["neutral"])[cand.family] / 5.0,
    }
    frac["_catalyst_source"] = cat_src
    frac["_rs_missing"] = rs_pct is None or bool(np.isnan(rs_pct))
    return frac


def total(frac: dict, weights: dict = SPEC_WEIGHTS) -> float:
    return float(sum(weights[c] * frac[c] for c in COMPONENTS))


# ---------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _nnls_ridge(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 400) -> np.ndarray:
    n, p = X.shape
    beta = np.zeros(p)
    XtX, Xty = X.T @ X, X.T @ y
    for _ in range(iters):
        for j in range(p):
            r = Xty[j] - XtX[j] @ beta + XtX[j, j] * beta[j]
            beta[j] = max(0.0, r / (XtX[j, j] + lam * n))
    return beta


def calibrate(train: list[tuple[dict, float]], test: list[tuple[dict, float]]) -> dict:
    """train/test: [(fractions, realised R)]. Returns the weights to use and the evidence for them."""
    out = {"weights": dict(SPEC_WEIGHTS), "adopted": False, "reason": "", "n_train": len(train), "n_test": len(test)}
    if len(train) < 200 or len(test) < 100:
        out["reason"] = "too few trades to calibrate; spec weights kept"
        return out
    Xtr = np.array([[f[c] for c in COMPONENTS] for f, _ in train])
    ytr = np.clip(np.array([r for _, r in train]), -2.0, 5.0)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    beta = _nnls_ridge((Xtr - mu) / sd, ytr - ytr.mean(), lam=0.5)
    if beta.sum() <= 0:
        out["reason"] = "no component carried a positive weight on the train half; spec weights kept"
        return out
    fitted = 100.0 * beta / beta.sum()
    spec = np.array([SPEC_WEIGHTS[c] for c in COMPONENTS])
    w = 0.5 * spec + 0.5 * fitted
    w = np.clip(w, 0.5 * spec, 2.0 * spec)
    w = 100.0 * w / w.sum()
    cand_w = dict(zip(COMPONENTS, [float(x) for x in w]))
    Xte = [f for f, _ in test]
    yte = np.array([r for _, r in test])
    ytr_raw = np.array([r for _, r in train])
    rho = lambda wts, rows, y: _spearman(np.array([total(f, wts) for f in rows]), y)
    out.update(
        fitted_raw=dict(zip(COMPONENTS, [float(x) for x in fitted])),
        calibrated=cand_w,
        spearman_train_spec=rho(SPEC_WEIGHTS, [f for f, _ in train], ytr_raw),
        spearman_train_cal=rho(cand_w, [f for f, _ in train], ytr_raw),
        spearman_test_spec=rho(SPEC_WEIGHTS, Xte, yte),
        spearman_test_cal=rho(cand_w, Xte, yte),
    )
    if out["spearman_test_cal"] >= out["spearman_test_spec"]:
        out["weights"], out["adopted"] = cand_w, True
        out["reason"] = "calibrated weights rank the held-out half at least as well as the spec weights"
    else:
        out["reason"] = "calibrated weights ranked the held-out half WORSE; spec weights kept"
    return out
