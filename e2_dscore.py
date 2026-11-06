"""
e2_dscore.py — detector-specific scoring (ACTION 1, 6 Nov).

Why: one weight set cannot score a gap-driven setup and a base-driven setup. The composite's components are the
same seven fractions (trend, rs, volume, base, catalyst, sector, regime); what changes per detector is how much each
one is worth.

Procedure (fixed before any result; nothing here looks at the live signals):
  1. PRIORS      the qualitative weighting the instruction gives per detector (high / medium / low / dominant) becomes
                 numbers once, in LEVEL below. Detectors the instruction does not name get a prior from their family,
                 declared below with the rest. A component the instruction does not mention gets DEFAULT_LEVEL.
  2. FIT         on the TRAIN half only, a non-negative ridge of realised R on the standardised fractions, per detector.
  3. SHRINK      the weights used are (1 - lam) * prior + lam * fit, lam in LAM_GRID chosen by expanding-window
                 time-block cross-validation INSIDE the train half. Fewer than MIN_FIT_N train trades: lam = 0 (prior only).
  4. COMPARE     weights alone make each detector's raw score 0-100 on its own scale; a slot allocator needs scores
                 that mean the same thing across detectors, so the final score maps the raw score to EXPECTED R:
                     E = mu_d + slope * (raw - raw_mean_d)         mu_d: train mean R of the detector, shrunk to the
                                                                   global mean with SHRINK_K pseudo-trades
                     score = clamp(50 + 100 * E, 0, 100)           slope: pooled within-detector slope of R on raw
                 The mapping is reported separately from the weights so its contribution is visible.
  5. TEST        the held-out half is scored once, by a model that never saw it.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from e2_scorer import COMPONENTS, SPEC_WEIGHTS, _nnls_ridge, _spearman

LEVEL = {"high": 28.0, "medium": 16.0, "low": 6.0, "dominant": 45.0}
DEFAULT_LEVEL = 10.0
LAM_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_FIT_N = 120
SHRINK_K = 30.0
CV_BLOCKS = 4
MIN_CV_N = 40

H, M, L, D = "high", "medium", "low", "dominant"
_BREAKOUT_BASE = {"base": H, "trend": H, "volume": H, "catalyst": L}

# Named by the instruction (6 Nov 18:51 IST); components it does not mention take DEFAULT_LEVEL.
PRIORS: dict[str, dict[str, str]] = {
    "episodic_pivot": {"catalyst": H, "volume": H, "base": L, "trend": M, "rs": M, "sector": L, "regime": M},
    "gap_and_go": {"catalyst": H, "volume": H, "base": L, "trend": M, "rs": M},
    "vcp": dict(_BREAKOUT_BASE),
    "cup_handle": dict(_BREAKOUT_BASE),
    "ascending_base": {"base": H, "trend": M, "volume": H},
    "pocket_pivot": {"volume": D, "base": L},
    "pullback_ema": {"trend": H, "rs": H, "volume": M},
    "flag_pennant": {"base": H, "catalyst": H, "volume": H},
    # Declared here for the detectors the instruction left to "the backtest disaggregation": their FAMILY's prior, which
    # the train-half fit then moves (it is not their final weighting).
    "flat_base": dict(_BREAKOUT_BASE), "base_on_base": dict(_BREAKOUT_BASE), "asc_triangle": dict(_BREAKOUT_BASE),
    "sym_triangle": dict(_BREAKOUT_BASE), "desc_triangle": dict(_BREAKOUT_BASE),
    "high_tight_flag": {"base": H, "trend": H, "rs": H, "volume": H, "catalyst": M},
    "double_bottom": {"base": H, "volume": H, "trend": L, "regime": M},
    "rounding_bottom": {"base": H, "volume": H, "trend": L, "regime": M},
    "inverse_hs": {"base": H, "volume": H, "trend": L, "regime": M},
    "undercut_reclaim": {"base": M, "volume": H, "trend": L, "regime": M},
    "breakout_retest": {"trend": H, "rs": H, "volume": M, "base": M},
    "momentum_burst": {"volume": H, "rs": H, "trend": M, "catalyst": M, "base": L},
}
# detectors reported by e2_actions that are variants of a named one
ALIAS = {"vcp_armed": "vcp", "vcp_armed_p60": "vcp"}


def prior_weights(detector: str) -> dict[str, float]:
    spec = PRIORS.get(ALIAS.get(detector, detector))
    if spec is None:
        return dict(SPEC_WEIGHTS)
    raw = np.array([LEVEL[spec[c]] if c in spec else DEFAULT_LEVEL for c in COMPONENTS], float)
    w = 100.0 * raw / raw.sum()
    return dict(zip(COMPONENTS, (float(x) for x in w)))


def _vec(w: dict[str, float]) -> np.ndarray:
    return np.array([w[c] for c in COMPONENTS], float)


def raw_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


def _fit(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    yc = np.clip(y, -2.0, 5.0)
    beta = _nnls_ridge((X - mu) / sd, yc - yc.mean(), lam=0.5)
    if beta.sum() <= 0:
        return None
    return 100.0 * beta / beta.sum()


def _blend(prior: np.ndarray, fit: np.ndarray | None, lam: float) -> np.ndarray:
    if fit is None or lam == 0.0:
        return prior.copy()
    w = (1.0 - lam) * prior + lam * fit
    return 100.0 * w / w.sum()


def choose_lambda(X: np.ndarray, y: np.ndarray, order: np.ndarray, prior: np.ndarray) -> tuple[float, dict]:
    """Expanding-window CV over CV_BLOCKS time blocks of the TRAIN half: fit on blocks[:k], score block k."""
    n = len(y)
    if n < MIN_FIT_N:
        return 0.0, {"reason": f"n_train {n} < {MIN_FIT_N}: prior only"}
    idx = order
    blocks = np.array_split(idx, CV_BLOCKS)
    res: dict[float, list[float]] = {lam: [] for lam in LAM_GRID}
    for k in range(1, CV_BLOCKS):
        tr = np.concatenate(blocks[:k])
        te = blocks[k]
        if len(tr) < MIN_CV_N or len(te) < MIN_CV_N:
            continue
        f = _fit(X[tr], y[tr])
        for lam in LAM_GRID:
            w = _blend(prior, f, lam)
            rho = _spearman(X[te] @ w, y[te])
            if rho == rho:
                res[lam].append(rho)
    means = {lam: (float(np.mean(v)) if v else float("nan")) for lam, v in res.items()}
    ok = {lam: m for lam, m in means.items() if m == m}
    if not ok:
        return 0.0, {"reason": "no usable CV fold: prior only"}
    best = max(ok.items(), key=lambda kv: (round(kv[1], 6), -kv[0]))[0]   # ties -> the smaller lam
    return float(best), {"cv_spearman_by_lam": {str(k): (round(v, 4) if v == v else None) for k, v in means.items()}}


class Model:
    def __init__(self):
        self.det: dict[str, dict] = {}
        self.slope = 0.0
        self.global_mean = 0.0
        self.meta: dict = {}

    # ---- scoring -----------------------------------------------------------------
    def _entry(self, detector: str) -> dict:
        d = ALIAS.get(detector, detector)
        e = self.det.get(d)
        if e is None:
            w = prior_weights(d)
            e = {"weights": w, "mu": self.global_mean, "raw_mean": float(np.dot(_vec(w), np.full(len(COMPONENTS), 0.5)))}
        return e

    def weights(self, detector: str) -> dict[str, float]:
        return self._entry(detector)["weights"]

    def raw(self, detector: str, frac: dict) -> float:
        w = self.weights(detector)
        return float(sum(w[c] * frac[c] for c in COMPONENTS))

    def score(self, detector: str, frac: dict, mapped: bool = True) -> float:
        r = self.raw(detector, frac)
        if not mapped:
            return r
        e = self._entry(detector)
        exp_r = e["mu"] + self.slope * (r - e["raw_mean"])
        return float(min(100.0, max(0.0, 50.0 + 100.0 * exp_r)))

    # ---- (de)serialisation -----------------------------------------------------------
    def to_json(self) -> dict:
        return {"version": 1, "components": COMPONENTS, "slope": self.slope, "global_mean": self.global_mean,
                "detectors": self.det, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Model":
        m = cls()
        m.det, m.slope, m.global_mean, m.meta = d["detectors"], float(d["slope"]), float(d["global_mean"]), d.get("meta", {})
        return m


def fit_model(rows: list[tuple[str, dict, float, np.datetime64]], detectors: list[str] | None = None) -> Model:
    """rows: (detector, fractions, realised R, signal date) — the TRAIN set only."""
    by: dict[str, list] = defaultdict(list)
    for det, frac, r, dt in rows:
        by[ALIAS.get(det, det)].append((np.array([frac[c] for c in COMPONENTS], float), float(r), dt))
    allr = np.array([r for v in by.values() for _, r, _ in v], float)
    m = Model()
    m.global_mean = float(allr.mean()) if len(allr) else 0.0
    cv_info = {}
    num = den = 0.0
    for det in sorted(set(by) | set(detectors or [])):
        prior = _vec(prior_weights(det))
        data = by.get(det, [])
        if data:
            X = np.array([x for x, _, _ in data]); y = np.array([r for _, r, _ in data]); dts = np.array([d for _, _, d in data])
            order = np.argsort(dts, kind="stable")
            lam, info = choose_lambda(X, y, order, prior)
            fit = _fit(X, y) if lam > 0 else None
            w = _blend(prior, fit, lam)
            raw = X @ w
            mu = (len(y) * float(y.mean()) + SHRINK_K * m.global_mean) / (len(y) + SHRINK_K)
            rc = raw - raw.mean()
            num += float((rc * (np.clip(y, -3, 6) - np.clip(y, -3, 6).mean())).sum())
            den += float((rc * rc).sum())
            m.det[det] = {"weights": dict(zip(COMPONENTS, (round(float(x), 3) for x in w))), "mu": float(mu),
                          "raw_mean": float(raw.mean()), "lam": lam, "n_train": int(len(y)),
                          "prior": dict(zip(COMPONENTS, (round(float(x), 3) for x in prior)))}
            cv_info[det] = info
        else:
            m.det[det] = {"weights": dict(zip(COMPONENTS, (round(float(x), 3) for x in prior))), "mu": m.global_mean,
                          "raw_mean": float(prior @ np.full(len(COMPONENTS), 0.5)), "lam": 0.0, "n_train": 0,
                          "prior": dict(zip(COMPONENTS, (round(float(x), 3) for x in prior)))}
            cv_info[det] = {"reason": "no train trades: prior only"}
    m.slope = max(num / den, 1e-6) if den > 0 else 1e-6
    m.meta = {"cv": cv_info, "levels": LEVEL, "default_level": DEFAULT_LEVEL, "lam_grid": list(LAM_GRID), "min_fit_n": MIN_FIT_N,
              "shrink_k": SHRINK_K}
    return m


# ---------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------
def spearman_pooled(scores: np.ndarray, y: np.ndarray) -> float | None:
    v = _spearman(np.asarray(scores, float), np.asarray(y, float))
    return None if v != v else round(float(v), 4)


def spearman_within(dets: list[str], scores: np.ndarray, y: np.ndarray, min_n: int = 30) -> dict:
    """n-weighted mean of the per-detector Spearman: how well the score orders trades INSIDE a detector."""
    out, num, den = {}, 0.0, 0
    for d in sorted(set(dets)):
        ix = np.array([i for i, x in enumerate(dets) if x == d])
        if len(ix) < min_n:
            continue
        rho = _spearman(scores[ix], y[ix])
        if rho == rho:
            out[d] = {"n": int(len(ix)), "rho": round(float(rho), 4)}
            num += rho * len(ix)
            den += len(ix)
    return {"weighted_mean": (round(num / den, 4) if den else None), "by_detector": out}


def dump(model: Model, path: str) -> None:
    with open(path, "w") as f:
        json.dump(model.to_json(), f, indent=1, sort_keys=True)


def load(path: str) -> Model:
    with open(path) as f:
        return Model.from_json(json.load(f))
