"""
e2_core.py — shared types and helpers for engine 2 (the multi-detector engine).

Engine 2 is built ALONGSIDE the live engine and shares none of its decision
code. It runs in shadow mode (records, never publishes) until a validation
report says otherwise.

DESIGN RULES (these are what the tests assert)
  * Causal. Every value at bar i uses bars <= i only. A pivot at j is "known"
    only from j+k (its confirmation bar).
  * One entry convention for every detector: a Candidate is CONFIRMED on the
    close of bar i; the order is a market-on-open on bar i+1. That is executable
    from a post-close scan and carries no intrabar look-ahead (the volume that
    confirms a breakout is only known at the close).
  * Detectors never reject for regime, fundamentals, RS or score. A detector
    answers one question: "does bar i complete this pattern?".
  * Every threshold is a named constant, fixed from the pattern's textbook
    definition BEFORE any result existed (pre-registered), overridable by env
    var for experiments but never tuned against the window that judges it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


def fenv(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def ienv(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# ---------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------
@dataclass
class Candidate:
    symbol: str
    detector: str
    idx: int                      # bar on whose close the pattern is confirmed
    date: np.datetime64
    pivot: float                  # the level that was broken / reference level
    stop: float                   # STRUCTURAL stop (the setup's invalidation price)
    quality: float                # 0-100 base/pattern quality, detector-specific
    family: str                   # breakout | reversal | momentum | continuation
    meta: dict = field(default_factory=dict)
    # filled later by the pipeline
    sector: str | None = None
    score: float | None = None
    score_parts: dict | None = None


# Families drive the regime's setup-preference weighting.
FAMILY = {
    "vcp": "breakout", "flat_base": "breakout", "base_on_base": "breakout",
    "cup_handle": "breakout", "ascending_base": "breakout", "flag_pennant": "breakout",
    "asc_triangle": "breakout", "sym_triangle": "breakout", "desc_triangle": "breakout",
    "high_tight_flag": "breakout", "double_bottom": "reversal", "rounding_bottom": "reversal",
    "inverse_hs": "reversal", "undercut_reclaim": "reversal",
    "pullback_ema": "continuation", "breakout_retest": "continuation",
    "episodic_pivot": "momentum", "momentum_burst": "momentum", "pocket_pivot": "momentum",
    "gap_and_go": "momentum",
}


# ---------------------------------------------------------------------
# Small numeric helpers (all causal unless stated)
# ---------------------------------------------------------------------
def rolling_max(a: np.ndarray, w: int) -> np.ndarray:
    """max over a[i-w+1 .. i]; NaN until the window is full."""
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        from numpy.lib.stride_tricks import sliding_window_view
        out[w - 1:] = sliding_window_view(a, w).max(axis=1)
    return out


def rolling_min(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        from numpy.lib.stride_tricks import sliding_window_view
        out[w - 1:] = sliding_window_view(a, w).min(axis=1)
    return out


def rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    if len(a) >= w:
        c = np.cumsum(np.insert(a.astype(float), 0, 0.0))
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def shift(a: np.ndarray, k: int) -> np.ndarray:
    """value k bars ago (NaN at the start)."""
    out = np.full(len(a), np.nan)
    if k < len(a):
        out[k:] = a[:-k] if k else a
    return out


def ema(a: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(a))
    if not len(a):
        return out
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def swing_points(arr: np.ndarray, k: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Fractal swing lows/highs: bar j is a swing low if it is the FIRST minimum of
    a[j-k .. j+k] (first, so a plateau yields one pivot). Returns (j, confirm)
    where confirm = j + k is the first bar on which the pivot is knowable.
    """
    n = len(arr)
    if n < 2 * k + 1:
        return np.array([], dtype=int), np.array([], dtype=int)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(arr, 2 * k + 1)
    pos = w.argmin(axis=1) if kind == "low" else w.argmax(axis=1)
    j = np.nonzero(pos == k)[0] + k
    return j, j + k


def linfit(y: np.ndarray, x: np.ndarray | None = None) -> tuple[float, float]:
    """least-squares slope, intercept"""
    if x is None:
        x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    xm, ym = x.mean(), y.mean()
    d = ((x - xm) ** 2).sum()
    if d == 0:
        return 0.0, ym
    s = ((x - xm) * (y - ym)).sum() / d
    return float(s), float(ym - s * xm)


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def band_score(v: float, lo: float, hi: float, soft: float) -> float:
    """100 inside [lo, hi]; falls linearly to 0 over `soft` outside it."""
    if v is None or np.isnan(v):
        return 0.0
    if lo <= v <= hi:
        return 100.0
    d = (lo - v) if v < lo else (v - hi)
    return clamp(100.0 * (1.0 - d / soft)) if soft > 0 else 0.0
