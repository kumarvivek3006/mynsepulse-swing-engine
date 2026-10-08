"""
e2_det_common.py — pieces shared by the detectors.

Every function takes a Bars (e2_features) and bar indices; nothing reads index
> i. `confirm_i` below is always the bar on whose close the pattern completes.
"""
from __future__ import annotations

import numpy as np

from e2_core import Candidate, FAMILY, clamp, linfit
from e2_features import Bars

WARMUP = 130        # first bar a detector may confirm on (6 months of history)


def make(b: Bars, name: str, i: int, pivot: float, stop: float, quality: float, **meta) -> Candidate:
    return Candidate(symbol=b.symbol, detector=name, idx=i, date=b.date[i], pivot=float(pivot),
                     stop=float(stop), quality=clamp(quality), family=FAMILY[name], meta=meta)


def breakout_quality(b: Bars, i: int, shape: float, dryup: float | None = None) -> float:
    """
    One common formula so detector qualities are comparable: how well the shape
    matches its definition (0-100, supplied by the detector), whether volume had
    dried up in the base, how strong the breakout volume was, and where the bar
    closed in its range.
    """
    rv = b.relvol[i]
    vol = 0.0 if np.isnan(rv) else min(rv / 3.0, 1.0) * 100.0
    cp = b.close_pos[i] * 100.0
    du = 50.0 if dryup is None else clamp(dryup)
    return 0.5 * shape + 0.2 * du + 0.2 * vol + 0.1 * cp


def dryup_score(b: Bars, lo: int, hi: int) -> float:
    """100 when the volume inside [lo, hi) is well below its own 50-bar norm, 0 when above."""
    if hi <= lo:
        return 50.0
    base = b.vavg50[hi] if hi < b.n and not np.isnan(b.vavg50[hi]) else np.nan
    if np.isnan(base) or base <= 0:
        return 50.0
    r = float(np.mean(b.v[lo:hi])) / base
    return clamp(100.0 * (1.3 - r) / 0.7)          # r=0.6 -> 100, r=1.3 -> 0


def vol_slope(v: np.ndarray) -> float:
    """normalised least-squares slope of volume (per bar, as a fraction of mean)."""
    if len(v) < 3 or np.mean(v) <= 0:
        return 0.0
    s, _ = linfit(v.astype(float))
    return float(s / np.mean(v))


def zigzag(h: np.ndarray, l: np.ndarray, s: int, e: int, thr: float) -> list[tuple[int, str, float]]:
    """
    Alternating swing highs/lows over bars [s, e] (inclusive), a reversal of
    >= thr (fraction) confirming a pivot. Starts from bar s taken as a PEAK.
    The trailing, unconfirmed extreme is returned as the last pivot, so a
    contraction still in progress is measured rather than ignored.
    """
    piv = [(s, "H", float(h[s]))]
    mode = "down"                     # looking for a trough after the peak
    ext_i, ext_p = s, float(l[s])
    for t in range(s + 1, e + 1):
        if mode == "down":
            if l[t] < ext_p:
                ext_i, ext_p = t, float(l[t])
            elif h[t] >= ext_p * (1 + thr) and ext_p > 0:
                piv.append((ext_i, "L", ext_p))
                mode, ext_i, ext_p = "up", t, float(h[t])
        else:
            if h[t] > ext_p:
                ext_i, ext_p = t, float(h[t])
            elif l[t] <= ext_p * (1 - thr):
                piv.append((ext_i, "H", ext_p))
                mode, ext_i, ext_p = "down", t, float(l[t])
    piv.append((ext_i, "L" if mode == "down" else "H", ext_p))
    return piv


def line_through(points: list[tuple[int, float]]) -> tuple[float, float]:
    xs = np.array([p[0] for p in points], float)
    ys = np.array([p[1] for p in points], float)
    return linfit(ys, xs)


def touches(points: list[tuple[int, float]], slope: float, icpt: float, tol: float) -> int:
    n = 0
    for x, y in points:
        ref = slope * x + icpt
        if ref > 0 and abs(y - ref) / ref <= tol:
            n += 1
    return n


def breakout_candidates(b: Bars, lo: int, hi: int, min_relvol: float, lookback: int = 5) -> np.ndarray:
    """
    Cheap necessary condition shared by every breakout-family detector: the bar
    closes above the highest high of the previous `lookback` bars on at least
    `min_relvol` times normal volume. Any pivot built from >= lookback bars
    satisfies it, so no real breakout is skipped.
    """
    idx = np.arange(max(lo, lookback + 1), hi)
    if idx.size == 0:
        return idx
    prior = np.array([b.h[i - lookback:i].max() for i in idx])
    ok = (b.c[idx] > prior) & (b.relvol[idx] >= min_relvol)
    return idx[ok]


def record_highs(h: np.ndarray, end: int, max_back: int) -> list[int]:
    """
    Every bar j in [end-max_back, end) whose high is strictly above every high after it up to
    end-1 — i.e. every bar that could be "the base high" for some base length. Nearest first.
    Replaces a coarse grid of window lengths: no base length is skipped because it fell between
    two grid points.
    """
    out, m = [], -np.inf
    for j in range(end - 1, max(end - max_back, 0) - 1, -1):
        if h[j] > m:
            out.append(j)
            m = h[j]
    return out


def saucer_candidates(h: np.ndarray, l: np.ndarray, end: int, min_len: int, max_len: int, min_left: int = 5,
                      lead: int = 10):
    """
    Every (left_rim, bottom) pair that could bound a cup or saucer ending just before `end`:
      bottom   a bar whose low is below every low after it (up to end-1)         — the true floor
      left rim a bar whose high is >= every high between it and the bottom, reached before any
               lower low appears to its left, and >= the `lead` bars before it (so a point
               half-way down a slope is not mistaken for the shoulder)           — the true shoulder
    Returned as (p, cb) with end - p in [min_len, max_len] and cb - p >= min_left.
    No window-length grid: a shape of any length is enumerated.
    """
    out = []
    lo_edge = max(end - max_len, 0)
    m = np.inf
    for cb in range(end - 1, lo_edge - 1, -1):
        if l[cb] >= m:
            continue
        m = l[cb]
        top = h[cb]
        for p in range(cb - 1, lo_edge - 1, -1):
            if l[p] < l[cb]:
                break
            if h[p] > top:
                top = h[p]
                if end - p >= min_len and cb - p >= min_left and p >= lead and h[p] >= h[p - lead:p].max():
                    out.append((p, cb))
    return out


def best_line(pts: list[tuple[int, float]], side: str, tol: float, min_gap: int = 5):
    """
    The trend line a triangle's boundary actually is: the line through two of the swing points that
    no other swing point breaks through (a SUPPORT line has no swing low below it by more than
    `tol`; a RESISTANCE line no swing high above it), with the most touches (points within `tol`).
    Minor swing points that sit on the safe side of the line are ignored, not disqualifying.
    Returns (slope, intercept, touches, first_touch_x) or None.
    """
    best = None
    for a in range(len(pts)):
        for c in range(a + 1, len(pts)):
            (x1, y1), (x2, y2) = pts[a], pts[c]
            if x2 - x1 < min_gap:
                continue
            k = (y2 - y1) / (x2 - x1)
            m = y1 - k * x1
            ok, touch, first = True, 0, None
            for x, y in pts:
                ref = k * x + m
                if ref <= 0:
                    ok = False
                    break
                dev = (y - ref) / ref
                if (side == "support" and dev < -tol) or (side == "resistance" and dev > tol):
                    ok = False
                    break
                if abs(dev) <= tol:
                    touch += 1
                    first = x if first is None else first
            if ok and touch >= 2 and (best is None or (touch, x2 - x1) > (best[2], best[4])):
                best = (k, m, touch, first, x2 - x1)
    return None if best is None else best[:4]
