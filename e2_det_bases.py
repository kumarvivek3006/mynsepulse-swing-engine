"""
e2_det_bases.py — thirteen base / consolidation breakout detectors.

Existing (re-expressed on the shared interface from the textbook definitions
the live engine's own comments cite):
  vcp, flat_base, base_on_base, double_bottom, rounding_bottom, inverse_hs, high_tight_flag
Corrected (written to the spec; the old failure modes are in e2_diagnosis.py):
  cup_handle, ascending_base, flag_pennant, asc_triangle
New triangles:
  sym_triangle, desc_triangle (long side: the upside break only)

Convention: bar i is the breakout bar (close above the pivot on enough volume);
the pivot, structure and stop are all measured from bars < i. A pivot window W
is the number of bars before i searched for the base's high.

Thresholds are pre-registered constants (see e2_core docstring).
"""
from __future__ import annotations

import numpy as np

from e2_core import Candidate, band_score, clamp, fenv, ienv, linfit
from e2_det_common import (WARMUP, breakout_candidates, breakout_quality, dryup_score, line_through,
                           best_line, make, record_highs, saucer_candidates, touches, vol_slope, zigzag)
from e2_features import Bars

WINDOWS = (25, 35, 50, 70, 100, 140, 200)

# ---- pre-registered definitions --------------------------------------
BRK_VOL = fenv("E2_BRK_VOL", 1.4)               # generic breakout volume (x 50-bar average)
VOL15 = fenv("E2_VOL15", 1.5)                   # patterns whose spec says 1.5x
VCP_ZZ_ATR = fenv("E2_VCP_ZZ_ATR", 2.0)         # a swing must exceed 2 daily ATRs to count: smaller wiggles are noise
VCP_RATIO, VCP_FINAL, VCP_ZZ, VCP_MAX_DEPTH = fenv("E2_VCP_RATIO", 0.7), fenv("E2_VCP_FINAL", 8.0), fenv("E2_VCP_ZZ", 0.035), fenv("E2_VCP_MAX_DEPTH", 40.0)
FB_MAX_DEPTH, FB_MIN_LEN, FB_PRIOR_RUN, FB_CLOSE_RANGE = fenv("E2_FB_MAX_DEPTH", 15.0), ienv("E2_FB_MIN_LEN", 25), fenv("E2_FB_PRIOR_RUN", 0.30), fenv("E2_FB_CLOSE_RANGE", 0.12)
BOB_PRIOR, BOB_MAX_DEPTH, BOB_MIN_LEN, BOB_PRIOR_DEPTH = ienv("E2_BOB_PRIOR", 40), fenv("E2_BOB_MAX_DEPTH", 20.0), ienv("E2_BOB_MIN_LEN", 15), fenv("E2_BOB_PRIOR_DEPTH", 30.0)
BOB_PRIOR_LEN, BOB_RUN_IN, BOB_MAX_ADV, BOB_ON_TOP = ienv("E2_BOB_PRIOR_LEN", 30), fenv("E2_BOB_RUN_IN", 0.15), fenv("E2_BOB_MAX_ADV", 0.25), fenv("E2_BOB_ON_TOP", 0.97)
DB_K, DB_TOL, DB_MIN_SEP, DB_MAX_SEP, DB_RALLY, DB_FRESH = ienv("E2_DB_K", 4), fenv("E2_DB_TOL", 0.03), ienv("E2_DB_MIN_SEP", 15), ienv("E2_DB_MAX_SEP", 60), fenv("E2_DB_RALLY", 0.10), ienv("E2_DB_FRESH", 25)
RB_MIN, RB_MAX, RB_DEPTH_LO, RB_DEPTH_HI, RB_R2 = ienv("E2_RB_MIN", 40), ienv("E2_RB_MAX", 150), fenv("E2_RB_DEPTH_LO", 15.0), fenv("E2_RB_DEPTH_HI", 40.0), fenv("E2_RB_R2", 0.6)
IHS_K, IHS_HEAD, IHS_SHOULDER, IHS_MIN, IHS_MAX, IHS_FRESH = ienv("E2_IHS_K", 4), fenv("E2_IHS_HEAD", 0.03), fenv("E2_IHS_SHOULDER", 0.15), ienv("E2_IHS_MIN", 30), ienv("E2_IHS_MAX", 150), ienv("E2_IHS_FRESH", 30)
HTF_GAIN, HTF_RETRACE, HTF_FLAG_MIN, HTF_FLAG_MAX, HTF_POLE_MIN, HTF_POLE_MAX = fenv("E2_HTF_GAIN", 1.0), fenv("E2_HTF_RETRACE", 0.25), ienv("E2_HTF_FLAG_MIN", 15), ienv("E2_HTF_FLAG_MAX", 25), ienv("E2_HTF_POLE_MIN", 20), ienv("E2_HTF_POLE_MAX", 40)
CUP_MIN_LEN, CUP_DEPTH_LO, CUP_DEPTH_HI = ienv("E2_CUP_MIN_LEN", 35), fenv("E2_CUP_DEPTH_LO", 0.15), fenv("E2_CUP_DEPTH_HI", 0.35)
CUP_H_LO, CUP_H_HI, CUP_H_MIN, CUP_H_MAX, CUP_RIM, CUP_HVOL = fenv("E2_CUP_H_LO", 0.05), fenv("E2_CUP_H_HI", 0.15), ienv("E2_CUP_H_MIN", 5), ienv("E2_CUP_H_MAX", 25), fenv("E2_CUP_RIM", 0.10), fenv("E2_CUP_HVOL", 0.85)
CUP_MAX_LEN = 260
FB_MAX_LEN, BOB_MAX_LEN = 120, 80
AB_K, AB_GAP_LO, AB_GAP_HI, AB_DEPTH_LO, AB_DEPTH_HI, AB_FRESH = ienv("E2_AB_K", 4), ienv("E2_AB_GAP_LO", 15), ienv("E2_AB_GAP_HI", 25), fenv("E2_AB_DEPTH_LO", 0.03), fenv("E2_AB_DEPTH_HI", 0.15), ienv("E2_AB_FRESH", 30)
FP_POLE_LO, FP_POLE_HI, FP_POLE_MIN_BARS, FP_POLE_MAX_BARS = fenv("E2_FP_POLE_LO", 0.30), fenv("E2_FP_POLE_HI", 1.00), ienv("E2_FP_POLE_MIN_BARS", 20), ienv("E2_FP_POLE_MAX_BARS", 40)
FP_FLAG_MIN, FP_FLAG_MAX, FP_DEPTH = ienv("E2_FP_FLAG_MIN", 10), ienv("E2_FP_FLAG_MAX", 30), fenv("E2_FP_DEPTH", 0.5)
TRI_K, TRI_TOUCH_TOL, TRI_APEX_MAX, TRI_MIN_LEN = ienv("E2_TRI_K", 2), fenv("E2_TRI_TOUCH_TOL", 0.015), ienv("E2_TRI_APEX_MAX", 20), ienv("E2_TRI_MIN_LEN", 15)
TRI_WINDOWS = tuple(range(20, 121, 5))        # every window length in 5-bar steps: no triangle falls between grid points
U_SHARE = fenv("E2_U_SHARE", 0.35)             # a U spends >= this share of its bars in its bottom quarter


def _pivot_window(b: Bars, i: int, W: int):
    s = i - W
    if s < 0:
        return None
    p = s + int(np.argmax(b.h[s:i]))
    return p, float(b.h[p])


def _u_share(closes: np.ndarray) -> float:
    """Share of closes in the bottom quarter of their own range. Ideal parabola ~0.50, a V ~0.25;
    the threshold sits at the geometric midpoint, fixed from those two shapes, not from results."""
    lo_, hi_ = float(closes.min()), float(closes.max())
    if hi_ <= lo_:
        return 0.0
    return float((closes <= lo_ + 0.25 * (hi_ - lo_)).mean())


def _trend_ctx(b: Bars, i: int) -> bool:
    """price above a rising-ish 50 and above the 150: the only context any base needs"""
    return bool(b.c[i] > b.sma50[i] and b.sma50[i] > b.sma150[i])


# =====================================================================
def vcp(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    for i in breakout_candidates(b, max(lo, 160), hi, VOL15):
        if np.isnan(b.high52[i]) or not _trend_ctx(b, i):
            continue
        for p in record_highs(b.h, i, 200):
            pivot = float(b.h[p])
            if b.c[i] <= pivot or i - p < 25 or pivot < 0.85 * b.high52[i]:
                continue
            zz = zigzag(b.h, b.l, p, i - 1, max(VCP_ZZ, VCP_ZZ_ATR * float(b.atr_pct[i]) / 100.0))
            depths = [(zz[k][2] - zz[k + 1][2]) / zz[k][2] * 100 for k in range(len(zz) - 1)
                      if zz[k][1] == "H" and zz[k + 1][1] == "L"]
            if len(depths) < 2 or depths[0] > VCP_MAX_DEPTH:
                continue
            if any(nx > pv * VCP_RATIO for pv, nx in zip(depths, depths[1:])) or depths[-1] >= VCP_FINAL:
                continue
            shape = 0.5 * band_score(depths[-1], 1.0, 6.0, 6.0) + 0.5 * min(len(depths) / 4.0, 1.0) * 100
            q = breakout_quality(b, i, shape, dryup_score(b, max(p, i - 10), i))
            out.append(make(b, "vcp", int(i), pivot, b.l[i - 10:i].min(), q,
                            contractions=[round(d, 1) for d in depths], base_len=int(i - p)))
            break
    return out


# =====================================================================
def flat_base(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, BRK_VOL):
        for p in record_highs(b.h, i, FB_MAX_LEN):
            pivot = float(b.h[p])
            L = i - p
            if b.c[i] <= pivot or L < FB_MIN_LEN or pivot < 0.85 * b.high52[i]:
                continue
            base_lo = float(b.l[p:i].min())
            depth = (pivot - base_lo) / pivot * 100
            cl = b.c[p:i]
            if depth > FB_MAX_DEPTH or (cl.max() - cl.min()) / cl.max() > FB_CLOSE_RANGE:
                continue
            run_from = float(b.l[max(0, p - 126):p].min()) if p > 20 else np.nan
            if np.isnan(run_from) or pivot / run_from - 1 < FB_PRIOR_RUN:
                continue                                               # a flat base consolidates a PRIOR advance
            shape = 0.6 * band_score(depth, 5, 12, 8) + 0.4 * band_score((cl.max() - cl.min()) / cl.max() * 100, 2, 8, 6)
            q = breakout_quality(b, i, shape, dryup_score(b, p, i))
            out.append(make(b, "flat_base", int(i), pivot, base_lo, q, depth_pct=round(depth, 1), base_len=int(L)))
            break
    return out


# =====================================================================
def base_on_base(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    """
    A second base forming on top of a first one after only a short advance (IBD: "base on base").
      base 2  pivot p = a record high >= 15 bars back, depth <= 20%, broken out of at bar i
      base 1  an earlier high H1 (the top of its own base, 6-60 bars before p, broken out of before p) with
              - a consolidation behind it: the 30 bars to H1 stay within 30% of it and the stock was
                already within 15% of H1 at the start of them (a base, not a run)
              - base 2 on top of it: pivot2 > H1, no more than 25% above it, and base 2's low >= 97% of H1
    """
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP + 40), hi, BRK_VOL):
        done = False
        for p in record_highs(b.h, i, BOB_MAX_LEN):
            pivot = float(b.h[p])
            L = i - p
            if b.c[i] <= pivot or L < BOB_MIN_LEN or p < BOB_PRIOR + 30:
                continue
            base_lo = float(b.l[p:i].min())
            depth = (pivot - base_lo) / pivot * 100
            if depth > BOB_MAX_DEPTH:
                continue
            for q in range(p - 60, p - 5):
                if q < BOB_PRIOR_LEN:
                    continue
                H1 = float(b.h[q])
                if H1 >= pivot or pivot > H1 * (1 + BOB_MAX_ADV) or base_lo < BOB_ON_TOP * H1:
                    continue
                if H1 < float(b.h[q - BOB_PRIOR_LEN:q].max()):
                    continue                                           # q must be the top of its own base
                up = np.nonzero(b.c[q + 1:p + 1] > H1)[0]
                if not len(up):
                    continue                                           # price must close above H1 (the first base's breakout) before p
                k = q + 1 + int(up[0])                                 # the breakout bar of base 1
                if k > q + 1 and float(b.h[q + 1:k].max()) > H1:
                    continue                                           # a higher high before it: q was not the top of base 1
                seg = slice(q - BOB_PRIOR_LEN, q + 1)
                if (H1 - float(b.l[seg].min())) / H1 * 100 > BOB_PRIOR_DEPTH or b.c[q - BOB_PRIOR_LEN] < H1 * (1 - BOB_RUN_IN):
                    continue                                           # the stretch behind H1 must have been a base, not a run
                shape = 0.5 * band_score(depth, 4, 12, 8) + 0.5 * clamp(100 * (base_lo / H1 - BOB_ON_TOP) / 0.10)
                out.append(make(b, "base_on_base", int(i), pivot, base_lo, breakout_quality(b, i, shape, dryup_score(b, p, i)),
                                depth_pct=round(depth, 1), prior_high=round(H1, 2)))
                done = True
                break
            if done:
                break
    return out


# =====================================================================
def double_bottom(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    jl, conf = b.swings("low", DB_K)
    for i in breakout_candidates(b, max(lo, WARMUP), hi, BRK_VOL):
        cand = jl[(conf <= i - 1) & (jl >= i - 2 * DB_MAX_SEP - DB_FRESH)]
        found = None
        for x in range(len(cand) - 1, 0, -1):
            bb = int(cand[x])
            if i - bb > DB_FRESH + DB_MIN_SEP:
                break
            for y in range(x - 1, -1, -1):
                a = int(cand[y])
                if not (DB_MIN_SEP <= bb - a <= DB_MAX_SEP):
                    continue
                la, lb = float(b.l[a]), float(b.l[bb])
                if abs(lb - la) / la > DB_TOL:
                    continue
                mid = float(b.h[a:bb + 1].max())
                if mid < max(la, lb) * (1 + DB_RALLY) or b.c[i] <= mid:
                    continue
                if (b.c[bb + 1:i] > mid).any():
                    continue                                           # already broke out earlier
                found = (a, bb, mid, la, lb)
                break
            if found:
                break
        if not found:
            continue
        a, bb, mid, la, lb = found
        sym = 100 - clamp(abs(lb - la) / la / DB_TOL * 100)
        v1 = float(b.v[max(0, a - 2):a + 3].mean()); v2 = float(b.v[max(0, bb - 2):bb + 3].mean())
        shape = 0.5 * sym + 0.5 * (100 if v2 <= v1 else 50)
        out.append(make(b, "double_bottom", int(i), mid, min(la, lb), breakout_quality(b, i, shape, dryup_score(b, bb, i)),
                        low1=round(la, 2), low2=round(lb, 2), sep=int(bb - a)))
    return out


# =====================================================================
def rounding_bottom(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, BRK_VOL):
        for p, cb in saucer_candidates(b.h, b.l, i, RB_MIN, RB_MAX, min_left=8):
            L = i - p
            rim = float(b.h[p:i].max())                                # break the HIGHER of the two rims
            if b.c[i] <= rim or float(b.h[p]) < 0.9 * rim or i - cb < 8:
                continue
            seg_l = b.l[p:i]
            depth = (float(b.h[p]) - float(seg_l.min())) / float(b.h[p]) * 100
            if not (RB_DEPTH_LO <= depth <= RB_DEPTH_HI):
                continue
            x = np.arange(L, dtype=float)
            y = b.c[p:i]
            coef = np.polyfit(x, y, 2)
            if coef[0] <= 0:
                continue                                               # must be a U, not an inverted U
            fit = np.polyval(coef, x)
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - float(((y - fit) ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
            vertex = -coef[1] / (2 * coef[0]) / L
            if r2 < RB_R2 or not (0.25 <= vertex <= 0.75) or _u_share(y) < U_SHARE:
                continue
            shape = 0.6 * clamp(100 * (r2 - 0.4) / 0.5) + 0.4 * band_score(vertex, 0.4, 0.6, 0.25)
            out.append(make(b, "rounding_bottom", int(i), rim, seg_l.min(), breakout_quality(b, i, shape, dryup_score(b, p, i)),
                            r2=round(r2, 2), depth_pct=round(depth, 1), base_len=int(L)))
            break
    return out


# =====================================================================
def inverse_hs(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    jl, conf = b.swings("low", IHS_K)
    for i in breakout_candidates(b, max(lo, WARMUP), hi, BRK_VOL):
        cand = [int(x) for x in jl[(conf <= i - 1) & (jl >= i - IHS_MAX)]]
        found = None
        for rs in reversed(cand):
            if i - rs > IHS_FRESH:
                continue
            for hd in reversed([x for x in cand if x < rs]):
                for ls in reversed([x for x in cand if x < hd and IHS_MIN <= rs - x <= IHS_MAX]):
                    lL, lH, lR = float(b.l[ls]), float(b.l[hd]), float(b.l[rs])
                    if b.l[ls:rs + 1].min() < lH:
                        continue                                       # the head is the lowest point of the pattern
                    if not (lH < min(lL, lR) * (1 - IHS_HEAD)):
                        continue                                       # head clearly lowest
                    if abs(lL - lR) / min(lL, lR) > IHS_SHOULDER:
                        continue                                       # shoulders comparable
                    left, right = hd - ls, rs - hd
                    if not (0.5 <= right / max(left, 1) <= 2.0):
                        continue                                       # time symmetry
                    p1i = ls + int(np.argmax(b.h[ls:hd + 1])); p2i = hd + int(np.argmax(b.h[hd:rs + 1]))
                    p1, p2 = float(b.h[p1i]), float(b.h[p2i])
                    if p1i == p2i:
                        continue
                    slope = (p2 - p1) / (p2i - p1i)
                    neck = p2 + slope * (i - p2i)
                    if b.c[i] <= neck or b.c[i] <= min(p1, p2):
                        continue
                    found = (ls, hd, rs, lL, lH, lR, neck)
                    break
                if found:
                    break
            if found:
                break
        if not found:
            continue
        ls, hd, rs, lL, lH, lR, neck = found
        sym = 0.5 * (100 - clamp(abs(lL - lR) / min(lL, lR) / IHS_SHOULDER * 100)) + 0.5 * clamp(100 * ((min(lL, lR) - lH) / lH) / 0.15)
        out.append(make(b, "inverse_hs", int(i), neck, lR, breakout_quality(b, i, sym, dryup_score(b, rs, i)),
                        head=round(lH, 2), neckline=round(neck, 2)))
    return out


# =====================================================================
def high_tight_flag(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, VOL15):
        t = i - HTF_FLAG_MAX + int(np.argmax(b.h[i - HTF_FLAG_MAX:i]))
        F = i - t
        if not (HTF_FLAG_MIN <= F <= HTF_FLAG_MAX) or t - HTF_POLE_MAX < 0:
            continue
        PT = float(b.h[t])
        FH = float(b.h[t + 1:i].max()) if F > 1 else PT
        FL = float(b.l[t + 1:i].min())
        if b.c[i] <= FH or (PT - FL) / PT > HTF_RETRACE:
            continue
        q0 = t - HTF_POLE_MAX + int(np.argmin(b.l[t - HTF_POLE_MAX:t - HTF_POLE_MIN + 1]))
        if PT / float(b.l[q0]) - 1 < HTF_GAIN:
            continue
        shape = 0.5 * clamp(100 * (PT / float(b.l[q0]) - 1) / 1.5) + 0.5 * clamp(100 * (1 - (PT - FL) / PT / HTF_RETRACE))
        out.append(make(b, "high_tight_flag", int(i), FH, FL, breakout_quality(b, i, shape, dryup_score(b, t + 1, i)),
                        pole_gain=round(PT / float(b.l[q0]) - 1, 2), flag_len=int(F)))
    return out


# =====================================================================
def cup_handle(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    """
    Spec: cup >= 7 weeks, depth 15-35%, rounded (U), right rim within 10% of the left rim,
    handle 5-25 bars in the UPPER HALF of the cup, handle depth 5-15% on declining volume,
    breakout above the HANDLE high on >= 1.5x volume.
    """
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, VOL15):
        done = False
        for hl in range(CUP_H_MIN, CUP_H_MAX + 1):
            hs = i - hl
            HH = float(b.h[hs:i].max())
            if b.c[i] <= HH or b.h[hs] < HH:
                continue                                               # the handle starts at its own high (the right rim)
            HL = float(b.l[hs:i].min())
            hdepth = (HH - HL) / HH
            if not (CUP_H_LO <= hdepth <= CUP_H_HI):
                continue
            for p, cb in saucer_candidates(b.h, b.l, hs, CUP_MIN_LEN, CUP_MAX_LEN):
                LR = float(b.h[p])
                cup_len = hs - p
                CL = float(b.l[cb])
                if b.h[cb:hs + 1].max() > HH:
                    continue                                           # a higher rim peak exists: the handle starts there, not here
                depth = (LR - CL) / LR
                if not (CUP_DEPTH_LO <= depth <= CUP_DEPTH_HI):
                    continue
                if HH < LR * (1 - CUP_RIM):
                    continue                                           # right rim must recover to within 10% of the left
                if HL < CL + 0.5 * (LR - CL):
                    continue                                           # handle in the upper half of the cup
                y = b.c[p:hs]
                x = np.arange(len(y), dtype=float)
                coef = np.polyfit(x, y, 2)
                if coef[0] <= 0:
                    continue
                fit = np.polyval(coef, x)
                ss = float(((y - y.mean()) ** 2).sum())
                r2 = 1 - float(((y - fit) ** 2).sum()) / ss if ss > 0 else 0.0
                if r2 < 0.4 or _u_share(y) < U_SHARE:
                    continue                                           # a V, not a U
                if float(b.v[hs:i].mean()) > CUP_HVOL * float(b.v[p:hs].mean()):
                    continue                                           # handle volume must be declining
                shape = (0.4 * clamp(100 * (r2 - 0.3) / 0.6) + 0.3 * band_score(depth * 100, 20, 30, 10)
                         + 0.3 * band_score(hdepth * 100, 6, 12, 5))
                out.append(make(b, "cup_handle", int(i), HH, HL, breakout_quality(b, i, shape, dryup_score(b, hs, i)),
                                cup_len=int(cup_len), cup_depth=round(depth * 100, 1), handle_depth=round(hdepth * 100, 1),
                                handle_len=int(hl), r2=round(r2, 2)))
                done = True
                break
            if done:
                break
    return out


# =====================================================================
def ascending_base(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    """
    Spec: 3 pullbacks, each low higher than the last, each 3-5 weeks (15-25 bars)
    after the previous, each pullback < 15% (and a real one: >= 3%), volume
    declining through the base, breakout above the base high on volume.
    Any three swing lows may form the pattern (micro-swings between them are allowed),
    but each low must be the lowest point of its own leg.
    """
    out = []
    jl, conf = b.swings("low", AB_K)
    for i in breakout_candidates(b, max(lo, WARMUP), hi, BRK_VOL):
        cand = [int(x) for x in jl[(conf <= i - 1) & (jl >= i - 3 * AB_GAP_HI - AB_FRESH)]]
        found = None
        for t3 in reversed(cand):
            if i - t3 > AB_FRESH:
                continue
            for t2 in reversed([x for x in cand if AB_GAP_LO <= t3 - x <= AB_GAP_HI]):
                for t1 in reversed([x for x in cand if AB_GAP_LO <= t2 - x <= AB_GAP_HI]):
                    l1, l2, l3 = float(b.l[t1]), float(b.l[t2]), float(b.l[t3])
                    if not (l1 < l2 * 0.995 and l2 < l3 * 0.995):
                        continue                                       # each low higher than the last
                    if b.l[t1:t2 + 1].min() < l1 or b.l[t2:t3 + 1].min() < l2 or b.l[t3:i].min() < l3:
                        continue                                       # each low is the floor of its own leg
                    ok = True
                    for prev, t in ((t1 - AB_GAP_HI, t1), (t1, t2), (t2, t3)):
                        pk = float(b.h[max(prev, 0):t + 1].max())
                        d = (pk - float(b.l[t])) / pk
                        if not (AB_DEPTH_LO <= d < AB_DEPTH_HI):
                            ok = False
                            break
                    if not ok:
                        continue
                    base_hi = float(b.h[max(t1 - 10, 0):i].max())
                    if b.c[i] <= base_hi:
                        continue
                    vv = b.v[max(t1 - AB_GAP_HI, 0):t3 + 1]
                    if vv[len(vv) // 2:].mean() >= vv[:len(vv) // 2].mean():
                        continue                                       # volume must be declining through the base
                    found = (t1, t2, t3, l1, l2, l3, base_hi)
                    break
                if found:
                    break
            if found:
                break
        if not found:
            continue
        t1, t2, t3, l1, l2, l3, base_hi = found
        shape = clamp(100 * ((l3 / l1 - 1) / 0.10))
        out.append(make(b, "ascending_base", int(i), base_hi, l3, breakout_quality(b, i, shape, dryup_score(b, t1, i)),
                        lows=[round(l1, 2), round(l2, 2), round(l3, 2)], span=int(i - t1)))
    return out


# =====================================================================
def flag_pennant(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    """
    Spec: pole +30-100% in 4-8 weeks (20-40 bars), flag 2-6 weeks (10-30 bars), flag depth
    < 50% of the pole, volume declining in the flag, breakout above the flag high on >= 1.5x.
    """
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, VOL15):
        t = i - FP_FLAG_MAX + int(np.argmax(b.h[i - FP_FLAG_MAX:i]))
        F = i - t
        if not (FP_FLAG_MIN <= F <= FP_FLAG_MAX) or t - FP_POLE_MAX_BARS < 0:
            continue
        PT = float(b.h[t])
        FH = float(b.h[t + 1:i].max()) if F > 1 else PT
        FL = float(b.l[t + 1:i].min())
        if b.c[i] <= FH:
            continue
        w0, w1 = t - FP_POLE_MAX_BARS, t - FP_POLE_MIN_BARS
        q0 = w0 + int(np.argmin(b.l[w0:w1 + 1]))
        PL = float(b.l[q0])
        gain = PT / PL - 1
        if not (FP_POLE_LO <= gain < FP_POLE_HI):
            continue
        if (PT - FL) >= FP_DEPTH * (PT - PL):
            continue                                                   # flag must stay shallow relative to the pole
        pole_v = float(b.v[q0:t + 1].mean()); flag_v = b.v[t + 1:i]
        if flag_v.mean() >= pole_v:
            continue                                                   # volume must fall away from the pole's into the flag
        shape = 0.5 * clamp(100 * (gain - 0.25) / 0.5) + 0.5 * clamp(100 * (1 - (PT - FL) / (PT - PL) / FP_DEPTH))
        out.append(make(b, "flag_pennant", int(i), FH, FL, breakout_quality(b, i, shape, dryup_score(b, t + 1, i)),
                        pole_gain=round(gain, 2), flag_len=int(F), flag_depth_of_pole=round((PT - FL) / (PT - PL), 2)))
    return out


# =====================================================================
# Triangles
# =====================================================================
def _swing_pts(b: Bars, kind: str, s: int, e: int):
    j, conf = b.swings(kind, TRI_K)
    arr = b.h if kind == "high" else b.l
    m = (j >= s) & (conf <= e)
    return [(int(x), float(arr[x])) for x in j[m]]


def _apex(slope_a, icpt_a, slope_b, icpt_b):
    d = slope_a - slope_b
    return None if abs(d) < 1e-12 else (icpt_b - icpt_a) / d


def asc_triangle(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, VOL15):
        for W in TRI_WINDOWS:
            s = i - W
            if s < 0:
                continue
            H = _swing_pts(b, "high", s, i - 1)
            Lw = _swing_pts(b, "low", s, i - 1)
            if len(H) < 2 or len(Lw) < 2:
                continue
            R = max(p for _, p in H)
            top = [(x, p) for x, p in H if p >= R * (1 - TRI_TOUCH_TOL)]
            if len(top) < 2 or top[-1][0] - top[0][0] < 8:
                continue                                               # flat top: >= 2 touches, not adjacent
            if b.c[i] <= R:
                continue
            ln = best_line(Lw, "support", TRI_TOUCH_TOL)
            if ln is None or ln[0] <= 0:
                continue                                               # rising bottom with >= 2 touches
            sl, ic, ntouch, first = ln
            x0 = min(top[0][0], first)
            if i - x0 < TRI_MIN_LEN:
                continue
            ap = _apex(0.0, R, sl, ic)
            if ap is None or not (0 <= ap - i <= TRI_APEX_MAX):
                continue                                               # apex within 4 weeks
            if not _vol_declines(b, x0, i):
                continue                                               # volume declining into the apex
            shape = 0.5 * clamp(100 * (1 - (ap - i) / TRI_APEX_MAX)) + 0.5 * clamp(100 * len(top) / 4.0)
            out.append(make(b, "asc_triangle", int(i), R, Lw[-1][1], breakout_quality(b, i, shape, dryup_score(b, x0, i)),
                            flat_top=round(R, 2), touches=len(top), rising_touches=ntouch, bars_to_apex=int(ap - i)))
            break
    return out


def _vol_declines(b: Bars, x0: int, i: int) -> bool:
    half = x0 + (i - x0) // 2
    return bool(b.v[half:i].mean() < b.v[x0:half].mean())


def sym_triangle(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, BRK_VOL):
        for W in TRI_WINDOWS:
            s = i - W
            if s < 0:
                continue
            H = _swing_pts(b, "high", s, i - 1)
            Lw = _swing_pts(b, "low", s, i - 1)
            if len(H) < 2 or len(Lw) < 2:
                continue
            up = best_line(H, "resistance", 0.02)
            dn = best_line(Lw, "support", 0.02)
            if up is None or dn is None or up[0] >= 0 or dn[0] <= 0:
                continue                                               # falling highs, rising lows
            su, iu, tu, fu = up
            sl, il, tl, fl = dn
            upper = su * i + iu
            if b.c[i] <= upper:
                continue
            ap = _apex(su, iu, sl, il)
            x0 = min(fu, fl)
            if ap is None or ap <= i or i - x0 < TRI_MIN_LEN or not (0.4 <= (i - x0) / (ap - x0) <= 1.0):
                continue
            if not _vol_declines(b, x0, i):
                continue
            shape = 0.5 * clamp(100 * (i - x0) / (ap - x0)) + 0.5 * clamp(100 * (tu + tl) / 8.0)
            out.append(make(b, "sym_triangle", int(i), upper, Lw[-1][1], breakout_quality(b, i, shape, dryup_score(b, x0, i)),
                            upper_line=round(upper, 2), touches=tu + tl))
            break
    return out


def desc_triangle(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    """Long side only: the upside break of a descending triangle (the textbook-bearish
    breakdown is not tradable in a cash-delivery book; see breakdown_count)."""
    out = []
    for i in breakout_candidates(b, max(lo, WARMUP), hi, VOL15):
        for W in TRI_WINDOWS:
            s = i - W
            if s < 0:
                continue
            H = _swing_pts(b, "high", s, i - 1)
            Lw = _swing_pts(b, "low", s, i - 1)
            if len(H) < 2 or len(Lw) < 2:
                continue
            S = min(p for _, p in Lw)
            flat = [(x, p) for x, p in Lw if p <= S * (1 + TRI_TOUCH_TOL)]
            if len(flat) < 2 or flat[-1][0] - flat[0][0] < 8:
                continue
            up = best_line(H, "resistance", TRI_TOUCH_TOL * 1.5)
            if up is None or up[0] >= 0:
                continue
            su, iu, tu, fu = up
            upper = su * i + iu
            if b.c[i] <= upper:
                continue
            ap = _apex(su, iu, 0.0, S)
            x0 = min(fu, flat[0][0])
            if ap is None or not (0 <= ap - i <= TRI_APEX_MAX) or i - x0 < TRI_MIN_LEN:
                continue
            if not _vol_declines(b, x0, i):
                continue
            shape = 0.5 * clamp(100 * (1 - (ap - i) / TRI_APEX_MAX)) + 0.5 * clamp(100 * len(flat) / 4.0)
            out.append(make(b, "desc_triangle", int(i), upper, S, breakout_quality(b, i, shape, dryup_score(b, x0, i)),
                            support=round(S, 2), touches=len(flat), direction="long_upside_break"))
            break
    return out


def breakdown_count(b: Bars, lo: int, hi: int) -> int:
    """Informational only: descending-triangle DOWNSIDE breaks (not tradable long-only)."""
    n = 0
    for i in range(max(lo, WARMUP), hi):
        if not (b.relvol[i] >= VOL15 and b.c[i] < b.l[i - 5:i].min()):
            continue
        for W in TRI_WINDOWS:
            s = i - W
            if s < 0:
                continue
            Lw = _swing_pts(b, "low", s, i - 1)
            H = _swing_pts(b, "high", s, i - 1)
            if len(Lw) < 2 or len(H) < 2:
                continue
            S = min(p for _, p in Lw)
            flat = [(x, p) for x, p in Lw if p <= S * (1 + TRI_TOUCH_TOL)]
            su, iu = line_through(H)
            if len(flat) >= 2 and su < 0 and b.c[i] < S:
                n += 1
                break
    return n
