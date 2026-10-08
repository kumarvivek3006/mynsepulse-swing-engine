"""
e2_det_momentum.py — seven event/momentum/continuation detectors.

  episodic_pivot   gap >= 4%, volume >= 5x, close in the top 25% of the range
  momentum_burst   range > 1.5 ATR, close top 25%, after a 3+ bar quiet spell, vol >= 1.5x
  pocket_pivot     up day, volume above the largest down-day volume of the prior 10 bars, close in the top half
  pullback_ema     uptrend, pullback to the 8/20 EMA, reversal candle, quieter volume
  undercut_reclaim wick below a confirmed swing low, close back above it, volume >= 1.5x
  gap_and_go       gap >= 3%, holds 2+ bars, closes above the gap day's high
  breakout_retest  breakout, pullback to the pivot holding within 2%, resumption on volume

Signature: fn(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]; evaluates
confirmation bars lo <= i < hi. All thresholds are pre-registered constants.
"""
from __future__ import annotations

import numpy as np

from e2_core import Candidate, band_score, clamp, fenv, ienv, shift
from e2_det_common import WARMUP, dryup_score, make
from e2_features import Bars

# ---- pre-registered definitions --------------------------------------
EP_GAP, EP_VOL, EP_CLOSEPOS = fenv("E2_EP_GAP", 4.0), fenv("E2_EP_VOL", 5.0), fenv("E2_EP_CLOSEPOS", 0.75)
MB_RANGE_ATR, MB_CLOSEPOS, MB_VOL = fenv("E2_MB_RANGE_ATR", 1.5), fenv("E2_MB_CLOSEPOS", 0.75), fenv("E2_MB_VOL", 1.5)
MB_QUIET_BARS, MB_QUIET_RANGE_ATR = ienv("E2_MB_QUIET_BARS", 3), fenv("E2_MB_QUIET_RANGE_ATR", 1.0)
PP_LOOK, PP_CLOSEPOS = ienv("E2_PP_LOOK", 10), fenv("E2_PP_CLOSEPOS", 0.5)
PB_HIGH_WIN, PB_MIN_BARS, PB_MAX_BARS = ienv("E2_PB_HIGH_WIN", 12), ienv("E2_PB_MIN_BARS", 2), ienv("E2_PB_MAX_BARS", 8)
PB_MIN_DEPTH, PB_MAX_DEPTH = fenv("E2_PB_MIN_DEPTH", 0.03), fenv("E2_PB_MAX_DEPTH", 0.18)
PB_TOUCH_TOL, PB_HOLD = fenv("E2_PB_TOUCH_TOL", 0.004), fenv("E2_PB_HOLD", 0.985)
UR_K, UR_LOOKBACK, UR_MIN_UNDERCUT = ienv("E2_UR_K", 5), ienv("E2_UR_LOOKBACK", 80), fenv("E2_UR_MIN_UNDERCUT", 0.003)
UR_VOL, UR_INTACT = fenv("E2_UR_VOL", 1.5), fenv("E2_UR_INTACT", 0.99)
GG_GAP, GG_HOLD, GG_MAX_WAIT = fenv("E2_GG_GAP", 3.0), ienv("E2_GG_HOLD", 2), ienv("E2_GG_MAX_WAIT", 8)
BR_PIVOT_WIN, BR_VOL, BR_RETEST_WIN = ienv("E2_BR_PIVOT_WIN", 40), fenv("E2_BR_VOL", 1.3), ienv("E2_BR_RETEST_WIN", 15)
BR_TOL, BR_RESUME_WIN, BR_RESUME_VOL = fenv("E2_BR_TOL", 0.02), ienv("E2_BR_RESUME_WIN", 3), fenv("E2_BR_RESUME_VOL", 1.0)


def _ok(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs:
        m &= ~np.isnan(a)
    return m


# =====================================================================
def episodic_pivot(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    """
    Gap-up on heavy volume closing strong. `catalyst` is recorded, not required:
    historical news/results data does not exist for the backtest window, so
    requiring it would make the detector untestable. When the live context
    supplies event dates the candidate is marked confirmed and the scorer pays
    for it; otherwise it is marked as a volume/gap proxy.
    """
    out = []
    events = ctx.get("catalyst_dates")           # np.datetime64 array or None
    idx = np.arange(max(lo, WARMUP), hi)
    m = ((b.gap_pct[idx] >= EP_GAP) & (b.relvol[idx] >= EP_VOL) & (b.close_pos[idx] >= EP_CLOSEPOS))
    for i in idx[m]:
        confirmed = False
        if events is not None and len(events):
            confirmed = bool(np.any(np.abs((events - b.date[i]).astype("timedelta64[D]").astype(int)) <= 1))
        q = (40 * min(b.gap_pct[i] / 12.0, 1) + 35 * min(b.relvol[i] / 15.0, 1) + 25 * b.close_pos[i])
        out.append(make(b, "episodic_pivot", int(i), b.h[i], b.l[i], q,
                        gap_pct=round(float(b.gap_pct[i]), 2), relvol=round(float(b.relvol[i]), 2),
                        catalyst="confirmed" if confirmed else "proxy"))
    return out


# =====================================================================
def momentum_burst(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    narrow = b.range_atr <= MB_QUIET_RANGE_ATR
    quiet = np.ones(b.n, bool)
    for k in range(1, MB_QUIET_BARS + 1):
        quiet &= shift(narrow.astype(float), k) == 1.0
    idx = np.arange(max(lo, WARMUP), hi)
    m = ((b.range_atr[idx] > MB_RANGE_ATR) & (b.close_pos[idx] >= MB_CLOSEPOS) & (b.c[idx] > b.c_prev[idx])
         & (b.relvol[idx] >= MB_VOL) & quiet[idx])
    for i in idx[m]:
        box_hi = float(b.h[i - MB_QUIET_BARS:i].max())
        if b.c[i] <= box_hi:
            continue                                     # a burst must clear the quiet spell's high
        box_lo = float(b.l[i - MB_QUIET_BARS:i].min())
        tight = 100 * (1 - min((box_hi - box_lo) / max(b.atr_prev[i], 1e-9) / 3.0, 1))
        q = 0.3 * tight + 0.3 * min(b.range_atr[i] / 3.0, 1) * 100 + 0.2 * min(b.relvol[i] / 4.0, 1) * 100 + 0.2 * b.close_pos[i] * 100
        out.append(make(b, "momentum_burst", int(i), box_hi, b.l[i], q,
                        range_atr=round(float(b.range_atr[i]), 2), relvol=round(float(b.relvol[i]), 2)))
    return out


# =====================================================================
def pocket_pivot(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    """
    Up day whose volume exceeds the HIGHEST down-day volume of the prior 10
    bars. If there was no down day in those 10 bars the comparison is against
    the 20-bar average volume instead (otherwise any up day would qualify).
    """
    out = []
    idx = np.arange(max(lo, WARMUP), hi)
    pre = (b.c[idx] > b.c_prev[idx]) & (b.close_pos[idx] >= PP_CLOSEPOS)
    for i in idx[pre]:
        w = slice(i - PP_LOOK, i)
        down = b.c[w] < shift(b.c, 1)[w]
        if down.any():
            ref = float(b.v[w][down].max())
        else:
            ref = float(b.vavg20[i]) if not np.isnan(b.vavg20[i]) else np.nan
        if np.isnan(ref) or b.v[i] <= ref:
            continue
        stop = float(b.l[i - 4:i + 1].min())
        q = 0.4 * min(b.v[i] / ref / 2.0, 1) * 100 + 0.3 * b.close_pos[i] * 100 + 0.3 * (100 if b.c[i] > b.sma50[i] else 40)
        out.append(make(b, "pocket_pivot", int(i), b.h[i], stop, q,
                        vol_vs_down=round(float(b.v[i] / ref), 2), no_down_day=bool(not down.any())))
    return out


# =====================================================================
def pullback_ema(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    for i in range(max(lo, WARMUP), hi):
        if not (b.c[i] > b.sma50[i] and b.sma50_up[i] and b.ema20[i] > b.sma50[i]):
            continue                                                    # trend up
        if not (b.c[i] > b.o[i] and b.c[i] > b.c[i - 1] and b.close_pos[i] >= 0.5):
            continue                                                    # reversal candle
        w0 = i - PB_HIGH_WIN + 1
        hi_idx = w0 + int(np.argmax(b.h[w0:i + 1]))
        since = i - hi_idx
        if not (PB_MIN_BARS <= since <= PB_MAX_BARS):
            continue
        hh = float(b.h[hi_idx])
        pb_low = float(b.l[hi_idx + 1:i + 1].min())
        depth = (hh - pb_low) / hh
        if not (PB_MIN_DEPTH <= depth <= PB_MAX_DEPTH):
            continue
        # touch of the 8 or 20 EMA on the signal bar or the 2 bars before it, with NO close through the 20 over those bars
        window = [j for j in (i, i - 1, i - 2) if j > hi_idx]
        if any(b.c[j] < b.ema20[j] * PB_HOLD for j in window):
            continue
        touched, which = False, None
        for j in window:
            for name, e in (("ema8", b.ema8[j]), ("ema20", b.ema20[j])):
                if b.l[j] <= e * (1 + PB_TOUCH_TOL):
                    touched, which = True, name
                    break
            if touched:
                break
        if not touched:
            continue
        pb_vol = float(np.mean(b.v[hi_idx + 1:i])) if i - hi_idx > 1 else float(b.v[i - 1])
        ref = float(b.vavg20[i]) if not np.isnan(b.vavg20[i]) else np.nan
        if np.isnan(ref) or pb_vol > 0.9 * ref:
            continue                                                    # volume declining into the pullback
        stop = float(b.l[max(hi_idx + 1, i - 2):i + 1].min())
        shape = 0.5 * band_score(depth * 100, 5, 12, 8) + 0.5 * band_score(float(b.ema20[i] / b.ema20[i - 5] - 1) * 100, 1, 6, 4)
        q = 0.5 * shape + 0.25 * clamp(100 * (0.9 - pb_vol / ref) / 0.5) + 0.25 * b.close_pos[i] * 100
        out.append(make(b, "pullback_ema", int(i), hh, stop, q, touched=which,
                        depth_pct=round(depth * 100, 2), bars_down=int(since)))
    return out


# =====================================================================
def undercut_reclaim(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    jl, conf = b.swings("low", UR_K)
    idx = np.arange(max(lo, WARMUP), hi)
    pre = (b.relvol[idx] >= UR_VOL) & (b.c[idx] > b.l[idx])
    for i in idx[pre]:
        usable = jl[(conf <= i - 1) & (jl >= i - UR_LOOKBACK)]
        best = None
        for j in usable:
            lvl = float(b.l[j])
            if not (b.l[i] < lvl * (1 - UR_MIN_UNDERCUT) and b.c[i] > lvl):
                continue
            if i - j <= UR_K:
                continue
            if float(b.c[j + 1:i].min()) < lvl * UR_INTACT:
                continue                                                # level already broken on a close: not a first undercut
            if best is None or lvl > best:
                best = lvl
        if best is None:
            continue
        depth = (best - b.l[i]) / best * 100
        q = (0.35 * band_score(depth, 0.5, 5, 4) + 0.25 * min(b.relvol[i] / 3.0, 1) * 100
             + 0.2 * b.close_pos[i] * 100 + 0.2 * (100 if b.c[i] > b.o[i] else 40))
        out.append(make(b, "undercut_reclaim", int(i), best, b.l[i], q,
                        undercut_pct=round(float(depth), 2), level=round(best, 2)))
    return out


# =====================================================================
def gap_and_go(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    gaps = np.nonzero(b.gap_pct >= GG_GAP)[0]
    for g in gaps:
        if g < WARMUP:
            continue
        gap_fill = float(b.c_prev[g])
        gh = float(b.h[g])
        for i in range(g + GG_HOLD + 1, min(g + GG_MAX_WAIT + 1, b.n)):
            if i < lo or i >= hi:
                continue
            seg = slice(g + 1, i)
            if (b.l[seg] < gap_fill).any() or (b.c[seg] > gh).any():
                break                                                   # gap filled, or already broke out earlier
            if (b.c[seg] < b.o[g] * 0.97).any():
                break                                                   # held on paper but faded
            if b.c[i] > gh:
                q = (0.35 * min(b.gap_pct[g] / 8.0, 1) * 100 + 0.25 * min(b.relvol[g] / 5.0, 1) * 100
                     + 0.2 * b.close_pos[i] * 100 + 0.2 * dryup_score(b, g + 1, i))
                out.append(make(b, "gap_and_go", int(i), gh, b.l[g], q,
                                gap_pct=round(float(b.gap_pct[g]), 2), days_held=int(i - g - 1)))
                break
    return out


# =====================================================================
def breakout_retest(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    out = []
    n = b.n
    # breakout bars: close above the prior BR_PIVOT_WIN-bar high on >= BR_VOL x volume
    prior_hi = np.full(n, np.nan)
    for t in range(BR_PIVOT_WIN, n):
        prior_hi[t] = b.h[t - BR_PIVOT_WIN:t].max()
    brk = np.nonzero((b.c > prior_hi) & (b.relvol >= BR_VOL))[0]
    used = set()
    for bo in brk:
        if bo < WARMUP:
            continue
        P = float(prior_hi[bo])
        for i in range(bo + 3, min(bo + BR_RETEST_WIN + BR_RESUME_WIN + 1, n)):
            if i < lo or i >= hi or (bo, ) in used:
                continue
            # retest: some bar j in (bo, i] dipped to within tolerance of the pivot without breaking it
            js = range(bo + 2, i)
            rt = [j for j in js if P * (1 - BR_TOL) <= b.l[j] <= P * (1 + BR_TOL) and b.c[j] >= P * (1 - BR_TOL / 2)]
            if not rt:
                continue
            j0 = rt[-1]                          # the most recent retest bar
            if i - j0 > BR_RESUME_WIN or j0 - bo > BR_RETEST_WIN:
                continue
            if (b.c[bo + 1:i] < P * (1 - BR_TOL)).any():
                continue                                                # the pivot did not hold
            # resumption: closes above the prior high, up on the day, volume at least normal
            if not (b.c[i] > b.h[i - 1] and b.c[i] > b.c_prev[i] and b.relvol20[i] >= BR_RESUME_VOL):
                continue
            stop = float(b.l[j0:i + 1].min())
            q = (0.35 * band_score(abs(b.l[j0] / P - 1) * 100, 0, 1.0, 2.0) + 0.25 * min(b.relvol[bo] / 3.0, 1) * 100
                 + 0.2 * dryup_score(b, bo + 1, j0 + 1) + 0.2 * b.close_pos[i] * 100)
            out.append(make(b, "breakout_retest", int(i), P, stop, q,
                            pivot_level=round(P, 2), bars_since_breakout=int(i - bo)))
            used.add((bo, ))
            break
    return out
