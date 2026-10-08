"""
e2_recognizers.py — an INDEPENDENT second implementation of each pattern, used only to count
how many setups of each kind exist ("expected") so the detectors can be measured against it.

Independence, honestly stated: this file imports nothing from e2_det_*, builds its own moving
averages/ATR from raw OHLCV slices (simple, not Wilder/EMA-recursive where the detectors use
those), finds swing points by brute-force window scans instead of the shared fractal helper,
and measures shapes on CLOSES where the detectors use highs/lows. It takes the same written
spec numbers (cup 15-35%, flag pole 30-100%, ...) but a different algorithm. It is still the
same author reading the same spec, so a shared misreading of the spec would pass both; what it
does catch is implementation error in either one. Where the spec is silent (e.g. breakout
volume for a base) it uses 1.5x, the value the spec states for the patterns that do state one.

recognize(b, i) -> bool: "does bar i complete this pattern?", using bars <= i only.
"""
from __future__ import annotations

import numpy as np

from e2_features import Bars

BV = 1.5            # breakout volume multiple used where the spec gives none


# -------------------------------------------------------------- small helpers (raw slices)
def _avgv(b: Bars, i: int, n: int = 50) -> float:
    s = b.v[max(0, i - n):i]
    return float(s.mean()) if len(s) else np.nan


def _relv(b: Bars, i: int) -> float:
    a = _avgv(b, i)
    return b.v[i] / a if a and a > 0 else 0.0


def _brk(b: Bars, i: int, vol: float = BV, look: int = 5) -> bool:
    return bool(i > look and b.c[i] > b.h[i - look:i].max() and _relv(b, i) >= vol)


def _sma(a: np.ndarray, i: int, n: int) -> float:
    return float(a[i - n + 1:i + 1].mean()) if i - n + 1 >= 0 else np.nan


def _ewm(a: np.ndarray, span: int) -> np.ndarray:
    k = 2.0 / (span + 1)
    out = np.empty(len(a)); out[0] = a[0]
    for j in range(1, len(a)):
        out[j] = out[j - 1] + k * (a[j] - out[j - 1])
    return out


def _atr_sma(b: Bars, i: int, n: int = 14) -> float:
    j = np.arange(i - n, i)
    if j[0] < 1:
        return np.nan
    tr = np.maximum.reduce([b.h[j] - b.l[j], np.abs(b.h[j] - b.c[j - 1]), np.abs(b.l[j] - b.c[j - 1])])
    return float(tr.mean())


def _swing_idx(a: np.ndarray, s: int, e: int, k: int, low: bool) -> list[int]:
    """brute force: j in [s, e-k] that is the strict extreme of a[j-k:j+k+1] (first occurrence)."""
    out = []
    for j in range(max(s, k), e - k + 1):
        w = a[j - k:j + k + 1]
        if (low and a[j] == w.min() and np.argmin(w) == k) or ((not low) and a[j] == w.max() and np.argmax(w) == k):
            out.append(j)
    return out


def _zig_closes(c: np.ndarray, s: int, e: int, thr: float):
    """alternating swing highs/lows of closes[s..e] with reversal >= thr; returns list of (idx, price, 'H'|'L')."""
    pts, hi_i, lo_i, direction = [], s, s, 0
    for j in range(s + 1, e + 1):
        if c[j] > c[hi_i]:
            hi_i = j
        if c[j] < c[lo_i]:
            lo_i = j
        if direction >= 0 and c[j] <= c[hi_i] * (1 - thr):
            pts.append((hi_i, c[hi_i], "H")); direction = -1; lo_i = j
        elif direction <= 0 and c[j] >= c[lo_i] * (1 + thr):
            pts.append((lo_i, c[lo_i], "L")); direction = 1; hi_i = j
    pts.append((hi_i if direction >= 0 else lo_i, c[hi_i] if direction >= 0 else c[lo_i], "H" if direction >= 0 else "L"))
    return pts


# -------------------------------------------------------------- momentum / continuation
def r_episodic_pivot(b: Bars, i: int) -> bool:
    if i < 51:
        return False
    gap = b.o[i] / b.c[i - 1] - 1
    rng = b.h[i] - b.l[i]
    return bool(gap >= 0.04 and _relv(b, i) >= 5 and rng > 0 and (b.c[i] - b.l[i]) / rng >= 0.75)


def r_momentum_burst(b: Bars, i: int) -> bool:
    if i < 20:
        return False
    atr = _atr_sma(b, i)
    rng = b.h[i] - b.l[i]
    if not (atr > 0 and rng > 1.5 * atr and rng > 0 and (b.c[i] - b.l[i]) / rng >= 0.75 and b.c[i] > b.c[i - 1]):
        return False
    if _relv(b, i) < 1.5:
        return False
    prior = range(i - 3, i)
    if any((b.h[j] - b.l[j]) > 1.0 * _atr_sma(b, j) for j in prior):
        return False
    return bool(b.c[i] > b.h[i - 3:i].max())


def r_pocket_pivot(b: Bars, i: int) -> bool:
    if i < 25 or not (b.c[i] > b.c[i - 1]):
        return False
    rng = b.h[i] - b.l[i]
    if rng <= 0 or (b.c[i] - b.l[i]) / rng < 0.5:
        return False
    dn = [b.v[j] for j in range(i - 10, i) if b.c[j] < b.c[j - 1]]
    ref = max(dn) if dn else float(b.v[i - 20:i].mean())
    return bool(b.v[i] > ref)


def r_pullback_ema(b: Bars, i: int) -> bool:
    if i < 60:
        return False
    e8, e20 = _ewm(b.c[:i + 1], 8), _ewm(b.c[:i + 1], 20)
    s50, s50_10 = _sma(b.c, i, 50), _sma(b.c, i - 10, 50)
    if not (b.c[i] > s50 > s50_10 and e20[i] > s50):
        return False
    if not (b.c[i] > b.o[i] and b.c[i] > b.c[i - 1] and (b.h[i] - b.l[i]) > 0 and (b.c[i] - b.l[i]) / (b.h[i] - b.l[i]) >= 0.5):
        return False
    w = b.h[i - 11:i + 1]
    top = i - 11 + int(np.argmax(w))
    if not (2 <= i - top <= 8):
        return False
    depth = (b.h[top] - b.l[top + 1:i + 1].min()) / b.h[top]
    if not (0.03 <= depth <= 0.18):
        return False
    hit = any(b.l[j] <= e8[j] * 1.004 or b.l[j] <= e20[j] * 1.004 for j in (i, i - 1, i - 2) if j > top) and \
        all(b.c[j] >= e20[j] * 0.985 for j in (i, i - 1, i - 2) if j > top)
    if not hit:
        return False
    pv = b.v[top + 1:i].mean() if i - top > 1 else b.v[i - 1]
    return bool(pv <= 0.9 * b.v[i - 20:i].mean())


def r_undercut_reclaim(b: Bars, i: int) -> bool:
    if i < 90 or _relv(b, i) < 1.5 or not (b.c[i] > b.l[i]):
        return False
    for j in _swing_idx(b.l, i - 80, i - 1, 5, True):
        if j + 5 >= i:
            continue
        lvl = b.l[j]
        if b.l[i] < lvl * 0.997 and b.c[i] > lvl and (j + 1 >= i or b.c[j + 1:i].min() >= lvl * 0.99):
            return True
    return False


def r_gap_and_go(b: Bars, i: int) -> bool:
    for g in range(i - 8, i - 2):
        if g < 2 or b.o[g] / b.c[g - 1] - 1 < 0.03:
            continue
        mid = slice(g + 1, i)
        if (b.l[mid] < b.c[g - 1]).any() or (b.c[mid] > b.h[g]).any() or (b.c[mid] < 0.97 * b.o[g]).any():
            continue
        if b.c[i] > b.h[g]:
            return True
    return False


def _br_at(b: Bars, i: int, bo: int) -> bool:
    P = b.h[bo - 40:bo].max()
    for j in range(bo + 2, i):
        if i - j <= 3 and P * 0.98 <= b.l[j] <= P * 1.02 and b.c[j] >= P * 0.99 and (b.c[bo + 1:i] >= P * 0.98).all():
            if b.c[i] > b.h[i - 1] and b.c[i] > b.c[i - 1] and b.v[i] >= b.v[i - 20:i].mean():
                return True
    return False


def r_breakout_retest(b: Bars, i: int) -> bool:
    """a breakout is retested once: the FIRST resumption after its retest is the setup, later bars are not new ones"""
    for bo in range(i - 18, i - 2):
        if bo < 45:
            continue
        P = b.h[bo - 40:bo].max()
        if not (b.c[bo] > P and _relv(b, bo) >= 1.3):
            continue
        if _br_at(b, i, bo) and not any(_br_at(b, i2, bo) for i2 in range(bo + 3, i)):
            return True
    return False


# -------------------------------------------------------------- bases
def r_flat_base(b: Bars, i: int) -> bool:
    if not _brk(b, i, 1.4):
        return False
    for L in range(25, 71):
        s = i - L
        if s < 130:
            break
        hi, lo = b.h[s:i].max(), b.l[s:i].min()
        cl = b.c[s:i]
        if i - (s + int(np.argmax(b.h[s:i]))) < 25:
            continue                                   # the base is the time SINCE its high, not the window length
        if b.c[i] <= hi or (hi - lo) / hi > 0.15 or (cl.max() - cl.min()) / cl.max() > 0.12:
            continue
        h52 = b.h[max(0, i - 252):i].max()
        if hi < 0.85 * h52:
            continue
        prior_low = b.l[max(0, s - 126):s].min()
        if hi / prior_low - 1 >= 0.30:
            return True
    return False


def _atr_wilder_pct(b: Bars, i: int) -> float:
    """14-bar Wilder ATR as a % of the close, as of bar i (own implementation of the spec's volatility yardstick)."""
    h, l, c = b.h[:i + 1], b.l[:i + 1], b.c[:i + 1]
    if len(c) < 16:
        return np.nan
    tr = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    a = tr[:14].mean()                      # seed with the simple mean, then Wilder smoothing
    for x in tr[14:]:
        a = (a * 13 + x) / 14
    return float(a / c[-1] * 100)


def _zig_hl(h: np.ndarray, l: np.ndarray, s: int, e: int, thr: float):
    """
    alternating swings of the high/low range from a peak at s; returns [(idx, price, 'H'|'L')].
    A bar that sets a new extreme cannot also confirm the reversal (the order of its high and low is unknown).
    """
    pts = [(s, h[s], "H")]
    top, bot, ti, bi, down = h[s], l[s], s, s, True
    for j in range(s + 1, e + 1):
        if down:
            if l[j] < bot:
                bot, bi = l[j], j
            elif h[j] >= bot * (1 + thr):
                pts.append((bi, bot, "L")); down, top, ti = False, h[j], j
        else:
            if h[j] > top:
                top, ti = h[j], j
            elif l[j] <= top * (1 - thr):
                pts.append((ti, top, "H")); down, bot, bi = True, l[j], j
    pts.append((bi, bot, "L") if down else (ti, top, "H"))
    return pts


def r_vcp(b: Bars, i: int) -> bool:
    """
    Spec: breakout bar i (close above the 5-bar high, relvol >= 1.5, close > sma50 > sma150) over a base whose pivot is
    ANY earlier high p (25-200 bars back, above every later high) within 15% of the 52w high; the swings between p and
    i (high/low range, a swing must reverse >= max(3.5%, 2 x ATR%)) give contraction depths that each shrink to
    <= 0.7 of the one before, with the first <= 40% and the last < 8%.
    """
    if not _brk(b, i, BV) or i < 160:
        return False
    if not (b.c[i] > _sma(b.c, i, 50) > _sma(b.c, i, 150)):
        return False
    h52 = b.h[max(0, i - 252):i].max()
    thr = max(0.035, 2.0 * _atr_wilder_pct(b, i) / 100.0)
    for p in range(max(0, i - 200), i - 24):
        if b.h[p] <= b.h[p + 1:i].max() or b.c[i] <= b.h[p] or b.h[p] < 0.85 * h52:
            continue
        z = _zig_hl(b.h, b.l, p, i - 1, thr)
        d = [(z[k][1] - z[k + 1][1]) / z[k][1] * 100 for k in range(len(z) - 1) if z[k][2] == "H" and z[k + 1][2] == "L"]
        if len(d) >= 2 and d[0] <= 40 and all(y <= 0.7 * x for x, y in zip(d, d[1:])) and d[-1] < 8:
            return True
    return False


def r_base_on_base(b: Bars, i: int) -> bool:
    """second base on top of a first (brute force over both pivots)"""
    if not _brk(b, i, 1.4) or i < 200:
        return False
    for p in range(i - 80, i - 14):
        if p < 70 or b.h[p] < b.h[p:i].max() or b.c[i] <= b.h[p]:
            continue
        lo2 = b.l[p:i].min()
        if (b.h[p] - lo2) / b.h[p] > 0.20:
            continue
        for q in range(p - 60, p - 5):
            if q < 30:
                continue
            H1 = b.h[q]
            if not (H1 < b.h[p] <= 1.25 * H1 and lo2 >= 0.97 * H1):
                continue
            if H1 < b.h[q - 30:q].max():
                continue
            broke = [k for k in range(q + 1, p + 1) if b.c[k] > H1]
            if not broke or (broke[0] > q + 1 and b.h[q + 1:broke[0]].max() > H1):
                continue
            if (H1 - b.l[q - 30:q + 1].min()) / H1 <= 0.30 and b.c[q - 30] >= 0.85 * H1:
                return True
    return False


def r_double_bottom(b: Bars, i: int) -> bool:
    if not _brk(b, i, 1.4):
        return False
    lows = _swing_idx(b.l, i - 140, i - 1, 4, True)
    for y in range(len(lows) - 1, 0, -1):
        if i - lows[y] > 40:
            break
        for x in range(y - 1, -1, -1):
            a, c2 = lows[x], lows[y]
            if not (15 <= c2 - a <= 60) or abs(b.l[c2] - b.l[a]) / b.l[a] > 0.03:
                continue
            mid = b.h[a:c2 + 1].max()
            if mid >= max(b.l[a], b.l[c2]) * 1.10 and b.c[i] > mid and not (b.c[c2 + 1:i] > mid).any():
                return True
    return False


def _saucer_ok(b: Bars, p: int, end: int, depth_lo: float, depth_hi: float, rim_tol: float = 0.9) -> tuple[bool, float]:
    """is [p, end) a saucer whose left rim is bar p?  (brute force over rim positions; closes for curvature)"""
    if p < 10 or b.h[p] < b.h[p - 10:p].max():
        return False, 0.0
    bot = p + int(np.argmin(b.l[p:end]))
    if bot - p < 5 or b.h[p] < b.h[p:bot + 1].max() or end - bot < 8:
        return False, 0.0
    rim = b.h[p:end].max()
    depth = (b.h[p] - b.l[bot]) / b.h[p]
    if not (depth_lo <= depth <= depth_hi) or b.h[p] < rim_tol * rim:
        return False, 0.0
    c = b.c[p:end]
    if (c <= c.min() + 0.25 * (c.max() - c.min())).mean() < 0.35:
        return False, 0.0
    x = np.arange(len(c), dtype=float)
    q = np.polyfit(x, c, 2)
    fit = np.polyval(q, x)
    r2 = 1 - ((c - fit) ** 2).sum() / ((c - c.mean()) ** 2).sum()
    return bool(q[0] > 0 and r2 >= 0.4), float(r2)


def r_rounding_bottom(b: Bars, i: int) -> bool:
    if not _brk(b, i, 1.4):
        return False
    for p in range(i - 150, i - 39):
        if p < 10:
            continue
        rim = b.h[p:i].max()
        if b.c[i] <= rim:
            continue
        ok, r2 = _saucer_ok(b, p, i, 0.15, 0.40)
        if not ok or r2 < 0.6:
            continue
        c = b.c[p:i]
        vertex = -np.polyfit(np.arange(len(c), dtype=float), c, 2)[1] / (2 * np.polyfit(np.arange(len(c), dtype=float), c, 2)[0]) / len(c)
        if 0.25 <= vertex <= 0.75:
            return True
    return False


def r_inverse_hs(b: Bars, i: int) -> bool:
    if not _brk(b, i, 1.4):
        return False
    lows = _swing_idx(b.l, i - 150, i - 1, 4, True)
    for r in reversed(lows):
        if i - r > 30:
            break
        for h in [x for x in lows if x < r]:
            for l in [x for x in lows if x < h and 30 <= r - x <= 150]:
                if b.l[l:r + 1].min() < b.l[h]:
                    continue
                sh = min(b.l[l], b.l[r])
                if not (b.l[h] < sh * 0.97 and abs(b.l[l] - b.l[r]) / sh <= 0.15 and 0.5 <= (r - h) / max(h - l, 1) <= 2.0):
                    continue
                p1i, p2i = l + int(np.argmax(b.h[l:h + 1])), h + int(np.argmax(b.h[h:r + 1]))
                if p1i == p2i:
                    continue
                slope = (b.h[p2i] - b.h[p1i]) / (p2i - p1i)
                neck = b.h[p2i] + slope * (i - p2i)
                if b.c[i] > neck and b.c[i] > min(b.h[p1i], b.h[p2i]):
                    return True
    return False


def r_high_tight_flag(b: Bars, i: int) -> bool:
    if not _brk(b, i, BV):
        return False
    for F in range(15, 26):
        t = i - F
        if t < 45:
            continue
        top = b.h[t:i].max()                       # the flag starts at (and includes) the pole's top
        t = t + int(np.argmax(b.h[t:i]))
        fl_lo = b.l[t + 1:i].min() if t + 1 < i else b.l[t]
        fh = b.h[t + 1:i].max() if t + 1 < i else top
        if not (15 <= i - t <= 25) or b.c[i] <= fh or (top - fl_lo) / top > 0.25:
            continue
        lo = b.l[t - 40:t - 19].min()
        if top / lo - 1 >= 1.0:
            return True
    return False


def r_cup_handle(b: Bars, i: int) -> bool:
    if not _brk(b, i, BV):
        return False
    for hl in range(5, 26):
        hs = i - hl
        hh = b.h[hs:i].max()
        if b.c[i] <= hh or b.h[hs] < hh:
            continue
        hl_low = b.l[hs:i].min()
        if not (0.05 <= (hh - hl_low) / hh <= 0.15):
            continue
        for p in range(hs - 260, hs - 34):
            if p < 10:
                continue
            ok, _ = _saucer_ok(b, p, hs, 0.15, 0.35, rim_tol=0.0)
            if not ok:
                continue
            bot = p + int(np.argmin(b.l[p:hs]))
            lr, cl = b.h[p], b.l[bot]
            if hh < 0.9 * lr or hl_low < cl + 0.5 * (lr - cl) or b.h[bot:hs + 1].max() > hh:
                continue
            if b.v[hs:i].mean() <= 0.85 * b.v[p:hs].mean():
                return True
    return False


def r_ascending_base(b: Bars, i: int) -> bool:
    if not _brk(b, i, 1.4):
        return False
    lows = _swing_idx(b.l, i - 105, i - 1, 4, True)
    for t3 in reversed(lows):
        if i - t3 > 30:
            continue
        for t2 in [x for x in lows if 15 <= t3 - x <= 25]:
            for t1 in [x for x in lows if 15 <= t2 - x <= 25]:
                l1, l2, l3 = b.l[t1], b.l[t2], b.l[t3]
                if not (l1 < 0.995 * l2 and l2 < 0.995 * l3):
                    continue
                if b.l[t1:t2 + 1].min() < l1 or b.l[t2:t3 + 1].min() < l2 or b.l[t3:i].min() < l3:
                    continue
                deps = []
                for a0, t in ((t1 - 25, t1), (t1, t2), (t2, t3)):
                    pk = b.h[max(a0, 0):t + 1].max()
                    deps.append((pk - b.l[t]) / pk)
                if not all(0.03 <= d < 0.15 for d in deps):
                    continue
                if b.c[i] <= b.h[max(t1 - 10, 0):i].max():
                    continue
                v = b.v[max(t1 - 25, 0):t3 + 1]
                if v[len(v) // 2:].mean() < v[:len(v) // 2].mean():
                    return True
    return False


def r_flag_pennant(b: Bars, i: int) -> bool:
    if not _brk(b, i, BV):
        return False
    for F in range(10, 31):
        t0 = i - F
        if t0 < 42:
            continue
        t = t0 + int(np.argmax(b.h[t0:i]))
        if not (10 <= i - t <= 30):
            continue
        top = b.h[t]
        fh = b.h[t + 1:i].max() if t + 1 < i else top
        fl = b.l[t + 1:i].min() if t + 1 < i else b.l[t]
        if b.c[i] <= fh:
            continue
        w = b.l[t - 40:t - 19]
        pl = w.min()
        q0 = t - 40 + int(np.argmin(w))
        gain = top / pl - 1
        if not (0.30 <= gain < 1.0) or (top - fl) >= 0.5 * (top - pl):
            continue
        if b.v[t + 1:i].mean() < b.v[q0:t + 1].mean():
            return True
    return False


def _tri_points(b: Bars, i: int, W: int):
    s = i - W
    hs = [(j, b.h[j]) for j in _swing_idx(b.h, s, i - 1, 2, False)]
    ls = [(j, b.l[j]) for j in _swing_idx(b.l, s, i - 1, 2, True)]
    return hs, ls


def _trim_fit(pts, side: str, tol: float):
    """regression line through swing points, trimmed of points on the 'safe' side until it settles;
    valid when >= 2 points touch it and none breaks through by more than 2 x tol"""
    keep = list(pts)
    for _ in range(6):
        if len(keep) < 2:
            return None
        x = np.array([p[0] for p in keep], float); y = np.array([p[1] for p in keep], float)
        k, m = np.polyfit(x, y, 1)
        dev = [(p[1] - (k * p[0] + m)) / (k * p[0] + m) for p in keep]
        far = [j for j, d in enumerate(dev) if (d > tol if side == "support" else d < -tol)]
        if not far:
            break
        keep = [p for j, p in enumerate(keep) if j not in far]
    else:
        return None
    if len(keep) < 2:
        return None
    for p in pts:
        d = (p[1] - (k * p[0] + m)) / (k * p[0] + m)
        if (side == "support" and d < -2 * tol) or (side == "resistance" and d > 2 * tol):
            return None
    first = min(p[0] for p in keep)
    return k, m, len(keep), first


def _vol_declines(b: Bars, x0: int, i: int) -> bool:
    h = x0 + (i - x0) // 2
    return bool(b.v[h:i].mean() < b.v[x0:h].mean())


def r_asc_triangle(b: Bars, i: int) -> bool:
    if not _brk(b, i, BV):
        return False
    for W in range(20, 121, 5):
        hs, ls = _tri_points(b, i, W)
        if len(hs) < 2 or len(ls) < 2:
            continue
        R = max(p for _, p in hs)
        top = [p for p in hs if p[1] >= R * 0.985]
        if len(top) < 2 or top[-1][0] - top[0][0] < 8 or b.c[i] <= R:
            continue
        ln = _trim_fit(ls, "support", 0.015)
        if ln is None or ln[0] <= 0:
            continue
        k, m, n, first = ln
        x0 = min(top[0][0], first)
        apex = (R - m) / k
        if i - x0 >= 15 and 0 <= apex - i <= 20 and _vol_declines(b, x0, i):
            return True
    return False


def r_sym_triangle(b: Bars, i: int) -> bool:
    if not _brk(b, i, 1.4):
        return False
    for W in range(20, 121, 5):
        hs, ls = _tri_points(b, i, W)
        if len(hs) < 2 or len(ls) < 2:
            continue
        up, dn = _trim_fit(hs, "resistance", 0.02), _trim_fit(ls, "support", 0.02)
        if up is None or dn is None or up[0] >= 0 or dn[0] <= 0:
            continue
        ku, mu, _, fu = up; kl, ml, _, fl = dn
        if b.c[i] <= ku * i + mu:
            continue
        x0 = min(fu, fl)
        apex = (ml - mu) / (ku - kl)
        if apex > i and i - x0 >= 15 and 0.4 <= (i - x0) / (apex - x0) <= 1.0 and _vol_declines(b, x0, i):
            return True
    return False


def r_desc_triangle(b: Bars, i: int) -> bool:
    if not _brk(b, i, BV):
        return False
    for W in range(20, 121, 5):
        hs, ls = _tri_points(b, i, W)
        if len(hs) < 2 or len(ls) < 2:
            continue
        S = min(p for _, p in ls)
        flat = [p for p in ls if p[1] <= S * 1.015]
        if len(flat) < 2 or flat[-1][0] - flat[0][0] < 8:
            continue
        up = _trim_fit(hs, "resistance", 0.0225)
        if up is None or up[0] >= 0 or b.c[i] <= up[0] * i + up[1]:
            continue
        x0 = min(up[3], flat[0][0])
        apex = (S - up[1]) / up[0]
        if 0 <= apex - i <= 20 and i - x0 >= 15 and _vol_declines(b, x0, i):
            return True
    return False


RECOGNIZERS = {
    "episodic_pivot": r_episodic_pivot, "momentum_burst": r_momentum_burst, "pocket_pivot": r_pocket_pivot,
    "pullback_ema": r_pullback_ema, "undercut_reclaim": r_undercut_reclaim, "gap_and_go": r_gap_and_go,
    "breakout_retest": r_breakout_retest, "flat_base": r_flat_base, "vcp": r_vcp, "base_on_base": r_base_on_base,
    "double_bottom": r_double_bottom, "rounding_bottom": r_rounding_bottom, "inverse_hs": r_inverse_hs,
    "high_tight_flag": r_high_tight_flag, "cup_handle": r_cup_handle, "ascending_base": r_ascending_base,
    "flag_pennant": r_flag_pennant, "asc_triangle": r_asc_triangle, "sym_triangle": r_sym_triangle,
    "desc_triangle": r_desc_triangle,
}

# cheap necessary conditions, so a recognizer is only evaluated where it could possibly be true
_MOMENTUM_ONLY = {"episodic_pivot", "momentum_burst", "pocket_pivot", "pullback_ema", "undercut_reclaim",
                  "gap_and_go", "breakout_retest"}


def expected_bars(b: Bars, name: str, lo: int, hi: int) -> list[int]:
    """bars in [lo, hi) on which the independent recognizer says the pattern completes (3-bar cooldown, first kept)."""
    fn = RECOGNIZERS[name]
    out: list[int] = []
    last = -10
    for i in range(max(lo, 130), hi):
        if name not in _MOMENTUM_ONLY:
            # every base pattern breaks above the previous 5 bars' high on elevated volume
            if not (b.c[i] > b.h[i - 5:i].max() and _relv(b, i) >= 1.4):
                continue
        try:
            ok = fn(b, i)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
            ok = False
        if ok and i - last > 3:
            out.append(i)
            last = i
    return out
