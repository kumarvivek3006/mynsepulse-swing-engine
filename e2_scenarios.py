"""
e2_scenarios.py — planted-setup charts with a known answer.

Each Scenario is a synthetic OHLCV series with ONE bar `at` on which the named detector
either must fire (fire=True) or must NOT fire (fire=False: the same shape with exactly one
defining property pushed outside its definition — a near miss).

Used by test_engine2.py, and by the server-side validation run as a self-check: if a
detector stops finding its own textbook shape in production, that is a visible failure.
These prove a detector implements its written definition. They say nothing about
whether the definition makes money; that is what the 3-year run on real data measures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from e2_synth import Chart


@dataclass
class Scenario:
    detector: str
    label: str
    df: pd.DataFrame
    at: int
    fire: bool


SHIFT = 0          # build_all(shift=k) re-rolls the noise of every scenario: same shapes, different bars


def _adv(seed: int, n: int = 250, d: float = 0.0025, sigma: float = 0.01) -> Chart:
    return Chart(seed + SHIFT).drift(n, d, sigma)


def _parab(c: Chart, L: float, depth: float, N: int, end: float = 1.0, vlo: float = 0.7, sig: float = 0.004):
    for t in range(1, N + 1):
        u = (t - N / 2) / (N / 2)
        tilt = 1 - (1 - end) * (t / N)
        p = L * tilt * (1 - depth * (1 - u * u)) * (1 + sig * c.rng.standard_normal())
        c.bar(p, vmult=vlo + (1 - vlo) * u * u)


def _vshape(c: Chart, L: float, depth: float, N: int):
    for t in range(1, N + 1):
        u = abs((t - N / 2) / (N / 2))
        c.bar(L * (1 - depth * (1 - u)) * (1 + 0.004 * c.rng.standard_normal()), vmult=0.8)


def _sc(det, label, c: Chart, fire: bool, at: int | None = None) -> Scenario:
    return Scenario(det, label, c.frame(), c.last if at is None else at, fire)


# ---------------------------------------------------------------- momentum family
def _ep(seed, gap, vm, high_close=True):
    c = _adv(seed, 260, 0.0008, 0.012); p = c.closes[-1]
    if high_close:
        c.bar(p * (1 + gap + 0.025), vmult=vm, open_=p * (1 + gap), high=p * (1 + gap + 0.028), low=p * (1 + gap - 0.005))
    else:
        c.bar(p * (1 + gap - 0.02), vmult=vm, open_=p * (1 + gap), high=p * (1 + gap + 0.025), low=p * (1 + gap - 0.025))
    return c


def _mb(seed, up):
    c = _adv(seed, 260, 0.0008, 0.012)
    for _ in range(4):
        p = c.closes[-1]; c.bar(p * 1.001, vmult=0.7, high=p * 1.004, low=p * 0.997)
    p = c.closes[-1]
    c.bar(p * (1 + up), vmult=2.2, open_=p * 1.002, high=p * (1 + up + 0.002), low=p * 0.999)
    return c


def _pp(seed, vm):
    c = _adv(seed, 260, 0.0008, 0.012)
    for _ in range(6):
        p = c.closes[-1]; c.bar(p * 0.99, vmult=1.2)
    for _ in range(3):
        p = c.closes[-1]; c.bar(p * 1.004, vmult=0.9)
    p = c.closes[-1]; c.bar(p * 1.025, vmult=vm, high=p * 1.026)
    return c


def _ur(seed, reclaim):
    c = _adv(seed, 200, 0.0008, 0.010)
    c.path([(15, c.closes[-1] * 1.12), (10, c.closes[-1] * 1.0), (15, c.closes[-1] * 1.1)], sigma=0.003)
    lvl = min(c.closes[-30:])
    c.path([(8, c.closes[-1] * 1.05), (8, lvl * 1.01)], sigma=0.003)
    close = lvl * 1.012 if reclaim else lvl * 0.985
    c.bar(close, vmult=2.2, open_=lvl * 0.995, low=lvl * 0.965, high=max(close, lvl) * 1.02)
    return c


def _gg(seed, gap, hold_ok):
    c = _adv(seed, 260, 0.0011, 0.011); p = c.closes[-1]
    c.bar(p * (1 + gap + 0.005), vmult=3, open_=p * (1 + gap), high=p * (1 + gap + 0.01), low=p * (1 + gap - 0.005))
    top = p * (1 + gap + 0.008)
    c.bar(top, vmult=0.8, low=top * 0.995)
    c.bar(top * 1.001, vmult=0.8, low=(top * 0.995 if hold_ok else p * 0.99))
    c.bar(top * 1.0, vmult=0.8, low=top * 0.995)
    c.bar(p * (1 + gap + 0.025), vmult=1.6)
    return c


def _br(seed, retest_depth):
    c = Chart(seed + SHIFT).drift(220, 0.0006, 0.009)
    c.path([(30, c.closes[-1])], sigma=0.003)
    P = max(c.closes[-40:])
    c.bar(P * 1.03, vmult=2.5, high=P * 1.035)
    c.path([(3, P * 1.05), (4, P * 1.004)], sigma=0.002, vmult=0.6)
    c.bar(P * (1 - retest_depth + 0.003), vmult=0.6, low=P * (1 - retest_depth))
    c.bar(P * 1.022, vmult=1.4, high=P * 1.025)
    return c


def _pb(seed, uptrend):
    if uptrend:
        c = Chart(seed + SHIFT).drift(220, 0.0008, 0.008)
        c.path([(25, c.closes[-1] * 1.25)], sigma=0.004)
    else:                                   # same pullback-and-reversal candle, but inside a persistent downtrend
        c = Chart(seed + SHIFT).drift(245, -0.002, 0.008)
        c.path([(6, c.closes[-1] * 1.06)], sigma=0.004)
    top = c.closes[-1]
    c.path([(5, top * 0.93)], sigma=0.004, vmult=0.6)
    c.bar(top * 0.945, vmult=0.6)
    c.bar(top * 0.975, vmult=1.0)
    return c


# ---------------------------------------------------------------- base family
def _flat(seed, depth):
    c = _adv(seed); c.path([(35, c.closes[-1] * 1.35)], sigma=0.006); P = c.closes[-1]
    c.path([(8, P * (1 - depth)), (8, P * 0.995), (7, P * (1 - depth * 1.2)), (7, P * 0.99)], sigma=0.004, vmult=0.6)
    c.bar(P * 1.03, vmult=2.2, high=P * 1.032)
    return c


def _vcp(seed, ratios_ok):
    c = _adv(seed); c.path([(35, c.closes[-1] * 1.3)], sigma=0.006); P = c.closes[-1]
    if ratios_ok:
        pts = [(7, .82 * P), (9, .97 * P), (7, .87 * P), (8, .985 * P), (5, .94 * P), (5, .99 * P)]
    else:
        pts = [(7, .90 * P), (9, .97 * P), (7, .80 * P), (8, .985 * P), (5, .90 * P), (5, .99 * P)]   # contractions widen
    c.path(pts, sigma=0.003, vmult=0.6)
    c.bar(P * 1.025, vmult=2.2)
    return c


def _cup(seed, depth, shape="u", hd=0.92):
    c = _adv(seed); c.path([(30, c.closes[-1] * 1.3)], sigma=0.006); L = c.closes[-1]
    if shape == "u":
        _parab(c, L, depth, 75, end=0.97)
    else:
        _vshape(c, L, depth, 75)
    r = c.closes[-1]
    c.path([(5, r * 1.005), (10, r * hd)], sigma=0.003, vmult=0.55)
    c.bar(r * 1.03, vmult=2.2, high=r * 1.032)
    return c


def _asc(seed, rising):
    c = _adv(seed); c.path([(25, c.closes[-1] * 1.3)], sigma=0.006); A = c.closes[-1]
    lows = (.90, .93, .97) if rising else (.90, .88, .86)
    pts = [(10, lows[0] * A), (10, 1.02 * A), (10, lows[1] * A), (10, 1.05 * A), (10, lows[2] * A), (10, 1.07 * A)]
    c.path(pts, sigma=0.003, vmult=[1.0, 1.0, 0.8, 0.8, 0.55, 0.55])
    c.bar(A * 1.095, vmult=2.0, high=A * 1.097)
    return c


def _db(seed, asym):
    c = _adv(seed); c.path([(20, c.closes[-1] * 1.2)], sigma=0.006); P = c.closes[-1]
    second = .805 if not asym else .72
    c.path([(15, .80 * P), (12, .92 * P), (12, second * P), (10, .91 * P)], sigma=0.003, vmult=[1.2, 0.9, 0.8, 0.8])
    c.bar(P * 0.945, vmult=2.0)
    return c


def _rb(seed, shape):
    c = _adv(seed); c.path([(15, c.closes[-1] * 1.2)], sigma=0.006); L = c.closes[-1]
    (_parab(c, L, 0.25, 90) if shape == "u" else _vshape(c, L, 0.25, 90))
    c.bar(L * 1.03, vmult=2.0, high=L * 1.032)
    return c


def _ihs(seed, head_low):
    c = _adv(seed); c.path([(15, c.closes[-1] * 1.15)], sigma=0.006); P = c.closes[-1]
    head = .78 if head_low else .875                       # a head that is not clearly lower than the shoulders
    c.path([(12, .88 * P), (8, .96 * P), (14, head * P), (14, .97 * P), (10, .88 * P), (8, .96 * P)],
           sigma=0.003, vmult=[1.0, 0.9, 1.1, 0.8, 0.7, 0.7])
    c.bar(P * 1.0, vmult=2.0, high=P * 1.005)
    return c


def _htf(seed, gain, retrace):
    c = _adv(seed, 200); B0 = c.closes[-1]
    c.path([(30, B0 * gain)], sigma=0.008, vmult=1.6); T = c.closes[-1]
    c.path([(6, T * (1 - retrace * 0.8)), (12, T * (1 - retrace))], sigma=0.004, vmult=0.5)
    c.bar(T * 1.03, vmult=2.0, high=T * 1.04)
    return c


def _flag(seed, gain, depth):
    c = _adv(seed, 200); B0 = c.closes[-1]
    c.path([(25, B0 * gain)], sigma=0.008, vmult=1.8); T = c.closes[-1]
    c.path([(5, T * (1 - depth * 0.7)), (10, T * (1 - depth))], sigma=0.004, vmult=0.5)
    c.bar(T * 1.03, vmult=2.0, high=T * 1.04)
    return c


def _asc_tri(seed, flat_top):
    c = _adv(seed, 220); B0 = c.closes[-1]; R = B0 * 1.1
    tops = [R, R * .998, R, R * .995] if flat_top else [R, R * 0.97, R * 0.94, R * 0.91]
    c.path([(10, tops[0]), (12, R * .91), (12, tops[1]), (10, R * .945), (10, tops[2]), (8, R * .972), (6, tops[3])],
           sigma=0.003, vmult=[1.2, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5])
    c.bar(R * 1.025, vmult=2.0, high=R * 1.027)
    return c


def _sym_tri(seed):
    c = _adv(seed, 220); B0 = c.closes[-1]; U = B0 * 1.1
    c.path([(10, U), (10, U * .82), (10, U * .96), (8, U * .86), (8, U * .93), (6, U * .89), (5, U * .915)],
           sigma=0.003, vmult=[1.2, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5])
    c.bar(U * 0.965, vmult=2.0)
    return c


def _desc_tri(seed):
    c = _adv(seed, 220); B0 = c.closes[-1]; H = B0 * 1.1
    c.path([(8, H), (10, H * .82), (10, H * .95), (8, H * .83), (8, H * .90), (6, H * .82), (5, H * .87)],
           sigma=0.003, vmult=[1.2, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5])
    c.bar(H * 0.92, vmult=2.0)
    return c


def _bob(seed):
    c = _adv(seed, 200); B0 = c.closes[-1]
    c.path([(25, B0 * 1.3)], sigma=0.005); P1 = c.closes[-1]
    c.path([(10, P1 * .92), (10, P1 * .97), (10, P1 * 1.0)], sigma=0.003, vmult=0.7)     # base 1: 30 bars, ~8% deep
    c.path([(6, P1 * 1.07)], sigma=0.003, vmult=1.4)                                      # breakout and a short advance
    c.path([(8, P1 * 1.015), (8, P1 * 1.06), (6, P1 * 1.04)], sigma=0.003, vmult=0.6)     # base 2 sits on base 1
    c.bar(P1 * 1.095, vmult=2.0)
    return c


def build_all(shift: int = 0) -> list[Scenario]:
    global SHIFT
    SHIFT = shift
    S: list[Scenario] = []
    # --- momentum / continuation -------------------------------------
    S += [_sc("episodic_pivot", "gap 7.5% on 9x volume, strong close", _ep(3, 0.075, 9), True),
          _sc("episodic_pivot", "NEAR MISS gap 3% (< 4%)", _ep(3, 0.03, 9), False),
          _sc("episodic_pivot", "NEAR MISS volume 3x (< 5x)", _ep(3, 0.075, 3), False),
          _sc("episodic_pivot", "NEAR MISS weak close", _ep(3, 0.075, 9, high_close=False), False)]
    S += [_sc("momentum_burst", "4.5% burst out of a quiet spell, 2.2x", _mb(4, 0.045), True),
          _sc("momentum_burst", "NEAR MISS range only ~1.2 ATR", _mb(4, 0.012), False)]
    S += [_sc("pocket_pivot", "up day above the largest down-day volume", _pp(5, 2.0), True),
          _sc("pocket_pivot", "NEAR MISS volume below the largest down day", _pp(5, 1.0), False)]
    S += [_sc("undercut_reclaim", "wick 3% under swing low, closes back above", _ur(6, True), True),
          _sc("undercut_reclaim", "NEAR MISS closes below the level", _ur(6, False), False)]
    S += [_sc("gap_and_go", "4% gap, holds 3 bars, clears day-1 high", _gg(7, 0.04, True), True),
          _sc("gap_and_go", "NEAR MISS gap fills during the hold", _gg(7, 0.04, False), False),
          _sc("gap_and_go", "NEAR MISS gap only 2%", _gg(7, 0.02, True), False)]
    S += [_sc("breakout_retest", "pivot retest holds, resumes on volume", _br(8, 0.01), True),
          _sc("breakout_retest", "NEAR MISS retest falls 6% through the pivot", _br(8, 0.06), False)]
    S += [_sc("pullback_ema", "uptrend pullback to 8/20 EMA, reversal candle", _pb(9, True), True),
          _sc("pullback_ema", "NEAR MISS same shape in a downtrend", _pb(9, False), False)]
    # --- bases -------------------------------------------------------
    S += [_sc("flat_base", "6% deep, 28 bars, after a 35% run", _flat(1, 0.04), True),
          _sc("flat_base", "NEAR MISS 25% deep", _flat(1, 0.22), False)]
    S += [_sc("vcp", "contractions 18.5 / 10.9 / 5.2", _vcp(2, True), True),
          _sc("vcp", "NEAR MISS contractions that widen", _vcp(2, False), False)]
    S += [_sc("cup_handle", "U cup 25%, 14-bar handle, 2.2x breakout", _cup(3, 0.25), True, None),
          _sc("cup_handle", "NEAR MISS V-shaped cup", _cup(3, 0.25, "v"), False),
          _sc("cup_handle", "NEAR MISS cup only 10% deep", _cup(3, 0.10), False),
          _sc("cup_handle", "NEAR MISS cup 45% deep", _cup(3, 0.45), False),
          _sc("cup_handle", "NEAR MISS handle 20% deep", _cup(3, 0.25, hd=0.80), False)]
    S += [_sc("ascending_base", "three higher lows ~20 bars apart", _asc(4, True), True),
          _sc("ascending_base", "NEAR MISS lows keep falling", _asc(4, False), False)]
    S += [_sc("double_bottom", "two lows within 1%, 24 bars apart", _db(5, False), True),
          _sc("double_bottom", "NEAR MISS second low 10% lower", _db(5, True), False)]
    S += [_sc("rounding_bottom", "90-bar saucer, 25% deep", _rb(6, "u"), True),
          _sc("rounding_bottom", "NEAR MISS V instead of saucer", _rb(6, "v"), False)]
    S += [_sc("inverse_hs", "head 11% below shoulders", _ihs(7, True), True),
          _sc("inverse_hs", "NEAR MISS head barely below shoulders", _ihs(7, False), False)]
    S += [_sc("high_tight_flag", "+130% pole, 18-bar flag, 10% retrace", _htf(8, 2.3, 0.10), True),
          _sc("high_tight_flag", "NEAR MISS 40% retrace", _htf(8, 2.3, 0.40), False),
          _sc("high_tight_flag", "NEAR MISS pole only +50%", _htf(8, 1.5, 0.10), False)]
    S += [_sc("flag_pennant", "+60% pole, 15-bar flag, 9% down", _flag(9, 1.6, 0.09), True),
          _sc("flag_pennant", "NEAR MISS pole only +15%", _flag(9, 1.15, 0.05), False),
          _sc("flag_pennant", "NEAR MISS flag gives back > 50% of the pole", _flag(9, 1.5, 0.30), False)]
    S += [_sc("asc_triangle", "flat top 3 touches, rising lows, apex near", _asc_tri(10, True), True),
          _sc("asc_triangle", "NEAR MISS falling tops (no flat top)", _asc_tri(10, False), False)]
    S += [_sc("sym_triangle", "converging highs and lows", _sym_tri(11), True)]
    S += [_sc("desc_triangle", "flat support, falling highs, UPSIDE break", _desc_tri(12), True)]
    S += [_sc("base_on_base", "second base sitting on the first", _bob(13), True)]
    return S
