"""
e2_regime.py — one continuous 0-100 market-regime score. It MODULATES; it never blocks.

Components (weights fixed up front, not tuned):
  index trend          30   NIFTY500 vs its 50/200-day averages and their slopes
  breadth              25   % of the universe above its 50-day and 200-day average
  volatility (VIX)     15   inverse of INDIAVIX's percentile within its own last 252 sessions
  sector participation 15   % of sectors whose composite is above its own 50-day average
  advance/decline      15   10-day advancers/decliners ratio and the A/D line vs its 50-day mean
Sector DISPERSION (cross-sectional spread of 3-month sector returns) is recorded next to the
score but not scored: high dispersion is ambiguous (rotation can be healthy or defensive).

Bands (spec): risk_off 0-25 -> size 0.5R, max 4 positions; neutral 25-60 -> 0.75R, max 6;
risk_on 60-100 -> 1.0R, max 8. Setup preference enters only through `regime_fit` (0-5 points
in the composite score): risk_off prefers reversal/continuation (pullback) setups, risk_on
prefers breakout/momentum, neutral treats every setup equally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

W = {"index": 30.0, "breadth": 25.0, "vix": 15.0, "sector": 15.0, "ad": 15.0}
BANDS = [(25.0, "risk_off", 0.50, 4), (60.0, "neutral", 0.75, 6), (101.0, "risk_on", 1.00, 8)]

REGIME_FIT = {
    "risk_on":  {"breakout": 5.0, "momentum": 5.0, "continuation": 3.0, "reversal": 1.0},
    "neutral":  {"breakout": 3.0, "momentum": 3.0, "continuation": 3.0, "reversal": 3.0},
    "risk_off": {"breakout": 1.0, "momentum": 1.0, "continuation": 5.0, "reversal": 5.0},
}


def label(score: float) -> tuple[str, float, int]:
    if score is None or np.isnan(score):
        return "neutral", 0.75, 6
    for hi, name, mult, mx in BANDS:
        if score < hi:
            return name, mult, mx
    return "risk_on", 1.0, 8


def _ramp(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0) * 100.0


def regime_frame(index_close: pd.Series, vix: pd.Series | None, close: pd.DataFrame,
                 sector_comp: pd.DataFrame) -> pd.DataFrame:
    """
    All inputs are indexed by trade date; every value on date d uses data <= d.
    close: dates x symbols (adjusted closes of the universe).
    """
    d = close.index
    ix = index_close.reindex(d).ffill()
    s50, s200 = ix.rolling(50).mean(), ix.rolling(200).mean()
    comp_idx = (40.0 * (ix > s200) + 30.0 * (ix > s50) + 15.0 * (s50 > s200)
                + 15.0 * (s50 > s50.shift(10))).where(s200.notna())

    ma50, ma200 = close.rolling(50).mean(), close.rolling(200).mean()
    ok50, ok200 = ma50.notna(), ma200.notna()
    b50 = ((close > ma50) & ok50).sum(axis=1) / ok50.sum(axis=1).replace(0, np.nan) * 100
    b200 = ((close > ma200) & ok200).sum(axis=1) / ok200.sum(axis=1).replace(0, np.nan) * 100
    comp_breadth = (0.6 * _ramp(b50, 20, 80) + 0.4 * _ramp(b200.fillna(b50), 20, 80))

    if vix is not None and len(vix.dropna()):
        v = vix.reindex(d).ffill()
        comp_vix = 100.0 - v.rolling(252, min_periods=60).rank(pct=True) * 100.0
        vix_known = True
    else:
        comp_vix, vix_known = pd.Series(50.0, index=d), False

    if sector_comp is not None and sector_comp.shape[1] >= 3:
        sc = sector_comp.reindex(d)
        above = (sc > sc.rolling(50).mean())
        have = sc.rolling(50).mean().notna().sum(axis=1).replace(0, np.nan)
        comp_sector = above.sum(axis=1) / have * 100.0
        r3 = sc / sc.shift(63) - 1.0
        dispersion = r3.std(axis=1)
    else:
        comp_sector, dispersion = pd.Series(50.0, index=d), pd.Series(np.nan, index=d)

    chg = close.diff()
    adv, dec = (chg > 0).sum(axis=1), (chg < 0).sum(axis=1)
    ratio10 = adv.rolling(10).sum() / dec.rolling(10).sum().replace(0, np.nan)
    ad_line = (adv - dec).cumsum()
    comp_ad = 0.6 * _ramp(ratio10, 0.7, 1.4) + 0.4 * (100.0 * (ad_line > ad_line.rolling(50).mean()))

    parts = pd.DataFrame({"index": comp_idx, "breadth": comp_breadth, "vix": comp_vix,
                          "sector": comp_sector, "ad": comp_ad}, index=d)
    # a component that has no history yet counts as neutral (50) rather than dragging the score down
    score = sum(parts[k].fillna(50.0) * w for k, w in W.items()) / sum(W.values())
    lab = [label(x) for x in score]
    out = parts.copy()
    out["score"] = score
    out["state"] = [x[0] for x in lab]
    out["size_mult"] = [x[1] for x in lab]
    out["max_positions"] = [x[2] for x in lab]
    out["dispersion"] = dispersion
    out["vix_known"] = vix_known
    out["warm"] = s200.notna().values        # False while the 200-day average does not exist yet
    return out
