"""
e2_swing_shadow.py — SWING candidates that run SHADOW-ONLY inside the live scan (owner's order of 18 Nov: oversold_dip, shadow-only).

A separate registry on purpose. These detectors are NOT in e2_detectors.DETECTORS, so the position engine's backtest, validation report, definition studies and
whitelist logic never see them. Only e2_live.scan (rows, scored and logged) and e2_fwd (forward-return test) call them. Guarantees, all enforced in code:
  - never selected into the book and never published: e2_live.is_shadow() is True for every name here whatever E2_SHADOW_DETECTORS / E2_PUBLISH_DETECTORS say;
  - marked "shadow" (why_not shadow_detector) in the scan's rows, and flagged x=swing_shadow in the forward log;
  - their forward-log R is simulated with the POSITION engine's exits (the swing exits are not designed yet), so it is kept out of the position engine's
    forward summary and reported in its own bucket, labelled as such;
  - E2_SWING_SHADOW=0 switches them off entirely.

oversold_dip (definition fixed 18 Nov before the entry test, unchanged): signal day i with close above the 200-day average, 14-day RSI <= 30 and a close at least
8% below the close 5 sessions earlier; one signal per episode (nothing of the same condition in the previous 10 sessions); entry at the next open.
18 Nov entry test, volatility-matched, 3-year window: n 179, matched excess +1.05% at 3 sessions and +0.78% at 5 (both halves positive), -0.26% at 10.
"""
from __future__ import annotations

import os

import numpy as np

from e2_core import Candidate, clamp
from e2_det_common import WARMUP
from e2_features import Bars

ENABLED = os.environ.get("E2_SWING_SHADOW", "1") == "1"
OS_RSI, OS_DROP5, OS_COOLDOWN = 30.0, -8.0, 10
FAMILY_OF = {"oversold_dip": "reversal"}               # the regime-fit table knows breakout | reversal | momentum | continuation


def oversold_dip(b: Bars, ctx: dict, lo: int, hi: int) -> list[Candidate]:
    cond = np.zeros(b.n, bool)
    for i in range(max(WARMUP, 205), b.n):
        if np.isnan(b.rsi14[i]) or np.isnan(b.sma200[i]) or b.c[i - 5] <= 0:
            continue
        cond[i] = bool(b.rsi14[i] <= OS_RSI and 100.0 * (b.c[i] / b.c[i - 5] - 1.0) <= OS_DROP5 and b.c[i] > b.sma200[i])
    out, last = [], -10 ** 9
    for i in range(b.n):
        if cond[i]:
            if i - last > OS_COOLDOWN and lo <= i < hi:
                drop5 = 100.0 * (b.c[i] / b.c[i - 5] - 1.0)
                out.append(Candidate(symbol=b.symbol, detector="oversold_dip", idx=i, date=b.date[i], pivot=float(b.c[i]), stop=float(b.l[i]),
                                     quality=clamp(50.0), family=FAMILY_OF["oversold_dip"],
                                     meta={"rsi14": round(float(b.rsi14[i]), 1), "drop_5d_pct": round(drop5, 2),
                                           "shadow_only": "swing candidate; the plan shown is the position engine's default, swing exits are not designed"}))
            last = i
    return out


SWING_DETECTORS = {"oversold_dip": oversold_dip}


def detect(b: Bars, lo: int, hi: int | None = None) -> list[Candidate]:
    """Candidates on bars [lo, hi). Empty when switched off. A detector that raises is skipped (the position engine's detectors are never affected)."""
    if not ENABLED:
        return []
    hi = b.n if hi is None else min(hi, b.n)
    out: list[Candidate] = []
    for fn in SWING_DETECTORS.values():
        try:
            out.extend(fn(b, {}, lo, hi))
        except Exception:                                              # noqa: BLE001
            import logging
            logging.getLogger("engine2.swing_shadow").exception("swing shadow detector failed (position engine unaffected)")
    return out
