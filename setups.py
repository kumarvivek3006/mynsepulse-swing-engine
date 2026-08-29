"""
Setups — Gates 4 to 7, and the score.

Everything here rests on one rule: entry, stop and target must each be a
price that already exists on the chart. A pivot is a bar's high. A stop
sits under a swing low that actually formed. A target is either the base
height projected, or a prior high someone actually sold into. Nothing is
derived by taking a percentage of the entry.

The consequence is that setups get rejected for having no valid level,
and that is intended. A stock in a clean uptrend with no definable pivot
is not a trade; it is a stock you missed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BASE_MIN_SESSIONS = int(os.environ.get("BASE_MIN_SESSIONS", "15"))
BASE_MAX_SESSIONS = int(os.environ.get("BASE_MAX_SESSIONS", "120"))
BREAKOUT_VOL_MULT = float(os.environ.get("BREAKOUT_VOL_MULT", "1.5"))
# How close to the pivot a stock must sit to be worth arming an order on.
ARMED_MAX_DISTANCE_PCT = float(os.environ.get("ARMED_MAX_DISTANCE_PCT", "4.0"))
MIN_RR = float(os.environ.get("MIN_RR", "1.0"))
MIN_STOP_ATR_MULT = float(os.environ.get("MIN_STOP_ATR_MULT", "0.75"))
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "8"))
PRIOR_UPTREND_PCT = float(os.environ.get("PRIOR_UPTREND_PCT", "25"))
# Bars over which the prior advance is measured. The classic trend-template
# looks at the whole advance into the base, not a fixed two-month window —
# a stock that ran 40% over five months and then consolidated is a valid
# base, and a 60-bar window wrongly rejects it.
PRIOR_UPTREND_WINDOW = int(os.environ.get("PRIOR_UPTREND_WINDOW", "120"))
ENTRY_BUFFER = 0.0025


# ---------------------------------------------------------------------
@dataclass
class Base:
    pattern: str
    start_idx: int
    pivot_idx: int
    pivot: float
    base_low: float
    depth_pct: float
    duration: int
    volume_dryup: float          # final-third volume / base average
    contraction_ratio: float     # last contraction / first contraction
    prior_uptrend_pct: float
    quality: float = 0.0


@dataclass
class Setup:
    symbol: str
    setup_type: str
    pattern: str
    entry: float
    stop: float
    t1: float
    t2: float
    r_multiple_t1: float
    base: Base
    stop_basis: str
    t1_basis: str
    score_total: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    # Diagnostic only — no gate reads this. See extension_metrics().
    extension: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


class Rejected(Exception):
    def __init__(self, gate: str, reason: str, detail: dict | None = None):
        self.gate, self.reason, self.detail = gate, reason, detail or {}
        super().__init__(f"{gate}:{reason}")


# ---------------------------------------------------------------------
# Swing points — the raw material for every level
# ---------------------------------------------------------------------
def swing_lows(df: pd.DataFrame, span: int = 3) -> list[int]:
    """A low with `span` higher lows either side. Confirmed, not provisional."""
    lows = df["low"].values
    out = []
    for i in range(span, len(lows) - span):
        window = lows[i - span:i + span + 1]
        if lows[i] == window.min() and (window > lows[i]).sum() >= span:
            out.append(i)
    return out


def swing_highs(df: pd.DataFrame, span: int = 3) -> list[int]:
    highs = df["high"].values
    out = []
    for i in range(span, len(highs) - span):
        window = highs[i - span:i + span + 1]
        if highs[i] == window.max() and (window < highs[i]).sum() >= span:
            out.append(i)
    return out


# ---------------------------------------------------------------------
# Gate 4 — base detection
# ---------------------------------------------------------------------
def detect_base(df: pd.DataFrame, exclude_last: int = 1) -> Base:
    """
    The base is formed by the bars BEFORE the trigger.

    If the most recent bar is a breakout, it holds the highest high in the
    window — so including it would make the breakout bar itself the pivot,
    and the "pivot must sit in the older part of the base" rule would
    reject every setup we actually want. The trigger bar is excluded here
    and evaluated separately in detect_trigger().
    """
    df = df.iloc[:-exclude_last] if exclude_last else df
    n = len(df)
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    volumes = df["volume"].values

    best: Base | None = None

    # Try several base lengths and keep the highest-quality structure.
    # Why each candidate window failed, so a zero-signal day is diagnosable
    # rather than a shrug.
    fail_counts: dict[str, int] = {}

    def _note(reason: str) -> None:
        fail_counts[reason] = fail_counts.get(reason, 0) + 1

    for lookback in range(BASE_MIN_SESSIONS,
                          min(BASE_MAX_SESSIONS, n - PRIOR_UPTREND_WINDOW // 2) + 1, 5):
        seg_start = n - lookback
        seg_high, seg_low = highs[seg_start:], lows[seg_start:]

        pivot_rel = int(np.argmax(seg_high))
        pivot = float(seg_high[pivot_rel])
        pivot_idx = seg_start + pivot_rel

        # The pivot is resistance to break out THROUGH, so it must sit in
        # the older part of the base. A high made yesterday is not a level
        # the stock has been coiling under.
        if pivot_rel > lookback * 0.7:
            _note("pivot_too_recent")
            continue

        base_low = float(seg_low.min())
        if base_low <= 0:
            _note("bad_low")
            continue
        depth = (pivot - base_low) / pivot * 100
        if depth > 35:
            _note("base_too_deep")
            continue

        # Prior uptrend: a base with nothing to consolidate is not a base.
        prior_window = closes[max(0, seg_start - PRIOR_UPTREND_WINDOW):seg_start + 1]
        if len(prior_window) < 20:
            _note("prior_window_too_short")
            continue
        prior_gain = (prior_window[-1] / prior_window.min() - 1) * 100
        if prior_gain < PRIOR_UPTREND_PCT:
            _note("weak_prior_uptrend")
            continue

        third = max(lookback // 3, 3)
        base_vol = volumes[seg_start:].mean()
        dryup = float(volumes[-third:].mean() / base_vol) if base_vol > 0 else 1.0

        first_range = seg_high[:third].max() - seg_low[:third].min()
        last_range = seg_high[-third:].max() - seg_low[-third:].min()
        contraction = float(last_range / first_range) if first_range > 0 else 1.0

        pattern = _classify(seg_high, seg_low, depth, lookback, contraction)

        quality = (
            (1.0 - min(depth / 35, 1.0)) * 30           # tighter is better
            + max(0.0, 1.0 - dryup) * 25               # volume drying up
            + max(0.0, 1.0 - contraction) * 25         # ranges contracting
            + min(lookback / 60, 1.0) * 10             # duration
            + min(prior_gain / 60, 1.0) * 10           # strength into the base
        )

        candidate = Base(pattern, seg_start, pivot_idx, pivot, base_low, depth,
                         lookback, dryup, contraction, prior_gain, quality)
        if best is None or candidate.quality > best.quality:
            best = candidate

    if best is None:
        dominant = max(fail_counts, key=fail_counts.get) if fail_counts else "no_windows"
        raise Rejected("gate4", f"no_base_{dominant}", dict(fail_counts))
    return best


def _classify(seg_high, seg_low, depth, duration, contraction) -> str:
    third = max(duration // 3, 3)
    lows_first = seg_low[:third].min()
    lows_last = seg_low[-third:].min()
    highs_first = seg_high[:third].max()
    highs_last = seg_high[-third:].max()

    flat_resistance = abs(highs_last - highs_first) / highs_first < 0.03 if highs_first else False
    rising_lows = lows_last > lows_first * 1.02

    if contraction < 0.6 and depth < 25:
        return "vcp"
    if flat_resistance and rising_lows:
        return "asc_triangle"
    if depth <= 15:
        return "flat_base"
    if depth > 15 and lows_last > lows_first:
        return "cup_handle"
    return "consolidation"


# ---------------------------------------------------------------------
# Gate 5 — trigger
# ---------------------------------------------------------------------
def detect_trigger(df: pd.DataFrame, base: Base) -> str:
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    atr = float(last["atr14"])
    vol50 = float(last["vol50"]) if pd.notna(last["vol50"]) else 0.0

    # Breakout
    if last["close"] > base.pivot:
        if vol50 <= 0 or last["volume"] < vol50 * BREAKOUT_VOL_MULT:
            raise Rejected("gate5", "breakout_without_volume",
                           {"volume_mult": round(float(last["volume"]) / vol50, 2) if vol50 else None})
        if rng > 0 and (last["close"] - last["low"]) / rng < 0.66:
            raise Rejected("gate5", "weak_close_in_range")
        # Exhaustion: a huge range with a long upper wick is supply, not demand.
        if atr > 0 and rng > 3 * atr:
            upper_wick = float(last["high"] - last["close"])
            if rng > 0 and upper_wick / rng > 0.5:
                raise Rejected("gate5", "exhaustion_candle")
        return "breakout"

    # Pullback into the 20 EMA while still inside the base
    ema20 = float(last["ema20"]) if pd.notna(last["ema20"]) else None
    if ema20 and abs(float(last["close"]) - ema20) / ema20 < 0.03:
        prev = df.iloc[-2]
        bullish = (
            last["close"] > last["open"]
            and last["close"] > (prev["high"] + prev["low"]) / 2
            and last["low"] <= prev["low"]
        )
        if bullish and (vol50 == 0 or last["volume"] < vol50):
            return "pullback"

    # --- armed: coiling under the pivot, no trigger yet ------------------
    #
    # This is the state a swing trader actually acts on. A breakout is only
    # visible after the close, so acting on a confirmed one means buying the
    # next open and paying the gap. An armed setup is published BEFORE the
    # move, with the entry as a resting stop order above the pivot — the
    # market fills you when it breaks, or it never triggers and the signal
    # expires. Nothing is missed and no gap is paid.
    close = float(last["close"])
    distance_pct = (base.pivot / close - 1) * 100

    if 0 <= distance_pct <= ARMED_MAX_DISTANCE_PCT:
        # Must still be constructive: sitting in the upper half of the base
        # and holding the 20 EMA. A stock at the bottom of its base is not
        # coiling, it is failing.
        midpoint = (base.pivot + base.base_low) / 2
        if close < midpoint:
            raise Rejected("gate5", "lower_half_of_base",
                           {"close": round(close, 2), "midpoint": round(midpoint, 2)})
        if pd.notna(last["ema20"]) and close < float(last["ema20"]):
            raise Rejected("gate5", "below_20ema")
        # Volume must have dried up in the base — supply leaving, not arriving.
        if base.volume_dryup > 1.1:
            raise Rejected("gate5", "no_volume_dryup",
                           {"dryup": round(base.volume_dryup, 2)})
        return "armed"

    raise Rejected("gate5", "no_trigger",
                   {"distance_to_pivot_pct": round(distance_pct, 2)})


# ---------------------------------------------------------------------
# Level derivation — organic only
# ---------------------------------------------------------------------
def derive_levels(df: pd.DataFrame, base: Base, setup_type: str) -> dict:
    last = df.iloc[-1]
    atr = float(last["atr14"])

    # Armed and breakout both enter on a stop order above the pivot; only a
    # pullback entry keys off the confirmation bar's high.
    anchor = float(last["high"]) if setup_type == "pullback" else base.pivot
    entry = anchor * (1 + ENTRY_BUFFER)

    # --- stop: the tightest structural level that is still real ---------
    candidates: list[tuple[float, str]] = []
    for idx in swing_lows(df.iloc[base.start_idx:], span=3):
        candidates.append((float(df["low"].iloc[base.start_idx + idx]), "base_swing_low"))
    candidates.append((float(last["low"]), "trigger_bar_low"))
    candidates.append((base.base_low, "base_low"))
    if pd.notna(last["ema20"]) and float(last["ema20"]) < entry:
        candidates.append((float(last["ema20"]), "ema20"))

    # Tightest first; step outward until one clears the ATR noise floor.
    viable = sorted({(round(p, 2), b) for p, b in candidates if p < entry},
                    key=lambda x: -x[0])
    stop = stop_basis = None
    for price, basis in viable:
        if (entry - price) >= MIN_STOP_ATR_MULT * atr:
            stop, stop_basis = price * 0.999, basis
            break

    if stop is None:
        raise Rejected("gate7", "no_structural_stop_beyond_noise",
                       {"atr14": round(atr, 2), "tightest": viable[0][0] if viable else None})

    risk = entry - stop
    if risk / entry * 100 > MAX_STOP_PCT:
        raise Rejected("gate7", "stop_too_wide",
                       {"stop_pct": round(risk / entry * 100, 2)})

    # --- targets: measured move vs the nearest real overhead supply ------
    measured = base.pivot + (base.pivot - base.base_low)

    supply = None
    for idx in swing_highs(df.iloc[:base.start_idx], span=3):
        high = float(df["high"].iloc[idx])
        if high > entry * 1.01:
            supply = high if supply is None else min(supply, high)

    if supply is not None and supply < measured:
        t1, t1_basis, t2 = supply, "overhead_supply", measured
    else:
        t1, t1_basis = measured, "measured_move"
        t2 = supply if supply and supply > measured else measured + (measured - base.pivot) * 0.5

    r_multiple = (t1 - entry) / risk
    if r_multiple < MIN_RR:
        raise Rejected("gate7", "insufficient_rr",
                       {"r_multiple": round(r_multiple, 2), "floor": MIN_RR,
                        "t1_basis": t1_basis})

    return {"entry": round(entry, 2), "stop": round(stop, 2),
            "t1": round(t1, 2), "t2": round(t2, 2),
            "r_multiple_t1": round(r_multiple, 2),
            "stop_basis": stop_basis, "t1_basis": t1_basis}


def extension_metrics(df: pd.DataFrame, entry: float) -> dict:
    """
    How stretched the entry is. RECORDED, NOT ACTED ON.

    Extension is the obvious next filter and I have deliberately not made
    it one. Distance from the 20 EMA correlates with momentum, so a cut
    here would reject the strongest breakouts before it rejected any
    mediocre ones — the same failure mode as an RSI ceiling, measured
    differently. It also partly duplicates checks that already exist: an
    extended stock usually has a distant structural stop, which already
    fails the 8% stop-width limit or the R:R floor.

    So this logs the numbers and changes nothing. When the backtest runs,
    the question to ask is whether extended entries actually underperformed.
    If they did, add the filter with a threshold taken from that answer
    rather than from my judgement.
    """
    last = df.iloc[-1]
    atr = float(last["atr14"])
    ema20 = float(last["ema20"]) if pd.notna(last["ema20"]) else None
    sma50 = float(last["sma50"]) if pd.notna(last["sma50"]) else None
    high52 = float(last["high52"]) if pd.notna(last["high52"]) else None

    return {
        "atr_above_ema20": round((entry - ema20) / atr, 2)
        if ema20 and atr > 0 else None,
        "pct_above_ema20": round((entry / ema20 - 1) * 100, 2) if ema20 else None,
        "pct_above_sma50": round((entry / sma50 - 1) * 100, 2) if sma50 else None,
        "pct_from_52w_high": round((entry / high52 - 1) * 100, 2) if high52 else None,
        "atr_pct_of_price": round(atr / entry * 100, 2) if atr and entry else None,
    }


# ---------------------------------------------------------------------
# Scoring — weights reflect what actually decides a swing trade
# ---------------------------------------------------------------------
WEIGHTS = {"trigger": 20, "base": 20, "trend": 18, "rs": 15,
           "trade_math": 12, "fundamentals": 10, "news": 5}


def score_setup(df: pd.DataFrame, base: Base, setup_type: str, levels: dict,
                rs63: float | None, rs126: float | None,
                fundamentals_ready: bool = False) -> tuple[float, dict]:
    last = df.iloc[-1]
    vol50 = float(last["vol50"]) if pd.notna(last["vol50"]) else 0.0
    rng = float(last["high"] - last["low"])

    vol_mult = (float(last["volume"]) / vol50) if vol50 else 0.0
    close_pos = ((float(last["close"]) - float(last["low"])) / rng) if rng > 0 else 0.5
    if setup_type == "breakout":
        trigger = min(vol_mult / 3.0, 1.0) * 0.6 + close_pos * 0.4
    elif setup_type == "pullback":
        trigger = 0.55 + close_pos * 0.25
    else:
        # Armed: there is no trigger bar yet, so this block grades readiness
        # — how tightly it is coiling and how far the volume has dried up —
        # rather than pretending to grade a breakout that has not occurred.
        proximity = 1.0 - min(max((base.pivot / float(last["close"]) - 1) * 100, 0)
                              / ARMED_MAX_DISTANCE_PCT, 1.0)
        trigger = 0.45 + proximity * 0.30 + max(0.0, 1.0 - base.volume_dryup) * 0.25

    base_q = base.quality / 100

    close = float(last["close"])
    trend = np.mean([
        1.0 if close > last["sma50"] > last["sma150"] > last["sma200"] else 0.4,
        min(max(float(last["sma200_slope25"]) / close * 40, 0), 1),
        1.0 - min(abs(close / float(last["high52"]) - 1) / 0.25, 1.0),
    ])

    rs_score = np.mean([
        min(max((rs63 or 0) / 30, 0), 1),
        min(max((rs126 or 0) / 50, 0), 1),
    ])

    math_score = min(max((levels["r_multiple_t1"] - MIN_RR) / 2.0, 0), 1) * 0.7 + 0.3

    # Absent inputs score zero, not a neutral default. A missing check is
    # not a passed check, and the total must reflect that.
    fundamentals = 0.0 if not fundamentals_ready else 0.5
    news = 0.0

    parts = {"trigger": trigger, "base": base_q, "trend": float(trend),
             "rs": float(rs_score), "trade_math": math_score,
             "fundamentals": fundamentals, "news": news}

    breakdown = {k: round(v * WEIGHTS[k], 1) for k, v in parts.items()}
    breakdown["max_possible"] = sum(
        WEIGHTS[k] for k in WEIGHTS
        if not (k == "fundamentals" and not fundamentals_ready) and k != "news"
    )
    return round(sum(breakdown[k] for k in parts), 1), breakdown


# ---------------------------------------------------------------------
def build_setup(symbol: str, df: pd.DataFrame, rs63: float | None,
                rs126: float | None, fundamentals_ready: bool = False) -> Setup:
    base = detect_base(df)
    setup_type = detect_trigger(df, base)
    levels = derive_levels(df, base, setup_type)
    total, breakdown = score_setup(df, base, setup_type, levels, rs63, rs126,
                                   fundamentals_ready)

    extension = extension_metrics(df, levels["entry"])

    return Setup(
        symbol=symbol, setup_type=setup_type, pattern=base.pattern,
        entry=levels["entry"], stop=levels["stop"], t1=levels["t1"], t2=levels["t2"],
        r_multiple_t1=levels["r_multiple_t1"], base=base,
        stop_basis=levels["stop_basis"], t1_basis=levels["t1_basis"],
        score_total=total, score_breakdown=breakdown,
        extension=extension,
        notes=[f"base {base.duration}d, depth {base.depth_pct:.1f}%",
               f"stop from {levels['stop_basis']}", f"T1 from {levels['t1_basis']}"],
    )
