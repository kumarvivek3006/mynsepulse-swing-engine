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
MIN_BASE_DEPTH_PCT = float(os.environ.get("MIN_BASE_DEPTH_PCT", "4"))

# Flag/pennant: a genuinely different animal from the 15-120 session bases
# above. A short, tight consolidation (1-3 weeks) immediately after a
# STEEP, RECENT advance — the flagpole. Below BASE_MIN_SESSIONS, so it
# cannot be found by widening the existing search; it needs its own path.
FLAG_MIN_SESSIONS = int(os.environ.get("FLAG_MIN_SESSIONS", "5"))
FLAG_MAX_SESSIONS = int(os.environ.get("FLAG_MAX_SESSIONS", "15"))
FLAGPOLE_LOOKBACK = int(os.environ.get("FLAGPOLE_LOOKBACK", "20"))
FLAGPOLE_MIN_GAIN_PCT = float(os.environ.get("FLAGPOLE_MIN_GAIN_PCT", "20"))
FLAG_MIN_DEPTH_PCT = float(os.environ.get("FLAG_MIN_DEPTH_PCT", "2"))
FLAG_MAX_DEPTH_PCT = float(os.environ.get("FLAG_MAX_DEPTH_PCT", "15"))
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
    # A VCP is a shape PLUS a contraction. Kept separate so the shape label
    # is never overwritten by the quality.
    contracting: bool = False
    # OBV rising while price is flat/consolidating — the textbook Wyckoff
    # accumulation signature. Real evidence a base is being bought into,
    # distinct from a stock that is merely quiet because nobody wants it.
    obv_rising: bool = False


@dataclass
class Setup:
    symbol: str
    setup_type: str
    pattern: str
    entry: float
    stop: float
    t1: float
    t2: float | None
    r_multiple_t1: float
    base: Base
    stop_basis: str
    t1_basis: str
    t2_basis: str | None = None
    score_total: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    # Diagnostic only — no gate reads this. See extension_metrics().
    extension: dict = field(default_factory=dict)
    provisional: bool = False
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


def _find_local_minima(low_arr, span: int) -> list[int]:
    """Same test as swing_lows(), on a raw array — a low with `span` higher
    lows either side. Used for shoulder/head detection below."""
    out = []
    n = len(low_arr)
    for i in range(span, n - span):
        window = low_arr[i - span:i + span + 1]
        if low_arr[i] == window.min() and (window > low_arr[i]).sum() >= span:
            out.append(i)
    return out


def _is_inverse_head_shoulders(seg_high, seg_low, duration: int) -> bool:
    """
    Three troughs, not one: left shoulder, head, right shoulder — the head
    the deepest of the three, both shoulders comparable to each other, the
    head clearly separated in time from both.

    The pivot detect_base already computes (highest high in the older 70%
    of the window) needs no special handling for this pattern: in a
    well-formed inverse H&S the neckline peak between the head and right
    shoulder IS that highest high, since it sits later than the left
    shoulder's peak and before the final 30% cutoff. So this only refines
    the LABEL — entry, stop and target are computed identically to every
    other pattern, from the same pivot and base low.
    """
    span = max(2, duration // 15)
    raw_troughs = _find_local_minima(seg_low, span)
    if len(raw_troughs) < 3:
        return False

    # Adjacent indices commonly both register as "local minima" either side
    # of a discretized trough's true bottom — tested directly: a smooth
    # parabolic dip produced two adjacent trough indices, and without this
    # merge step the triple-matching below paired both points from the SAME
    # dip as if they were separate shoulders, never reaching the real head.
    troughs = [raw_troughs[0]]
    for idx in raw_troughs[1:]:
        if idx - troughs[-1] <= span:
            if seg_low[idx] < seg_low[troughs[-1]]:
                troughs[-1] = idx
        else:
            troughs.append(idx)
    if len(troughs) < 3:
        return False

    for i in range(len(troughs) - 2):
        left, head, right = troughs[i], troughs[i + 1], troughs[i + 2]
        left_low, head_low, right_low = seg_low[left], seg_low[head], seg_low[right]

        if not (head_low < left_low and head_low < right_low):
            continue                                    # head must be deepest

        left_depth = left_low - head_low
        right_depth = right_low - head_low
        if left_depth <= 0 or right_depth <= 0:
            continue
        if min(left_depth, right_depth) / max(left_depth, right_depth) < 0.5:
            continue                                    # shoulders too dissimilar

        if (head - left) < 3 or (right - head) < 3:
            continue                                    # not enough separation
        if right > duration - 3:
            continue                                    # need room after the pattern completes

        return True
    return False


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
# Measurement-only alternatives to the live "best_quality" search, added to
# test whether the composite scoring formula itself is the problem. detect_base
# was measured against outcomes for the first time via base_quality_quartile:
# its own TOP-rated bucket (q4) was the WORST performer in the backtest,
# consistently across both halves (-0.761R, -0.407R), and four separate
# follow-up checks (liquidity, depth, prior-move strength, volume asymmetry)
# each found the badness uniform across every sub-slice rather than
# concentrated in one — meaning no single ingredient explains it, which
# points at the SELECTION MECHANISM (best-of-~20-windows) rather than any one
# factor in the formula.
#
# "best_quality" (default) is BYTE-IDENTICAL to the original function and is
# what the live engine calls. The other two only run inside the backtest.
BASE_SELECTION_STRATEGIES = ("best_quality", "first_valid", "fixed_window")
FIXED_BASE_WINDOW = int(os.environ.get("FIXED_BASE_WINDOW", "45"))


def detect_base(df: pd.DataFrame, exclude_last: int = 1,
                strategy: str = "best_quality") -> Base:
    """
    The base is formed by the bars BEFORE the trigger.

    If the most recent bar is a breakout, it holds the highest high in the
    window — so including it would make the breakout bar itself the pivot,
    and the "pivot must sit in the older part of the base" rule would
    reject every setup we actually want. The trigger bar is excluded here
    and evaluated separately in detect_trigger().

    strategy:
      "best_quality" (live default) — scan every window, keep the one
        scoring highest on the composite formula. Unchanged from before.
      "first_valid" — scan windows shortest-to-longest (most recent base
        first) and take the FIRST one clearing every validity check, no
        quality ranking at all. Mirrors a trader taking the most recent
        legitimate base rather than grading twenty of them against a
        formula.
      "fixed_window" — try exactly ONE lookback (FIXED_BASE_WINDOW
        sessions). No search at all.
    """
    df = df.iloc[:-exclude_last] if exclude_last else df
    n = len(df)
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    volumes = df["volume"].values
    obv = df["obv"].values if "obv" in df.columns else None

    best: Base | None = None

    # Try several base lengths and keep the highest-quality structure.
    # Why each candidate window failed, so a zero-signal day is diagnosable
    # rather than a shrug.
    fail_counts: dict[str, int] = {}

    def _note(reason: str) -> None:
        fail_counts[reason] = fail_counts.get(reason, 0) + 1

    max_lookback = min(BASE_MAX_SESSIONS, n - PRIOR_UPTREND_WINDOW // 2)
    if strategy == "fixed_window":
        lookback_values = [FIXED_BASE_WINDOW] if FIXED_BASE_WINDOW <= max_lookback else []
    else:
        lookback_values = range(BASE_MIN_SESSIONS, max_lookback + 1, 5)

    for lookback in lookback_values:
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
        if depth < MIN_BASE_DEPTH_PCT:
            # A 1% range over 15 sessions is not a base, it is noise. This
            # is a setup-quality question, separate from position sizing:
            # the ATR floor downstream already protects sizing by rejecting
            # a stop too tight to clear normal daily range, but that leaves
            # the SETUP itself validated as if the noise were a real base.
            # It was not — there was no genuine contraction to measure.
            _note("base_too_shallow")
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

        # Measuring low-to-end rewards a crash-and-recover exactly as much as
        # a genuine advance: a stock falling 200 -> 120 then returning to 175
        # scored a "46% prior uptrend" while sitting 12% BELOW where it began.
        # A base is only a base if the stock reached NEW ground first, so the
        # pivot must exceed the highest close of the window's earlier half.
        earlier_half = prior_window[:max(len(prior_window) // 2, 5)]
        if len(earlier_half) and pivot <= float(earlier_half.max()):
            _note("no_new_ground")
            continue

        third = max(lookback // 3, 3)
        # Baseline is the earlier two-thirds, not the whole base. Including
        # the quiet final third in its own denominator diluted the very
        # comparison the test exists to make.
        early_vol = volumes[seg_start:-third].mean() if lookback > third else 0.0
        dryup = float(volumes[-third:].mean() / early_vol) if early_vol > 0 else 1.0

        first_range = seg_high[:third].max() - seg_low[:third].min()
        last_range = seg_high[-third:].max() - seg_low[-third:].min()
        contraction = float(last_range / first_range) if first_range > 0 else 1.0

        pattern = _classify(seg_high, seg_low, depth, lookback, contraction)
        contracting = bool(contraction < 0.6 and depth < 25)

        # Deliberately NOT added into the quality score below. That formula
        # is the one measured to be inverted — its own top-rated quartile
        # was the worst performer in the backtest — and mixing a genuine
        # signal into an already-unreliable one risks diluting the signal
        # rather than fixing the formula. This is surfaced as a visible,
        # separate fact instead: did cumulative buying pressure actually
        # rise over the life of this base, real accumulation, versus a
        # stock that is merely flat because nobody is trading it.
        obv_rising = bool(
            obv is not None and obv[-1] > obv[seg_start]) if obv is not None else False

        quality = (
            (1.0 - min(depth / 35, 1.0)) * 30           # tighter is better
            + max(0.0, 1.0 - dryup) * 25               # volume drying up
            + max(0.0, 1.0 - contraction) * 25         # ranges contracting
            + min(lookback / 60, 1.0) * 10             # duration
            + min(prior_gain / 60, 1.0) * 10           # strength into the base
        )

        candidate = Base(pattern, seg_start, pivot_idx, pivot, base_low, depth,
                         lookback, dryup, contraction, prior_gain, quality,
                         contracting, obv_rising)

        if strategy == "first_valid":
            # Stop at the first legitimate window — no ranking against the
            # (now suspect) composite score.
            best = candidate
            break
        if best is None or candidate.quality > best.quality:
            best = candidate

    if best is None:
        dominant = max(fail_counts, key=fail_counts.get) if fail_counts else "no_windows"
        raise Rejected("gate4", f"no_base_{dominant}", dict(fail_counts))
    return best


def _is_cup_and_handle(seg_high, seg_low, duration: int,
                       highs_first: float, highs_last: float) -> bool:
    """
    Real O'Neil criteria — this is the fix, not the proxy it replaces.

    The previous test was `lows_last > lows_first`: any base deeper than
    15% whose ending low sat above its starting low. That has no shape
    requirement at all and labels a wide range of non-cup structures —
    ascending bases, partial recoveries, plain consolidations — as
    "cup-and-handle". The pattern backtested at -0.36R because the SAMPLE
    was wrong, not because cup-and-handle itself is weak; O'Neil's own
    research treats it as one of the four reliable base types.

    Four real checks, each against a documented criterion:

      1. Right side reaches back near the left side (within ~15%) — a base
         recovering to well below its own starting high is a partial
         recovery, not a cup.
      2. The base's deepest point sits in the cup, not the handle — a
         genuine handle never makes a new low.
      3. The handle stays in the UPPER HALF of the cup's full range.
      4. The handle is shallower than half the cup's own depth — O'Neil:
         typically 10-15%, never approaching the cup's depth.
    """
    handle_len = max(5, min(duration // 4, 15))
    if duration - handle_len < 10:
        return False        # not enough cup left once the handle is set aside

    cup_portion = seg_low[:-handle_len]
    cup_low_idx = int(np.argmin(cup_portion))
    cup_low = float(cup_portion[cup_low_idx])
    cup_high = float(seg_high[:-handle_len].max())
    handle_low = float(seg_low[-handle_len:].min())
    handle_high = float(seg_high[-handle_len:].max())
    if cup_high <= 0 or cup_low <= 0 or handle_high <= 0:
        return False

    # A genuine cup DECLINES first, then recovers. A monotonically rising
    # series — a plain ascending base or triangle — has its minimum at the
    # very start almost by definition, and would otherwise pass every check
    # below with no cup shape at all: tested directly against a real
    # ascending triangle (flat top, steadily rising lows, no dip), it was
    # wrongly accepted as a cup until this check was added. Requiring the
    # low to sit meaningfully inside the window, not at the edge, is what
    # actually enforces "declined, then recovered" rather than "rose".
    if cup_low_idx < len(cup_portion) * 0.10:
        return False

    if highs_first <= 0 or highs_last < highs_first * 0.85:
        return False                                    # check 1

    if handle_low < cup_low:
        return False                                    # check 2

    cup_mid = cup_low + 0.5 * (cup_high - cup_low)
    if handle_low < cup_mid:
        return False                                    # check 3

    cup_depth = (cup_high - cup_low) / cup_high
    handle_depth = (handle_high - handle_low) / handle_high
    if handle_depth > cup_depth * 0.5:
        return False                                    # check 4

    return True


def _classify(seg_high, seg_low, depth, duration, contraction) -> str:
    third = max(duration // 3, 3)
    lows_first = seg_low[:third].min()
    lows_last = seg_low[-third:].min()
    highs_first = seg_high[:third].max()
    highs_last = seg_high[-third:].max()

    flat_resistance = abs(highs_last - highs_first) / highs_first < 0.03 if highs_first else False
    rising_lows = lows_last > lows_first * 1.02

    # Shape is decided by geometry alone. Contraction is a separate quality,
    # reported alongside rather than overriding the shape.
    #
    # cup_and_handle is checked before asc_triangle deliberately: it is the
    # more specific, harder-to-satisfy claim (four real geometric checks
    # against one loose pair). A genuine cup can incidentally also satisfy
    # the flat-resistance-plus-rising-lows test — tested directly, a
    # constructed textbook cup was swallowed by asc_triangle when that ran
    # first, and the specific test never got a chance to fire.
    if depth <= 15:
        return "flat_base"
    if _is_inverse_head_shoulders(seg_high, seg_low, duration):
        # Tried before cup_and_handle: three well-formed troughs is a more
        # specific, harder-to-satisfy claim than a single dip, so it gets
        # first refusal the same way cup_and_handle gets first refusal over
        # the looser asc_triangle test below.
        return "inverse_head_shoulders"
    if _is_cup_and_handle(seg_high, seg_low, duration, highs_first, highs_last):
        return "cup_handle"
    if flat_resistance and rising_lows:
        return "asc_triangle"
    if lows_last > lows_first:
        # A genuine staircase of higher lows that fails the cup criteria
        # above — O'Neil's fourth classic base type, previously folded
        # into the cup_handle mislabel with no distinct identity of its own.
        return "ascending_base"
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

    # Pullback into the 20 EMA while still inside the base.
    #
    # The prior test only checked distance from the average — abs(close -
    # ema)/ema < 3% — which fires whether price is above or below it, and
    # never checks the pullback against the base at all. A stock sliding
    # DOWN through its base on the way to breaking it prints the same
    # bullish reversal bar as one holding support 3% above the average.
    #
    # Three real conditions now gate it, all read from price that printed:
    #   1. Close sits AT or just ABOVE the average (0% to +3%), not below —
    #      a pullback that has broken the average is not "into" it.
    #   2. The average itself is rising over the last 10 sessions — the
    #      trend the pullback is buying into must still be intact.
    #   3. The bar's low held above the base low — a pullback that breaks
    #      the floor of its own base has invalidated the base, not paused.
    ema20 = float(last["ema20"]) if pd.notna(last["ema20"]) else None
    ema20_prior = (float(df["ema20"].iloc[-11])
                  if len(df) > 10 and pd.notna(df["ema20"].iloc[-11]) else None)

    if ema20:
        near_ema = ema20 <= float(last["close"]) <= ema20 * 1.03
        ema_rising = ema20_prior is not None and ema20 > ema20_prior
        held_base = float(last["low"]) >= base.base_low

        if near_ema and ema_rising and held_base:
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

        # A single-bar close-vs-midpoint test can pass a stock that spent
        # its recent sessions trading in the LOWER half before one bar
        # recovered. Tested: a base dipping to its low, sitting there for a
        # week, then rallying to close above the midpoint passed the
        # existing check, because that check only looks at the last bar.
        # This looks at the PRICE RANGE of the same final third of the base
        # the volume dry-up is measured over — if that whole window traded
        # in the lower half, the "dry-up" is a stock going quiet on the way
        # down, not a stock coiling near resistance.
        third = max(base.duration // 3, 3)
        recent_bars = df.iloc[-third:]
        recent_mid = float((recent_bars["high"].max() + recent_bars["low"].min()) / 2)
        base_mid = (base.pivot + base.base_low) / 2
        if recent_mid < base_mid:
            raise Rejected("gate5", "dryup_in_lower_half",
                           {"recent_mid": round(recent_mid, 2),
                            "base_mid": round(base_mid, 2)})
        return "armed"

    raise Rejected("gate5", "no_trigger",
                   {"distance_to_pivot_pct": round(distance_pct, 2)})


# ---------------------------------------------------------------------
# Level derivation — organic only
# ---------------------------------------------------------------------
def derive_levels(df: pd.DataFrame, base: Base, setup_type: str,
                  last_bar_incomplete: bool = False) -> dict:
    """
    last_bar_incomplete=True during an intraday run.

    The base, pivot, base low and overhead supply already come from closed
    bars — detect_base excludes the final bar. So the only levels that can
    leak from an unfinished session are a stop taken from the trigger bar's
    low, or from the 20 EMA, both of which still move. Excluding those two
    makes an ARMED setup fully derived from completed data, which is what
    lets it be published mid-session as a real level rather than a guess.
    """
    last = df.iloc[-1]
    atr = float(last["atr14"])

    # Entry must be a level the market has NOT already left behind.
    #
    # Anchoring every breakout to the pivot produced triggers below the
    # current price: a bar gapping 6% through the pivot gave an entry 5%
    # under the close. Live that fills at the open as a chase, and the R:R
    # shown was computed from a price you could not get. Worse, risk was
    # measured entry-to-stop while the real fill sat far higher — 4.7x the
    # planned risk in testing, which position sizing then treated as small.
    #
    # The honest anchor is the higher of the pivot and the trigger bar's
    # high. Both are prices that printed. Nothing is manufactured, and an
    # extended breakout now fails the R:R floor and stop-width limit on its
    # own arithmetic rather than needing a new threshold.
    if setup_type.startswith("pullback"):
        anchor = float(last["high"])
    else:
        anchor = max(base.pivot, float(last["high"]))
    entry = anchor * (1 + ENTRY_BUFFER)

    # --- stop: the tightest structural level that is still real ---------
    candidates: list[tuple[float, str]] = []
    for idx in swing_lows(df.iloc[base.start_idx:], span=3):
        candidates.append((float(df["low"].iloc[base.start_idx + idx]), "base_swing_low"))
    candidates.append((base.base_low, "base_low"))

    if last_bar_incomplete:
        # The forming bar's low and EMA both still move, so use the last
        # CLOSED bar instead. Same kind of level, equally tight, but fixed.
        # Falling back only to the base low would widen risk enough to fail
        # the R:R floor and quietly suppress most intraday setups.
        if len(df) >= 2:
            candidates.append((float(df["low"].iloc[-2]), "prev_bar_low"))
            prev_ema = df["ema20"].iloc[-2]
            if pd.notna(prev_ema) and float(prev_ema) < entry:
                candidates.append((float(prev_ema), "ema20_prev_close"))
    else:
        candidates.append((float(last["low"]), "trigger_bar_low"))
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
    #
    # T1 uses the measuring principle — base height projected from the
    # pivot — which is standard technical analysis, not an invented number.
    # T2 previously padded to "measured + (measured-pivot)*0.5" whenever no
    # second level existed above the measured move. That 0.5 multiplier
    # never traded; it is exactly the kind of manufactured level removed
    # from the trail (breakeven, ATR chandelier) and it does not belong
    # here either. A stock breaking to new highs — often the strongest
    # setups, since nothing overhead has ever stopped it — now gets T1
    # only. No second target is invented in its place.
    measured = base.pivot + (base.pivot - base.base_low)

    supply = None
    for idx in swing_highs(df.iloc[:base.start_idx], span=3):
        high = float(df["high"].iloc[idx])
        if high > entry * 1.01:
            supply = high if supply is None else min(supply, high)

    if supply is not None and supply < measured:
        t1, t1_basis = supply, "overhead_supply"
        t2, t2_basis = measured, "measured_move"
    elif supply is not None and supply > measured:
        t1, t1_basis = measured, "measured_move"
        t2, t2_basis = supply, "overhead_supply"
    else:
        t1, t1_basis = measured, "measured_move"
        t2, t2_basis = None, None

    r_multiple = (t1 - entry) / risk
    if r_multiple < MIN_RR:
        raise Rejected("gate7", "insufficient_rr",
                       {"r_multiple": round(r_multiple, 2), "floor": MIN_RR,
                        "t1_basis": t1_basis})

    return {"entry": round(entry, 2), "stop": round(stop, 2),
            "t1": round(t1, 2), "t2": round(t2, 2) if t2 is not None else None,
            "r_multiple_t1": round(r_multiple, 2),
            "stop_basis": stop_basis, "t1_basis": t1_basis, "t2_basis": t2_basis}


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


def _score_fundamentals(snap) -> float | None:
    """
    Grade the two things we actually hold: promoter behaviour and the
    revenue/profit trend. Returns None when there is no data, which keeps
    the block out of the achievable ceiling entirely rather than awarding
    a token half-mark nobody can improve on.
    """
    if snap is None or not getattr(snap, "has_data", False):
        return None

    parts: list[float] = []

    if snap.promoter_pct is not None and snap.promoter_pct_2q_ago is not None:
        change = snap.promoter_pct - snap.promoter_pct_2q_ago
        # Promoters adding is the strongest signal available; flat is fine;
        # selling is already a Gate 2 veto at 2pp, so anything reaching here
        # is a mild drift.
        parts.append(1.0 if change > 0.25 else 0.6 if change > -0.5 else 0.2)

    rev, pat = snap.revenue_trend, snap.pat_trend
    if len(rev) >= 3:
        growth = (rev[-1] / rev[-3] - 1) if rev[-3] > 0 else 0
        parts.append(min(max(growth / 0.20, 0.0), 1.0))
    if len(pat) >= 3 and pat[-3] > 0:
        growth = pat[-1] / pat[-3] - 1
        parts.append(min(max(growth / 0.25, 0.0), 1.0))

    return (sum(parts) / len(parts)) if parts else None


def score_setup(df: pd.DataFrame, base: Base, setup_type: str, levels: dict,
                rs63: float | None, rs126: float | None,
                snap=None) -> tuple[float, dict]:
    last = df.iloc[-1]
    vol50 = float(last["vol50"]) if pd.notna(last["vol50"]) else 0.0
    rng = float(last["high"] - last["low"])

    vol_mult = (float(last["volume"]) / vol50) if vol50 else 0.0
    close_pos = ((float(last["close"]) - float(last["low"])) / rng) if rng > 0 else 0.5
    if setup_type.startswith("breakout"):
        trigger = min(vol_mult / 3.0, 1.0) * 0.6 + close_pos * 0.4
    elif setup_type.startswith("pullback"):
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

    # A block with no data is excluded from the ceiling rather than scored
    # zero out of its full weight. Scoring it zero would punish the stock
    # for our missing ingestion; counting it in the ceiling would make the
    # maximum unreachable. Neither is honest, so the weight is removed.
    fundamentals = _score_fundamentals(snap)
    news = None                      # no source ingested

    parts = {"trigger": trigger, "base": base_q, "trend": float(trend),
             "rs": float(rs_score), "trade_math": math_score,
             "fundamentals": fundamentals, "news": news}

    breakdown = {k: (round(v * WEIGHTS[k], 1) if v is not None else 0.0)
                 for k, v in parts.items()}
    breakdown["max_possible"] = sum(WEIGHTS[k] for k, v in parts.items()
                                    if v is not None)
    breakdown["blocks_unscored"] = [k for k, v in parts.items() if v is None]
    return round(sum(breakdown[k] for k in parts), 1), breakdown


def detect_flag_pennant(df: pd.DataFrame, exclude_last: int = 1) -> Base:
    """
    A short, tight consolidation immediately after a steep, recent advance.

    Structurally different from detect_base's 15-120 session search, which
    is tuned for genuine bases — flags typically run 1-3 weeks, below
    BASE_MIN_SESSIONS entirely, and what defines one is not the
    consolidation's own shape but the sharp move that PRECEDES it. Two
    requirements neither detect_base nor any of its pattern labels check:
    a flagpole (a steep, recent advance immediately before the
    consolidation) and volume that contracts inside the flag relative to
    the pole that formed it.

    Called only as a fallback in build_setup, when the standard base search
    finds nothing — this can only ADD coverage for a structurally distinct,
    shorter-timeframe setup, never override or compete with an
    already-valid longer base.
    """
    df = df.iloc[:-exclude_last] if exclude_last else df
    n = len(df)
    highs, lows, closes = df["high"].values, df["low"].values, df["close"].values
    volumes = df["volume"].values

    fail_counts: dict[str, int] = {}

    def _note(reason: str) -> None:
        fail_counts[reason] = fail_counts.get(reason, 0) + 1

    best: Base | None = None

    for flag_len in range(FLAG_MIN_SESSIONS, FLAG_MAX_SESSIONS + 1):
        if n - flag_len < FLAGPOLE_LOOKBACK + 5:
            _note("insufficient_history")
            continue

        flag_start = n - flag_len
        flag_high, flag_low = highs[flag_start:], lows[flag_start:]

        pivot = float(flag_high.max())
        pivot_idx = flag_start + int(np.argmax(flag_high))
        base_low = float(flag_low.min())
        if base_low <= 0 or pivot <= 0:
            _note("bad_low")
            continue

        depth = (pivot - base_low) / pivot * 100
        if depth > FLAG_MAX_DEPTH_PCT:
            _note("flag_too_deep")
            continue
        if depth < FLAG_MIN_DEPTH_PCT:
            _note("flag_too_shallow")
            continue

        # The flagpole: a steep, RECENT advance immediately before the flag,
        # not merely "some advance sometime in the last 120 sessions" — the
        # generic prior-uptrend check used elsewhere is too loose for what
        # specifically defines this pattern.
        pole_start = max(0, flag_start - FLAGPOLE_LOOKBACK)
        pole_window = closes[pole_start:flag_start + 1]
        if len(pole_window) < 5:
            _note("no_flagpole_room")
            continue
        pole_gain = (pole_window[-1] / pole_window.min() - 1) * 100
        if pole_gain < FLAGPOLE_MIN_GAIN_PCT:
            _note("weak_flagpole")
            continue

        # Volume should contract inside the flag relative to the pole that
        # formed it — the pole runs on demand, the flag on its absence.
        pole_vol = volumes[pole_start:flag_start].mean() if flag_start > pole_start else 0.0
        flag_vol = volumes[flag_start:].mean()
        dryup = float(flag_vol / pole_vol) if pole_vol > 0 else 1.0
        if dryup > 1.0:
            _note("no_volume_dryup")
            continue

        quality = (
            (1.0 - min(depth / FLAG_MAX_DEPTH_PCT, 1.0)) * 40
            + max(0.0, 1.0 - dryup) * 30
            + min(pole_gain / 60, 1.0) * 30
        )

        candidate = Base("flag_pennant", flag_start, pivot_idx, pivot, base_low,
                         depth, flag_len, dryup, dryup, pole_gain, quality,
                         True, False)
        if best is None or candidate.quality > best.quality:
            best = candidate

    if best is None:
        dominant = max(fail_counts, key=fail_counts.get) if fail_counts else "no_windows"
        raise Rejected("gate4", f"no_flag_{dominant}", dict(fail_counts))
    return best


# ---------------------------------------------------------------------
TRANSITION_MIN_BASE_SESSIONS = int(
    os.environ.get("TRANSITION_MIN_BASE_SESSIONS", "60"))
TRANSITION_VOL_MULT = float(os.environ.get("TRANSITION_VOL_MULT", "2.0"))


def build_setup(symbol: str, df: pd.DataFrame, rs63: float | None,
                rs126: float | None, snap=None, transition: bool = False,
                last_bar_incomplete: bool = False,
                base_strategy: str = "best_quality") -> Setup:
    """
    transition=True applies the Stage 1->2 profile: a longer base and a
    heavier volume break. The trend filter is looser on that path, so the
    price-action evidence has to be stronger to compensate. Loosening one
    test without tightening another is how a second path becomes a back
    door.

    base_strategy is measurement-only — see detect_base. The live engine
    never passes anything but the default.
    """
    try:
        base = detect_base(df, strategy=base_strategy)
    except Rejected as base_rej:
        # Fallback only on the live path. The alternate strategies
        # (first_valid, fixed_window) exist purely to compare window
        # SELECTION within detect_base's own search — letting a flag
        # fallback fire underneath them would let identical flag signals
        # leak into all three, diluting exactly the comparison that
        # backtest is measuring.
        if base_strategy != "best_quality":
            raise
        base = detect_flag_pennant(df)

    if transition:
        if base.duration < TRANSITION_MIN_BASE_SESSIONS:
            raise Rejected("gate4", "transition_base_too_short",
                           {"duration": base.duration,
                            "required": TRANSITION_MIN_BASE_SESSIONS})
        last = df.iloc[-1]
        vol50 = float(last["vol50"]) if pd.notna(last["vol50"]) else 0.0
        if last["close"] > base.pivot and (
                vol50 <= 0 or last["volume"] < vol50 * TRANSITION_VOL_MULT):
            raise Rejected("gate5", "transition_volume_insufficient",
                           {"required_mult": TRANSITION_VOL_MULT,
                            "actual": round(float(last["volume"]) / vol50, 2)
                            if vol50 else None})

    setup_type = detect_trigger(df, base)
    if transition:
        setup_type = f"{setup_type}_transition"
    levels = derive_levels(df, base, setup_type, last_bar_incomplete)
    total, breakdown = score_setup(df, base, setup_type, levels, rs63, rs126, snap)

    extension = extension_metrics(df, levels["entry"])

    # Levels are never provisional intraday: with last_bar_incomplete set,
    # every one is drawn from a closed bar. What IS unconfirmed is whether
    # the trigger holds to the close — a breakout can finish in the lower
    # half of its range, or the volume pace can fade. So this flags an
    # unconfirmed TRIGGER, not an unreliable level, which is what the
    # earlier "provisional" naming wrongly implied.
    provisional = last_bar_incomplete and not setup_type.startswith("armed")

    return Setup(
        symbol=symbol, setup_type=setup_type, pattern=base.pattern,
        entry=levels["entry"], stop=levels["stop"], t1=levels["t1"], t2=levels["t2"],
        r_multiple_t1=levels["r_multiple_t1"], base=base,
        stop_basis=levels["stop_basis"], t1_basis=levels["t1_basis"],
        t2_basis=levels.get("t2_basis"),
        score_total=total, score_breakdown=breakdown,
        extension=extension, provisional=provisional,
        notes=[f"base {base.duration}d, depth {base.depth_pct:.1f}%"
               + (", contracting" if base.contracting else ""),
               f"stop from {levels['stop_basis']}", f"T1 from {levels['t1_basis']}"]
              + ([f"T2 from {levels['t2_basis']}"] if levels.get('t2_basis') else
                 ["no second target — no overhead level beyond the measured move"]),
    )
