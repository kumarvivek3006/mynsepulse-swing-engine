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

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# gates.py imports nothing from this package, so this direction creates no
# cycle. weekly_volume_surge lives there because that is where the other
# weekly/indicator logic already sits.
from gates import (WEEKLY_VOL_CHECK_ENABLED, WEEKLY_VOL_MULT,
                   weekly_volume_surge)

BASE_MIN_SESSIONS = int(os.environ.get("BASE_MIN_SESSIONS", "15"))
MIN_BASE_DEPTH_PCT = float(os.environ.get("MIN_BASE_DEPTH_PCT", "6"))
# Spec (Prompt 7): VCP's final contraction should be tightest, under ~8%.
VCP_FINAL_CONTRACTION_PCT = float(os.environ.get("VCP_FINAL_CONTRACTION_PCT", "8"))

# ---------------------------------------------------------------------
# Evidence-derived gates (Run #21, sign-stable across both halves).
#
# EVERY ONE IS SWITCHABLE. These were each measured in ISOLATION; turning
# on 20 simultaneously is a different and untested claim, and several of
# these same findings flipped sign in earlier runs. Defaults are on so the
# re-run measures the full stack, but any single gate can be disabled from
# the environment when the data says it does not hold.
# ---------------------------------------------------------------------
BREAKOUT_RSI_GATE = os.environ.get("BREAKOUT_RSI_GATE", "true").lower() == "true"
BREAKOUT_RSI_LO = float(os.environ.get("BREAKOUT_RSI_LO", "70"))
BREAKOUT_RSI_HI = float(os.environ.get("BREAKOUT_RSI_HI", "80"))

ARMED_MIN_DISTANCE_PCT = float(os.environ.get("ARMED_MIN_DISTANCE_PCT", "2.0"))

PRIOR_MOVE_REJECT_LO = float(os.environ.get("PRIOR_MOVE_REJECT_LO", "25"))
PRIOR_MOVE_REJECT_HI = float(os.environ.get("PRIOR_MOVE_REJECT_HI", "35"))

MINERVINI_GATE = os.environ.get("MINERVINI_GATE", "true").lower() == "true"
MINERVINI_RS_FLOOR = float(os.environ.get("MINERVINI_RS_FLOOR", "70"))

RS_MIN_GATE = os.environ.get("RS_MIN_GATE", "true").lower() == "true"
# These are PERCENTAGE POINTS of outperformance vs the index, not
# percentiles. 20pp over 63 days is roughly top-decile and 30pp over 126
# days top-5% — far stricter than the evidence supports. The 3-year run
# justifies HAVING a floor (bottom RS quintiles were sign-stable negative);
# it says nothing about this floor. Set near top-quartile outperformance
# and let the re-run identify where the separation actually is.
RS63_FLOOR = float(os.environ.get("RS63_FLOOR", "8"))
RS126_FLOOR = float(os.environ.get("RS126_FLOOR", "10"))

# Named HARD floor to distinguish it from scan.py's regime-scaled
# MIN_SCORE/MIN_SCORE_NEUTRAL/MIN_SCORE_RISK_OFF, which is the primary live
# gate. Two gates sharing one reason code made attribution ambiguous.
SETUP_HARD_FLOOR = float(os.environ.get("SETUP_HARD_FLOOR", "55"))

DELIVERY_GATE = os.environ.get("DELIVERY_GATE", "false").lower() == "true"
DELIVERY_MIN_PCT = float(os.environ.get("DELIVERY_MIN_PCT", "30"))

OBV_GATE = os.environ.get("OBV_GATE", "true").lower() == "true"

# asc_triangle and flag_pennant were both sign-stable negative in the 3-year
# run. _is_ascending_triangle has since been rebuilt (it previously counted
# local minima without checking they RISE, so descending troughs passed), and
# the flag detector gained a pole-retracement limit and the spec's 1.8x
# volume bar — so the negative result may already be reversed. But that
# cannot be known without a re-run, and until it is these should not trade.
# Set REJECT_NEGATIVE_PATTERNS=false to measure them again.
REJECT_NEGATIVE_PATTERNS = os.environ.get(
    "REJECT_NEGATIVE_PATTERNS", "true").lower() == "true"
# cup_handle added: 17 trades, 1 winner, -0.954R, and MFE averaging 0.4R
# means they go DOWN immediately rather than rallying and stopping out.
# The mechanism is entry timing and is not yet understood, so the pattern
# is disabled as a SIGNAL while remaining fully detected and labelled —
# the 17 trades stay as a diagnostic set, and detection continues so the
# population keeps growing for analysis.
#
# This is not a fix and is not presented as one. CUP_HANDLE_ENABLED=true
# restores it.
NEGATIVE_PATTERNS = {"asc_triangle", "flag_pennant"}
if os.environ.get("CUP_HANDLE_ENABLED", "false").lower() != "true":
    NEGATIVE_PATTERNS = NEGATIVE_PATTERNS | {"cup_handle"}
# ascending_base: negative in every run, ~-0.5R on ~21 fills. Its
# diagnosis is NOT complete — the trail forensics (trail_activated,
# stop_at_mfe, mae/mfe_bar_index) require a run on migration 011, which
# has not happened. Disabled until that diagnosis lands and a fix is
# validated, per the acceptance criteria. Detection continues, so the
# population keeps growing for analysis.
if os.environ.get("ASCENDING_BASE_ENABLED", "false").lower() != "true":
    NEGATIVE_PATTERNS = NEGATIVE_PATTERNS | {"ascending_base"}

# ---------------------------------------------------------------------
# Why DELIVERY_GATE defaults to false
#
# The spec (Layer 8) requires delivery >= 30% on the breakout day, and
# armed_delivery.confirmed was sign-stable positive. It is off because
# delivery_pct is not on the indicator frame — it lives in its own table and
# the join is not wired. Turning it on before that would reject EVERY
# breakout on missing data rather than on evidence.
#
# Flip to true once: (a) delivery_pct is joined onto the frame, and (b) the
# armed_delivery.confirmed bucket has n >= 100 after a re-run.
#
# PRIOR_MOVE_REJECT_LO/HI (25-35%) overlaps PRIOR_UPTREND_PCT (60) and is
# currently redundant. Kept deliberately: if the 60% floor is ever lowered
# for testing, the specific losing band stays excluded.
# ---------------------------------------------------------------------
# Rounding bottom is a LONG base by definition — 8-30 weeks per spec.
ROUNDING_MIN_SESSIONS = int(os.environ.get("ROUNDING_MIN_SESSIONS", "40"))
# High tight flag (spec): pole +100% in 4-8 weeks, flag 3-5 weeks, <25% retrace.
HTF_MIN_POLE_GAIN_PCT = float(os.environ.get("HTF_MIN_POLE_GAIN_PCT", "100"))
HTF_POLE_MIN_SESSIONS = int(os.environ.get("HTF_POLE_MIN_SESSIONS", "20"))
HTF_POLE_MAX_SESSIONS = int(os.environ.get("HTF_POLE_MAX_SESSIONS", "40"))
HTF_FLAG_MIN_SESSIONS = int(os.environ.get("HTF_FLAG_MIN_SESSIONS", "15"))
HTF_FLAG_MAX_SESSIONS = int(os.environ.get("HTF_FLAG_MAX_SESSIONS", "25"))
HTF_MAX_RETRACE_PCT = float(os.environ.get("HTF_MAX_RETRACE_PCT", "25"))
# Base-on-base: how far back to look for the prior consolidation.
BASE_ON_BASE_LOOKBACK = int(os.environ.get("BASE_ON_BASE_LOOKBACK", "40"))

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
# Spec requires >1.8x on the flag breakout — a higher bar than the generic
# 1.5x, because a flag is a short pause inside an established move and the
# resumption should carry more conviction than an ordinary base breakout.
FLAG_BREAKOUT_VOL_MULT = float(os.environ.get("FLAG_BREAKOUT_VOL_MULT", "1.8"))
# Spec: flag must not retrace more than 50% of the flagpole. A deeper
# pullback is a reversal, not a continuation pause.
FLAG_MAX_POLE_RETRACE_PCT = float(
    os.environ.get("FLAG_MAX_POLE_RETRACE_PCT", "50"))
BASE_MAX_SESSIONS = int(os.environ.get("BASE_MAX_SESSIONS", "120"))
BREAKOUT_VOL_MULT = float(os.environ.get("BREAKOUT_VOL_MULT", "1.5"))
# How close to the pivot a stock must sit to be worth arming an order on.
ARMED_MAX_DISTANCE_PCT = float(os.environ.get("ARMED_MAX_DISTANCE_PCT", "4.0"))
MIN_RR = float(os.environ.get("MIN_RR", "1.5"))
MIN_STOP_ATR_MULT = float(os.environ.get("MIN_STOP_ATR_MULT", "0.75"))
MAX_STOP_PCT = float(os.environ.get("MAX_STOP_PCT", "8"))
PRIOR_UPTREND_PCT = float(os.environ.get("PRIOR_UPTREND_PCT", "60"))
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
    # Records the strategy detect_base ACTUALLY ran with. Proves the
    # parameter arrived rather than relying on it having been passed.
    strategy_used: str | None = None
    # C4 shape diagnostics. Populated for every base, whatever the pattern,
    # so the LOSING population can be characterised rather than only the
    # accepted one. Nothing gates on these — cup_handle and ascending_base
    # remain fully tradeable while the mechanism is investigated.
    shape_diag: dict = field(default_factory=dict)


@dataclass
class Setup:
    """
    A constructed setup with its levels, score and provenance.

    strategy_requested vs strategy_used — these are NOT interchangeable:

      strategy_requested — the top-level config value handed to
        build_setup (BASE_STRATEGY, or whatever a caller passed).

      base.strategy_used — the strategy that ACTUALLY selected the base.
        None when the flag/pennant fallback produced it, since flag
        detection is not a base strategy.

    Diagnostic comparisons must read base.strategy_used. Reading
    strategy_requested only tells you what was asked for, which is exactly
    the mistake that let a self-comparison go unnoticed in Run #28.
    """
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
    strategy_requested: str | None = None
    trigger_diag: dict = field(default_factory=dict)
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
# "best_quality" is BYTE-IDENTICAL to the original function and is retained
# for A/B MEASUREMENT ONLY — it is no longer what the live engine calls. The
# live default became "first_valid" (see DEFAULT_BASE_STRATEGY below) once
# the q4 inversion was traced to this selector. This comment previously
# still described best_quality as the live path, which stopped being true
# that same round.
BASE_SELECTION_STRATEGIES = ("best_quality", "first_valid", "fixed_window",
                             "first_valid_quality_gated")
# Fourth option: take the first valid window, but only if it also clears a
# minimum quality bar. first_valid can accept a technically-valid but poor
# base purely because it appeared first; best_quality's own top quartile
# loses. This sits between them — earliest acceptable rather than earliest
# or best-scoring.
FIRST_VALID_MIN_QUALITY = float(os.environ.get("FIRST_VALID_MIN_QUALITY", "40"))

# ---------------------------------------------------------------------
# C4 diagnostic thresholds (measurement only — nothing gates on these)
#
# HANDLE SLOPE: the handle is "rising"/"falling" when its net move exceeds
# +/-1% OF THE HANDLE'S OWN RANGE, flat otherwise. Expressed relative to
# the handle rather than to price so a 2% handle on a 3000-rupee stock and
# on a 30-rupee stock classify the same way. The diagnosis must be robust
# to +/-0.5% here; if it is not, the finding is a threshold artefact and
# must be reported as such.
HANDLE_SLOPE_FLAT_PCT = float(os.environ.get("HANDLE_SLOPE_FLAT_PCT", "1.0"))
#
# CUP ROUNDING: bars whose low sits in the LOWER THIRD of (cup_high -
# cup_low). U-shape >= 5 such bars, V-shape <= 2. A U spends time at the
# bottom; a V touches once and leaves.
CUP_ROUNDING_U_MIN_BARS = int(os.environ.get("CUP_ROUNDING_U_MIN_BARS", "5"))
CUP_ROUNDING_V_MAX_BARS = int(os.environ.get("CUP_ROUNDING_V_MAX_BARS", "2"))
# Live default is first_valid, NOT best_quality.
#
# base_quality_quartile.q4 — the top bucket of the quality formula's own
# ranking — was sign-stable NEGATIVE across both halves of both 3-year runs
# (-0.698/-0.660 and -0.698/-0.616). best_quality is the mechanism that
# selects q4. The depth term was separately found to be scoring backwards
# (rating a 2% base above a 20% one, when 15-25% is the measured winning
# band) and has been rebuilt to band-scoring — but until a re-run shows the
# inversion is actually gone, the live path should not use the selector that
# produced it. best_quality stays available for A/B measurement.
# Live default: first_valid.
#
# Switched from first_valid_quality_gated at the 2026-09-13 freeze, BY THE
# AMENDED ADOPT RULE rather than by judgement:
#
#   "prefer the one with higher TOTAL R unless another beats by >0.05R
#    expectancy AND retains at least 90% of the fills"
#
#   first_valid                +52.1R total, +0.416R exp, 125 fills, 4/6 WF
#   first_valid_quality_gated  +51.2R total, +0.388R exp, 132 fills, 4/6 WF
#
# first_valid has the higher total R and nothing overrides it: the only
# rival, first_valid_quality_gated, is 0.028R LOWER on expectancy, so the
# >0.05R override clause cannot fire. Cost is 7 fewer fills over 3 years.
#
# (An earlier reading treated the 0.05R clause as a bar the total-R winner
# must itself clear. It is not — it is the condition under which a
# DIFFERENT strategy displaces the total-R winner.)
#
# All four strategies remain available via BASE_STRATEGY.
DEFAULT_BASE_STRATEGY = os.environ.get("BASE_STRATEGY", "first_valid")
FIXED_BASE_WINDOW = int(os.environ.get("FIXED_BASE_WINDOW", "45"))


def detect_base(df: pd.DataFrame, exclude_last: int = 1,
                strategy: str = DEFAULT_BASE_STRATEGY) -> Base:
    """
    The base is formed by the bars BEFORE the trigger.

    If the most recent bar is a breakout, it holds the highest high in the
    window — so including it would make the breakout bar itself the pivot,
    and the "pivot must sit in the older part of the base" rule would
    reject every setup we actually want. The trigger bar is excluded here
    and evaluated separately in detect_trigger().

    strategy:
      "best_quality" — scan every window, keep the one scoring highest on
        the composite formula. Byte-identical to the original function, but
        NO LONGER THE LIVE DEFAULT: its own top-rated quartile was the worst
        performer across both halves of both 3-year runs. Retained for A/B
        measurement against the alternatives below.
      "first_valid" (LIVE DEFAULT, via DEFAULT_BASE_STRATEGY) — scan
        windows shortest-to-longest (most recent base first) and take the
        FIRST one clearing every validity check, no
        quality ranking at all. Mirrors a trader taking the most recent
        legitimate base rather than grading twenty of them against a
        formula.
      "first_valid_quality_gated" (can reject with
        no_base_quality_below_floor, which is DISTINCT from
        no_base_no_windows: the former means windows were valid but all
        scored under FIRST_VALID_MIN_QUALITY, the latter that none were
        valid at all) — scan shortest-to-longest like
        first_valid, but skip windows scoring below FIRST_VALID_MIN_QUALITY
        and take the first that clears it. Sits between the other two:
        first_valid can accept a technically-valid but poor base purely
        because it appeared first, while best_quality's own top quartile
        measured worst. Earliest ACCEPTABLE rather than earliest or
        best-scoring.
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
        # 25-35% was sign-stable NEGATIVE across both halves of the 3-year
        # run, and the old 25% floor put the threshold exactly inside that
        # losing band — the detector was selecting for it.
        if PRIOR_MOVE_REJECT_LO <= prior_gain <= PRIOR_MOVE_REJECT_HI:
            _note("prior_move_near_floor")
            continue
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

        pattern = _classify(seg_high, seg_low, depth, lookback,
                            volumes[seg_start:])

        # Two patterns need the full frame (a prior base, or a flagpole
        # before the base), not just the base slice _classify receives, so
        # they are resolved here. Both are MORE specific than whatever
        # _classify returned, so they override it — a high tight flag would
        # otherwise be labelled flat_base and lose the thing that makes it
        # worth trading.
        if _is_high_tight_flag(df, seg_start, pivot, base_low, lookback):
            pattern = "high_tight_flag"
        elif _is_base_on_base(df, seg_start, base_low, pivot):
            pattern = "base_on_base"
        # Derived from the PATTERN, and computed AFTER the overrides above —
        # two edits landed out of order and left this unconditionally False,
        # so every base including genuine VCPs reported contracting=False and
        # the whole attribution dimension was dead.
        contracting = (pattern == "vcp")

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

        # Depth and duration score toward the spec's IDEAL BANDS, not
        # monotonically. This was "(1 - depth/35) * 30 — tighter is better",
        # which peaks at depth 0 and rates a 2% base (28/30) far above a 20%
        # base (13/30). Prompt 7 states the ideal depth is 15-25%, and the
        # engine's own backtest agrees with the spec rather than the formula:
        #
        #   15-25% depth -> +0.157R (n=224)
        #    6-15% depth -> -0.014R (n=401)
        #   under 6%     -> -0.230R (n=35)
        #
        # This matters more than a scoring tweak: `quality` is what
        # best_quality uses to CHOOSE which candidate window becomes the
        # base. A formula that prefers shallower is one plausible reason the
        # top-rated quartile (q4) has been the worst performer in every
        # backtest — it was selecting for the wrong thing, and selecting
        # hardest on exactly the trades it rated highest.
        depth_score = _band_score(depth, 15.0, 25.0, 35.0) * 30
        duration_score = _band_score(lookback, 25.0, 40.0, 120.0) * 10

        quality = (
            depth_score
            + max(0.0, 1.0 - dryup) * 25               # volume drying up
            + max(0.0, 1.0 - contraction) * 25         # ranges contracting
            + duration_score
            + min(prior_gain / 60, 1.0) * 10           # strength into the base
        )

        # Measured for cup_handle and ascending_base regardless of which
        # label won — a base rejected as a cup is exactly the population
        # the diagnosis needs.
        shape_diag: dict = {}
        if pattern in ("cup_handle", "rounding_bottom", "ascending_base",
                       "double_bottom", "consolidation"):
            try:
                if pattern == "ascending_base":
                    shape_diag = _ascending_base_diagnostics(
                        seg_high, seg_low, lookback)
                else:
                    shape_diag = _cup_handle_diagnostics(
                        seg_high, seg_low, lookback)
            except Exception as exc:          # diagnostics must never break a scan
                log.debug("shape_diag failed: %s", exc)
                shape_diag = {}

        candidate = Base(pattern, seg_start, pivot_idx, pivot, base_low, depth,
                         lookback, dryup, contraction, prior_gain, quality,
                         contracting, obv_rising, strategy_used=strategy,
                         shape_diag=shape_diag)

        if strategy == "first_valid":
            # Stop at the first legitimate window — no ranking against the
            # (now suspect) composite score.
            best = candidate
            break
        if strategy == "first_valid_quality_gated":
            if candidate.quality >= FIRST_VALID_MIN_QUALITY:
                best = candidate
                break
            # Counted. Without this the window was discarded silently and
            # the eventual failure surfaced as no_base_no_windows — which is
            # indistinguishable from "the loop never ran" and "nothing was
            # valid". A window that passed every validity check and failed
            # only the quality floor is a different diagnosis entirely.
            _note("quality_below_floor")
            continue
        if best is None or candidate.quality > best.quality:
            best = candidate

    if best is None:
        dominant = max(fail_counts, key=fail_counts.get) if fail_counts else "no_windows"
        raise Rejected("gate4", f"no_base_{dominant}", dict(fail_counts))
    return best


def _cup_handle_diagnostics(seg_high, seg_low, duration: int) -> dict:
    """
    Shape measurements for the C4 investigation. Computed for EVERY base
    that reaches the cup test, pass or fail, so the failing population can
    be characterised rather than only the accepted one.

    Nothing gates on these. cup_handle and ascending_base stay tradeable;
    this exists to find out WHY they lose -25R over three years.
    """
    out: dict = {}
    handle_len = max(5, min(duration // 4, 15))
    if duration - handle_len < 10 or len(seg_low) < duration:
        return out

    cup_portion_low = seg_low[:-handle_len]
    cup_portion_high = seg_high[:-handle_len]
    if len(cup_portion_low) == 0:
        return out

    cup_low_idx = int(np.argmin(cup_portion_low))
    cup_low = float(cup_portion_low[cup_low_idx])
    cup_high = float(cup_portion_high.max())
    handle_lows = seg_low[-handle_len:]
    handle_highs = seg_high[-handle_len:]

    out["cup_low_idx_in_window"] = cup_low_idx
    out["handle_start_idx"] = duration - handle_len
    out["handle_end_idx"] = duration - 1
    out["handle_length"] = handle_len

    if cup_high > cup_low > 0:
        # Handle slope, as a fraction of the handle's own range.
        h_range = float(handle_highs.max() - handle_lows.min())
        net = float(handle_lows[-1] - handle_lows[0])
        if h_range > 0:
            slope_pct = net / h_range * 100
            out["handle_slope_pct"] = round(slope_pct, 2)
            if slope_pct > HANDLE_SLOPE_FLAT_PCT:
                out["handle_slope"] = "rising"
            elif slope_pct < -HANDLE_SLOPE_FLAT_PCT:
                out["handle_slope"] = "falling"
            else:
                out["handle_slope"] = "flat"

        h_high = float(handle_highs.max())
        h_low = float(handle_lows.min())
        out["handle_depth_pct"] = round((h_high - h_low) / h_high * 100, 2) if h_high > 0 else None

        # Rounding: bars resting in the lower third of the cup's range.
        lower_third = cup_low + (cup_high - cup_low) / 3.0
        bars_low = int((cup_portion_low <= lower_third).sum())
        out["cup_rounding_bars"] = bars_low
        out["cup_shape"] = ("U" if bars_low >= CUP_ROUNDING_U_MIN_BARS
                            else "V" if bars_low <= CUP_ROUNDING_V_MAX_BARS
                            else "intermediate")
        out["cup_duration"] = duration - handle_len
        out["cup_depth_pct"] = round((cup_high - cup_low) / cup_high * 100, 2)

        # At least one higher low inside the cup, after its bottom.
        after = cup_portion_low[cup_low_idx + 1:]
        out["cup_has_higher_low"] = bool(len(after) >= 2 and any(
            after[i] > after[i - 1] for i in range(1, len(after))))

        # Right side vs left side — already enforced, recorded per trade.
        third = max(duration // 3, 3)
        lhs = float(seg_high[:third].max())
        rhs = float(seg_high[-third:].max())
        out["right_vs_left_pct"] = round((rhs / lhs - 1) * 100, 2) if lhs > 0 else None

    return out


def _ascending_base_diagnostics(seg_high, seg_low, duration: int) -> dict:
    """
    Shape measurements for ascending_base (Task 2.5). Same principle:
    measure, do not gate.
    """
    out: dict = {}
    span = max(2, duration // 12)
    troughs = _find_local_minima(seg_low, span)
    out["pullback_count"] = len(troughs)
    if len(troughs) < 2:
        return out

    lows = [float(seg_low[t]) for t in troughs]
    base_low = float(seg_low.min())
    out["first_pullback_low"] = round(lows[0], 2)
    out["last_pullback_low"] = round(lows[-1], 2)
    out["base_low"] = round(base_low, 2)
    out["first_pullback_above_base_low"] = bool(lows[0] > base_low * 1.001)
    out["has_higher_low_between"] = bool(
        len(lows) >= 3 and any(lows[i] > lows[i - 1] for i in range(1, len(lows) - 1)))
    if lows[0] > 0:
        rise = (lows[-1] / lows[0] - 1) * 100
        out["last_vs_first_pullback_pct"] = round(rise, 2)
        # "Materially higher" vs "roughly flat" — 2% of the first low.
        out["pullbacks_materially_rising"] = bool(rise > 2.0)
    return out


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


def _band_score(value: float, ideal_lo: float, ideal_hi: float,
                hard_max: float) -> float:
    """
    1.0 inside the ideal band, tapering to 0 outside it.

    Used where a spec gives an IDEAL RANGE rather than "more is better" —
    base depth (15-25%) and duration (5-8 weeks). A monotonic score cannot
    express "too little is also bad", which is precisely the error that made
    the old depth term rate a 2% base above a 20% one.
    """
    if value <= 0:
        return 0.0
    if ideal_lo <= value <= ideal_hi:
        return 1.0
    if value < ideal_lo:
        return max(0.0, value / ideal_lo)
    if value >= hard_max:
        return 0.0
    return max(0.0, 1.0 - (value - ideal_hi) / (hard_max - ideal_hi))


def _count_contractions(seg_high, seg_low, duration: int) -> list[float]:
    """
    Depths of each successive pullback inside the base, oldest first.

    Spec (Prompt 7) defines a VCP as >=2 contractions, each under 0.7x the
    prior one, with the final contraction tightest. The engine only ever
    computed a single first-third vs last-third range ratio and exposed it
    as a boolean `contracting` flag — that cannot tell a genuine stepwise
    volatility contraction from a base that merely happens to end quieter
    than it started, and VCP never existed as a pattern LABEL at all.
    """
    span = max(2, duration // 12)
    troughs = _find_local_minima(seg_low, span)
    if len(troughs) < 2:
        return []

    # Merge adjacent near-duplicate troughs (same reason as the H&S detector:
    # a discretized dip commonly registers two adjacent local minima).
    merged = [troughs[0]]
    for idx in troughs[1:]:
        if idx - merged[-1] <= span:
            if seg_low[idx] < seg_low[merged[-1]]:
                merged[-1] = idx
        else:
            merged.append(idx)

    depths = []
    for i, t in enumerate(merged):
        # Peak preceding this trough, back to the previous trough.
        start = merged[i - 1] if i else 0
        if t <= start:
            continue
        peak = float(seg_high[start:t + 1].max())
        low = float(seg_low[t])
        if peak > 0:
            depths.append((peak - low) / peak * 100)
    return depths


def _is_vcp(seg_high, seg_low, depth, duration: int) -> bool:
    """Spec: >=2 contractions, each < 0.7x the prior, final tightest (<8%)."""
    depths = _count_contractions(seg_high, seg_low, duration)
    if len(depths) < 2:
        return False
    for prev, nxt in zip(depths, depths[1:]):
        if prev <= 0 or nxt > prev * 0.7:
            return False
    return depths[-1] < VCP_FINAL_CONTRACTION_PCT


def _is_double_bottom(seg_high, seg_low, duration: int,
                      seg_vol=None) -> bool:
    """
    Spec: two lows within 2%, 4-12 weeks apart, VOLUME HIGHER on the second
    reversal. Previously fell into `consolidation` entirely.

    The volume rule is the part that distinguishes a real double bottom from
    two coincidental equal lows: the second test should attract more buying
    than the first, showing demand arriving rather than the level simply
    being touched again on apathy.
    """
    span = max(2, duration // 12)
    troughs = _find_local_minima(seg_low, span)
    if len(troughs) < 2:
        return False

    for i in range(len(troughs) - 1):
        for j in range(i + 1, len(troughs)):
            a, b = troughs[i], troughs[j]
            if b - a < max(5, duration // 6):
                continue                       # too close together in time
            low_a, low_b = float(seg_low[a]), float(seg_low[b])
            if low_a <= 0:
                continue
            if abs(low_b - low_a) / low_a > 0.02:
                continue                       # lows not within 2%
            mid_peak = float(seg_high[a:b + 1].max())
            # A real W needs a genuine recovery between the two feet.
            if mid_peak <= max(low_a, low_b) * 1.05:
                continue

            # Volume higher on the second reversal (spec). Compared over a
            # small window around each low so a single quiet bar does not
            # decide it.
            if seg_vol is not None and len(seg_vol) == len(seg_low):
                w = max(2, duration // 20)
                v1 = float(seg_vol[max(0, a - w):a + w + 1].mean())
                v2 = float(seg_vol[max(0, b - w):b + w + 1].mean())
                if v1 > 0 and v2 <= v1:
                    continue
            return True
    return False


def _is_rounding_bottom(seg_high, seg_low, depth, duration: int) -> bool:
    """
    Spec: a slow, smooth U over a long base, depth 20-40%, low in the middle.

    Distinguished from a cup by having NO handle — the right side runs
    straight up into the pivot without a final shallow pullback.
    """
    if not (20 <= depth <= 40) or duration < ROUNDING_MIN_SESSIONS:
        return False

    low_idx = int(seg_low.argmin())
    # Low must sit in the middle half of the base, not at either edge.
    if not (duration * 0.25 <= low_idx <= duration * 0.75):
        return False

    # Smoothness: both sides should descend/ascend without a deep spike
    # against the trend. Measured as the worst counter-move on each side
    # relative to that side's own range.
    left, right = seg_low[:low_idx + 1], seg_low[low_idx:]
    if len(left) < 3 or len(right) < 3:
        return False
    for side in (left, right[::-1]):
        rng = float(side.max() - side.min())
        if rng <= 0:
            return False
        worst = 0.0
        running = float(side[0])
        for v in side:
            worst = max(worst, float(v) - running) if v > running else worst
            running = min(running, float(v))
        if worst / rng > 0.4:
            return False
    return True


def _is_high_tight_flag(df: pd.DataFrame, seg_start: int, pivot: float,
                        base_low: float, duration: int) -> bool:
    """
    Spec: flagpole +100% in 4-8 weeks, flag 3-5 weeks retracing under 25%.

    The rarest and most explosive of O'Neil's patterns, and a genuinely
    different shape from the ordinary flag already detected: that one wants
    a 20% pole over 20 sessions, this demands a DOUBLE over 20-40 sessions.
    Treating them as one pattern would bury the rare, powerful case inside
    the common one.
    """
    if not (HTF_FLAG_MIN_SESSIONS <= duration <= HTF_FLAG_MAX_SESSIONS):
        return False
    if pivot <= 0:
        return False

    retrace = (pivot - base_low) / pivot * 100
    if retrace > HTF_MAX_RETRACE_PCT:
        return False

    closes = df["close"].values
    pole_start = max(0, seg_start - HTF_POLE_MAX_SESSIONS)
    pole_window = closes[pole_start:seg_start + 1]
    if len(pole_window) < HTF_POLE_MIN_SESSIONS:
        return False

    pole_gain = (float(pole_window[-1]) / float(pole_window.min()) - 1) * 100
    return pole_gain >= HTF_MIN_POLE_GAIN_PCT


def _is_base_on_base(df: pd.DataFrame, seg_start: int, base_low: float,
                     pivot: float) -> bool:
    """
    Spec: a base forming on top of a prior base — consolidation of prior
    gains rather than a fresh advance. Very bullish continuation.

    Detected as: a PRIOR consolidation immediately before this one, whose
    own high sits at or below this base's low. That is what "on top of"
    means structurally — the new base has not given back the prior one's
    range.
    """
    if seg_start < BASE_ON_BASE_LOOKBACK + 10 or base_low <= 0:
        return False

    prior = df.iloc[seg_start - BASE_ON_BASE_LOOKBACK:seg_start]
    if len(prior) < 15:
        return False

    prior_high = float(prior["high"].max())
    prior_low = float(prior["low"].min())
    if prior_high <= 0 or prior_low <= 0:
        return False

    prior_depth = (prior_high - prior_low) / prior_high * 100
    # The prior stretch must itself have been a consolidation, not a run.
    if prior_depth > 30:
        return False

    # This base sits ON TOP of it: the new base's floor is at or above the
    # prior base's high (allowing a small overlap).
    return base_low >= prior_high * 0.97 and pivot > prior_high


def _is_ascending_triangle(seg_high, seg_low, duration: int,
                           highs_first: float, highs_last: float,
                           lows_first: float, lows_last: float) -> bool:
    """
    Spec requires at least 2 touches on the flat top AND 2 rising lows —
    a touch count, not a shape approximation.

    The previous test compared only the first third's extremes against the
    last third's: flat_resistance if those two highs were within 3%, and
    rising_lows if the last third's low exceeded the first's. That passes
    on shapes with no triangle in them at all — a single spike early and a
    single spike late, with nothing between, satisfies it. Counting actual
    touches of the resistance line is what distinguishes a real triangle
    from two coincidental highs.
    """
    resistance = float(seg_high.max())
    if resistance <= 0:
        return False

    # A "touch" is any bar reaching within 1.5% of the resistance line.
    touches = int((seg_high >= resistance * 0.985).sum())
    if touches < 2:
        return False

    # Rising lows: split the base into halves and require the later half's
    # low to sit meaningfully above the earlier half's, with at least two
    # distinct swing lows forming the rising trendline.
    if lows_first <= 0 or lows_last <= lows_first * 1.02:
        return False

    # _find_local_minima returns ALL local minima, rising or falling. The
    # old check only counted them — so a DESCENDING series of troughs
    # satisfied "ascending triangle". Each trough must actually be higher
    # than the one before it.
    span = max(2, duration // 12)
    troughs = _find_local_minima(seg_low, span)
    if len(troughs) < 2:
        return False
    if not all(seg_low[troughs[i + 1]] > seg_low[troughs[i]]
               for i in range(len(troughs) - 1)):
        return False

    # Flat top: the resistance must genuinely be flat, not sloping.
    return abs(highs_last - highs_first) / highs_first < 0.03 if highs_first else False


def _classify(seg_high, seg_low, depth, duration,
              seg_vol=None) -> str:
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
    # Ordered most-specific first. Each test below is a harder claim than the
    # ones after it, so it gets first refusal — the same precedence rule that
    # had to be applied when a genuine cup was being swallowed by the looser
    # ascending-triangle test.
    # VCP checked BEFORE flat_base. A genuine VCP with depth <=15% was
    # being labelled flat_base and losing its identity — and the spec's
    # tightest final contraction (<8%) can occur in shallower bases too.
    if _is_vcp(seg_high, seg_low, depth, duration):
        # A real stepwise volatility contraction: >=2 pullbacks, each under
        # 0.7x the prior. Previously VCP existed only as a boolean flag
        # derived from a single first-third vs last-third range ratio, and
        # never as a pattern label — so it could not be measured at all.
        return "vcp"
    if depth <= 15:
        return "flat_base"
    if _is_inverse_head_shoulders(seg_high, seg_low, duration):
        return "inverse_head_shoulders"
    if _is_double_bottom(seg_high, seg_low, duration, seg_vol):
        return "double_bottom"
    if _is_cup_and_handle(seg_high, seg_low, duration, highs_first, highs_last):
        return "cup_handle"
    if _is_rounding_bottom(seg_high, seg_low, depth, duration):
        # Checked AFTER cup_handle: a rounding bottom is a cup without a
        # handle, so anything with a valid handle should be labelled a cup.
        return "rounding_bottom"
    if flat_resistance and rising_lows:
        return "asc_triangle" if _is_ascending_triangle(
            seg_high, seg_low, duration, highs_first, highs_last,
            lows_first, lows_last) else "consolidation"
    if lows_last > lows_first:
        # A genuine staircase of higher lows that fails the cup criteria
        # above — O'Neil's fourth classic base type, previously folded
        # into the cup_handle mislabel with no distinct identity of its own.
        return "ascending_base"
    return "consolidation"


# ---------------------------------------------------------------------
# Gate 5 — trigger
# ---------------------------------------------------------------------
def detect_trigger(df: pd.DataFrame, base: Base,
                   last_bar_incomplete: bool = False) -> str:
    """
    Classify the trigger on the most recent bar: breakout, pullback, or armed.

    INTRADAY BAR POLICY — deliberate, and not a hybrid accident.

    When last_bar_incomplete is True, the volume test reads the FORMING bar
    while the RSI test reads the last CLOSED bar. That looks inconsistent
    but follows one rule: a gate may read a forming bar only if the quantity
    it tests is MONOTONIC within the bar.

      * Volume only accumulates. A forming bar already at 2x its average
        will still be at or above 2x when the session closes, so a pass
        cannot later become a fail. Reading it early is safe and is
        genuinely current information.

      * RSI is not monotonic. It can read 75 at 14:00 and close at 62. A
        gate acting on that decides on a number that never existed at any
        close — so it reads the last confirmed bar instead.

    The alternative of reading the closed bar for BOTH would test yesterday's
    volume against a breakout that happened today, which is meaningless. The
    alternative of reading the forming bar for both re-admits the RSI
    problem that provisional=True warns about but does not prevent.

    This is the same monotonicity argument weekly_volume_surge() uses for
    partial weeks, applied at the daily scale.

    SCOPE: the monotonicity rule governs FILTERS — quantities asked "is this
    stock good enough", such as RSI and volume. It does NOT govern
    TRIGGER-SHAPE checks (close above pivot, bullish candle, close position
    in range, exhaustion wick). Those define what the current bar IS, so
    they are inherently same-bar and continue to read the forming bar. A
    trigger-shape check on the previous bar would be asking whether
    YESTERDAY broke out, which is a different question entirely.
    """
    last = df.iloc[-1]
    rng = float(last["high"] - last["low"])
    atr = float(last["atr14"])
    vol50 = float(last["vol50"]) if pd.notna(last["vol50"]) else 0.0

    # Breakout
    if last["close"] > base.pivot:
        # A flag breakout carries a higher bar than an ordinary base
        # breakout (spec: 1.8x vs 1.5x). A flag is a brief pause inside an
        # already-running move, so the resumption should show more
        # conviction than a breakout from a long, quiet base.
        required_mult = (FLAG_BREAKOUT_VOL_MULT
                         if base.pattern == "flag_pennant"
                         else BREAKOUT_VOL_MULT)
        if vol50 <= 0 or last["volume"] < vol50 * required_mult:
            raise Rejected("gate5", "breakout_without_volume",
                           {"volume_mult": round(float(last["volume"]) / vol50, 2) if vol50 else None,
                            "required": required_mult,
                            "pattern": base.pattern})

        # breakout_rsi_70_80 was the ONLY sub-signal positive in both halves
        # (+0.281R / +0.513R); breakout_other_rsi was sign-stable negative.
        # Without this the engine cannot tell them apart — every breakout
        # was undifferentiated.
        if BREAKOUT_RSI_GATE:
            # Same reasoning as the retest path: gate on a closed bar's RSI.
            use_closed = last_bar_incomplete and len(df) >= 2
            rsi_bar = df.iloc[-2] if use_closed else df.iloc[-1]
            rsi = float(rsi_bar["rsi14"]) if pd.notna(rsi_bar.get("rsi14")) else None
            if rsi is None or not (BREAKOUT_RSI_LO <= rsi <= BREAKOUT_RSI_HI):
                raise Rejected("gate5", "breakout_rsi_outside_band",
                               {"rsi14": round(rsi, 1) if rsi is not None else None,
                                "band": [BREAKOUT_RSI_LO, BREAKOUT_RSI_HI],
                                "used_closed_bar": use_closed})

        if DELIVERY_GATE:
            dp = last.get("delivery_pct")
            dp = float(dp) if dp is not None and pd.notna(dp) else None
            if dp is None or dp < DELIVERY_MIN_PCT:
                raise Rejected("gate5", "breakout_delivery_too_low",
                               {"delivery_pct": dp, "required": DELIVERY_MIN_PCT})
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
            # Testing a REAL level, not an arbitrary one-day comparison.
            #
            # The prior test was `last.low <= prev.low` — comparing only to
            # yesterday. A multi-day pullback's actual low often printed
            # several sessions earlier; a single-day comparison can pass on
            # a random daily wiggle that never tested any real support, or
            # miss the genuine reversal because it happened before "prev".
            # The confirmed-swing-low finder already used for stop
            # selection gives a real reference: today's low should sit at
            # or just above the most recent one, showing an actual retest,
            # not an arbitrary undercut of one prior close.
            recent_lows = swing_lows(df.iloc[base.start_idx:], span=2)
            reference_low = (float(df["low"].iloc[base.start_idx + recent_lows[-1]])
                             if recent_lows else None)

            prev = df.iloc[-2]
            testing_real_level = (
                reference_low is not None and float(last["low"]) <= reference_low * 1.02
                if reference_low is not None
                else float(last["low"]) <= float(prev["low"])   # no confirmed low yet — fall back
            )

            bullish = (
                last["close"] > last["open"]
                and last["close"] > (prev["high"] + prev["low"]) / 2
                and testing_real_level
            )
            # Requiring volume BELOW average on the reversal bar was
            # backward: light volume during the drift down is constructive,
            # but on the actual reversal day, renewed demand showing up as
            # volume is a BETTER signal than a quiet bounce, not a worse
            # one. That requirement previously excluded exactly the
            # stronger pullbacks — reversals confirmed by real buying.
            if bullish:
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

    # tight_0_2pct was sign-stable negative, wide_2_4pct positive. The old
    # 0-4% window contained both, so half of every armed signal came from
    # the measurably losing sub-band.
    if ARMED_MIN_DISTANCE_PCT <= distance_pct <= ARMED_MAX_DISTANCE_PCT:
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

        # Today's OWN candle shape — position checks above establish WHERE
        # the close sits; this checks what happened DURING the day to get
        # there. Tested directly: a day that rallied 3% intraday toward the
        # pivot and gave nearly all of it back, closing red, still cleared
        # every position check above and was armed with no warning. The
        # breakout branch already screens its trigger bar this way
        # (weak_close_in_range, exhaustion_candle); the armed branch never
        # examines its own bar at all.
        if atr > 0 and rng > 0.5 * atr:
            upper_wick_pct = (float(last["high"]) - close) / rng if rng > 0 else 0.0
            closed_red = close < float(last["open"])
            if upper_wick_pct > 0.5 and closed_red:
                raise Rejected("gate5", "distribution_candle",
                               {"upper_wick_pct": round(upper_wick_pct * 100, 1)})

        return "armed"

    if 0 <= distance_pct < ARMED_MIN_DISTANCE_PCT:
        raise Rejected("gate5", "armed_distance_too_tight",
                       {"distance_to_pivot_pct": round(distance_pct, 2),
                        "min_required": ARMED_MIN_DISTANCE_PCT})

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
    if setup_type.startswith("pullback") or setup_type == "breakout_retest":
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
    elif setup_type.startswith("pullback") or setup_type == "breakout_retest":
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


RETEST_LOOKBACK_SESSIONS = int(os.environ.get("RETEST_LOOKBACK_SESSIONS", "15"))
RETEST_MAX_UNDERCUT_PCT = float(os.environ.get("RETEST_MAX_UNDERCUT_PCT", "3.0"))


def detect_breakout_retest(df: pd.DataFrame, exclude_last: int = 1,
                           base_strategy: str = DEFAULT_BASE_STRATEGY,
                           last_bar_incomplete: bool = False) -> Base:
    """
    Base -> genuine breakout above the pivot -> controlled pullback BACK
    down to that same level -> reversal today. Anchored to the pivot
    itself, not a moving average — a level that has already proven itself
    as resistance and is now being tested as support.

    Needs its OWN, wider exclusion window, and this is not optional.
    detect_base's standard search only holds out the trigger bar
    (exclude_last=1), so a breakout-then-pullback sequence spanning several
    bars would otherwise sit INSIDE the base search itself. Tested
    directly: the window search found a short, tight recent candidate that
    already included the breakout's own high as its pivot, which then
    failed as "pivot_too_recent" — the search had absorbed the very
    breakout this pattern needs to have happened BEFORE the base's pivot
    was set. Excluding the whole retest window up front, then locating the
    original base only in what remains, is what keeps the two phases from
    corrupting each other.
    """
    df_visible = df.iloc[:-exclude_last] if exclude_last else df
    n = len(df_visible)
    # Full PRIOR_UPTREND_WINDOW, not half: detect_base needs the whole
    # window of prior closes, so a // 2 threshold let calls through that
    # then failed inside detect_base with prior_window_too_short — a
    # misleading reason code pointing at the wrong cause.
    if n < RETEST_LOOKBACK_SESSIONS + BASE_MIN_SESSIONS + PRIOR_UPTREND_WINDOW:
        raise Rejected("gate4", "retest_insufficient_history")

    original = df_visible.iloc[:-RETEST_LOOKBACK_SESSIONS]
    # Strategy threaded through. This called detect_base with no strategy,
    # inheriting its best_quality default — so the retest detector kept
    # using the one selector the code documents as producing the q4
    # inversion, even after the live path moved to first_valid.
    base = detect_base(original, exclude_last=0, strategy=base_strategy)

    retest_window = df_visible.iloc[len(original):]
    if len(retest_window) < 2:
        raise Rejected("gate4", "retest_window_too_short")

    broke_out = retest_window["close"] > base.pivot
    if not broke_out.any():
        raise Rejected("gate4", "retest_no_breakout_found")

    first_breakout_pos = int(broke_out.to_numpy().nonzero()[0][0])
    since_breakout = retest_window.iloc[first_breakout_pos + 1:]
    if len(since_breakout) < 1:
        raise Rejected("gate4", "retest_no_pullback_yet")

    # Volume is what separates a genuine retest from a failing breakout, and
    # this detector had NO volume logic at all — verified by inspection. Two
    # separate requirements, both from the strategy spec:
    #
    #   1. The breakout itself must have carried real demand (>1.5x the
    #      50-day average). A breakout on ordinary volume is the exact
    #      "false breakout" case this whole pattern exists to avoid — and
    #      without this check, the retest logic was happily confirming
    #      retests of breakouts that never had conviction behind them.
    #   2. The pullback must be QUIETER than the breakout. A retest on
    #      volume equal to or above the breakout is distribution — sellers
    #      are hitting it, not a healthy pause.
    breakout_bar = retest_window.iloc[first_breakout_pos]
    breakout_vol = float(breakout_bar["volume"])
    bo_vol50 = float(breakout_bar["vol50"]) if pd.notna(breakout_bar.get("vol50")) else 0.0

    if bo_vol50 <= 0 or breakout_vol < bo_vol50 * BREAKOUT_VOL_MULT:
        raise Rejected("gate4", "retest_breakout_without_volume",
                       {"volume_mult": round(breakout_vol / bo_vol50, 2)
                        if bo_vol50 else None,
                        "required": BREAKOUT_VOL_MULT})

    # Compared against the retest window's PEAK, not its mean.
    #
    # The mean masks exactly what this check exists to catch: traced on a
    # constructed case, retest bars of [1M, 1M, 3M, 3M, 3M] averaged 2.2M
    # against a 2.5M breakout and passed — while three of the five bars ran
    # 20% ABOVE the breakout's own volume. That is sellers hitting the
    # retest, which is precisely the distribution signature the spec's
    # "retest volume < breakout volume" rule is meant to reject. A single
    # heavy bar in the pullback invalidates the retest regardless of how
    # quiet the others were.
    retest_peak_vol = float(since_breakout["volume"].max())
    if breakout_vol > 0 and retest_peak_vol >= breakout_vol:
        raise Rejected("gate4", "retest_volume_not_lower",
                       {"retest_peak_vol": int(retest_peak_vol),
                        "breakout_vol": int(breakout_vol)})

    # A shallow undercut is the classic, often stronger version of this
    # setup — a brief shakeout below the old resistance before reclaiming
    # it. Anything deeper is a failed retest, not a valid one.
    retest_low = float(since_breakout["low"].min())
    if retest_low < base.pivot * (1 - RETEST_MAX_UNDERCUT_PCT / 100):
        raise Rejected("gate4", "retest_broke_down",
                       {"retest_low": round(retest_low, 2),
                        "pivot": round(base.pivot, 2)})

    last = df.iloc[-1]     # the actual trigger bar, not df_visible's tail
    close = float(last["close"])
    near_pivot = base.pivot * (1 - RETEST_MAX_UNDERCUT_PCT / 100) <= close <= base.pivot * 1.02
    reversed_here = close > float(last["open"]) and float(last["low"]) <= base.pivot * 1.01

    if not (near_pivot and reversed_here):
        raise Rejected("gate5", "retest_not_confirmed_today",
                       {"close": round(close, 2), "pivot": round(base.pivot, 2)})

    # A retest is a breakout variant, but build_setup calls this BEFORE
    # detect_trigger and returns early on success — so retests were skipping
    # the RSI band entirely, the one sub-signal measured positive in both
    # halves. Separate reason code so attribution can tell the two paths
    # apart.
    if BREAKOUT_RSI_GATE:
        # Read the last CLOSED bar intraday. RSI on a forming bar moves
        # through the session, so a 14:00 reading can sit inside the band
        # and finish outside it — the gate would be deciding on a value
        # that never existed at any close. provisional=True warns the
        # consumer, but a gate should not act on an unconfirmed number.
        use_closed = last_bar_incomplete and len(df) >= 2
        rsi_bar = df.iloc[-2] if use_closed else df.iloc[-1]
        rsi = float(rsi_bar["rsi14"]) if pd.notna(rsi_bar.get("rsi14")) else None
        if rsi is None or not (BREAKOUT_RSI_LO <= rsi <= BREAKOUT_RSI_HI):
            raise Rejected("gate5", "retest_rsi_outside_band",
                           {"rsi14": round(rsi, 1) if rsi is not None else None,
                            "band": [BREAKOUT_RSI_LO, BREAKOUT_RSI_HI],
                            # Reports what was ACTUALLY read, not what was
                            # requested — the two diverge if len(df) < 2.
                            "used_closed_bar": use_closed})

    # The base's SHAPE is whatever detect_base found (flat_base, cup_handle,
    # VCP, ...) — retest describes how we are entering, not what the
    # structure looked like. Returning the original base unchanged keeps
    # that distinction intact.
    return base


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

        # Spec: a flag retracing more than half its own flagpole is a
        # reversal, not a pause. Depth was previously bounded only as a
        # percentage of price (FLAG_MAX_DEPTH_PCT), which says nothing
        # about how much of the preceding MOVE has been given back — a 12%
        # dip is trivial after a 60% pole and fatal after a 20% one.
        pole_low = float(pole_window.min())
        pole_height = pivot - pole_low
        if pole_height > 0:
            retrace_pct = (pivot - base_low) / pole_height * 100
            if retrace_pct > FLAG_MAX_POLE_RETRACE_PCT:
                _note("flag_retraced_too_much")
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

        # contracting=False: that field now means "is a VCP" everywhere
        # else. A flag is a momentum pause, not a Minervini contraction, and
        # marking it True put every flag into the VCP bucket for attribution.
        #
        # strategy_used is intentionally None here — flag detection is not a
        # base strategy, so there is nothing to record. A diagnostic seeing
        # strategy_requested=<x> alongside strategy_used=None should read
        # this as "the flag fallback fired", not "the parameter was lost".
        candidate = Base("flag_pennant", flag_start, pivot_idx, pivot, base_low,
                         depth, flag_len, dryup, dryup, pole_gain, quality,
                         False, False)
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


# Counts Minervini passes where the RS component was skipped for want of a
# percentile. Drained by the scan into its summary.
# Module-level, so it accumulates across EVERY call in the process — not
# per scan. Two faults followed from that, and 795 on a 500-symbol scan is
# the visible symptom of both:
#
#  1. Never reset at the START of a scan. It was only cleared when the
#     summary READ it, so a scan that raised before reaching the summary
#     left its count to be inherited by the next one. The intraday slots
#     run hourly, so a day's counts compound.
#
#  2. Counted per CALL, not per SYMBOL. build_setup runs more than once
#     for the same symbol — the breakout-retest path calls it, the
#     standard path calls it again, and the backtest additionally calls it
#     once per alternative base strategy. A symbol reaching Minervini
#     three ways counted three times.
#
# Now a SET of symbols, reset explicitly at scan start. The count can no
# longer exceed the universe size, which is what makes it readable.
_MINERVINI_NO_RS: set = set()


def minervini_no_rs_reset() -> None:
    """Called at scan start. Without this the counter is cumulative across
    every scan in the process lifetime."""
    _MINERVINI_NO_RS.clear()


def minervini_no_rs_count(reset: bool = False) -> int:
    """
    Distinct symbols that cleared Minervini with the RS check SKIPPED for
    want of a percentile.

    reset defaults to False now: reading a diagnostic should not mutate it.
    Resetting on read meant the value depended on whether anything else had
    read it first.
    """
    n = len(_MINERVINI_NO_RS)
    if reset:
        _MINERVINI_NO_RS.clear()
    return n


# Set by build_setup so _minervini_ok can attribute the skip to a symbol
# without changing its signature everywhere it is called.
_minervini_symbol_hint: list = ["?"]


def _minervini_ok(last, rs_rank_pct: float | None) -> tuple[bool, str | None]:
    """
    All 8 Minervini trend-template rules, as a GATE.

    rs_rank_pct is a PERCENTILE (0-100) against the universe — Minervini's
    "RS rating > 70" means top 30% of all stocks.

    This previously received rs126, which relative_strength() returns as
    OUTPERFORMANCE VS THE INDEX IN PERCENTAGE POINTS. Comparing that to a
    floor of 70 demanded a stock beat the Nifty by 70 points over six
    months — a silent over-filter that would reject nearly everything. The
    units were wrong, not the threshold.

    When no percentile is available the RS check is SKIPPED rather than
    failed, so a missing input never masquerades as a rule violation.
    """
    # NaN guard first: comparing against NaN yields False, which would
    # report a genuine-looking rule failure when the real cause is a stock
    # with insufficient history.
    required = ["sma50", "sma150", "sma200", "sma200_slope25", "low52", "high52"]
    for field in required:
        value = last.get(field)
        if value is None or pd.isna(value):
            return False, f"missing_{field}"

    checks = {
        "close_above_150": last["close"] > last["sma150"],
        "close_above_200": last["close"] > last["sma200"],
        "150_above_200": last["sma150"] > last["sma200"],
        "200_rising": last["sma200_slope25"] > 0,
        "50_above_150": last["sma50"] > last["sma150"],
        "50_above_200": last["sma50"] > last["sma200"],
        "close_above_50": last["close"] > last["sma50"],
        "30pct_above_52w_low": last["close"] >= float(last["low52"]) * 1.30,
        "within_25pct_of_52w_high": last["close"] >= float(last["high52"]) * 0.75,
    }
    # Skipped, not failed, when the percentile is unavailable — a missing
    # input should not masquerade as a rule violation. The consequence is
    # asymmetric though: freshly-listed names without 127 bars of history
    # bypass the RS component entirely. Reported so a suspiciously high
    # pass-rate on new listings is traceable rather than mysterious.
    rs_skipped = rs_rank_pct is None
    if not rs_skipped:
        checks["rs_rank_above_floor"] = rs_rank_pct >= MINERVINI_RS_FLOOR

    for name, ok in checks.items():
        if not ok:
            return False, name

    # Counted AFTER the checks loop. It previously incremented at the point
    # the RS check was skipped — before any rule had been evaluated — so
    # every stock that later failed on close_above_50 or any other rule was
    # counted as a "pass without RS". The summary key promises passes; it
    # was reporting arrivals.
    if rs_skipped:
        # Symbol-keyed: build_setup can run several times for one symbol
        # (retest path, standard path, per-strategy in the backtest).
        _MINERVINI_NO_RS.add(_minervini_symbol_hint[0])
    return True, None


def build_setup(symbol: str, df: pd.DataFrame, rs63: float | None,
                rs126: float | None, snap=None, transition: bool = False,
                last_bar_incomplete: bool = False,
                base_strategy: str = DEFAULT_BASE_STRATEGY,
                rs_rank_pct: float | None = None) -> Setup:
    """
    transition=True applies the Stage 1->2 profile: a longer base and a
    heavier volume break. The trend filter is looser on that path, so the
    price-action evidence has to be stronger to compensate. Loosening one
    test without tightening another is how a second path becomes a back
    door.

    base_strategy is measurement-only — see detect_base. The live engine
    never passes anything but the default.
    """
    last_bar = df.iloc[-1]

    # Gates run BEFORE any detection — a stock failing the trend template
    # should never reach pattern detection at all, and running detection
    # first wastes the work and muddies attribution.
    #
    # RS floor: rs_quintile q4 was sign-stable negative, and RS was computed
    # on every signal but gated on nowhere.
    # `or`, not `and`: rejecting only when BOTH metrics were weak let a
    # setup through on one strong timeframe while the other sat far below
    # floor. The bottom RS quintile was sign-stable negative.
    if RS_MIN_GATE:
        if ((rs63 is None or rs63 < RS63_FLOOR)
                or (rs126 is None or rs126 < RS126_FLOOR)):
            raise Rejected("gate3", "rs_too_low",
                           {"rs63": rs63, "rs126": rs126,
                            "floors": [RS63_FLOOR, RS126_FLOOR]})

    if MINERVINI_GATE:
        # rs_rank_pct is a true universe percentile, computed cross-
        # sectionally by the caller. rs126 is NOT interchangeable with it.
        ok, failed = _minervini_ok(last_bar, rs_rank_pct)
        if not ok:
            raise Rejected("gate3", "minervini_template_failed",
                           {"failed_rule": failed})
    # Breakout retest is tried FIRST, not as a fallback after the standard
    retest_base: Base | None = None
    setup_type: str | None = None

    # search. It needs its own wider exclusion window from the very start —
    # trying it only after detect_base has already run would be too late,
    # since detect_base's own search would already have looked at (and
    # potentially been corrupted by) the retest bars themselves. Skipped
    # entirely under the Stage 1->2 transition profile: retest presupposes
    # an ALREADY-confirmed breakout from an established base, which is the
    # opposite premise of a stock still emerging from Stage 1.
    if not transition:
        # Detection ONLY here — no levels, no scoring, no early return.
        #
        # The retest branch used to build levels, score, and return a Setup
        # inline. That early return skipped every gate defined below it:
        # NEGATIVE_PATTERNS and OBV_GATE never ran for retests, so a retest
        # on a banned pattern fired normally. Detecting here and falling
        # through to one shared tail removes the whole class of bug rather
        # than re-adding each gate in a second place.
        try:
            retest_base = detect_breakout_retest(
                df, base_strategy=base_strategy,
                last_bar_incomplete=last_bar_incomplete)
        except Rejected:
            retest_base = None

    if retest_base is not None:
        base = retest_base
        setup_type = "breakout_retest"
    else:
        try:
            base = detect_base(df, strategy=base_strategy)
        except Rejected:
            # The flag fallback is blocked only for fixed_window, which
            # exists purely to measure ONE lookback and would be corrupted
            # by a fallback firing underneath it.
            #
            # This previously read `if base_strategy != "best_quality"`,
            # written when best_quality was the default. Once first_valid
            # became the live default that condition inverted: the live path
            # started raising instead of falling back, and every flag and
            # pennant setup was silently lost.
            # `or transition`: a flag can never be valid for the Stage 1->2
            # path, so running the detector there produced a flag_pennant
            # that was immediately rejected below. Skipping it outright is
            # cheaper and states the intent at the point of decision.
            if base_strategy == "fixed_window" or transition:
                raise
            base = detect_flag_pennant(df)

        if transition:
            if base.duration < TRANSITION_MIN_BASE_SESSIONS:
                raise Rejected("gate4", "transition_base_too_short",
                               {"duration": base.duration,
                                "required": TRANSITION_MIN_BASE_SESSIONS})
            last = df.iloc[-1]
            vol50 = float(last["vol50"]) if pd.notna(last["vol50"]) else 0.0
            if last["close"] > base.pivot:
                # Weinstein's breakout rule is a WEEKLY volume expansion
                # (>2x the 50-week average), not a daily one. Daily is kept
                # as a floor alongside it so the breakout bar itself shows
                # real demand rather than riding an earlier surge.
                if vol50 <= 0 or last["volume"] < vol50 * TRANSITION_VOL_MULT:
                    raise Rejected("gate5", "transition_volume_insufficient",
                                   {"required_mult": TRANSITION_VOL_MULT,
                                    "actual": round(float(last["volume"]) / vol50, 2)
                                    if vol50 else None})

                if WEEKLY_VOL_CHECK_ENABLED:
                    weekly_ok, weekly_mult = weekly_volume_surge(df, WEEKLY_VOL_MULT)
                    if not weekly_ok:
                        raise Rejected("gate5",
                                       "transition_weekly_volume_insufficient",
                                       {"required_weekly_mult": WEEKLY_VOL_MULT,
                                        "actual_weekly_mult": weekly_mult})

    # --- shared gates: both paths reach these ---------------------------
    #
    # Pattern rejection runs BEFORE trigger detection and level derivation:
    # there is no point finding a trigger on a banned pattern, and the
    # rejection reason is cleaner without levels attached.
    if REJECT_NEGATIVE_PATTERNS and base.pattern in NEGATIVE_PATTERNS:
        raise Rejected("gate6", "pattern_currently_negative",
                       {"pattern": base.pattern})

    if OBV_GATE and base.pattern in ("vcp", "flat_base", "ascending_base"):
        if not base.obv_rising:
            raise Rejected("gate5", "obv_not_rising", {"pattern": base.pattern})

    if setup_type is None:
        setup_type = detect_trigger(df, base, last_bar_incomplete)
        if transition:
            setup_type = f"{setup_type}_transition"
    levels = derive_levels(df, base, setup_type, last_bar_incomplete)
    total, breakdown = score_setup(df, base, setup_type, levels, rs63, rs126, snap)

    log.debug(
        "strategy_trace symbol=%s requested=%s used=%s "
        "base_start=%s base_duration=%s base_quality=%.2f",
        symbol, base_strategy, base.strategy_used,
        base.start_idx, base.duration, base.quality,
    )

    # Entry-timing diagnostics. The cup_handle evidence showed MFE
    # averaging 0.4R across 17 trades — they go down immediately rather
    # than rallying and stopping out, which is an entry-timing signature,
    # not a stop-placement one. Recorded for EVERY setup so the failing
    # population can be compared against the working one.
    _last = df.iloc[-1]
    _vol50 = float(_last["vol50"]) if pd.notna(_last.get("vol50")) else 0.0
    _ema20 = float(_last["ema20"]) if pd.notna(_last.get("ema20")) else None
    _rng = float(_last["high"]) - float(_last["low"])
    trigger_diag = {
        "trigger_volume_vs_50d": (round(float(_last["volume"]) / _vol50, 2)
                                  if _vol50 > 0 else None),
        "pct_above_ema20_at_entry": (
            round((float(_last["close"]) / _ema20 - 1) * 100, 2)
            if _ema20 else None),
        # 0 = closed on its low, 1 = closed on its high. A breakout closing
        # in the lower half of its own bar is being sold into.
        "trigger_close_position_in_range": (
            round((float(_last["close"]) - float(_last["low"])) / _rng, 3)
            if _rng > 0 else None),
    }

    extension = extension_metrics(df, levels["entry"])

    # Levels are never provisional intraday: with last_bar_incomplete set,
    # every one is drawn from a closed bar. What IS unconfirmed is whether
    # the trigger holds to the close — a breakout can finish in the lower
    # half of its range, or the volume pace can fade. So this flags an
    # unconfirmed TRIGGER, not an unreliable level, which is what the
    # earlier "provisional" naming wrongly implied.
    # Nothing gated on the score before this: a 20/100 setup was published
    # identically to an 80/100 one.
    if total < SETUP_HARD_FLOOR:
        raise Rejected("gate8", "score_below_hard_floor",
                       {"score": total, "floor": SETUP_HARD_FLOOR})

    provisional = last_bar_incomplete and not setup_type.startswith("armed")

    return Setup(
        symbol=symbol, setup_type=setup_type, pattern=base.pattern,
        entry=levels["entry"], stop=levels["stop"], t1=levels["t1"], t2=levels["t2"],
        r_multiple_t1=levels["r_multiple_t1"], base=base,
        stop_basis=levels["stop_basis"], t1_basis=levels["t1_basis"],
        t2_basis=levels.get("t2_basis"),
        score_total=total, score_breakdown=breakdown,
        extension=extension, provisional=provisional,
        strategy_requested=base_strategy,
        trigger_diag=trigger_diag,
        notes=[f"base {base.duration}d, depth {base.depth_pct:.1f}%"
               + (", contracting" if base.contracting else ""),
               f"stop from {levels['stop_basis']}", f"T1 from {levels['t1_basis']}"]
              + ([f"T2 from {levels['t2_basis']}"] if levels.get('t2_basis') else
                 ["no second target — no overhead level beyond the measured move"]),
    )
