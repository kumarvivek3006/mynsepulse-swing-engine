"""
Sequential gates — elimination, not scoring.

A discretionary trader does not grade 500 stocks. They discard almost all
of them on a handful of disqualifying facts, then look hard at what is
left. These gates do the same, in order, and every rejection is recorded
with a reason so the discard pile can be audited later.

Implemented here: Gate 0 (tradability), Gate 1 (market regime), Gate 3
(trend structure).

Gate 2 (fundamental vetoes) is deliberately absent. It needs quarterly
financials, shareholding and pledge data that we have not ingested yet.
It is logged as skipped on every run rather than silently omitted, so the
gap stays visible instead of being mistaken for a pass.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)

MIN_PRICE = float(os.environ.get("MIN_PRICE", "40"))
MIN_TURNOVER_CR = float(os.environ.get("MIN_TURNOVER_CR", "5"))
MIN_LISTING_SESSIONS = int(os.environ.get("MIN_LISTING_SESSIONS", "250"))


@dataclass
class GateResult:
    symbol: str
    passed: bool
    failed_gate: str | None = None
    reason: str | None = None
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: trade_date, open, high, low, close, volume."""
    df = df.sort_values("trade_date").reset_index(drop=True)
    c, h, l = df["close"], df["high"], df["low"]

    for n in (20, 50, 150, 200):
        df[f"sma{n}"] = c.rolling(n).mean()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()

    # True range and ATR(14), Wilder smoothing
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    df["vol50"] = df["volume"].rolling(50).mean()
    df["turnover_cr"] = (c * df["volume"]) / 1e7
    df["turnover20_cr"] = df["turnover_cr"].rolling(20).median()

    # RSI(14), Wilder smoothing. Added specifically to enforce the one
    # momentum-based finding that held sign in both halves of the backtest:
    # setups firing while RSI sits in 40-55 lose money consistently.
    # Overbought (70+) was the BEST zone measured, so this is not a general
    # RSI filter — it excludes only the weak middle, not high momentum.
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    # avg_loss == 0 with real gains is genuine strength (100). avg_loss == 0
    # with NO gains either is a dormant, flat series — neutral 50, not 100.
    # Missed this the first time; caught by the same flat-series check that
    # caught the identical bug in the backtest's copy of this calculation.
    dormant = (avg_loss == 0) & (avg_gain == 0)
    df["rsi14"] = df["rsi14"].where(avg_loss != 0, 100.0)
    df["rsi14"] = df["rsi14"].where(~dormant, 50.0)

    # On-Balance Volume — standard definition, no variation. Added to give
    # the engine a genuine way to tell "quiet because being accumulated"
    # from "quiet because abandoned", which an RSI level alone cannot do: a
    # stock can sit at RSI 45 either because smart money is patiently
    # buying into weakness, or because nobody is trading it at all. OBV
    # rising while price is flat is the textbook accumulation signature;
    # OBV flat or falling while price is flat is not.
    price_change = c.diff()
    obv_step = df["volume"].where(price_change > 0, 0.0)
    obv_step = obv_step.where(price_change >= 0, -df["volume"])
    # The first bar has no prior close, so price_change is NaN there. NaN
    # comparisons are False, which fell through to the "down day" branch and
    # subtracted volume from a bar that had no direction at all — caught by
    # checking the actual sequence against a hand-computed one.
    obv_step = obv_step.where(price_change.notna(), 0.0)
    df["obv"] = obv_step.cumsum()

    window52 = 250
    df["high52"] = h.rolling(window52, min_periods=100).max()
    df["low52"] = l.rolling(window52, min_periods=100).min()

    df["sma200_slope25"] = df["sma200"] - df["sma200"].shift(25)
    df["sma50_slope10"] = df["sma50"] - df["sma50"].shift(10)
    return df


def weekly_structure_ok(df: pd.DataFrame, weeks: int = 12) -> bool:
    """
    Weekly trend confirmation, Weinstein Stage 2.

    The previous test required the last 6 weeks to exceed the prior 6 on BOTH
    highs and lows. That rejects a stock for the pullbacks a base is made of:
    one lower weekly low anywhere in six weeks failed it. Measured on live
    data it removed 10-17 candidates a day — about 13% of everything clearing
    the moving-average tests — including stocks that then broke out days
    later, by which time the entry had gone.

    A six-week window is short enough that noise dominates. The canonical
    weekly test is Weinstein's: price above a RISING 30-week moving average.
    That is what defines Stage 2, and it tolerates the pullbacks within a
    base while still excluding stocks in downtrends.

    Three requirements, all trend properties rather than window comparisons:

      1. Weekly close above the 30-week moving average
      2. That average rising over the last 8 weeks
      3. Higher highs over a 13-week span, so the stock is making new ground

    Higher LOWS are deliberately not required over a short window. A base is
    a pullback; demanding it never makes one is demanding it never bases.
    """
    w = (df.set_index(pd.to_datetime(df["trade_date"]))
           .resample("W")
           .agg({"high": "max", "low": "min", "close": "last"})
           .dropna())
    if len(w) < 32:
        return False

    ma30 = w["close"].rolling(30).mean()
    if pd.isna(ma30.iloc[-1]) or pd.isna(ma30.iloc[-9]):
        return False

    above_ma = float(w["close"].iloc[-1]) > float(ma30.iloc[-1])
    ma_rising = float(ma30.iloc[-1]) > float(ma30.iloc[-9])

    span = min(13, len(w) // 2)
    recent_high = float(w["high"].iloc[-span:].max())
    prior_high = float(w["high"].iloc[-2 * span:-span].max())
    higher_highs = recent_high > prior_high

    # A guard against a stock that has broken down badly: the most recent
    # 13-week low must hold above the low of the period before it. This is
    # the same intent as the old higher-lows rule, measured over a span long
    # enough that a base pullback does not fail it.
    recent_low = float(w["low"].iloc[-span:].min())
    prior_low = float(w["low"].iloc[-2 * span:-span].min())
    not_broken = recent_low > prior_low * 0.90

    return bool(above_ma and ma_rising and higher_highs and not_broken)


# ---------------------------------------------------------------------
# Gate 1 — market regime (once per run)
# ---------------------------------------------------------------------
def evaluate_regime(nifty: pd.DataFrame, vix: pd.DataFrame,
                    breadth_pct: float) -> dict:
    """
    Risk-on / neutral / risk-off.

    Distribution days are counted the classic way: a down day on higher
    volume than the prior session. Index volume is unreliable on NSE, so
    when it is absent the count is skipped rather than guessed at, and the
    regime leans on trend, breadth and volatility instead.
    """
    nifty = add_indicators(nifty)
    last = nifty.iloc[-1]

    above20 = bool(last["close"] > last["sma20"]) if pd.notna(last["sma20"]) else False
    above50 = bool(last["close"] > last["sma50"]) if pd.notna(last["sma50"]) else False
    slope50 = bool(last["sma50_slope10"] > 0) if pd.notna(last["sma50_slope10"]) else False

    vix_level = float(vix.iloc[-1]["close"]) if len(vix) else float("nan")
    vix_10d = (float(vix.iloc[-1]["close"] - vix.iloc[-11]["close"])
               if len(vix) > 11 else 0.0)

    recent = nifty.tail(25)
    if recent["volume"].sum() > 0:
        down = recent["close"] < recent["close"].shift(1)
        heavier = recent["volume"] > recent["volume"].shift(1)
        distribution_days = int((down & heavier).sum())
    else:
        distribution_days = 0

    # Risk-off must mean actual deterioration, not merely a dip below the
    # 50 DMA. An earlier version made `not above50` sufficient on its own,
    # which suppressed every setup on a calm tape with VIX at 11, zero
    # distribution days and 67 stocks in clean Stage 2 uptrends. A single
    # moving average should not outvote breadth, volatility and supply.
    above200 = bool(last["close"] > last["sma200"]) if pd.notna(last["sma200"]) else True
    vix_spike = (not pd.isna(vix_level)) and vix_level > 25 and vix_10d > 0

    # Risk-off requires TWO independent confirmations. A single input has
    # twice now zeroed every signal on its own — first the 50 DMA, then the
    # 200 DMA while VIX sat at 11 with no distribution. One indicator
    # should not outvote every other measure of the tape.
    bearish = [
        breadth_pct < 35,
        not above200,
        distribution_days >= 6,
        vix_spike,
    ]
    if sum(bool(x) for x in bearish) >= 2:
        state = "risk_off"
    elif above20 and above50 and slope50 and breadth_pct >= 50 and distribution_days <= 4:
        state = "risk_on"
    else:
        state = "neutral"

    score = sum([above20, above50, slope50, breadth_pct >= 50,
                 pd.isna(vix_level) or vix_level < 20, distribution_days <= 4])

    return {
        "state": state,
        "nifty_close": float(last["close"]),
        "nifty_vs_20dma": float(last["close"] - last["sma20"]) if above20 or pd.notna(last["sma20"]) else None,
        "nifty_vs_50dma": float(last["close"] - last["sma50"]) if pd.notna(last["sma50"]) else None,
        "breadth_above_50dma": round(breadth_pct, 2),
        "vix": None if pd.isna(vix_level) else vix_level,
        "vix_10d_change": vix_10d,
        "distribution_days": distribution_days,
        "notes": {"score": score, "above20": above20, "above50": above50,
                  "above200": above200, "slope50": slope50,
                  "vix_spike": bool(vix_spike),
                  "bearish_confirmations": int(sum(bool(x) for x in bearish)),
                  "sma20": None if pd.isna(last["sma20"]) else float(last["sma20"]),
                  "sma50": None if pd.isna(last["sma50"]) else float(last["sma50"]),
                  "sma200": None if pd.isna(last["sma200"]) else float(last["sma200"])},
    }


# ---------------------------------------------------------------------
# Gate 0 — tradability
# ---------------------------------------------------------------------
def gate0_tradability(symbol: str, df: pd.DataFrame,
                      under_surveillance: bool) -> GateResult:
    if under_surveillance:
        return GateResult(symbol, False, "gate0", "surveillance",
                          {"list": "ASM/GSM"})

    if len(df) < MIN_LISTING_SESSIONS:
        return GateResult(symbol, False, "gate0", "insufficient_history",
                          {"sessions": len(df)})

    last = df.iloc[-1]

    if last["close"] < MIN_PRICE:
        return GateResult(symbol, False, "gate0", "price_below_floor",
                          {"close": float(last["close"]), "floor": MIN_PRICE})

    turnover = last["turnover20_cr"]
    if pd.isna(turnover) or turnover < MIN_TURNOVER_CR:
        return GateResult(symbol, False, "gate0", "illiquid",
                          {"turnover20_cr": None if pd.isna(turnover) else round(float(turnover), 2),
                           "floor": MIN_TURNOVER_CR})

    return GateResult(symbol, True, detail={"turnover20_cr": round(float(turnover), 2)})


# ---------------------------------------------------------------------
# Gate 2 — fundamental vetoes
#
# A veto, never a stock picker. It removes structurally unsound companies
# that gap against you on news; it does not rank anything.
#
# Two rules are implemented, both from data we actually hold. The pledge
# and balance-sheet vetoes in the spec are NOT here — NSE's shareholding
# endpoint carries no pledge figure, and cash flow / debt-equity / ROCE
# live in annual filings we have not ingested. Those return `None` from
# `gate2_missing_vetoes()` so the gap stays visible instead of being
# mistaken for a pass.
# ---------------------------------------------------------------------
PROMOTER_DROP_PP = float(os.environ.get("PROMOTER_DROP_PP", "2.0"))

# Spec (Prompts 1, 5, 10) requires RS rating > 70 as a hard gate. RS is
# computed and ranked on every signal today but never filtered on — Prompt
# 10 is effectively unimplemented as a strategy for that reason.
#
# Defaulted OFF because every backtest this engine has run showed RS
# carrying little signal AS SCORED, and turning it on would cut signal
# count materially on an unvalidated basis. A hard floor is a different
# claim from a ranking weight and has never been tested — so it ships
# switchable, and the backtest decides rather than my judgement.
RS_FLOOR_ENABLED = os.environ.get("RS_FLOOR_ENABLED", "false").lower() == "true"
RS_FLOOR_PCT = float(os.environ.get("RS_FLOOR_PCT", "70"))

# Weinstein breakout volume: >2x the 50-WEEK average, on the weekly candle.
# Applies to the Stage 1->2 transition path only. The daily-chart setups
# (VCP, retest, flag, triangle, cup-handle) keep their own daily multiples,
# which is what the strategy spec actually calls for in each case — weekly
# is the right measure for Weinstein specifically, not a global upgrade.
WEEKLY_VOL_MULT = float(os.environ.get("WEEKLY_VOL_MULT", "2.0"))
WEEKLY_VOL_CHECK_ENABLED = os.environ.get(
    "WEEKLY_VOL_CHECK_ENABLED", "true").lower() == "true"


def gate2_fundamentals(symbol: str, snap) -> GateResult:
    """
    snap is a FundamentalSnapshot, or None when we hold no data.

    No data means the gate cannot run. It is logged as skipped and the
    stock proceeds — but the signal is flagged so the score never implies
    a check that did not happen.
    """
    if snap is None or not snap.has_data:
        return GateResult(symbol, True, reason="gate2_no_data",
                          detail={"checked": False})

    # Promoter selling down is the single loudest governance signal
    # available to us. Two percentage points over two quarters is a
    # deliberate exit, not portfolio noise.
    if snap.promoter_pct is not None and snap.promoter_pct_2q_ago is not None:
        drop = snap.promoter_pct_2q_ago - snap.promoter_pct
        if drop >= PROMOTER_DROP_PP:
            return GateResult(symbol, False, "gate2", "promoter_holding_falling",
                              {"now": round(snap.promoter_pct, 2),
                               "two_quarters_ago": round(snap.promoter_pct_2q_ago, 2),
                               "drop_pp": round(drop, 2)})

    # Two consecutive quarters where revenue AND profit both fell.
    # Either alone is noise; both together is deterioration.
    rev, pat = snap.revenue_trend, snap.pat_trend
    if len(rev) >= 3 and len(pat) >= 3:
        rev_falling = rev[-1] < rev[-2] < rev[-3]
        pat_falling = pat[-1] < pat[-2] < pat[-3]
        if rev_falling and pat_falling:
            return GateResult(symbol, False, "gate2", "revenue_and_profit_declining",
                              {"revenue": [round(x, 1) for x in rev[-3:]],
                               "pat": [round(x, 1) for x in pat[-3:]]})

    return GateResult(symbol, True, detail={"checked": True,
                                            "promoter_pct": snap.promoter_pct})


def gate2_missing_vetoes() -> list[str]:
    """Vetoes in the spec that no ingested source can currently satisfy."""
    return [
        "promoter_pledge_above_25pct",   # needs SHP filing Column XIV
        "pledge_increased_qoq",          # needs SHP filing Column XIV
        "negative_operating_cash_flow",  # needs annual filings
        "debt_equity_above_2",           # needs annual filings
        "auditor_qualification",         # needs annual filings
        "receivable_days_rising",        # needs annual filings
    ]


# ---------------------------------------------------------------------
# Gate 3 — trend structure (Stage 2)
# ---------------------------------------------------------------------
def gate3_trend_structure(symbol: str, df: pd.DataFrame,
                          rs_rank_pct: float | None = None) -> GateResult:
    """
    rs_rank_pct is the stock's same-day RS percentile against the universe.
    Only enforced when RS_FLOOR_ENABLED — see the constant's note.
    """
    last = df.iloc[-1]
    needed = ["sma50", "sma150", "sma200", "high52", "low52",
              "sma200_slope25", "sma50_slope10"]
    if any(pd.isna(last[c]) for c in needed):
        return GateResult(symbol, False, "gate3", "indicators_incomplete")

    close = float(last["close"])
    checks = {
        "close_above_50": close > last["sma50"],
        # Minervini rule 1 requires price above BOTH the 150 and 200 DMA.
        # Only close_above_50 was checked, plus the MAs' ordering relative
        # to each other — which is a different claim. A stock declining
        # sharply can sit below its 200 DMA while the MA stack is still
        # stale and correctly ordered, so early breakdowns were passing a
        # gate whose whole purpose is to exclude them.
        "close_above_150": close > last["sma150"],
        "close_above_200": close > last["sma200"],
        "50_above_150": last["sma50"] > last["sma150"],
        "150_above_200": last["sma150"] > last["sma200"],
        "200_rising": last["sma200_slope25"] > 0,
        "50_rising": last["sma50_slope10"] > 0,
        "within_25pct_of_52w_high": close >= float(last["high52"]) * 0.75,
        "30pct_above_52w_low": close >= float(last["low52"]) * 1.30,
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        return GateResult(symbol, False, "gate3", failed[0],
                          {"failed_checks": failed})

    if not weekly_structure_ok(df):
        return GateResult(symbol, False, "gate3", "weekly_structure")

    # Minervini rule 8 / Prompt 10's core filter. Off by default.
    if RS_FLOOR_ENABLED and rs_rank_pct is not None and rs_rank_pct < RS_FLOOR_PCT:
        return GateResult(symbol, False, "gate3", "rs_below_floor",
                          {"rs_rank_pct": rs_rank_pct, "floor": RS_FLOOR_PCT})

    return GateResult(symbol, True, detail={
        "pct_from_52w_high": round((close / float(last["high52"]) - 1) * 100, 2),
        "pct_above_52w_low": round((close / float(last["low52"]) - 1) * 100, 2),
    })


# ---------------------------------------------------------------------
# Gate 3b — Stage 1 to Stage 2 transition (opt-in, additive)
#
# Gate 3 requires a stock to ALREADY be in Stage 2: price above a rising
# 200 DMA with 50 > 150 > 200. That is correct for continuation setups and
# it systematically excludes the stock emerging from a long Stage 1 base —
# the one that produces the longest rides, precisely because nobody is
# watching it yet.
#
# This is a SECOND path, not a relaxation of the first. A candidate must
# still clear Gate 0, and it must satisfy conditions Gate 3 never asks for:
# a 50/200 crossover already in place, a 200 DMA that has stopped falling
# AND is improving, and price holding above both. What it does not demand
# is that the 200 DMA already be rising over 25 sessions.
#
# Anything reaching here failed Gate 3, so nothing that passes today can be
# lost by enabling it.
# ---------------------------------------------------------------------
TRANSITION_MAX_200_DECLINE_PCT = float(
    os.environ.get("TRANSITION_MAX_200_DECLINE_PCT", "0.5"))
# Gate 3 requires 30% above the 52-week low. That figure is calibrated for a
# stock already advancing in Stage 2. A stock completing a Stage 1 base is
# necessarily still close to its low — the base IS near the low — so applying
# 30% here would reject almost every genuine transition and make this path
# decorative. 20% still excludes stocks sitting on their lows.
TRANSITION_MIN_ABOVE_52W_LOW_PCT = float(
    os.environ.get("TRANSITION_MIN_ABOVE_52W_LOW_PCT", "20"))


# ---------------------------------------------------------------------
# C3 / D4 — top-level regime gate
#
# ADDITIVE. evaluate_regime() below is untouched and still drives scoring;
# this is a separate gate that runs BEFORE Gate 1 and can stop a day
# producing signals at all. REGIME_GATE_ENABLED=false restores the prior
# behaviour exactly.
#
# Run #29 walk-forward: the engine won W4 and W6, lost W1, W3, W5. The
# score-based regime filter flips sign in the split test, so it does not
# separate them.
#
# NOT COMPUTABLE from current ingestion — follow-up ticket, do not block:
#   - sector dispersion (needs Nifty sector indices)
#   - smallcap/largecap ratio (needs Nifty Smallcap 100)
#   - advance/decline ratio (needs per-symbol daily A/D, not stored)
#   - new highs minus new lows (needs rolling 52w extremes per symbol)
# We ingest NIFTY50, NIFTY500 and INDIAVIX only. Building a proxy for any
# of these from what we have would be inventing a number.
# Default OFF. Run #31 measured the c1_and_c5_or_c7 default as INVERTED:
# taken expectancy -0.071R against skipped +0.455R. Both components are
# individually inverted too — above-200DMA and VIX<15 each select the worse
# half. A gate that reliably picks the wrong side is worse than no gate, so
# it ships disabled until a classifier earns it.
REGIME_GATE_ENABLED = os.environ.get(
    "REGIME_GATE_ENABLED", "false").lower() == "true"
REGIME_CLASSIFIER = os.environ.get("REGIME_CLASSIFIER", "c1_and_c5_or_c7")

# c7 (breadth > 55%) applied PER SIGNAL rather than as a day-level gate.
# It is the one classifier showing real separation (+0.508R between taken
# and skipped) even though it does not flip the walk-forward. A per-signal
# filter drops individual signals on weak-breadth days instead of blanking
# the whole day, so partial days remain possible.
BREADTH_FILTER_ENABLED = os.environ.get(
    "BREADTH_FILTER_ENABLED", "false").lower() == "true"
BREADTH_FILTER_MIN_PCT = float(os.environ.get("BREADTH_FILTER_MIN_PCT", "55"))
REGIME_VIX_LOW = float(os.environ.get("REGIME_VIX_LOW", "15"))
REGIME_VIX_HIGH = float(os.environ.get("REGIME_VIX_HIGH", "18"))
REGIME_BREADTH_LOW = float(os.environ.get("REGIME_BREADTH_LOW", "55"))
REGIME_BREADTH_HIGH = float(os.environ.get("REGIME_BREADTH_HIGH", "60"))


def regime_classifiers(nifty: pd.DataFrame, vix: pd.DataFrame | None,
                       breadth: float | None) -> dict:
    """
    The six computable single-metric classifiers plus three composites.

    Every one returns True (trade), False (skip) or None (cannot compute).
    None is NOT treated as False anywhere: a missing input must not silently
    become a skip decision, or a data outage reads as a bearish market.
    """
    out: dict = {}
    if nifty is None or len(nifty) < 200:
        return {"error": "insufficient nifty history"}

    close = float(nifty["close"].iloc[-1])
    sma50 = float(nifty["close"].rolling(50).mean().iloc[-1])
    sma200 = float(nifty["close"].rolling(200).mean().iloc[-1])
    sma50_prev = (float(nifty["close"].rolling(50).mean().iloc[-21])
                  if len(nifty) > 220 else None)

    out["c1_close_above_200dma"] = close > sma200
    out["c2_close_above_50_and_stack"] = bool(close > sma50 and sma50 > sma200)
    out["c3_50dma_rising_20d"] = (sma50 > sma50_prev) if sma50_prev else None

    vix_level = None
    if vix is not None and len(vix):
        try:
            vix_level = float(vix["close"].iloc[-1])
        except Exception:
            vix_level = None
    out["c5_vix_below_15"] = (vix_level < REGIME_VIX_LOW) if vix_level is not None else None
    out["c6_vix_below_18"] = (vix_level < REGIME_VIX_HIGH) if vix_level is not None else None

    out["c7_breadth_above_55"] = (breadth > REGIME_BREADTH_LOW) if breadth is not None else None
    out["c8_breadth_above_60"] = (breadth > REGIME_BREADTH_HIGH) if breadth is not None else None

    # Composites. Any None propagates rather than counting as False.
    c1, c3 = out["c1_close_above_200dma"], out["c3_50dma_rising_20d"]
    c2, c5 = out["c2_close_above_50_and_stack"], out["c5_vix_below_15"]
    c7 = out["c7_breadth_above_55"]

    out["c1_and_c5_or_c7"] = (
        bool(c1 and (c5 or c7)) if None not in (c5, c7) else None)
    out["c2_and_c7"] = bool(c2 and c7) if c7 is not None else None

    votes = [v for v in (c1, c3, c5, c7) if v is not None]
    out["majority_c1_c3_c5_c7"] = (
        (sum(votes) > len(votes) / 2) if len(votes) >= 3 else None)

    out["_inputs"] = {
        "nifty_close": round(close, 2), "sma50": round(sma50, 2),
        "sma200": round(sma200, 2), "vix": vix_level,
        "breadth_pct": round(breadth, 2) if breadth is not None else None,
    }
    return out


def regime_gate_passes(nifty: pd.DataFrame, vix: pd.DataFrame | None,
                       breadth: float | None,
                       classifier: str | None = None) -> tuple[bool, dict]:
    """
    Does the configured classifier permit trading today?

    An unavailable classifier result defaults to TRADE, not skip. A gate
    that silently blocks every day because an input is missing is worse
    than no gate — the engine would go quiet and look like a dead market.
    """
    name = classifier or REGIME_CLASSIFIER
    results = regime_classifiers(nifty, vix, breadth)
    if "error" in results:
        return True, {"decision": "trade", "reason": results["error"],
                      "classifier": name, "defaulted": True}

    verdict = results.get(name)
    if verdict is None:
        return True, {"decision": "trade", "reason": "classifier_not_computable",
                      "classifier": name, "defaulted": True,
                      "classifiers": results}
    return bool(verdict), {"decision": "trade" if verdict else "skip",
                           "classifier": name, "defaulted": False,
                           "classifiers": results}


def weekly_volume_surge(df: pd.DataFrame, mult: float) -> tuple[bool, float | None]:
    """
    Is the current week's volume above `mult` x the 50-week average?

    Weinstein's framework is weekly throughout, and his breakout rule is a
    weekly volume expansion — sustained accumulation across five sessions,
    not a single day's spike. Our daily 1.5x-of-50-day check is a proxy for
    that: a one-day news spike that fully reverses passes it, whereas a
    genuine week of institutional buying is a different and stronger claim.

    PARTIAL WEEKS: the current week is usually incomplete when this runs.
    Volume only ACCUMULATES within a week — it cannot shrink — so a partial
    week that already clears the threshold will still clear it at week's
    end. Requiring it to be already met is therefore conservative and can
    never be retroactively wrong. Nothing is prorated or projected: that
    would be inventing a number, the same error as comparing three days of
    volume against a five-day average.

    Returns (passed, observed_multiple).
    """
    w = (df.set_index(pd.to_datetime(df["trade_date"]))
           .resample("W")["volume"].sum()
           .dropna())
    if len(w) < 52:
        return False, None

    # Average EXCLUDES the current (possibly partial) week, so the week
    # being tested is never part of its own baseline.
    baseline = float(w.iloc[-51:-1].mean())
    if baseline <= 0:
        return False, None

    current = float(w.iloc[-1])
    observed = current / baseline
    return observed >= mult, round(observed, 2)


def gate3_stage1_transition(symbol: str, df: pd.DataFrame) -> GateResult:
    last = df.iloc[-1]
    needed = ["sma50", "sma150", "sma200", "high52", "low52", "sma200_slope25"]
    if any(pd.isna(last[c]) for c in needed):
        return GateResult(symbol, False, "gate3b", "indicators_incomplete")

    close = float(last["close"])
    sma200 = float(last["sma200"])

    # The 200 DMA must have stopped falling meaningfully, and must be
    # improving — a flat average that is still deteriorating is not a
    # transition, it is a pause in a downtrend.
    slope_now = float(last["sma200_slope25"])
    prior_slope = None
    if len(df) > 50 and pd.notna(df["sma200"].iloc[-26]) and pd.notna(df["sma200"].iloc[-51]):
        prior_slope = float(df["sma200"].iloc[-26] - df["sma200"].iloc[-51])

    checks = {
        "close_above_200": close > sma200,
        "close_above_50": close > float(last["sma50"]),
        "50_above_200": float(last["sma50"]) > sma200,       # crossover in place
        "200_not_falling": slope_now > -(close * TRANSITION_MAX_200_DECLINE_PCT / 100),
        "200_improving": prior_slope is None or slope_now > prior_slope,
        "within_25pct_of_52w_high": close >= float(last["high52"]) * 0.75,
        "above_52w_low": close >= float(last["low52"]) * (
            1 + TRANSITION_MIN_ABOVE_52W_LOW_PCT / 100),
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        return GateResult(symbol, False, "gate3b", failed[0],
                          {"failed_checks": failed})

    if not weekly_structure_ok(df):
        return GateResult(symbol, False, "gate3b", "weekly_structure")

    # Recorded here, enforced at trigger time in setups.py. The gate answers
    # "is this stock in a Stage 1->2 transition"; whether THIS bar's breakout
    # carries weekly volume is a trigger question, and enforcing it here
    # would wrongly reject a stock still quietly basing before its breakout.
    weekly_vol_ok, weekly_vol_mult = weekly_volume_surge(df, WEEKLY_VOL_MULT)

    return GateResult(symbol, True, detail={
        "sma200_slope25": round(slope_now, 2),
        "prior_slope": round(prior_slope, 2) if prior_slope is not None else None,
        "pct_above_200dma": round((close / sma200 - 1) * 100, 2),
        "pct_from_52w_high": round((close / float(last["high52"]) - 1) * 100, 2),
        "weekly_vol_surge": weekly_vol_ok,
        "weekly_vol_mult": weekly_vol_mult,
    })


# ---------------------------------------------------------------------
# Relative strength — used for scoring later, computed here
# ---------------------------------------------------------------------
def relative_strength(df: pd.DataFrame, bench: pd.DataFrame,
                      lookback: int = 63) -> float | None:
    if len(df) <= lookback or len(bench) <= lookback:
        return None
    stock = df["close"].iloc[-1] / df["close"].iloc[-1 - lookback] - 1
    index = bench["close"].iloc[-1] / bench["close"].iloc[-1 - lookback] - 1
    return round((stock - index) * 100, 2)
