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

    window52 = 250
    df["high52"] = h.rolling(window52, min_periods=100).max()
    df["low52"] = l.rolling(window52, min_periods=100).min()

    df["sma200_slope25"] = df["sma200"] - df["sma200"].shift(25)
    df["sma50_slope10"] = df["sma50"] - df["sma50"].shift(10)
    return df


def weekly_structure_ok(df: pd.DataFrame, weeks: int = 12) -> bool:
    """
    Higher highs and higher lows on the weekly chart.

    Resampled from daily rather than read from a weekly table so there is
    one source of truth for price. Requires the most recent swing to be
    above the prior one on both highs and lows — a stock making higher
    highs on lower lows is broadening, not trending.
    """
    w = (df.set_index(pd.to_datetime(df["trade_date"]))
           .resample("W")
           .agg({"high": "max", "low": "min", "close": "last"})
           .dropna())
    if len(w) < weeks:
        return False
    recent, prior = w.iloc[-weeks // 2:], w.iloc[-weeks:-weeks // 2]
    return (recent["high"].max() > prior["high"].max()
            and recent["low"].min() > prior["low"].min())


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
def gate3_trend_structure(symbol: str, df: pd.DataFrame) -> GateResult:
    last = df.iloc[-1]
    needed = ["sma50", "sma150", "sma200", "high52", "low52",
              "sma200_slope25", "sma50_slope10"]
    if any(pd.isna(last[c]) for c in needed):
        return GateResult(symbol, False, "gate3", "indicators_incomplete")

    close = float(last["close"])
    checks = {
        "close_above_50": close > last["sma50"],
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

    return GateResult(symbol, True, detail={
        "sma200_slope25": round(slope_now, 2),
        "prior_slope": round(prior_slope, 2) if prior_slope is not None else None,
        "pct_above_200dma": round((close / sma200 - 1) * 100, 2),
        "pct_from_52w_high": round((close / float(last["high52"]) - 1) * 100, 2),
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
