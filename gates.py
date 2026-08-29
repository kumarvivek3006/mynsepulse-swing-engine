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

    if (breadth_pct < 35) or (not above200) or distribution_days >= 6 or vix_spike:
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
                  "vix_spike": bool(vix_spike)},
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
# Relative strength — used for scoring later, computed here
# ---------------------------------------------------------------------
def relative_strength(df: pd.DataFrame, bench: pd.DataFrame,
                      lookback: int = 63) -> float | None:
    if len(df) <= lookback or len(bench) <= lookback:
        return None
    stock = df["close"].iloc[-1] / df["close"].iloc[-1 - lookback] - 1
    index = bench["close"].iloc[-1] / bench["close"].iloc[-1 - lookback] - 1
    return round((stock - index) * 100, 2)
