"""
Walk-forward backtest.

Replays the SAME pipeline the live engine runs — same gates, same base
detection, same level derivation, same score floors — over stored history,
then simulates what each signal would have done.

The point is one question: **does the 80+ score band actually outperform
the 65-79 band?** If it does not, the weights are decoration and the
conviction labels should never be turned on. NSE Pulse learned exactly this
about its own score, and removed the badge.

Three things make this honest rather than flattering:

  1. **No lookahead.** Indicators are causal — a rolling mean or EWM at bar
     i uses only bars up to i — so they are computed once over the full
     history and then SLICED. Slicing a causal series is identical to
     recomputing on the truncated series, and vastly cheaper. Gates only
     ever see df.iloc[:i+1].

  2. **Pessimistic fills.** Entry fills at max(trigger, that day's open), so
     a gap up costs you. A bar that touches both stop and target in the same
     session is recorded as a STOP, because intraday order is unknown and
     assuming otherwise inflates every result.

  3. **Costs applied.** Brokerage, STT, slippage. Ignoring them turns a
     losing system into a marginal one on paper.

Additive: nothing in the live path imports this module.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta

import pandas as pd

from gates import (
    add_indicators,
    evaluate_regime,
    gate0_tradability,
    gate3_trend_structure,
    relative_strength,
)
from ingest import connect
from setups import Rejected, build_setup

log = logging.getLogger(__name__)

MIN_BARS = 210
EXPIRY_SESSIONS = int(os.environ.get("SIGNAL_EXPIRY_SESSIONS", "5"))
MAX_HOLD_SESSIONS = int(os.environ.get("BACKTEST_MAX_HOLD", "40"))
# Round trip: brokerage + STT + exchange + stamp + slippage, as a fraction.
COST_PCT = float(os.environ.get("BACKTEST_COST_PCT", "0.25"))
SCALE_OUT_PCT = float(os.environ.get("SCALE_OUT_PCT", "50"))


def _band(score: float) -> str:
    return "high" if score >= 80 else "medium" if score >= 65 else "low"


def _simulate(fwd: pd.DataFrame, entry: float, stop: float, t1: float,
              t2: float | None) -> dict | None:
    """
    Walk forward bar by bar from the session after the signal.

    fwd columns: trade_date, open, high, low, close.
    Returns None if the trigger was never reached before expiry.
    """
    filled_at = None
    entry_idx = None

    for i in range(min(EXPIRY_SESSIONS, len(fwd))):
        bar = fwd.iloc[i]
        if bar["high"] >= entry:
            # Gap through the trigger fills at the open, not the trigger.
            filled_at = max(float(entry), float(bar["open"]))
            entry_idx = i
            break

    if filled_at is None:
        return None

    risk = filled_at - stop
    if risk <= 0:
        return None

    mfe = mae = 0.0
    scaled = False
    realised_r = 0.0
    remaining = 1.0

    for i in range(entry_idx, min(entry_idx + MAX_HOLD_SESSIONS, len(fwd))):
        bar = fwd.iloc[i]
        high, low = float(bar["high"]), float(bar["low"])
        mfe = max(mfe, (high - filled_at) / risk)
        mae = min(mae, (low - filled_at) / risk)

        hit_stop = low <= stop
        hit_t1 = high >= t1

        # Both in one session: intraday order is unknown, so assume the
        # worse. Assuming the target came first is how backtests lie.
        if hit_stop:
            exit_px = min(float(bar["open"]), stop) if float(bar["open"]) < stop else stop
            realised_r += remaining * (exit_px - filled_at) / risk
            return _close(fwd, entry_idx, i, filled_at, exit_px, realised_r,
                          "stop" if not scaled else "trail_stop", mfe, mae)

        if hit_t1 and not scaled:
            portion = SCALE_OUT_PCT / 100.0
            realised_r += portion * (t1 - filled_at) / risk
            remaining -= portion
            scaled = True
            stop = filled_at          # runner rides from the fill onward
            if remaining <= 0:
                return _close(fwd, entry_idx, i, filled_at, t1, realised_r,
                              "target", mfe, mae)

        if scaled and t2 and high >= t2:
            realised_r += remaining * (t2 - filled_at) / risk
            return _close(fwd, entry_idx, i, filled_at, t2, realised_r,
                          "target2", mfe, mae)

    # Time exit at the last close available.
    last_i = min(entry_idx + MAX_HOLD_SESSIONS, len(fwd)) - 1
    if last_i < entry_idx:
        return None
    exit_px = float(fwd.iloc[last_i]["close"])
    realised_r += remaining * (exit_px - filled_at) / risk
    return _close(fwd, entry_idx, last_i, filled_at, exit_px, realised_r,
                  "time", mfe, mae)


def _close(fwd, entry_idx, exit_idx, filled_at, exit_px, realised_r,
           reason, mfe, mae) -> dict:
    # Costs charged in R terms so they scale with the trade's own risk.
    risk_pct = abs(filled_at - exit_px) / filled_at if filled_at else 0
    cost_r = (COST_PCT / 100.0) * filled_at / max(abs(filled_at - exit_px), 1e-9) \
        * abs(realised_r) if risk_pct else 0
    return {
        "entry_date": fwd.iloc[entry_idx]["trade_date"],
        "entry_price": round(filled_at, 2),
        "exit_date": fwd.iloc[exit_idx]["trade_date"],
        "exit_price": round(float(exit_px), 2),
        "exit_reason": reason,
        "r_realised": round(realised_r - min(cost_r, 0.1), 3),
        "max_favourable_r": round(mfe, 2),
        "max_adverse_r": round(mae, 2),
        "bars_held": exit_idx - entry_idx + 1,
    }


def run_backtest(from_date: date, to_date: date, step: int = 1,
                 min_score_pct: float | None = None) -> dict:
    """
    Replay the pipeline day by day.

    Indicators are computed once per symbol; gates see truncated slices.
    Only Gate 0 and Gate 3 survivors reach base detection, which keeps the
    expensive part to roughly 13% of the universe per day.
    """
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    conn = connect()

    try:
        with conn.cursor() as cur:
            cur.execute("select symbol from symbols where is_active "
                        "and coalesce(series,'') <> 'INDEX' order by symbol")
            universe = [r[0] for r in cur.fetchall()]

            cur.execute("""
                select trade_date, adj_close, adj_high, adj_low, adj_open, volume
                from ohlcv_daily where symbol = 'NIFTY50' order by trade_date
            """)
            nrows = cur.fetchall()

        nifty = pd.DataFrame(nrows, columns=["trade_date", "close", "high",
                                             "low", "open", "volume"])
        for c in ("close", "high", "low", "open"):
            nifty[c] = pd.to_numeric(nifty[c])
        nifty["trade_date"] = pd.to_datetime(nifty["trade_date"])

        trading_days = [d.date() for d in nifty["trade_date"]
                        if from_date <= d.date() <= to_date]
        log.info("Backtest %s to %s: %d sessions, %d symbols",
                 from_date, to_date, len(trading_days), len(universe))

        # Regime per session, from the same function the live engine uses.
        regimes: dict[date, str] = {}
        for d in trading_days:
            idx = nifty[nifty["trade_date"] <= pd.Timestamp(d)]
            if len(idx) < 60:
                continue
            # Breadth is expensive to recompute historically; the regime is
            # driven mainly by trend and volatility, so it is approximated
            # here from the index alone. Flagged in the metrics as such.
            regimes[d] = evaluate_regime(idx.copy(), pd.DataFrame(), 50.0)["state"]

        trades: list[dict] = []
        day_set = set(trading_days)

        for sym in universe:
            with conn.cursor() as cur:
                cur.execute("""
                    select trade_date, adj_open, adj_high, adj_low, adj_close, volume
                    from ohlcv_daily where symbol = %s order by trade_date
                """, (sym,))
                rows = cur.fetchall()
            if len(rows) < MIN_BARS + 20:
                continue

            df = pd.DataFrame(rows, columns=["trade_date", "open", "high",
                                             "low", "close", "volume"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            for c in ("open", "high", "low", "close"):
                df[c] = pd.to_numeric(df[c])
            df["volume"] = pd.to_numeric(df["volume"]).fillna(0)
            df = add_indicators(df)          # once; slicing stays causal

            dates = df["trade_date"].dt.date.tolist()
            for i in range(MIN_BARS, len(df) - 1, step):
                d = dates[i]
                if d not in day_set:
                    continue

                window = df.iloc[:i + 1]
                if not gate0_tradability(sym, window, False).passed:
                    continue
                if not gate3_trend_structure(sym, window).passed:
                    continue

                try:
                    setup = build_setup(
                        sym, window,
                        relative_strength(window, nifty[nifty["trade_date"]
                                                        <= df["trade_date"].iloc[i]], 63),
                        relative_strength(window, nifty[nifty["trade_date"]
                                                        <= df["trade_date"].iloc[i]], 126),
                        None)
                except Rejected:
                    continue

                regime = regimes.get(d, "neutral")
                ceiling = float(setup.score_breakdown.get("max_possible", 100)) or 100
                floor_pct = min_score_pct if min_score_pct is not None else {
                    "risk_on": 65.0, "neutral": 72.0, "risk_off": 80.0}[regime]
                if setup.score_total < floor_pct * ceiling / 100.0:
                    continue

                fwd = df.iloc[i + 1:][["trade_date", "open", "high", "low", "close"]]
                result = _simulate(fwd, setup.entry, setup.stop, setup.t1, setup.t2)

                record = {
                    "symbol": sym, "signal_date": d,
                    "setup_type": setup.setup_type, "pattern": setup.pattern,
                    "score_total": setup.score_total,
                    "band": _band(setup.score_total), "regime": regime,
                    "entry_trigger": setup.entry, "stop_loss": setup.stop,
                    "t1": setup.t1, "t2": setup.t2,
                    "r_planned": setup.r_multiple_t1,
                }
                record.update(result or {"exit_reason": "never_triggered"})
                trades.append(record)

            log.debug("%s: %d signals so far", sym, len(trades))

        return {"trades": trades, "sessions": len(trading_days),
                "universe": len(universe)}

    finally:
        conn.close()


def summarise(trades: list[dict]) -> dict:
    """Metrics that answer the question, split the ways that matter."""
    def stats(subset):
        filled = [t for t in subset if t.get("r_realised") is not None]
        if not filled:
            return {"signals": len(subset), "filled": 0, "hit_rate": None,
                    "expectancy_r": None, "total_r": None, "avg_win": None,
                    "avg_loss": None}
        rs = [t["r_realised"] for t in filled]
        wins = [r for r in rs if r > 0]
        losses = [r for r in rs if r <= 0]
        return {
            "signals": len(subset),
            "filled": len(filled),
            "fill_rate": round(len(filled) / len(subset), 3),
            "hit_rate": round(len(wins) / len(rs), 3),
            "expectancy_r": round(sum(rs) / len(rs), 3),
            "total_r": round(sum(rs), 1),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        }

    out = {"overall": stats(trades)}
    for key in ("band", "regime", "setup_type", "pattern"):
        out[key] = {v: stats([t for t in trades if t.get(key) == v])
                    for v in sorted({t.get(key) for t in trades if t.get(key)})}

    # The question this exists to answer.
    bands = out["band"]
    hi = bands.get("high", {}).get("expectancy_r")
    mid = bands.get("medium", {}).get("expectancy_r")
    lo = bands.get("low", {}).get("expectancy_r")
    out["verdict"] = {
        "bands_separate": (hi is not None and mid is not None and hi > mid),
        "monotonic": (None not in (hi, mid, lo) and hi > mid > lo),
        "note": ("Score bands are only meaningful if high outperforms medium. "
                 "If they do not separate, the weights carry no information "
                 "and conviction labels must stay off."),
    }
    return out
