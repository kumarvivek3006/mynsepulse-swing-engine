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


def _rsi(closes, period: int = 14) -> float | None:
    """
    Wilder's RSI at the last bar. Computed here rather than in gates.py so
    nothing in the live engine changes.
    """
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_g, avg_l = gains / period, losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (period - 1) + max(d, 0.0)) / period
        avg_l = (avg_l * (period - 1) + max(-d, 0.0)) / period
    if avg_l == 0:
        # No losses AND no gains is a dormant series, not maximum strength.
        # Returning 100 there would classify every flat stock as extreme
        # overbought and poison the whole comparison.
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return round(100 - 100 / (1 + rs), 1)


def _rsi_zone(rsi: float | None) -> str:
    """
    Buckets chosen before looking at any result.

    I argued earlier that an RSI ceiling would reject the best setups,
    because strong stocks stay overbought for months. That was judgement,
    not evidence. These buckets let the data settle it: if 70+ genuinely
    underperforms in BOTH halves, I was wrong and a ceiling is justified.
    """
    if rsi is None:
        return "unknown"
    if rsi < 40:
        return "weak_under_40"
    if rsi < 55:
        return "neutral_40_55"
    if rsi < 70:
        return "healthy_55_70"
    if rsi < 80:
        return "overbought_70_80"
    return "extreme_80_plus"


def _band(score: float) -> str:
    return "high" if score >= 80 else "medium" if score >= 65 else "low"


# Exit variants, pre-specified. A small fixed set judged by the split-sample
# test — not a sweep. Sweeping dozens of combinations across 1,602 trades will
# always surface something that looks excellent and means nothing.
#
# The current live exit (baseline) truncates the right tail three ways: half
# the position sold at T1, the runner's stop jumped to breakeven, and a hard
# stop at 40 sessions. Trend systems earn from a few very large winners, and
# breakouts here show a 40.1% hit rate with only 1.48R average wins — high
# accuracy, small payoff, which is what cutting winners looks like.
EXIT_VARIANTS = {
    "baseline":        {"scale_pct": 50, "breakeven": True,  "trail": False, "max_hold": 40},
    "no_scale_out":    {"scale_pct": 0,  "breakeven": True,  "trail": False, "max_hold": 40},
    "no_breakeven":    {"scale_pct": 50, "breakeven": False, "trail": True,  "max_hold": 40},
    "let_it_run":      {"scale_pct": 0,  "breakeven": False, "trail": True,  "max_hold": 120},
    "trail_only_long": {"scale_pct": 33, "breakeven": False, "trail": True,  "max_hold": 120},
}


def _swing_low(lows: list, i: int) -> float | None:
    """Most recent confirmed swing low at or before i: two higher lows each side."""
    for j in range(i - 2, 1, -1):
        w = lows[j - 2:j + 3]
        if len(w) == 5 and lows[j] == min(w):
            return lows[j]
    return None


def _simulate_variant(fwd: pd.DataFrame, entry: float, stop: float, t1: float,
                      t2: float | None, cfg: dict) -> dict | None:
    """One exit policy. Entry logic is identical across variants."""
    filled_at = entry_idx = None
    for i in range(min(EXPIRY_SESSIONS, len(fwd))):
        bar = fwd.iloc[i]
        if bar["high"] >= entry:
            filled_at = max(float(entry), float(bar["open"]))
            entry_idx = i
            break
    if filled_at is None:
        return None

    risk = filled_at - stop
    if risk <= 0:
        return None

    max_hold = cfg["max_hold"]
    scale_pct = cfg["scale_pct"] / 100.0
    live_stop = stop
    remaining = 1.0
    realised = 0.0
    scaled = False
    mfe = mae = 0.0
    lows = [float(x) for x in fwd["low"].tolist()]

    for i in range(entry_idx, min(entry_idx + max_hold, len(fwd))):
        bar = fwd.iloc[i]
        high, low = float(bar["high"]), float(bar["low"])
        mfe = max(mfe, (high - filled_at) / risk)
        mae = min(mae, (low - filled_at) / risk)

        if low <= live_stop:
            realised += remaining * (live_stop - filled_at) / risk
            return _close(fwd, entry_idx, i, filled_at, live_stop, realised,
                          "stop" if not scaled else "trail_stop", mfe, mae)

        if scale_pct and not scaled and high >= t1:
            realised += scale_pct * (t1 - filled_at) / risk
            remaining -= scale_pct
            scaled = True
            if remaining <= 0:
                return _close(fwd, entry_idx, i, filled_at, t1, realised,
                              "target", mfe, mae)

        # Breakeven only where the variant asks for it.
        if cfg["breakeven"] and scaled and live_stop < filled_at:
            live_stop = filled_at

        # Structural trail: ride the most recent confirmed swing low. Only
        # ever raises the stop.
        if cfg["trail"] and (high - filled_at) / risk >= 1.0:
            sl = _swing_low(lows, i)
            if sl and sl > live_stop:
                live_stop = sl

        if not cfg["trail"] and not scale_pct and t2 and high >= t2:
            realised += remaining * (t2 - filled_at) / risk
            return _close(fwd, entry_idx, i, filled_at, t2, realised,
                          "target2", mfe, mae)

    last_i = min(entry_idx + max_hold, len(fwd)) - 1
    if last_i < entry_idx:
        return None
    exit_px = float(fwd.iloc[last_i]["close"])
    realised += remaining * (exit_px - filled_at) / risk
    return _close(fwd, entry_idx, last_i, filled_at, exit_px, realised,
                  "time", mfe, mae)


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


def _build_cross_section(conn, universe: list[str]) -> dict:
    """
    Cross-sectional rankings per session.

    The engine asks "is this chart good?" — an ABSOLUTE judgement on each
    stock in isolation. O'Neil, and the entire momentum literature, asks
    "is this stock stronger than the other 499?" — a RELATIVE one. Only the
    relative question has decades of out-of-sample evidence behind it.

    Builds, for every trading day:
      * each symbol's 126-day return percentile across the universe
      * each industry group's average return percentile

    RS is already computed on every signal and then ignored. This is what
    makes it usable as a filter rather than a decoration.
    """
    with conn.cursor() as cur:
        cur.execute("""
            select symbol, trade_date, adj_close
            from ohlcv_daily
            where symbol = any(%s)
            order by trade_date, symbol
        """, (universe,))
        rows = cur.fetchall()

        cur.execute("select symbol, industry from symbols "
                    "where industry is not null and symbol = any(%s)", (universe,))
        industry = dict(cur.fetchall())

    prices: dict[str, dict] = {}
    for sym, d, px in rows:
        prices.setdefault(sym, {})[d] = float(px)
    del rows

    all_dates = sorted({d for series in prices.values() for d in series})
    date_idx = {d: i for i, d in enumerate(all_dates)}

    LOOKBACK = 126
    rs_pct: dict[tuple, float] = {}
    grp_pct: dict[tuple, float] = {}

    for i in range(LOOKBACK, len(all_dates)):
        d, d_prev = all_dates[i], all_dates[i - LOOKBACK]
        returns = {}
        for sym, series in prices.items():
            now, then = series.get(d), series.get(d_prev)
            if now and then and then > 0:
                returns[sym] = now / then - 1
        if len(returns) < 50:
            continue

        ordered = sorted(returns.items(), key=lambda kv: kv[1])
        n = len(ordered)
        for rank, (sym, _) in enumerate(ordered):
            rs_pct[(sym, d)] = round(rank / (n - 1) * 100, 1)

        # Industry groups, ranked by their members' average return. Leaders
        # emerge from leading groups; `industry` was populated and never read.
        groups: dict[str, list] = {}
        for sym, r in returns.items():
            g = industry.get(sym)
            if g:
                groups.setdefault(g, []).append(r)
        if len(groups) >= 5:
            gavg = sorted(((g, sum(v) / len(v)) for g, v in groups.items()),
                          key=lambda kv: kv[1])
            gn = len(gavg)
            gmap = {g: round(rank / (gn - 1) * 100, 1)
                    for rank, (g, _) in enumerate(gavg)}
            for sym in returns:
                g = industry.get(sym)
                if g in gmap:
                    grp_pct[(sym, d)] = gmap[g]

    log.info("Cross-section built: %d sessions, %d industry groups",
             len(all_dates) - LOOKBACK, len({v for v in industry.values()}))
    return {"rs_pct": rs_pct, "group_pct": grp_pct, "industry": industry}


def _quintile(pct: float | None) -> str:
    if pct is None:
        return "unknown"
    if pct >= 80:
        return "q5_strongest"
    if pct >= 60:
        return "q4"
    if pct >= 40:
        return "q3"
    if pct >= 20:
        return "q2"
    return "q1_weakest"


def simulate_portfolio(trades: list[dict], max_positions: int = 8,
                       rank_by: str = "rs_pct", min_rs: float = 0.0) -> dict:
    """
    A portfolio, not a list of independent trades.

    The engine evaluates every setup in isolation and would have you holding
    all of them. Professional books are concentrated: a capped number of
    positions, the strongest candidates taken first, the rest declined.
    That changes results even with identical signals, because capacity
    forces selection.
    """
    dated = sorted([t for t in trades if t.get("entry_date") and t.get("exit_date")
                    and t.get("r_realised") is not None],
                   key=lambda t: (t["entry_date"], -(t.get(rank_by) or 0)))
    if not dated:
        return {"error": "no completed trades"}

    open_until: list = []
    taken, declined = [], 0

    for t in dated:
        if (t.get(rank_by) or 0) < min_rs:
            continue
        open_until = [d for d in open_until if d > t["entry_date"]]
        if len(open_until) >= max_positions:
            declined += 1
            continue
        open_until.append(t["exit_date"])
        taken.append(t)

    rs = [t["r_realised"] for t in taken]
    wins = [r for r in rs if r > 0]
    return {
        "max_positions": max_positions, "min_rs_percentile": min_rs,
        "taken": len(taken), "declined_no_capacity": declined,
        "hit_rate": round(len(wins) / len(rs), 3) if rs else None,
        "expectancy_r": round(sum(rs) / len(rs), 3) if rs else None,
        "total_r": round(sum(rs), 1) if rs else None,
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

        # The index's own trend state per session. Recorded alongside the
        # regime label because EVERY category flipped sign at the same date in
        # the split-sample test — which points at the market, not at any
        # pattern or setup type. A single, unambiguous condition is the way to
        # test that: was the Nifty above its own 200 DMA?
        nifty_ind = add_indicators(nifty.copy())
        index_state: dict[date, str] = {}
        for _, row in nifty_ind.iterrows():
            d = row["trade_date"].date()
            if pd.notna(row["sma200"]):
                index_state[d] = ("above_200dma" if row["close"] > row["sma200"]
                                  else "below_200dma")

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

        cross = _build_cross_section(conn, universe)
        rs_pct, grp_pct = cross["rs_pct"], cross["group_pct"]

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

                # Same signal, same entry, different exit policies. Any
                # difference is attributable to the exit alone.
                variants = {}
                for name, cfg in EXIT_VARIANTS.items():
                    v = _simulate_variant(fwd, setup.entry, setup.stop,
                                          setup.t1, setup.t2, cfg)
                    variants[name] = v["r_realised"] if v else None

                record = {
                    "symbol": sym, "signal_date": d,
                    "setup_type": setup.setup_type, "pattern": setup.pattern,
                    "contracting": "contracting" if setup.base.contracting
                                   else "not_contracting",
                    "score_total": setup.score_total,
                    "band": _band(setup.score_total), "regime": regime,
                    "index_state": index_state.get(d, "unknown"),
                    "rs_pct": rs_pct.get((sym, d)),
                    "rs_quintile": _quintile(rs_pct.get((sym, d))),
                    "group_pct": grp_pct.get((sym, d)),
                    "group_quintile": _quintile(grp_pct.get((sym, d))),
                    # Pre-specified intersection, not a general cross-tab.
                    # Crossing every setup type against every RSI zone would
                    # create 15 cells on 1,602 trades; something would look
                    # significant by chance alone. One named hypothesis only.
                    "breakout_rsi": (
                        "breakout_rsi_70_80"
                        if setup.setup_type.startswith("breakout")
                        and _rsi_zone(_rsi(window["close"].tolist()[-60:]))
                            == "overbought_70_80"
                        else "breakout_other_rsi"
                        if setup.setup_type.startswith("breakout")
                        else "not_breakout"),
                    "rsi": _rsi(window["close"].tolist()[-60:]),
                    "rsi_zone": _rsi_zone(_rsi(window["close"].tolist()[-60:])),
                    "entry_trigger": setup.entry, "stop_loss": setup.stop,
                    "t1": setup.t1, "t2": setup.t2,
                    "r_planned": setup.r_multiple_t1,
                }
                record.update(result or {"exit_reason": "never_triggered"})
                record["variants"] = variants
                trades.append(record)

            log.debug("%s: %d signals so far", sym, len(trades))

        return {"trades": trades, "sessions": len(trading_days),
                "universe": len(universe)}

    finally:
        conn.close()


def compare_exits(trades: list[dict]) -> dict:
    """
    Expectancy of each exit policy, on the SAME signals, in BOTH halves.

    Selection is held constant, so any difference is the exit. A variant only
    counts if it is positive in both halves — the same bar every other finding
    has had to clear.
    """
    dated = sorted([t for t in trades if t.get("signal_date") and t.get("variants")],
                   key=lambda t: t["signal_date"])
    if len(dated) < 40:
        return {"error": "too few signals"}
    mid = len(dated) // 2
    halves = {"first": dated[:mid], "second": dated[mid:]}

    out = {}
    for name in EXIT_VARIANTS:
        row = {}
        for half, subset in halves.items():
            rs = [t["variants"].get(name) for t in subset
                  if t["variants"].get(name) is not None]
            row[half] = {
                "filled": len(rs),
                "expectancy_r": round(sum(rs) / len(rs), 3) if rs else None,
                "total_r": round(sum(rs), 1) if rs else None,
                "hit_rate": round(sum(1 for r in rs if r > 0) / len(rs), 3) if rs else None,
                "avg_win": round(sum(r for r in rs if r > 0)
                                 / max(sum(1 for r in rs if r > 0), 1), 2) if rs else None,
            }
        allr = [t["variants"].get(name) for t in dated
                if t["variants"].get(name) is not None]
        row["full"] = {
            "filled": len(allr),
            "expectancy_r": round(sum(allr) / len(allr), 3) if allr else None,
            "total_r": round(sum(allr), 1) if allr else None,
        }
        ef, es = row["first"]["expectancy_r"], row["second"]["expectancy_r"]
        row["positive_in_both_halves"] = bool(
            ef is not None and es is not None and ef > 0 and es > 0)
        out[name] = row

    winners = [k for k, v in out.items() if v["positive_in_both_halves"]]
    out["verdict"] = {
        "positive_in_both_halves": winners,
        "note": ("Selection is identical across variants, so any difference is "
                 "the exit policy alone. A variant positive in only one half "
                 "is the same illusion the split test exists to catch."),
    }
    return out


def split_sample(trades: list[dict]) -> dict:
    """
    Does anything hold across BOTH halves of the period?

    The one-year run showed +0.113R; three years showed -0.106R. A result
    that reverses when the window moves is a property of the sample, not of
    the system. So every breakdown is recomputed on the first and second
    halves independently, and only findings with the same SIGN in both are
    reported as consistent.

    This is the guard against fitting to whichever slice happens to flatter
    the engine. Anything that fails it should not drive a change.
    """
    dated = sorted([t for t in trades if t.get("signal_date")],
                   key=lambda t: t["signal_date"])
    if len(dated) < 40:
        return {"error": "too few signals to split"}

    mid = len(dated) // 2
    first, second = dated[:mid], dated[mid:]
    boundary = str(dated[mid]["signal_date"])

    a, b = summarise(first), summarise(second)
    out = {
        "boundary_date": boundary,
        "first_half": {"n": len(first), "period":
                       f"{first[0]['signal_date']} to {first[-1]['signal_date']}",
                       "expectancy_r": a["overall"]["expectancy_r"]},
        "second_half": {"n": len(second), "period":
                        f"{second[0]['signal_date']} to {second[-1]['signal_date']}",
                        "expectancy_r": b["overall"]["expectancy_r"]},
        "consistent": {}, "inconsistent": {},
    }

    MIN_N = 25          # below this a half is noise, not evidence
    for key in ("band", "setup_type", "pattern", "regime", "index_state",
                "rsi_zone", "rs_quintile", "group_quintile", "contracting",
                "breakout_rsi"):
        for name in set(a.get(key, {})) | set(b.get(key, {})):
            sa, sb = a.get(key, {}).get(name), b.get(key, {}).get(name)
            if not sa or not sb:
                continue
            ea, eb = sa.get("expectancy_r"), sb.get("expectancy_r")
            na, nb = sa.get("filled") or 0, sb.get("filled") or 0
            if ea is None or eb is None:
                continue

            entry = {"first": ea, "second": eb, "n_first": na, "n_second": nb,
                     "underpowered": na < MIN_N or nb < MIN_N}
            label = f"{key}.{name}"
            # Same sign in both halves, and enough trades in each to mean it.
            if (ea > 0) == (eb > 0) and not entry["underpowered"]:
                out["consistent"][label] = entry
            else:
                out["inconsistent"][label] = entry

    positives = {k: v for k, v in out["consistent"].items()
                 if v["first"] > 0 and v["second"] > 0}
    # Does the engine only work with the index above its own 200 DMA? Tested
    # in BOTH halves, because a condition that only holds in one is the same
    # illusion the split test exists to catch.
    def _state(half, name):
        st = half.get("index_state", {}).get(name)
        return (st or {}).get("expectancy_r"), (st or {}).get("filled") or 0

    above_a, n_above_a = _state(a, "above_200dma")
    above_b, n_above_b = _state(b, "above_200dma")
    below_a, n_below_a = _state(a, "below_200dma")
    below_b, n_below_b = _state(b, "below_200dma")

    holds = (above_a is not None and above_b is not None
             and above_a > 0 and above_b > 0
             and n_above_a >= MIN_N and n_above_b >= MIN_N)

    out["index_filter"] = {
        "above_200dma": {"first": above_a, "second": above_b,
                         "n_first": n_above_a, "n_second": n_above_b},
        "below_200dma": {"first": below_a, "second": below_b,
                         "n_first": n_below_a, "n_second": n_below_b},
        "holds_in_both_halves": holds,
        "note": ("If trading only above the 200 DMA is positive in BOTH halves "
                 "with adequate sample, that is the first finding the data "
                 "actually supports. If not, the engine has no demonstrated "
                 "edge under any condition tested."),
    }

    out["verdict"] = {
        "overall_sign_stable": (out["first_half"]["expectancy_r"] or 0) > 0
                               == ((out["second_half"]["expectancy_r"] or 0) > 0),
        "consistently_positive": sorted(positives),
        "consistently_negative": sorted(k for k, v in out["consistent"].items()
                                        if v["first"] <= 0 and v["second"] <= 0),
        "note": ("Only findings with the same sign in both halves and at least "
                 f"{MIN_N} filled trades in each are treated as evidence. "
                 "Anything listed as inconsistent flipped when the window "
                 "moved and must not drive a change."),
    }
    return out


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
    for key in ("band", "regime", "setup_type", "pattern", "index_state",
                "rsi_zone", "rs_quintile", "group_quintile", "contracting",
                "breakout_rsi"):
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
