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
    regime_classifiers,
    add_indicators,
    evaluate_regime,
    gate0_tradability,
    gate3_trend_structure,
    relative_strength,
)
from ingest import connect
from setups import (BASE_SELECTION_STRATEGIES, DEFAULT_BASE_STRATEGY,
                    ENTRY_BUFFER, Rejected, build_setup)

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
# t2_exit is explicit per variant. It used to be inferred inside the
# simulator as `not cfg["trail"] and not scale_pct`, which for the baseline
# (scale_pct=50) silently evaluated False — so the baseline variant never
# took its T2 exit while the separate _simulate() function did. Two code
# paths both labelled "baseline" were modelling different policies, and the
# entire +0.058R vs +0.016R discrepancy between exit_variants.baseline and
# metrics.overall came from exactly that. Inferring behaviour from unrelated
# flags is how that hid; naming it prevents a repeat.
# Live exit policy. Changed from "baseline" to "let_it_run" after Run #28:
# three no-scale-out variants outperformed baseline by roughly 3x on
# full-period expectancy (0.072 -> 0.19-0.20), and the improvement held in
# BOTH halves. The individual winner varies run to run; the family is what
# is stable, so the default is set to a member of it rather than to whichever
# variant topped one run.
#
# EXIT_VARIANT=baseline restores the previous behaviour exactly.
EXIT_VARIANT = os.environ.get("EXIT_VARIANT", "let_it_run")

# How far a candidate strategy may trail the live default in one half and
# still be adoptable, provided it strictly beats in the other. Without this
# a 0.004R shortfall — noise on a 160-trade sample — was decisive.
ADOPT_TOLERANCE_R = float(os.environ.get("ADOPT_TOLERANCE_R", "0.05"))

EXIT_VARIANTS = {
    "baseline":        {"scale_pct": 50, "breakeven": True,  "trail": False, "max_hold": 40,  "t2_exit": True},
    "no_scale_out":    {"scale_pct": 0,  "breakeven": True,  "trail": False, "max_hold": 40,  "t2_exit": True},
    "no_breakeven":    {"scale_pct": 50, "breakeven": False, "trail": True,  "max_hold": 40,  "t2_exit": False},
    "let_it_run":      {"scale_pct": 0,  "breakeven": False, "trail": True,  "max_hold": 120, "t2_exit": False},
    "trail_only_long": {"scale_pct": 33, "breakeven": False, "trail": True,  "max_hold": 120, "t2_exit": False},
    # New: hold the full position to T2, scale there, trail the remainder.
    # Baseline scales at T1 and caps the winner early; let_it_run never
    # scales at all. This tests whether the gain comes from not scaling, or
    # simply from scaling LATER.
    "scale_out_at_t2": {"scale_pct": 50, "breakeven": False, "trail": True,  "max_hold": 120,
                        "t2_exit": False, "scale_at_t2": True},
}


def _swing_low(lows: list, i: int) -> float | None:
    """Most recent confirmed swing low at or before i: two higher lows each side."""
    for j in range(i - 2, 1, -1):
        w = lows[j - 2:j + 3]
        if len(w) == 5 and lows[j] == min(w):
            return lows[j]
    return None


def _simulate_variant(fwd: pd.DataFrame, entry: float, stop: float, t1: float,
                      t2: float | None, cfg: dict,
                      expiry_sessions: int | None = None) -> dict | None:
    """One exit policy. Entry logic is identical across variants."""
    filled_at = entry_idx = None
    # Explicit rather than read from the module global, so a what-if can
    # vary it without mutating engine state. Defaults to the live value.
    expiry = expiry_sessions if expiry_sessions is not None else EXPIRY_SESSIONS
    for i in range(min(expiry, len(fwd))):
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
    initial_stop = stop
    live_stop = stop
    # Trail forensics. exit_reason alone could not answer "did the trail
    # fire": the label read "stop" unless a scale-out had occurred, and
    # let_it_run has scale_pct=0, so EVERY exit was labelled "stop"
    # regardless of whether the stop had been moved. 162 stops / 0 trail
    # stops in Run #32 was a reporting artefact, not a finding.
    trail_activated = False
    mae_bar_index = mfe_bar_index = None
    stop_at_mfe = stop
    remaining = 1.0
    realised = 0.0
    scaled = False
    mfe = mae = 0.0
    lows = [float(x) for x in fwd["low"].tolist()]

    for i in range(entry_idx, min(entry_idx + max_hold, len(fwd))):
        bar = fwd.iloc[i]
        high, low = float(bar["high"]), float(bar["low"])
        bar_mfe = (high - filled_at) / risk
        bar_mae = (low - filled_at) / risk
        if bar_mfe > mfe:
            mfe, mfe_bar_index, stop_at_mfe = bar_mfe, i - entry_idx, live_stop
        if bar_mae < mae:
            mae, mae_bar_index = bar_mae, i - entry_idx

        if low <= live_stop:
            realised += remaining * (live_stop - filled_at) / risk
            # Labelled by whether the stop actually MOVED, not by whether a
            # scale-out happened. Those are different questions and the old
            # label answered the wrong one.
            reason = "trail_stop" if live_stop > initial_stop else "stop"
            return _close(fwd, entry_idx, i, filled_at, live_stop, realised,
                          reason, mfe, mae,
                          trail_activated=trail_activated,
                          initial_stop=initial_stop, stop_at_exit=live_stop,
                          stop_at_mfe=stop_at_mfe,
                          mae_bar_index=mae_bar_index,
                          mfe_bar_index=mfe_bar_index)

        # scale_at_t2 defers the partial exit to T2 instead of T1.
        scale_level = (t2 if cfg.get("scale_at_t2") and t2 else t1)
        if scale_pct and not scaled and high >= scale_level:
            realised += scale_pct * (scale_level - filled_at) / risk
            remaining -= scale_pct
            scaled = True
            if remaining <= 0:
                return _close(fwd, entry_idx, i, filled_at, scale_level, realised,
                              "target2" if cfg.get("scale_at_t2") else "target",
                              mfe, mae, trail_activated=trail_activated,
                              initial_stop=initial_stop, stop_at_exit=live_stop,
                              stop_at_mfe=stop_at_mfe,
                              mae_bar_index=mae_bar_index,
                              mfe_bar_index=mfe_bar_index)

        # Breakeven only where the variant asks for it.
        if cfg["breakeven"] and scaled and live_stop < filled_at:
            live_stop = filled_at

        # Structural trail: ride the most recent confirmed swing low. Only
        # ever raises the stop.
        if cfg["trail"] and (high - filled_at) / risk >= 1.0:
            sl = _swing_low(lows, i)
            if sl and sl > live_stop:
                live_stop = sl
                trail_activated = True

        if cfg.get("t2_exit") and t2 and high >= t2 and (scaled or not scale_pct):
            realised += remaining * (t2 - filled_at) / risk
            return _close(fwd, entry_idx, i, filled_at, t2, realised,
                          "target2", mfe, mae, trail_activated=trail_activated,
                          initial_stop=initial_stop, stop_at_exit=live_stop,
                          stop_at_mfe=stop_at_mfe,
                          mae_bar_index=mae_bar_index,
                          mfe_bar_index=mfe_bar_index)

    last_i = min(entry_idx + max_hold, len(fwd)) - 1
    if last_i < entry_idx:
        return None
    exit_px = float(fwd.iloc[last_i]["close"])
    realised += remaining * (exit_px - filled_at) / risk
    return _close(fwd, entry_idx, last_i, filled_at, exit_px, realised,
                  "time", mfe, mae, trail_activated=trail_activated,
                  initial_stop=initial_stop, stop_at_exit=live_stop,
                  stop_at_mfe=stop_at_mfe,
                  mae_bar_index=mae_bar_index, mfe_bar_index=mfe_bar_index)


def _simulate(fwd: pd.DataFrame, entry: float, stop: float, t1: float,
              t2: float | None) -> dict | None:
    """
    The headline simulation, now a thin wrapper over _simulate_variant.

    This was previously a SECOND full implementation of the same walk-forward
    logic, kept in parallel with _simulate_variant. The two drifted: the T2
    exit fired here but not in the baseline variant, so metrics.overall and
    exit_variants.baseline reported different results for identical trades
    and the exit comparison could not be trusted.

    One code path removes that entire class of bug rather than patching this
    instance of it.
    """
    return _simulate_variant(fwd, entry, stop, t1, t2,
                             EXIT_VARIANTS.get(EXIT_VARIANT,
                                               EXIT_VARIANTS["baseline"]))


def _close(fwd, entry_idx, exit_idx, filled_at, exit_px, realised_r,
           reason, mfe, mae, **forensics) -> dict:
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
        **forensics,
    }


_CLASSIFIER_CACHE: dict = {}


def _session_classifiers(nifty_ind: pd.DataFrame, d, breadth, vix_level) -> dict:
    """
    Every classifier's verdict for one session, cached per date.

    Uses the SAME regime_classifiers() the live gate calls, so backtest
    attribution and live behaviour cannot drift apart — a separate
    reimplementation here would be the base_strategy self-comparison bug
    all over again.
    """
    if d in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[d]

    hist = nifty_ind[nifty_ind["trade_date"] <= pd.Timestamp(d)]
    if len(hist) < 200:
        _CLASSIFIER_CACHE[d] = {}
        return {}

    vix_df = pd.DataFrame({"close": [vix_level]}) if vix_level is not None else None
    res = regime_classifiers(hist[["close"]].copy(), vix_df, breadth)
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    _CLASSIFIER_CACHE[d] = out
    return out


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


def _volume_asymmetry(window: pd.DataFrame, start_idx: int) -> float | None:
    """
    Buying pressure versus selling pressure WITHIN the base, as a ratio in
    [-1, 1]: (up-day volume - down-day volume) / total volume.

    The dry-up term in detect_base's quality formula only asks "how much
    total volume", never "on which days". A stock going quiet because
    smart money is patiently absorbing supply typically still shows more
    volume on its up days than its down days. A stock going quiet because
    nobody is interested shows no such asymmetry — volume is just uniformly
    thin. The formula currently scores both as identically "dried up",
    which is the leading hypothesis for why its top-rated quartile is the
    worst performer measured so far.
    """
    base_bars = window.iloc[start_idx:]
    if len(base_bars) < 5:
        return None
    closes = base_bars["close"].to_numpy()
    vols = base_bars["volume"].to_numpy()
    up_vol = float(vols[1:][closes[1:] > closes[:-1]].sum())
    down_vol = float(vols[1:][closes[1:] < closes[:-1]].sum())
    total = up_vol + down_vol
    if total <= 0:
        return None
    return (up_vol - down_vol) / total


def _asymmetry_band(a: float | None) -> str:
    if a is None:
        return "unknown"
    if a > 0.1:
        return "buying_pressure"
    if a < -0.1:
        return "selling_pressure"
    return "balanced"


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

        # Per-session breadth and VIX — needed for classifier attribution.
        # regimes[] above uses a hardcoded breadth of 50.0 because it only
        # needs a coarse label; the classifiers test breadth THRESHOLDS, so
        # a constant would make c7/c8 answer the same way on every session
        # and look like they separate nothing.
        breadth_by_date: dict[date, float] = {}
        with conn.cursor() as cur:
            cur.execute("""
                -- Window function in a CTE, aggregate in the outer query.
                -- These cannot be combined: avg(...) over (...) nested
                -- inside avg(...) is invalid SQL and failed the run in 8
                -- seconds with GroupingError.
                with dma as (
                    select o.symbol, o.trade_date, o.adj_close,
                           avg(o.adj_close) over (
                               partition by o.symbol order by o.trade_date
                               rows between 49 preceding and current row
                           ) as sma50,
                           count(*) over (
                               partition by o.symbol order by o.trade_date
                               rows between 49 preceding and current row
                           ) as bars
                    from ohlcv_daily o
                    join symbols s on s.symbol = o.symbol
                    where coalesce(s.series,'') <> 'INDEX'
                )
                select trade_date,
                       avg((adj_close > sma50)::int::float) * 100 as pct
                from dma
                -- bars = 50 excludes the warm-up window: a 3-bar average is
                -- not a 50DMA, and counting it would overstate early breadth.
                where sma50 is not null and bars = 50 and trade_date >= %s
                group by trade_date order by trade_date
            """, (from_date,))
            for d_, pct in cur.fetchall():
                if pct is not None:
                    breadth_by_date[d_] = float(pct)

            cur.execute("""
                select trade_date, adj_close from ohlcv_daily
                where symbol = 'INDIAVIX' and trade_date >= %s
                order by trade_date
            """, (from_date,))
            vix_by_date = {d_: float(v) for d_, v in cur.fetchall() if v is not None}
        log.info("Classifier inputs: breadth for %d sessions, VIX for %d",
                 len(breadth_by_date), len(vix_by_date))

        cross = _build_cross_section(conn, universe)
        rs_pct, grp_pct = cross["rs_pct"], cross["group_pct"]

        # Delivery-expansion state per (symbol, date), for the armed-quality
        # test below. Armed is 85% of every signal produced and the worst
        # performer in every run so far (-0.15R to -0.20R, three years
        # running) — it is the one setup type that asserts nothing has
        # happened yet, rather than reacting to a confirmed bar. Delivery
        # expansion is the one confirming signal the engine already computes
        # (for the Stage 1 transition path) but does not apply to ordinary
        # armed setups. This tests whether it should, WITHOUT changing what
        # the live engine gates on — measured first, same as everything else.
        with conn.cursor() as cur:
            cur.execute("""
                select symbol, trade_date, delivery_pct
                from delivery_daily
                where symbol = any(%s) and delivery_pct is not null
                order by symbol, trade_date
            """, (universe,))
            drows = cur.fetchall()
        delivery_by_symbol: dict[str, list] = {}
        for sym_, dd, pct in drows:
            delivery_by_symbol.setdefault(sym_, []).append((dd, float(pct)))
        del drows

        def _delivery_expanding(sym_: str, as_of: date) -> str:
            """'confirmed', 'not_confirmed', or 'unknown' (insufficient history)."""
            series = delivery_by_symbol.get(sym_)
            if not series:
                return "unknown"
            recent = [p for dd, p in series if dd <= as_of][-40:]
            if len(recent) < 20:
                return "unknown"
            last10 = sum(recent[-10:]) / min(10, len(recent[-10:]))
            prior = recent[:-10]
            if not prior:
                return "unknown"
            prior_avg = sum(prior) / len(prior)
            return "confirmed" if last10 > prior_avg * 1.1 else "not_confirmed"

        trades: list[dict] = []
        # One trade list per alternative base-selection strategy, built in
        # the SAME pass as the main loop below — gate0/gate3 filtering and
        # window slicing are identical regardless of how the base itself
        # gets chosen, so there is no reason to re-read OHLCV or re-run the
        # cheap gates three times. Only base selection and everything
        # downstream of it (trigger, levels, score, simulation) runs once
        # per strategy.
        # Alternatives are every strategy EXCEPT the one the main loop
        # already ran. This excluded "best_quality" by name, which was
        # correct only while best_quality was the default. Once the default
        # became first_valid the main trades were first_valid trades — and
        # compare_base_strategies still labelled them "best_quality" while
        # ALSO running first_valid as an alternative. The comparison was
        # first_valid against itself under two names, which is exactly why
        # Run #28 returned byte-identical buckets, splits and metrics.
        base_strategy_trades: dict[str, list[dict]] = {
            s: [] for s in BASE_SELECTION_STRATEGIES if s != DEFAULT_BASE_STRATEGY}
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
                        None,
                        # A true cross-sectional percentile. The Minervini
                        # gate needs this, not rs126 (which is percentage-
                        # point outperformance vs the index — different
                        # units entirely).
                        rs_rank_pct=rs_pct.get((sym, d)))
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

                # Alternative base-selection strategies, same window, same
                # regime/RS/score-floor logic — isolating base selection as
                # the only variable. A strategy may produce no setup at all
                # where the default did (different pivot, different trigger
                # outcome, different R:R) — that is itself part of what is
                # being measured, not an error.
                for strat_name in base_strategy_trades:
                    try:
                        alt_setup = build_setup(
                            sym, window,
                            relative_strength(window, nifty[nifty["trade_date"]
                                                            <= df["trade_date"].iloc[i]], 63),
                            relative_strength(window, nifty[nifty["trade_date"]
                                                            <= df["trade_date"].iloc[i]], 126),
                            None, base_strategy=strat_name)
                    except Rejected:
                        continue

                    alt_ceiling = float(alt_setup.score_breakdown.get(
                        "max_possible", 100)) or 100
                    if alt_setup.score_total < floor_pct * alt_ceiling / 100.0:
                        continue

                    alt_result = _simulate(fwd, alt_setup.entry, alt_setup.stop,
                                           alt_setup.t1, alt_setup.t2)
                    base_strategy_trades[strat_name].append({
                        "symbol": sym, "signal_date": d,
                        "setup_type": alt_setup.setup_type,
                        "score_total": alt_setup.score_total,
                        "band": _band(alt_setup.score_total), "regime": regime,
                        "strategy_used": alt_setup.base.strategy_used,
                        # Taken from the Setup that produced this signal, not
                        # from the harness env var: a trade whose base came
                        # from the flag fallback has strategy_used=None while
                        # strategy_requested still records what was asked for.
                        "strategy_requested": alt_setup.strategy_requested,
                        "base_start_idx": alt_setup.base.start_idx,
                        "base_duration": alt_setup.base.duration,
                        "base_quality": round(alt_setup.base.quality, 2),
                        **(alt_result or {"exit_reason": "never_triggered"}),
                    })

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
                    "breadth_pct": breadth_by_date.get(d),
                    "regime_classifiers": _session_classifiers(
                        nifty_ind, d, breadth_by_date.get(d), vix_by_date.get(d)),
                    "rs_pct": rs_pct.get((sym, d)),
                    "rs_quintile": _quintile(rs_pct.get((sym, d))),
                    "group_pct": grp_pct.get((sym, d)),
                    "group_quintile": _quintile(grp_pct.get((sym, d))),
                    # Base-quality quartile: the INTERNAL formula detect_base
                    # uses to pick which of ~20 candidate lookback windows to
                    # keep (tightness, dry-up, contraction, duration, prior
                    # gain) has never itself been checked against outcomes —
                    # only the downstream trade score has. If quality does
                    # not separate results either, the window-selection
                    # formula is picking noise that merely looks tidy.
                    # C1: recorded per trade so base selection can be
                    # attributed rather than inferred.
                    "strategy_used": setup.base.strategy_used,
                    "strategy_requested": setup.strategy_requested,
                    "base_start_idx": setup.base.start_idx,
                    "base_duration": setup.base.duration,
                    "base_quality": round(setup.base.quality, 2),
                    # C4 shape diagnostics, flattened for SQL.
                    "handle_slope": setup.base.shape_diag.get("handle_slope"),
                    "handle_slope_pct": setup.base.shape_diag.get("handle_slope_pct"),
                    "handle_depth_pct": setup.base.shape_diag.get("handle_depth_pct"),
                    "cup_shape": setup.base.shape_diag.get("cup_shape"),
                    "cup_rounding_bars": setup.base.shape_diag.get("cup_rounding_bars"),
                    "cup_low_idx_in_window": setup.base.shape_diag.get("cup_low_idx_in_window"),
                    "handle_start_idx": setup.base.shape_diag.get("handle_start_idx"),
                    "shape_diag": setup.base.shape_diag,
                    # (A) entry-timing
                    "trigger_volume_vs_50d": setup.trigger_diag.get("trigger_volume_vs_50d"),
                    "pct_above_ema20_at_entry": setup.trigger_diag.get("pct_above_ema20_at_entry"),
                    "trigger_close_position_in_range": setup.trigger_diag.get(
                        "trigger_close_position_in_range"),

                    "base_quality_quartile": _quintile(
                        min(setup.base.quality, 100)) if setup.base.quality is not None
                        else "unknown",

                    # Diagnostic-only, testing a specific hypothesis: the
                    # quality formula scored q4 as its BEST bucket and it is
                    # measurably the WORST performer (-0.568R, stable across
                    # both halves). One plausible mechanism — the formula
                    # rewards tightness and low volume heavily, and a stock
                    # nobody trades will score as "tight and quiet" exactly
                    # like one genuinely under accumulation. These three
                    # fields let that be checked directly rather than
                    # guessed at: is q4's failure concentrated in the
                    # shallowest, thinnest-liquidity, weakest-prior-move
                    # sub-segment?
                    "base_depth_band": (
                        "under_6pct" if setup.base.depth_pct < 6 else
                        "6_to_15pct" if setup.base.depth_pct < 15 else
                        "15_to_25pct" if setup.base.depth_pct < 25 else
                        "over_25pct"),
                    "liquidity_band": (
                        "under_10cr" if pd.isna(window["turnover20_cr"].iloc[-1])
                        or float(window["turnover20_cr"].iloc[-1]) < 10 else
                        "10_to_30cr" if float(window["turnover20_cr"].iloc[-1]) < 30 else
                        "over_30cr"),
                    "prior_move_band": (
                        "near_floor_25_35pct" if setup.base.prior_uptrend_pct < 35 else
                        "35_to_60pct" if setup.base.prior_uptrend_pct < 60 else
                        "over_60pct"),
                    "volume_asymmetry_band": _asymmetry_band(
                        _volume_asymmetry(window, setup.base.start_idx)),

                    # Armed-specific tests. Both null for breakout/pullback —
                    # only meaningful where the setup type has no confirmed
                    # trigger bar of its own.
                    "armed_delivery": (
                        _delivery_expanding(sym, d)
                        if setup.setup_type.startswith("armed") else "n/a"),
                    "armed_distance_band": (
                        ("tight_0_2pct" if (setup.base.pivot / float(window["close"].iloc[-1]) - 1) * 100 <= 2
                         else "wide_2_4pct")
                        if setup.setup_type.startswith("armed") else "n/a"),

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
                # result carries the trail forensics (trail_activated,
                # stop_at_exit, mae_bar_index, ...) from _simulate_variant.
                record.update(result or {"exit_reason": "never_triggered"})
                record["variants"] = variants
                trades.append(record)

            log.debug("%s: %d signals so far", sym, len(trades))

        return {"trades": trades, "sessions": len(trading_days),
                "universe": len(universe),
                "base_strategy_trades": base_strategy_trades}

    finally:
        conn.close()


def _exit_reason_counts(trades: list[dict], variant: str) -> dict:
    """
    Exit-reason distribution per variant.

    Only the live-path simulation records a per-trade exit_reason; the
    variant side-simulations record R alone. Reported for the live variant
    and left empty for the others rather than inventing a number, so an
    absent distribution is visibly absent.
    """
    if variant != EXIT_VARIANT:
        return {"note": "recorded for the live variant only"}
    counts: dict = {}
    for t in trades:
        if t["variants"].get(variant) is None:
            continue
        reason = t.get("exit_reason") or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


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
        wins = [r for r in allr if r > 0]
        losses = [r for r in allr if r <= 0]
        row["full"] = {
            "filled": len(allr),
            "expectancy_r": round(sum(allr) / len(allr), 3) if allr else None,
            "total_r": round(sum(allr), 1) if allr else None,
            # Per-variant detail required by the C2 manifest. Additive —
            # every field the previous output had is still present above.
            "hit_rate": round(len(wins) / len(allr), 3) if allr else None,
            "avg_winner_r": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loser_r": round(sum(losses) / len(losses), 2) if losses else None,
            "avg_bars_held": round(
                sum(t.get("bars_held") or 0 for t in dated
                    if t["variants"].get(name) is not None)
                / max(len(allr), 1), 1) if allr else None,
            "exit_reasons": _exit_reason_counts(dated, name),
        }
        row["is_live_default"] = (name == EXIT_VARIANT)
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


def diagnose_quality_quartile(trades: list[dict], quartile: str = "q4") -> dict:
    """
    Why does the quality formula's OWN top-rated bucket underperform?

    Filters to one quartile and breaks it down by three dimensions the
    formula does not see: base depth, 20-day liquidity, and the strength of
    the prior advance. This is a targeted follow-up on ONE specific finding
    (q4 at -0.568R, stable across both halves) — not a general sweep, and
    each sub-cut still needs the same split-sample discipline before being
    treated as an answer rather than a lead.
    """
    cohort = [t for t in trades if t.get("base_quality_quartile") == quartile]
    dated = sorted([t for t in cohort if t.get("signal_date")],
                   key=lambda t: t["signal_date"])
    if len(dated) < 20:
        return {"error": f"too few {quartile} signals to diagnose", "n": len(dated)}

    mid = len(dated) // 2
    halves = {"first": dated[:mid], "second": dated[mid:]}

    def stats(subset):
        rs = [t["r_realised"] for t in subset if t.get("r_realised") is not None]
        return {"n": len(subset), "filled": len(rs),
                "expectancy_r": round(sum(rs) / len(rs), 3) if rs else None,
                "hit_rate": round(sum(1 for r in rs if r > 0) / len(rs), 3) if rs else None}

    out = {"quartile": quartile, "cohort_size": len(cohort),
          "overall": stats(cohort)}

    for dim in ("base_depth_band", "liquidity_band", "prior_move_band",
                       "volume_asymmetry_band"):
        out[dim] = {}
        for val in sorted({t.get(dim) for t in cohort if t.get(dim)}):
            sub = [t for t in cohort if t.get(dim) == val]
            row = {"full": stats(sub)}
            for half_name, half_trades in halves.items():
                row[half_name] = stats([t for t in half_trades if t.get(dim) == val])
            out[dim][val] = row

    return out


def compare_base_strategies(main_trades: list[dict],
                            base_strategy_trades: dict[str, list[dict]]) -> dict:
    """
    Does an alternative base-selection rule beat the current one?

    base_quality_quartile.q4 — the live formula's own TOP-rated bucket — was
    the worst performer in the backtest, stable across both halves, and four
    separate follow-up checks (liquidity, depth, prior-move strength, volume
    asymmetry) each found the badness spread uniformly rather than
    concentrated in an identifiable confound. That points at the SELECTION
    MECHANISM — best of ~20 scored candidate windows — rather than any one
    input to the score.

    "best_quality" here is the CURRENT LIVE approach, included as the
    baseline so the comparison is apples-to-apples rather than trusting a
    number computed at a different time. A strategy is only worth adopting
    if it beats best_quality in BOTH halves — the same bar every other
    finding in this backtest has had to clear.
    """
    # Keyed by the strategy that ACTUALLY produced them. Hardcoding
    # "best_quality" here mislabelled the main group the moment the default
    # changed, and silently turned the comparison into a self-comparison.
    all_strategies = {DEFAULT_BASE_STRATEGY: main_trades, **base_strategy_trades}

    def stats(subset):
        rs = [t["r_realised"] for t in subset if t.get("r_realised") is not None]
        wins = [r for r in rs if r > 0]
        return {
            "signals": len(subset), "filled": len(rs),
            "hit_rate": round(len(wins) / len(rs), 3) if rs else None,
            "expectancy_r": round(sum(rs) / len(rs), 3) if rs else None,
            "total_r": round(sum(rs), 1) if rs else None,
        }

    out = {}
    for name, trs in all_strategies.items():
        dated = sorted([t for t in trs if t.get("signal_date")],
                       key=lambda t: t["signal_date"])
        if len(dated) < 40:
            out[name] = {"error": "too few signals", "n": len(dated)}
            continue
        mid = len(dated) // 2
        first, second = stats(dated[:mid]), stats(dated[mid:])
        out[name] = {
            "full": stats(dated),
            "first_half": first, "second_half": second,
            "positive_in_both_halves": bool(
                first["expectancy_r"] is not None and second["expectancy_r"] is not None
                and first["expectancy_r"] > 0 and second["expectancy_r"] > 0),
            "beats_live_default_both_halves": None,   # filled in below
        }

    baseline = out.get(DEFAULT_BASE_STRATEGY, {})
    for name, row in out.items():
        if name == DEFAULT_BASE_STRATEGY or "error" in row:
            continue
        bf, bs = baseline.get("first_half", {}), baseline.get("second_half", {})
        if (row["first_half"].get("expectancy_r") is not None
                and bf.get("expectancy_r") is not None
                and row["second_half"].get("expectancy_r") is not None
                and bs.get("expectancy_r") is not None):
            # Tolerance: beats-or-ties within ADOPT_TOLERANCE_R in BOTH
            # halves, and strictly beats in at least one.
            #
            # The strict rule rejected best_quality for trailing by 0.004R
            # in one half (+0.406 vs +0.410) while leading on aggregate and
            # being positive in both. A 0.004R gap on ~160 trades is noise,
            # not a preference — treating it as decisive made the rule
            # arbitrary at the margin.
            f_delta = row["first_half"]["expectancy_r"] - bf["expectancy_r"]
            s_delta = row["second_half"]["expectancy_r"] - bs["expectancy_r"]
            row["first_half_delta_r"] = round(f_delta, 4)
            row["second_half_delta_r"] = round(s_delta, 4)
            row["beats_live_default_both_halves"] = bool(
                f_delta >= -ADOPT_TOLERANCE_R
                and s_delta >= -ADOPT_TOLERANCE_R
                and (f_delta > 0 or s_delta > 0))

    # Task 5's comparison, produced automatically rather than by hand:
    # do the strategies actually pick DIFFERENT windows? Identical
    # distributions across all four would mean either genuine convergence
    # or a plumbing fault, and the quality spread distinguishes them.
    for name, trs in all_strategies.items():
        if "error" in out.get(name, {}):
            continue
        quals = [t["base_quality"] for t in trs if t.get("base_quality") is not None]
        starts = [t["base_start_idx"] for t in trs if t.get("base_start_idx") is not None]
        if quals:
            ordered = sorted(quals)
            out[name]["base_selection"] = {
                "n_with_quality": len(quals),
                "quality_mean": round(sum(quals) / len(quals), 2),
                "quality_median": round(ordered[len(ordered) // 2], 2),
                "quality_min": round(min(quals), 2),
                "quality_max": round(max(quals), 2),
                "distinct_start_idx": len(set(starts)),
            }

    # How many signals did each alternative select a DIFFERENT window for
    # than the live default? Zero across the board is the signature of a
    # plumbing fault.
    default_by_key = {
        (t.get("symbol"), t.get("signal_date")): t.get("base_start_idx")
        for t in all_strategies.get(DEFAULT_BASE_STRATEGY, [])}
    for name, trs in all_strategies.items():
        if name == DEFAULT_BASE_STRATEGY or "error" in out.get(name, {}):
            continue
        compared = differing = 0
        for t in trs:
            key = (t.get("symbol"), t.get("signal_date"))
            if key in default_by_key and t.get("base_start_idx") is not None:
                compared += 1
                if default_by_key[key] != t["base_start_idx"]:
                    differing += 1
        out[name]["vs_live_default"] = {
            "signals_compared": compared,
            "different_base_selected": differing,
            "pct_different": round(differing / compared * 100, 1) if compared else None,
        }

    winners = [n for n, r in out.items()
              if n != DEFAULT_BASE_STRATEGY
              and r.get("beats_live_default_both_halves")]

    # Deterministic tie-break. The rule was
    #   winners[0] if len(winners) == 1 else None
    # which returns null whenever TWO strategies qualify — reading as
    # "nothing qualified" when the truth was "two did and the code could
    # not choose". Run #34 reported adopt_candidate=null with both
    # first_valid and best_quality qualifying; the tolerance rule was
    # working and this line discarded the result.
    ADOPT_EXPECTANCY_MARGIN_R = float(
        os.environ.get("ADOPT_EXPECTANCY_MARGIN_R", "0.02"))
    # A strategy winning on expectancy by a hair while giving up a third of
    # the fills is not an improvement: Run #35's best_quality led by 0.025R
    # but cost 10R of total return and 35 fills. Expectancy alone ranked it
    # first, which is the wrong answer for a book that has to compound.
    ADOPT_STRONG_MARGIN_R = float(
        os.environ.get("ADOPT_STRONG_MARGIN_R", "0.05"))
    ADOPT_MIN_FILL_RETENTION = float(
        os.environ.get("ADOPT_MIN_FILL_RETENTION", "0.90"))

    adopt, tie, reason = None, [], None
    strict = [n for n in winners
              if (out[n].get("first_half_delta_r") or 0) > 0
              and (out[n].get("second_half_delta_r") or 0) > 0]

    if len(strict) == 1:
        adopt = strict[0]
        reason = "sole strategy beating strictly in both halves"
    elif winners:
        def _exp(n):
            return out[n].get("full", {}).get("expectancy_r") or -99

        def _total(n):
            return out[n].get("full", {}).get("total_r") or -99

        def _fills(n):
            return out[n].get("full", {}).get("filled") or 0

        # Default preference is TOTAL R, not expectancy. Expectancy ranks a
        # strategy that takes 10 excellent trades above one that takes 100
        # good ones — fine as a per-trade statistic, wrong as a choice of
        # what to run.
        by_total = sorted(winners, key=_total, reverse=True)
        top_total = by_total[0]
        max_fills = max(_fills(n) for n in winners) or 1

        # An expectancy winner overrides only if the edge is STRONG and it
        # has not thrown away the fills to get there.
        override = [
            n for n in winners
            if _exp(n) - _exp(top_total) >= ADOPT_STRONG_MARGIN_R
            and _fills(n) / max_fills >= ADOPT_MIN_FILL_RETENTION]

        if override:
            adopt = max(override, key=_exp)
            reason = (f"beats on expectancy by >= {ADOPT_STRONG_MARGIN_R}R "
                      f"while retaining >= {ADOPT_MIN_FILL_RETENTION:.0%} of fills")
        else:
            adopt = top_total
            reason = (f"highest total R ({_total(top_total)}) among strategies "
                      f"beating within {ADOPT_TOLERANCE_R}R in both halves; "
                      "no rival cleared the strong-margin + fill-retention test")
            tie = [n for n in winners
                   if abs(_total(n) - _total(top_total)) < 1.0 and n != top_total]
    else:
        reason = "no strategy qualified"

    # The live default is itself a candidate: if it leads, the answer is
    # "no change", which is different from "nothing qualified".
    live_exp = out.get(DEFAULT_BASE_STRATEGY, {}).get("full", {}).get("expectancy_r")
    recommended = adopt or DEFAULT_BASE_STRATEGY
    if (adopt and live_exp is not None
            and (out[adopt].get("full", {}).get("expectancy_r") or -99)
                - live_exp < ADOPT_EXPECTANCY_MARGIN_R):
        recommended = DEFAULT_BASE_STRATEGY
        reason += " (but does not clear the margin over the live default)"

    out["verdict"] = {
        "adopt_candidate": adopt,
        "recommended_strategy": recommended,
        "action_required": recommended != DEFAULT_BASE_STRATEGY,
        "tie_between": tie,
        "decision_reason": reason,
        "expectancy_margin_r": ADOPT_EXPECTANCY_MARGIN_R,
        "beats_baseline_both_halves": winners,
        "live_default": DEFAULT_BASE_STRATEGY,
        "adopt_tolerance_r": ADOPT_TOLERANCE_R,
        "note": ("A strategy must beat or tie (within "
                 f"{ADOPT_TOLERANCE_R}R) the CURRENT LIVE DEFAULT in "
                 "BOTH halves to be a candidate for live code. Beating it only "
                 "in the full-period average, or in one half, is the same "
                 "illusion the split test exists to catch."),
    }
    return out


def breadth_filter_windows(trades: list[dict], min_pct: float = 55.0,
                           windows: int = 6) -> dict:
    """
    (a) — per-window expectancy when ONLY c7-passing signals are counted.

    c7 shows the largest taken/skipped separation of any classifier
    (+0.508R) but does not flip the walk-forward as a day-level gate. The
    question this answers is narrower and more useful: do W1, W3 and W5 —
    the losing windows — actually improve once weak-breadth signals are
    dropped, or does the filter simply remove signals evenly and leave the
    sign unchanged?
    """
    dated = sorted([t for t in trades
                    if t.get("signal_date") and t.get("r_realised") is not None],
                   key=lambda t: t["signal_date"])
    with_breadth = [t for t in dated if t.get("breadth_pct") is not None]
    if len(with_breadth) < 20:
        return {"error": "insufficient breadth data",
                "n_with_breadth": len(with_breadth), "n_total": len(dated)}

    first, last = dated[0]["signal_date"], dated[-1]["signal_date"]
    step = max((last - first).days, 1) / windows

    def stat(rows):
        rs = [t["r_realised"] for t in rows]
        if not rs:
            return {"n": 0, "expectancy_r": None}
        return {"n": len(rs), "expectancy_r": round(sum(rs) / len(rs), 3),
                "total_r": round(sum(rs), 1)}

    out, improved = [], 0
    for w in range(windows):
        lo = first + timedelta(days=int(step * w))
        hi = first + timedelta(days=int(step * (w + 1)))
        in_w = [t for t in with_breadth if lo <= t["signal_date"] < hi]
        passing = [t for t in in_w if t["breadth_pct"] >= min_pct]
        blocked = [t for t in in_w if t["breadth_pct"] < min_pct]

        a, f = stat(in_w), stat(passing)
        better = (f["expectancy_r"] is not None and a["expectancy_r"] is not None
                  and f["expectancy_r"] > a["expectancy_r"])
        if better:
            improved += 1
        out.append({
            "window": w + 1, "from": str(lo), "to": str(hi),
            "unfiltered": a, "c7_passing": f, "c7_blocked": stat(blocked),
            "filter_improves_window": better,
            "sign_flipped_positive": bool(
                a["expectancy_r"] is not None and f["expectancy_r"] is not None
                and a["expectancy_r"] <= 0 < f["expectancy_r"]),
        })

    flipped = [w["window"] for w in out if w["sign_flipped_positive"]]
    return {
        "min_breadth_pct": min_pct,
        "windows": out,
        "windows_improved": improved,
        "windows_flipped_to_positive": flipped,
        "note": ("A window counts as flipped only if it was <=0 unfiltered "
                 "and >0 with the filter. Improving a window that was "
                 "already positive does not move the walk-forward."),
    }


def _window_outcomes(trades: list[dict], windows: int = 6) -> tuple[set, set, list]:
    """
    Which windows won and which lost, computed FROM THIS RUN.

    Previously hardcoded as {1,3,5} losing and {4,6} winning, taken from
    Run #29. Run #32 moved W3 to +0.023, so the static list declared a
    positive window "losing" and any classifier skipping it was judged
    against the wrong target. Window outcomes shift between runs; the
    reference must shift with them.
    """
    dated = sorted([t for t in trades
                    if t.get("signal_date") and t.get("r_realised") is not None],
                   key=lambda t: t["signal_date"])
    if not dated:
        return set(), set(), []

    first, last = dated[0]["signal_date"], dated[-1]["signal_date"]
    step = max((last - first).days, 1) / windows
    losing, winning, detail = set(), set(), []

    for w in range(windows):
        lo = first + timedelta(days=int(step * w))
        hi = first + timedelta(days=int(step * (w + 1)))
        rs = [t["r_realised"] for t in dated if lo <= t["signal_date"] < hi]
        if not rs:
            detail.append({"window": w + 1, "n": 0, "expectancy_r": None,
                           "outcome": "empty"})
            continue
        exp = sum(rs) / len(rs)
        (winning if exp > 0 else losing).add(w + 1)
        detail.append({"window": w + 1, "n": len(rs),
                       "expectancy_r": round(exp, 3),
                       "outcome": "winning" if exp > 0 else "losing"})
    return losing, winning, detail


def never_triggered_whatif(conn, run_id: int, extended_expiry: int = 20,
                           variant: str | None = None) -> dict:
    """
    Task 5.4 — what the never-triggered signals would have done.

    Reads stored signals and replays them through the SAME
    _simulate_variant the engine uses, with only the expiry window
    changed. Reimplementing the exit logic here would measure a copy of
    the engine rather than the engine.

    Nothing is written and no engine parameter changes.
    """
    cfg = EXIT_VARIANTS.get(variant or EXIT_VARIANT, EXIT_VARIANTS["let_it_run"])

    with conn.cursor() as cur:
        cur.execute("""
            select symbol, signal_date, setup_type, pattern,
                   entry_trigger, stop_loss, t1, t2
            from backtest_trades
            where run_id = %s and r_realised is null
              and entry_trigger is not null and stop_loss is not null
            order by signal_date
        """, (run_id,))
        rows = cur.fetchall()

    if not rows:
        return {"error": "no never-triggered signals with levels",
                "run_id": run_id}

    filled, never, by_bucket = [], 0, {}
    # Per setup type as well as per bucket. armed and pullback have already
    # diverged sharply on fill-later rate (40% vs 73%) and are different
    # populations; a combined expectancy would average the answer away.
    by_setup: dict = {}
    for sym, sig_date, setup_type, pattern, entry, stop, t1, t2 in rows:
        with conn.cursor() as cur:
            cur.execute("""
                select trade_date, adj_open, adj_high, adj_low, adj_close
                from ohlcv_daily
                where symbol = %s and trade_date > %s
                order by trade_date limit %s
            """, (sym, sig_date, extended_expiry + cfg["max_hold"]))
            bars = cur.fetchall()
        if len(bars) < 2:
            continue

        fwd = pd.DataFrame(bars, columns=["trade_date", "open", "high",
                                          "low", "close"])
        for c in ("open", "high", "low", "close"):
            fwd[c] = pd.to_numeric(fwd[c])

        # Search for the trigger ONLY within the extended window.
        #
        # This previously searched the whole frame, which holds
        # extended_expiry + max_hold bars (140 with the defaults) because
        # the simulator needs the tail to run the exit. A signal first
        # touching its trigger on session 87 was therefore counted as
        # "would fill" and bucketed day_11_20. That is the 36-vs-25
        # discrepancy against the SQL, which correctly limits to 20: the
        # endpoint was under-reporting never-triggered and over-reporting
        # late fills, with some "late fills" months away.
        search = fwd.iloc[:extended_expiry]
        hit = search.index[search["high"] >= float(entry)]
        if len(hit) == 0:
            never += 1
            continue
        session = int(hit[0]) + 1
        if session <= EXPIRY_SESSIONS:
            # Should already have filled — a data mismatch, not a finding.
            by_bucket.setdefault("would_have_filled_in_original_window", []).append(None)
            continue

        # Extended window passed explicitly — no global mutation, so a
        # concurrent scan cannot see a changed expiry.
        res = _simulate_variant(fwd, float(entry), float(stop),
                                float(t1) if t1 else float(entry) * 1.05,
                                float(t2) if t2 else None, cfg,
                                expiry_sessions=extended_expiry)

        if res and res.get("r_realised") is not None:
            bucket = "day_6_10" if session <= 10 else "day_11_20"
            rec = {"symbol": sym, "signal_date": sig_date,
                   "setup_type": setup_type, "pattern": pattern,
                   "trigger_session": session,
                   "r_realised": res["r_realised"],
                   "exit_reason": res["exit_reason"],
                   "max_favourable_r": res["max_favourable_r"],
                   "max_adverse_r": res["max_adverse_r"]}
            filled.append(rec)
            by_bucket.setdefault(bucket, []).append(rec)
            by_setup.setdefault(setup_type or "unknown", []).append(rec)

    def stat(recs):
        rs = [r["r_realised"] for r in recs if isinstance(r, dict)]
        if not rs:
            return {"n": len([r for r in recs if r is None]) or 0,
                    "expectancy_r": None}
        wins = [r for r in rs if r > 0]
        return {"n": len(rs), "expectancy_r": round(sum(rs) / len(rs), 3),
                "total_r": round(sum(rs), 2),
                "hit_rate": round(len(wins) / len(rs), 3)}

    return {
        "run_id": run_id,
        "exit_variant": variant or EXIT_VARIANT,
        "original_expiry_sessions": EXPIRY_SESSIONS,
        "extended_expiry_sessions": extended_expiry,
        "never_triggered_examined": len(rows),
        "never_trigger_in_20": never,
        "by_bucket": {k: stat(v) for k, v in by_bucket.items()},
        "by_setup_type": {k: stat(v) for k, v in by_setup.items()},
        "all_late_fills": stat(filled),
        "scope": ("The 39 LATE-FILL signals only. The 47 that never trigger "
                  "within 20 sessions are a DISJOINT population and are not "
                  "addressed here — 18 of those are narrow misses (<1% short "
                  "of trigger) and need a separate ENTRY_BUFFER experiment, "
                  "not an expiry one."),
        "note": ("Hypothetical. These trades were never taken. A positive "
                 "expectancy here means the 5-session expiry is discarding "
                 "edge; a negative one means it is working as intended."),
    }


def anchor_whatif(conn, run_id: int, entry_buffer: float = 0.0,
                  max_miss_pct: float = 1.0, lookahead: int = 20,
                  variant: str | None = None) -> dict:
    """
    Would the NARROW MISSES have filled and paid at a tighter anchor?

    A disjoint question from never_triggered_whatif: that one asks whether
    a LONGER WINDOW helps signals that eventually reached their trigger.
    This asks whether a CLOSER ENTRY helps signals that never reached it
    but came within max_miss_pct.

    WHAT IS ACTUALLY RECOMPUTED — and it is not the pivot.

    pivot is not stored in backtest_trades. What IS exactly recoverable is
    the ANCHOR:  anchor = entry_trigger / (1 + ENTRY_BUFFER).

    For a breakout the anchor is max(pivot, trigger_bar_high), so it can
    sit ABOVE the pivot; for a pullback it is the trigger bar's high and
    is unrelated to the pivot. Calling this a pivot test would overstate
    it. entry_buffer=0.0 therefore means "enter AT the anchor instead of
    0.25% above it" — which is the honest form of the question and still
    the right one, since the buffer is the part under our control.

    Nothing is written and no engine parameter changes.
    """
    cfg = EXIT_VARIANTS.get(variant or EXIT_VARIANT, EXIT_VARIANTS["let_it_run"])

    with conn.cursor() as cur:
        cur.execute("""
            select symbol, signal_date, setup_type, pattern,
                   entry_trigger, stop_loss, t1, t2
            from backtest_trades
            where run_id = %s and r_realised is null
              and entry_trigger is not null and stop_loss is not null
            order by signal_date
        """, (run_id,))
        rows = cur.fetchall()

    if not rows:
        return {"error": "no never-triggered signals with levels", "run_id": run_id}

    per_signal, filled = [], []
    examined = skipped_not_narrow = 0

    for sym, sig_date, setup_type, pattern, entry, stop, t1, t2 in rows:
        entry = float(entry)
        with conn.cursor() as cur:
            cur.execute("""
                select trade_date, adj_open, adj_high, adj_low, adj_close
                from ohlcv_daily
                where symbol = %s and trade_date > %s
                order by trade_date limit %s
            """, (sym, sig_date, lookahead + cfg["max_hold"]))
            bars = cur.fetchall()
        if len(bars) < 2:
            continue

        fwd = pd.DataFrame(bars, columns=["trade_date", "open", "high",
                                          "low", "close"])
        for c in ("open", "high", "low", "close"):
            fwd[c] = pd.to_numeric(fwd[c])

        window = fwd.iloc[:lookahead]
        best_high = float(window["high"].max())
        if best_high >= entry:
            continue                     # reached the original trigger; not a miss
        miss_pct = (best_high / entry - 1) * 100
        examined += 1
        if miss_pct < -abs(max_miss_pct):
            skipped_not_narrow += 1
            continue                     # not a NARROW miss

        anchor = entry / (1 + ENTRY_BUFFER)
        new_trigger = anchor * (1 + entry_buffer)
        would_fill = best_high >= new_trigger

        rec = {"symbol": sym, "signal_date": str(sig_date),
               "setup_type": setup_type, "pattern": pattern,
               "original_trigger": round(entry, 2),
               "anchor": round(anchor, 2),
               "new_trigger": round(new_trigger, 2),
               "best_high_in_window": round(best_high, 2),
               "miss_pct": round(miss_pct, 3),
               "would_fill": bool(would_fill), "exit_r": None}

        if would_fill:
            res = _simulate_variant(fwd, new_trigger, float(stop),
                                    float(t1) if t1 else new_trigger * 1.05,
                                    float(t2) if t2 else None, cfg,
                                    expiry_sessions=lookahead)
            if res and res.get("r_realised") is not None:
                rec["exit_r"] = res["r_realised"]
                rec["exit_reason"] = res["exit_reason"]
                filled.append(res["r_realised"])
        per_signal.append(rec)

    wins = [r for r in filled if r > 0]
    return {
        "run_id": run_id,
        "entry_buffer_tested": entry_buffer,
        "current_entry_buffer": ENTRY_BUFFER,
        "narrow_miss_threshold_pct": max_miss_pct,
        "exit_variant": variant or EXIT_VARIANT,
        "never_triggered_examined": examined,
        "excluded_not_narrow": skipped_not_narrow,
        "narrow_misses": len(per_signal),
        "signals_that_would_fill": sum(1 for r in per_signal if r["would_fill"]),
        "hypothetical_expectancy_r": (round(sum(filled) / len(filled), 3)
                                      if filled else None),
        "hypothetical_total_r": round(sum(filled), 2) if filled else None,
        "hypothetical_hit_rate": (round(len(wins) / len(filled), 3)
                                  if filled else None),
        "per_signal": per_signal,
        "caveat": ("Recomputed from the ANCHOR, not the pivot — pivot is not "
                   "stored. For breakouts anchor = max(pivot, trigger_bar_high) "
                   "and can exceed the pivot; for pullbacks it is the trigger "
                   "bar high. So this measures removing the 0.25% buffer, not "
                   "entering at the pivot."),
    }


def classifier_improves_walk_forward(trades: list[dict], windows: int,
                                     classifier_fn) -> tuple[bool, dict]:
    """
    Does filtering by this classifier INCREASE the count of positive
    walk-forward windows?

    Replaces skips_only_losing_windows, which was near-vacuous: three
    classifiers qualified in Run #34 by skipping W2 (n=3) while continuing
    to trade W5 — the one consistently losing window with real sample.
    Skipping a tiny window is not evidence of anything.

    This asks the question that actually matters: after filtering, are MORE
    windows positive than before? Run #34 unfiltered = 4.

    classifier_fn(trade) -> True (take), False (skip), or None (unknown).
    None is treated as TAKE, matching the live gate: a missing input must
    not silently become a skip decision.
    """
    dated = sorted([t for t in trades
                    if t.get("signal_date") and t.get("r_realised") is not None],
                   key=lambda t: t["signal_date"])
    if len(dated) < windows * 3:
        return False, {"error": "too few trades", "n": len(dated)}

    first, last = dated[0]["signal_date"], dated[-1]["signal_date"]
    step = max((last - first).days, 1) / windows

    def positive_windows(rows):
        pos, detail = 0, []
        for w in range(windows):
            lo = first + timedelta(days=int(step * w))
            hi = first + timedelta(days=int(step * (w + 1)))
            rs = [t["r_realised"] for t in rows if lo <= t["signal_date"] < hi]
            exp = round(sum(rs) / len(rs), 3) if rs else None
            if exp is not None and exp > 0:
                pos += 1
            detail.append({"window": w + 1, "n": len(rs), "expectancy_r": exp})
        return pos, detail

    kept = [t for t in dated if classifier_fn(t) is not False]
    unfiltered_pw, unfiltered_detail = positive_windows(dated)
    filtered_pw, filtered_detail = positive_windows(kept)

    return filtered_pw > unfiltered_pw, {
        "unfiltered_positive_windows": unfiltered_pw,
        "filtered_positive_windows": filtered_pw,
        "signals_kept": len(kept),
        "signals_skipped": len(dated) - len(kept),
        "unfiltered_windows": unfiltered_detail,
        "filtered_windows": filtered_detail,
        "qualifies": filtered_pw > unfiltered_pw,
    }


def w5_classifier_test(trades: list[dict], windows: int = 6) -> dict:
    """
    Task 4 — can any single classifier split W5 into a positive and a
    negative half?

    W5 is the only decisively negative window across every run (-0.658,
    -0.594, -0.609, -0.609 on n=16-18). It never flips. W1 and W4 are
    marginal and do flip, so they are not the target.

    Four singletons only. No composites: building one after four singletons
    fail is fitting to 16 trades.
    """
    dated = sorted([t for t in trades
                    if t.get("signal_date") and t.get("r_realised") is not None],
                   key=lambda t: t["signal_date"])
    if not dated:
        return {"error": "no trades"}

    first, last = dated[0]["signal_date"], dated[-1]["signal_date"]
    step = max((last - first).days, 1) / windows
    lo = first + timedelta(days=int(step * 4))      # W5 is index 4
    hi = first + timedelta(days=int(step * 5))
    w5 = [t for t in dated if lo <= t["signal_date"] < hi]
    if not w5:
        return {"error": "no trades in W5", "from": str(lo), "to": str(hi)}

    tests = {
        "w5_c1_close_above_200dma": lambda t: (t.get("regime_classifiers") or {}).get("c1_close_above_200dma"),
        "w5_c2_vix_below_15":       lambda t: (t.get("regime_classifiers") or {}).get("c5_vix_below_15"),
        "w5_c3_breadth_above_55":   lambda t: (t.get("regime_classifiers") or {}).get("c7_breadth_above_55"),
        "w5_c4_50dma_above_200dma": lambda t: (t.get("regime_classifiers") or {}).get("c2_close_above_50_and_stack"),
    }

    def stat(rows):
        rs = [t["r_realised"] for t in rows]
        if not rs:
            return {"n": 0, "expectancy_r": None}
        return {"n": len(rs), "expectancy_r": round(sum(rs) / len(rs), 3),
                "total_r": round(sum(rs), 2)}

    out = {"w5_from": str(lo), "w5_to": str(hi),
           "w5_overall": stat(w5), "classifiers": {}}
    separating = []

    for name, fn in tests.items():
        passing = [t for t in w5 if fn(t) is True]
        failing = [t for t in w5 if fn(t) is False]
        unknown = [t for t in w5 if fn(t) is None]
        p, f = stat(passing), stat(failing)
        # "Separates" means one side is actually POSITIVE — not merely
        # less negative. A classifier that splits -0.9 from -0.4 has found
        # nothing tradeable.
        sep = bool(p["expectancy_r"] is not None and p["expectancy_r"] > 0
                   and p["n"] >= 5)
        if sep:
            separating.append(name)
        out["classifiers"][name] = {
            "passing": p, "failing": f, "not_computable": stat(unknown),
            "separates_positive": sep,
        }

    out["separating_classifiers"] = separating
    out["verdict"] = (
        f"W5 separated by: {separating}" if separating else
        "W5 not distinguishable from current market data. Next investment "
        "is the four missing sources: sector dispersion, smallcap/largecap "
        "ratio, advance/decline, new-highs-minus-lows.")
    return out


def classifier_attribution(trades: list[dict], windows: int = 6) -> dict:
    """
    Tasks 4.2 / 4.3 — what each classifier would have done.

    For every single classifier and composite:
      - signals it would have TAKEN, and their expectancy
      - signals it would have SKIPPED, and their expectancy
      - how many of the six walk-forward windows it skips ENTIRELY

    That last figure is the decisive one. A classifier skipping more than
    two windows is over-fitted to this calibration data and is not a usable
    live filter, however good its expectancy split looks — it is closer to
    "trade only in 2024" than to a market condition. A classifier that
    skips exactly the losing windows while allowing the winning ones is the
    right gate.

    A skipped signal's expectancy is what the engine AVOIDED. High positive
    expectancy in the skipped bucket means the classifier is discarding
    good trades, which is a cost even when the taken bucket improves.
    """
    dated = sorted([t for t in trades
                    if t.get("signal_date") and t.get("r_realised") is not None
                    and t.get("regime_classifiers")],
                   key=lambda t: t["signal_date"])
    if len(dated) < 20:
        return {"error": "too few trades with classifier data", "n": len(dated)}

    # Same six windows walk_forward() uses, so the counts are comparable.
    first, last = dated[0]["signal_date"], dated[-1]["signal_date"]
    step = max((last - first).days, 1) / windows
    bounds = [(first + timedelta(days=int(step * w)),
               first + timedelta(days=int(step * (w + 1)))) for w in range(windows)]

    names = sorted({k for t in dated for k in t["regime_classifiers"]})
    out: dict = {}

    for name in names:
        taken = [t for t in dated if t["regime_classifiers"].get(name) is True]
        skipped = [t for t in dated if t["regime_classifiers"].get(name) is False]
        unknown = [t for t in dated if t["regime_classifiers"].get(name) is None]

        def stat(rows):
            rs = [t["r_realised"] for t in rows]
            if not rs:
                return {"n": 0, "expectancy_r": None, "total_r": None}
            return {"n": len(rs),
                    "expectancy_r": round(sum(rs) / len(rs), 3),
                    "total_r": round(sum(rs), 1),
                    "hit_rate": round(sum(1 for r in rs if r > 0) / len(rs), 3)}

        # A window counts as skipped only if the classifier blocks EVERY
        # signal in it — that is what "skips the window entirely" means.
        windows_skipped, window_detail = [], []
        for w, (lo, hi) in enumerate(bounds, start=1):
            in_w = [t for t in dated if lo <= t["signal_date"] < hi]
            if not in_w:
                window_detail.append({"window": w, "n": 0, "status": "empty"})
                continue
            t_in = [t for t in in_w if t["regime_classifiers"].get(name) is True]
            blocked = len(t_in) == 0
            if blocked:
                windows_skipped.append(w)
            window_detail.append({
                "window": w, "n": len(in_w), "would_take": len(t_in),
                "status": "SKIPPED" if blocked else "traded",
                "window_expectancy_r": round(
                    sum(t["r_realised"] for t in in_w) / len(in_w), 3),
            })

        out[name] = {
            "taken": stat(taken),
            "skipped": stat(skipped),
            "not_computable": stat(unknown),
            "windows_skipped_entirely": windows_skipped,
            "windows_skipped_count": len(windows_skipped),
            "window_detail": window_detail,
            # ">2 windows blocked = over-fitted" and "blocking exactly the
            # losing windows is the right gate" CONFLICT when there are three
            # losing windows, which is this dataset's case (W1, W3, W5).
            # Tested directly: a classifier blocking precisely {1,3,5} — the
            # stated success condition — was being flagged over-fitted and
            # discarded by the count rule alone.
            #
            # Resolution: blocking only losing windows is never over-fitting,
            # whatever the count. The count rule applies to everything else,
            # which is what it was for — catching a classifier that blocks
            # half the period indiscriminately.
            # Uses this run's losing set, filled in after the loop.
            "over_fitted": len(windows_skipped) > 2,
        }

    # Derived from THIS run rather than a fixed list — see _window_outcomes.
    LOSING, WINNING, window_outcomes = _window_outcomes(dated, windows)
    # Re-evaluate over_fitted now that LOSING is known: skipping only
    # losing windows is the success condition and cannot also be evidence
    # of over-fitting, whatever the count.
    for name, r in out.items():
        if isinstance(r, dict) and "windows_skipped_entirely" in r:
            sk = set(r["windows_skipped_entirely"])
            if sk and sk <= LOSING:
                r["over_fitted"] = False

    ideal = []
    for name, r in out.items():
        sk = set(r["windows_skipped_entirely"])
        # Checked BEFORE the count rule: skipping only losing windows is
        # the success condition, so it cannot simultaneously be evidence of
        # over-fitting.
        if sk and sk <= LOSING and not (sk & WINNING):
            ideal.append(name)

    out["_verdict"] = {
        "losing_windows": sorted(LOSING),
        "winning_windows": sorted(WINNING),
        "window_outcomes": window_outcomes,
        "derived_from": "this run's walk-forward, not a static list",
        # RETIRED — see classifier_improves_walk_forward. Near-vacuous:
        # qualified on skipping the n=3 W2 while still trading W5. Kept
        # visible so both criteria can be compared rather than silently
        # swapped.
        "skips_only_losing_windows_legacy": ideal,
        "over_fitted_classifiers": [n for n, r in out.items()
                                    if isinstance(r, dict) and r.get("over_fitted")],
        "note": ("A classifier blocking >2 windows is fitted to this "
                 "calibration data, not a market condition. One that blocks "
                 "only losing windows and keeps the winners is the gate; if "
                 "none qualifies, report the null result rather than forcing "
                 "one."),
    }
    return out


def walk_forward(trades: list[dict], windows: int = 6) -> dict:
    """
    Six consecutive 6-month windows, each scored on its own trades.

    split_sample() halves the period once, which a single favourable stretch
    can dominate — Run #27 is +0.072R overall but -0.323R / +0.472R by half,
    so the positive result rests entirely on the back end. Six windows show
    whether the edge recurs or happened once.

    NOTE ON "OPTIMISATION": no parameter is fitted per window here. Fitting
    thresholds on window N and testing on N+1 across ~175 trades would mean
    optimising on ~29 trades a window, which produces a number that looks
    like validation and is noise. This reports each window's OUT-OF-SAMPLE
    expectancy under one fixed configuration — the honest version of the
    same question, and the one whose pass/fail can be trusted.

    Pass criterion: at least 4 of 6 windows positive.
    """
    dated = sorted([t for t in trades
                    if t.get("signal_date") and t.get("r_realised") is not None],
                   key=lambda t: t["signal_date"])
    if len(dated) < windows * 5:
        return {"error": "too few trades to split", "n": len(dated),
                "windows": windows, "passed": False}

    first, last = dated[0]["signal_date"], dated[-1]["signal_date"]
    span_days = max((last - first).days, 1)
    step = span_days / windows

    out, positive = [], 0
    for w in range(windows):
        lo = first + timedelta(days=int(step * w))
        hi = first + timedelta(days=int(step * (w + 1)))
        subset = [t for t in dated
                  if lo <= t["signal_date"] < hi
                  or (w == windows - 1 and t["signal_date"] == last)]
        rs = [t["r_realised"] for t in subset]
        exp = round(sum(rs) / len(rs), 3) if rs else None
        if exp is not None and exp > 0:
            positive += 1
        out.append({
            "window": w + 1,
            "from": str(lo), "to": str(hi),
            "n": len(rs),
            "expectancy_r": exp,
            "total_r": round(sum(rs), 2) if rs else None,
            "hit_rate": round(sum(1 for r in rs if r > 0) / len(rs), 3) if rs else None,
        })

    underpowered = [w["window"] for w in out if (w["n"] or 0) < 15]
    # Positive windows that also have adequate sample. The plain count
    # cannot distinguish a pass carried by n=13 windows from one carried
    # by n=30 windows, and the sample is thinning as losing patterns are
    # disabled — so the two numbers are diverging precisely when the
    # distinction matters most.
    adequate_positive = sum(
        1 for w in out
        if (w.get("expectancy_r") or 0) > 0 and (w.get("n") or 0) >= 15)
    fully_dependent = (positive >= 4 and adequate_positive < 4)

    return {
        "windows": out,
        "positive_windows": positive,
        "adequate_positive_windows": adequate_positive,
        "min_n_for_adequate": 15,
        "pass_depends_on_underpowered_windows": fully_dependent,
        "required": 4,
        "passed": positive >= 4,
        "passed_on_adequate_sample": adequate_positive >= 4,
        "underpowered_windows": underpowered,
        "note": ("Pass needs 4 of 6 windows positive. Windows with n<15 are "
                 "flagged: a positive sign on a handful of trades is not "
                 "evidence, and a pass carried by them should be read as a "
                 "fail."),
    }


def winning_profile(trades: list[dict]) -> dict:
    """
    The subset the evidence actually supports, measured as one book.

    Defined BEFORE looking at its result, from findings that already held:
    flat_base was the best-performing pattern, and RSI 70-80 was the only
    sub-signal positive in both halves of Run #21.

    Acceptance: n >= 100, expectancy >= +0.3R, positive in BOTH halves.
    A subset that clears expectancy but fails the sign test is the same
    illusion the whole split-sample discipline exists to catch.
    """
    def in_profile(t):
        pattern = (t.get("pattern") or "")
        setup = (t.get("setup_type") or "")
        rsi_zone = t.get("rsi_zone")
        if pattern == "flat_base":
            return True
        if rsi_zone == "overbought_70_80":
            return True
        if setup.startswith("breakout") and t.get("breakout_rsi") == "breakout_rsi_70_80":
            return True
        return False

    subset = sorted([t for t in trades
                     if in_profile(t) and t.get("r_realised") is not None
                     and t.get("signal_date")],
                    key=lambda t: t["signal_date"])
    if not subset:
        return {"error": "no trades matched the profile", "n": 0, "accepted": False}

    def stats(rows):
        rs = [t["r_realised"] for t in rows]
        if not rs:
            return {"n": 0, "expectancy_r": None, "hit_rate": None, "total_r": None}
        wins = [r for r in rs if r > 0]
        return {"n": len(rs),
                "expectancy_r": round(sum(rs) / len(rs), 3),
                "hit_rate": round(len(wins) / len(rs), 3),
                "total_r": round(sum(rs), 2),
                "avg_win": round(sum(wins) / len(wins), 2) if wins else None}

    mid = len(subset) // 2
    full, a, b = stats(subset), stats(subset[:mid]), stats(subset[mid:])
    both_positive = (a["expectancy_r"] is not None and b["expectancy_r"] is not None
                     and a["expectancy_r"] > 0 and b["expectancy_r"] > 0)

    return {
        "definition": "flat_base OR rsi_zone=overbought_70_80 OR breakout_rsi_70_80",
        "full": full, "first_half": a, "second_half": b,
        "both_halves_positive": both_positive,
        "accepted": bool(full["n"] >= 100
                         and (full["expectancy_r"] or 0) >= 0.3
                         and both_positive),
        "acceptance": "n >= 100 AND expectancy >= +0.3R AND positive in both halves",
    }


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
                "breakout_rsi", "base_quality_quartile", "armed_delivery",
                # D2 shape diagnostics. Persisted to backtest_trades since
                # migration 008 but never cut in the metrics, which is why
                # the pattern breakdown showed no sign of them — the data
                # was there, the report was not asking for it.
                "handle_slope", "cup_shape",
                "armed_distance_band", "base_depth_band", "liquidity_band",
                "prior_move_band"):
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
                "breakout_rsi", "base_quality_quartile", "armed_delivery",
                # D2 shape diagnostics. Persisted to backtest_trades since
                # migration 008 but never cut in the metrics, which is why
                # the pattern breakdown showed no sign of them — the data
                # was there, the report was not asking for it.
                "handle_slope", "cup_shape",
                "armed_distance_band", "base_depth_band", "liquidity_band",
                "prior_move_band"):
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
