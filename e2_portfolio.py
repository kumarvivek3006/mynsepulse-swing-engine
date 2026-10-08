"""
e2_portfolio.py — turns a pile of independently-simulated trades into one book.

Every candidate has already been simulated (e2_manage). This layer only decides WHICH to take
and HOW BIG, day by day, in score order:

  size      1% of equity risked x regime multiplier x drawdown multiplier, divided by the
            trade's stop distance; a single position is capped at 20% of equity
  regime    max positions 4 / 6 / 8 (risk_off / neutral / risk_on), read on the SIGNAL date
  heat      sum of open initial risk <= 6R (a position stops counting once its stop is at
            break-even); at 1% per position this binds before the 8-position cap
  sector    at most 2 open positions per sector (an unmapped sector has no cap)
  corr      no new position whose 60-session return correlation with an open one exceeds 0.7
  drawdown  > 5% below the equity peak: size x0.75; > 10%: x0.5; > 15%: no new entries for a
            20-session cooldown (without a way back a halt would be permanent: nothing could ever
            recover the equity), then trading resumes at the scaled size; a further 5% loss halts again
Every rejection is counted by reason. Nothing else rejects a trade.

Order within a day: entries are decided on positions open at the START of the day, then exits
and marks are applied, so capital freed on a day is not reused the same day (conservative).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from e2_core import fenv, ienv
from e2_manage import Trade


@dataclass
class Params:
    equity0: float = 1_000_000.0
    risk: float = fenv("E2_RISK", 0.01)
    heat_cap: float = fenv("E2_HEAT_CAP", 6.0)
    sector_cap: int = ienv("E2_SECTOR_CAP", 2)
    corr_cap: float = fenv("E2_CORR_CAP", 0.70)
    corr_window: int = ienv("E2_CORR_WINDOW", 60)
    max_pos_value: float = fenv("E2_MAX_POS_VALUE", 0.20)
    dd_levels: tuple = ((0.05, 0.75), (0.10, 0.50))
    dd_halt: float = fenv("E2_DD_HALT", 0.15)
    halt_cooldown: int = ienv("E2_HALT_COOLDOWN", 20)       # sessions without new entries after a halt
    halt_rearm: float = fenv("E2_HALT_REARM", 0.05)         # after the cooldown, a further 5% loss halts again
    use_regime: bool = True


@dataclass
class Result:
    equity: pd.Series
    taken: list
    rejected: dict
    metrics: dict
    daily: pd.DataFrame
    halts: int = 0
    halted_sessions: int = 0


def _dd_mult(dd: float, p: Params) -> float:
    m = 1.0
    for lvl, mult in p.dd_levels:
        if dd > lvl:
            m = mult
    return m


def run(trades: list[Trade], regime: pd.DataFrame, returns: pd.DataFrame, dates: pd.DatetimeIndex,
        p: Params | None = None, start=None, end=None) -> Result:
    """
    trades   simulated trades (skipped ones are ignored here)
    regime   dates x [score,state,size_mult,max_positions]
    returns  dates x symbols daily returns (for the correlation cap)
    dates    the trading calendar to step through
    """
    p = p or Params()
    live = [t for t in trades if not t.skipped]
    by_entry: dict = {}
    for t in live:
        by_entry.setdefault(np.datetime64(t.entry_date, "D"), []).append(t)
    cal = [np.datetime64(d, "D") for d in dates if (start is None or d >= start) and (end is None or d <= end)]
    ret_idx = {np.datetime64(d, "D"): k for k, d in enumerate(returns.index)}
    ret_arr = returns.to_numpy(dtype=np.float32, na_value=np.nan)
    col = {c: j for j, c in enumerate(returns.columns)}
    reg_state = regime["state"].to_dict()
    reg_mult = regime["size_mult"].to_dict()
    reg_max = regime["max_positions"].to_dict()
    reg_by_day = {np.datetime64(k, "D"): k for k in regime.index}

    cash, peak = p.equity0, p.equity0
    open_pos: list[dict] = []
    taken: list[dict] = []
    rejected: dict[str, int] = {}
    rows = []
    last_equity = p.equity0

    def rej(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    def corr_ok(sym: str, sig_date, others: list[str]) -> bool:
        if sym not in col:
            return True
        k = ret_idx.get(np.datetime64(sig_date, "D"))
        if k is None or k < p.corr_window:
            return True
        a = ret_arr[k - p.corr_window + 1:k + 1, col[sym]]
        for o in others:
            if o not in col:
                continue
            b = ret_arr[k - p.corr_window + 1:k + 1, col[o]]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 30:
                continue
            c = np.corrcoef(a[m], b[m])[0, 1]
            if not np.isnan(c) and c > p.corr_cap:
                return False
        return True

    halt_until, halt_eq, halts, halted_sessions = -1, None, 0, 0
    for di, d in enumerate(cal):
        dd = max(0.0, 1.0 - last_equity / peak)
        if dd > p.dd_halt and di > halt_until and (halt_eq is None or last_equity < halt_eq * (1 - p.halt_rearm)):
            halt_until, halt_eq, halts = di + p.halt_cooldown, last_equity, halts + 1
        halted = di <= halt_until
        halted_sessions += int(halted)
        dd_mult = _dd_mult(dd, p)
        todays = by_entry.get(d, [])
        if todays:
            todays = sorted(todays, key=lambda t: (-(t.score if t.score is not None else -1), -t.quality))
        for t in todays:
            if halted:
                rej("drawdown_halt")
                continue
            sd = reg_by_day.get(np.datetime64(t.signal_date, "D"))
            mult = float(reg_mult.get(sd, 0.75)) if (p.use_regime and sd is not None) else (1.0 if not p.use_regime else 0.75)
            mx = int(reg_max.get(sd, 6)) if (p.use_regime and sd is not None) else (8 if not p.use_regime else 6)
            if len(open_pos) >= mx:
                rej("max_positions")
                continue
            heat_now = sum(q["heat"] for q in open_pos if not _past_be(q, d))
            risk_mult = mult * dd_mult
            if heat_now + risk_mult > p.heat_cap + 1e-9:
                rej("heat_cap")
                continue
            if t.sector is not None and sum(1 for q in open_pos if q["trade"].sector == t.sector) >= p.sector_cap:
                rej("sector_cap")
                continue
            if not corr_ok(t.symbol, t.signal_date, [q["trade"].symbol for q in open_pos]):
                rej("correlation")
                continue
            if any(q["trade"].symbol == t.symbol for q in open_pos):
                rej("already_in_symbol")
                continue
            equity_now = last_equity
            risk_amt = equity_now * p.risk * risk_mult
            intended = risk_amt / t.risk_frac if t.risk_frac > 0 else 0.0
            value = min(intended, p.max_pos_value * equity_now, cash)
            if value < 0.25 * intended or value <= 0:
                rej("no_cash")
                continue
            cash -= value
            open_pos.append({"trade": t, "value": value, "ptr": -1, "heat": risk_mult * value / intended})
            taken.append({"trade": t, "value": value, "risk_mult": risk_mult, "regime": reg_state.get(sd),
                          "scale": value / intended})
        # exits and marks at today's close
        still = []
        mtm = 0.0
        for q in open_pos:
            t: Trade = q["trade"]
            if np.datetime64(t.exit_date, "D") <= d:
                proceeds = q["value"] * (1.0 + t.ret)
                cash += proceeds
                q["pnl"] = proceeds - q["value"]
                q["closed"] = d
                for tk in reversed(taken):
                    if tk["trade"] is t:
                        tk["pnl"] = q["pnl"]
                        break
                continue
            md = t.mark_dates
            ptr = q["ptr"]
            while md is not None and ptr + 1 < len(md) and np.datetime64(md[ptr + 1], "D") <= d:
                ptr += 1
            q["ptr"] = ptr
            m = float(t.marks[ptr]) if ptr >= 0 else 0.0
            mtm += q["value"] * (1.0 + m)
            still.append(q)
        open_pos = still
        last_equity = cash + mtm
        peak = max(peak, last_equity)
        rows.append((d, last_equity, len(open_pos), mtm / last_equity if last_equity else 0.0, dd, dd_mult))

    daily = pd.DataFrame(rows, columns=["date", "equity", "positions", "exposure", "drawdown_prev", "dd_mult"]).set_index("date")
    eq = daily["equity"]
    return Result(eq, taken, rejected, metrics(eq, taken, p), daily, halts, halted_sessions)


def _past_be(q: dict, d) -> bool:
    t: Trade = q["trade"]
    return t.be_date is not None and np.datetime64(t.be_date, "D") <= d


def metrics(eq: pd.Series, taken: list[dict], p: Params) -> dict:
    if len(eq) < 2:
        return {"trades": len(taken)}
    r = eq.pct_change().dropna()
    years = len(eq) / 252.0
    peak = eq.cummax()
    mdd = float((1 - eq / peak).max())
    tot = float(eq.iloc[-1] / p.equity0 - 1)
    closed = [x for x in taken if "pnl" in x]
    rs = np.array([x["trade"].r for x in closed]) if closed else np.array([])
    pnl = np.array([x["pnl"] for x in closed]) if closed else np.array([])
    gross_w, gross_l = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "total_return_pct": round(100 * tot, 2),
        "cagr_pct": round(100 * ((1 + tot) ** (1 / years) - 1), 2) if years > 0 and tot > -1 else None,
        "max_drawdown_pct": round(100 * mdd, 2),
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else None,
        "trades_taken": len(taken), "trades_closed": len(closed),
        "win_rate_pct": round(100 * float((rs > 0).mean()), 1) if len(rs) else None,
        "avg_r": round(float(rs.mean()), 3) if len(rs) else None,
        "median_r": round(float(np.median(rs)), 3) if len(rs) else None,
        "profit_factor": round(float(gross_w / gross_l), 2) if gross_l > 0 else None,
        "avg_bars_held": round(float(np.mean([x["trade"].bars for x in closed])), 1) if closed else None,
        "trades_per_year": round(len(taken) / years, 1) if years > 0 else None,
    }
