"""
e2_manage.py — one trade, from the confirmed candidate to the last exit.

The path of a trade does NOT depend on the portfolio: every candidate gets simulated on
its own in R terms, and the portfolio layer then decides which of them to take and how
big. That separation is what lets per-detector statistics be reported on every candidate
(not only the ones the portfolio happened to pick).

Rules (spec) and the pre-registered choices where the spec leaves room:
  entry      market on the OPEN of the bar after confirmation (+ slippage)
  stop       the TIGHTEST of: structural stop, entry - 2 ATR, entry - 8%.
             Floor: never tighter than 1 ATR from entry (a stop inside one day's normal
             range is noise; the count of floored trades is reported).
  T1  2R     sell 50%; stop on the rest moves to break-even (effective NEXT bar)
  T2  3R     sell 25%; the last 25% then trails
  T3 trail   the LOOSER of EMA20 and the chandelier (highest close - 2 ATR), both read
             as of the previous close, ratcheting up only, never below break-even
  time stop  bar 20 after entry: if T1 has not filled and the close is under entry + 1R,
             exit on the next open
  events     results date in the next session -> exit at the close before it (when the
             calendar has one); a circuit-locked bar (high == low on a >= 4.5% move) can
             not be traded on, so nothing fills on it and the position is flattened on
             the next open; a gap in the symbol's own sessions (> 6 days) is a halt:
             exit on the first bar that trades.
  intrabar   a bar that touches both the stop and a target is resolved AGAINST us (stop first).
  gaps       stops that gap through fill at the open, not at the stop price.
  costs      0.30% round trip + 0.10% slippage per side.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from e2_core import Candidate, fenv, ienv
from e2_features import Bars

T1_R, T2_R = fenv("E2_T1_R", 2.0), fenv("E2_T2_R", 3.0)
T1_FRAC, T2_FRAC = fenv("E2_T1_FRAC", 0.50), fenv("E2_T2_FRAC", 0.25)
STOP_ATR, STOP_PCT, MIN_RISK_ATR = fenv("E2_STOP_ATR", 2.0), fenv("E2_STOP_PCT", 0.08), fenv("E2_MIN_RISK_ATR", 1.0)
TRAIL_ATR = fenv("E2_TRAIL_ATR", 2.0)
TIME_STOP_BARS, TIME_STOP_R = ienv("E2_TIME_STOP_BARS", 20), fenv("E2_TIME_STOP_R", 1.0)
MAX_HOLD = ienv("E2_MAX_HOLD", 150)
COST_RT, SLIP = fenv("E2_COST_RT", 0.0030), fenv("E2_SLIP", 0.0010)
CIRCUIT_MOVE, HALT_GAP_DAYS = fenv("E2_CIRCUIT_MOVE", 0.045), ienv("E2_HALT_GAP_DAYS", 6)


@dataclass
class Trade:
    symbol: str
    detector: str
    signal_idx: int
    signal_date: object
    skipped: str | None = None          # why no trade was opened (execution reasons only)
    entry_idx: int = -1
    entry_date: object = None
    entry: float = 0.0
    stop0: float = 0.0
    risk_frac: float = 0.0              # (entry - stop0) / entry: 1R as a fraction of price
    floored: bool = False               # stop distance raised to the 1 ATR floor
    exits: list = field(default_factory=list)    # (bar idx, price, fraction, reason)
    exit_idx: int = -1
    exit_date: object = None
    ret: float = 0.0                    # net return on the whole position, costs included
    r: float = 0.0                      # ret / risk_frac
    bars: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    reason: str = ""                    # how the LAST part left
    be_idx: int = -1                    # bar from which the stop sat at break-even or better
    be_date: object = None
    mark_dates: np.ndarray | None = None
    marks: np.ndarray | None = None     # cumulative return on the position at each close (gross of exit cost)
    open_at_end: bool = False
    score: float | None = None
    sector: str | None = None
    family: str = ""
    quality: float = 0.0
    meta: dict = field(default_factory=dict)
    frac: dict | None = None            # composite-score component fractions (filled by the pipeline)
    regime_state: str | None = None


def initial_stop(b: Bars, cand: Candidate, entry: float) -> tuple[float, bool]:
    """tightest of structural / ATR / pct, floored at MIN_RISK_ATR; returns (stop, floored)."""
    atr = float(b.atr14[cand.idx])
    opts = [entry * (1 - STOP_PCT)]
    if not np.isnan(atr):
        opts.append(entry - STOP_ATR * atr)
    if cand.stop < entry:
        opts.append(cand.stop)
    stop = max(opts)                                    # tightest = highest price below entry
    floored = False
    if not np.isnan(atr) and entry - stop < MIN_RISK_ATR * atr:
        stop, floored = entry - MIN_RISK_ATR * atr, True
    return float(stop), floored


def _locked(b: Bars, t: int) -> bool:
    if t < 1 or b.c_prev[t] <= 0:
        return False
    return bool(b.h[t] == b.l[t] and abs(b.c[t] / b.c_prev[t] - 1) >= CIRCUIT_MOVE)


def simulate(b: Bars, cand: Candidate, events: np.ndarray | None = None) -> Trade:
    """events: sorted np.datetime64[D] results dates for this symbol (or None when the calendar has none)."""
    tr = Trade(b.symbol, cand.detector, cand.idx, cand.date, family=cand.family, quality=cand.quality,
               score=cand.score, sector=cand.sector, meta=cand.meta)
    e = cand.idx + 1
    if e >= b.n:
        tr.skipped = "no_next_bar"
        return tr
    entry = float(b.o[e]) * (1 + SLIP)
    if _locked(b, e):
        tr.skipped = "entry_bar_circuit_locked"
        return tr
    stop, floored = initial_stop(b, cand, entry)
    if b.o[e] <= cand.stop or b.o[e] <= stop:
        tr.skipped = "open_below_stop"                  # the setup's invalidation level is already gone at the open
        return tr
    d_end = b.date[e + 1] if e + 1 < b.n else b.date[e] + np.timedelta64(1, "D")
    if events is not None and len(events) and _event_between(events, b.date[cand.idx], d_end):
        tr.skipped = "results_before_entry"            # announcement between the signal and the end of the entry session
        return tr
    R = entry - stop
    tr.entry_idx, tr.entry_date, tr.entry, tr.stop0 = e, b.date[e], entry, stop
    tr.risk_frac, tr.floored = R / entry, floored
    t1, t2 = entry + T1_R * R, entry + T2_R * R

    stage, rem, stop_now = 0, 1.0, stop
    realised = 0.0                      # sum of fraction * (exit/entry - 1)
    marks, mfe, mae = [], 0.0, 0.0
    flatten_next = None                 # reason to leave on the next open
    hh_close = entry
    last = min(b.n - 1, e + MAX_HOLD)

    def leave(t: int, px: float, frac: float, reason: str) -> None:
        nonlocal rem, realised
        px = px * (1 - SLIP)
        tr.exits.append((int(t), float(px), float(frac), reason))
        realised += frac * (px / entry - 1.0)
        rem -= frac

    for t in range(e, last + 1):
        k = t - e + 1
        if t > e and (b.date[t] - b.date[t - 1]).astype(int) > HALT_GAP_DAYS and flatten_next is None:
            leave(t, float(b.o[t]), rem, "halt_gap_exit")
            tr.reason = "halt_gap_exit"
            break
        if flatten_next is not None:
            leave(t, float(b.o[t]), rem, flatten_next)
            tr.reason = flatten_next
            break
        locked = _locked(b, t)
        if t > e and b.o[t] <= stop_now and not locked:
            leave(t, float(b.o[t]), rem, "stop_gap")
            tr.reason = "stop_gap"
            break
        if not locked:
            if b.l[t] <= stop_now:
                leave(t, stop_now, rem, "stop")
                tr.reason = "stop" if stage == 0 else ("breakeven_stop" if stage == 1 else "trail_stop")
                break
            if stage == 0 and b.h[t] >= t1:
                leave(t, max(t1, float(b.o[t])) if b.o[t] > t1 else t1, T1_FRAC, "T1")
                stage, tr.be_idx = 1, t + 1
                tr.be_date = b.date[min(t + 1, b.n - 1)]
                pending_stop = entry
            else:
                pending_stop = None
            if stage == 1 and b.h[t] >= t2:
                leave(t, max(t2, float(b.o[t])) if b.o[t] > t2 else t2, T2_FRAC, "T2")
                stage = 2
        else:
            pending_stop = None
            flatten_next = "circuit_exit"
        r_hi, r_lo = (b.h[t] - entry) / R, (b.l[t] - entry) / R
        mfe, mae = max(mfe, r_hi), min(mae, r_lo)
        marks.append(realised + rem * (b.c[t] / entry - 1.0))
        hh_close = max(hh_close, float(b.c[t]))
        # stops for the NEXT bar
        if pending_stop is not None:
            stop_now = max(stop_now, pending_stop)
        if stage == 2:
            trail = min(float(b.ema20[t]), hh_close - TRAIL_ATR * float(b.atr14[t]))
            stop_now = max(stop_now, trail, entry)
        # exits decided at this close
        if rem > 1e-9 and flatten_next is None:
            if events is not None and len(events) and t + 1 < b.n and _event_between(events, b.date[t], b.date[t + 1]):
                leave(t, float(b.c[t]), rem, "results_exit")
                tr.reason = "results_exit"
                break
            if k == TIME_STOP_BARS and stage == 0 and b.c[t] < entry + TIME_STOP_R * R:
                flatten_next = "time_stop"
        if t == last and rem > 1e-9:
            leave(t, float(b.c[t]), rem, "open_at_end" if t == b.n - 1 and k < MAX_HOLD else "max_hold")
            tr.reason = tr.exits[-1][3]
            tr.open_at_end = (t == b.n - 1 and k < MAX_HOLD)
        if rem <= 1e-9:
            break

    if rem > 1e-9:                      # flatten_next scheduled on the very last bar: close it at the close
        t = len(marks) + e - 1
        leave(t, float(b.c[t]), rem, "open_at_end")
        tr.reason, tr.open_at_end = "open_at_end", True
    tr.exit_idx = tr.exits[-1][0]
    tr.exit_date = b.date[tr.exit_idx]
    tr.bars = tr.exit_idx - e + 1
    tr.ret = realised - COST_RT
    tr.r = tr.ret / tr.risk_frac if tr.risk_frac > 0 else 0.0
    tr.mfe_r, tr.mae_r = float(mfe), float(mae)
    m = np.array(marks, float)
    tr.marks = m
    tr.mark_dates = b.date[e:e + len(m)]
    return tr


def _event_between(events: np.ndarray, d0, d1) -> bool:
    """is there a results date d with d0 <= d < d1 (announced after d0's close / before d1's session)?"""
    i = np.searchsorted(events, d0, side="left")
    return bool(i < len(events) and events[i] < d1)
