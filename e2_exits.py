"""
e2_exits.py — EXIT REDESIGN STUDY (owner's order of 6 Nov 21:42 IST): the same candidates, five different exit rulebooks.

Entries, detectors, gates and signal counts are untouched. Only what happens AFTER the market-on-open entry changes.
Costs are the live ones (0.30% round trip + 0.10% slippage per side). A bar that touches both the stop and a target
resolves against us; stops that gap through fill at the open; results-date, circuit and halt handling are the live rules.

Variants (written down before any result; ambiguous wording resolved as stated here):
  BASE  live rules: tightest of structural / 2 ATR / 8%, floored at 1 ATR; 2R sell 50% -> break-even; 3R sell 25%; last 25% trails
        (looser of EMA20 and highest close - 2 ATR); bar 20: exit next open if no 2R yet and close < entry + 1R.
  A     stop = entry - 2.0 ATR (no 8% cap, no structural tightening). Stop to break-even once the HIGH reaches +1.5R (effective next
        bar). Sell 50% at +2R. Remaining 50% trails the 10-EMA once the 2R partial has filled (never below break-even). Bar 15: exit
        next open if the 2R partial has not filled and the close is below entry + 0.5R.
  B     stop = entry - 1.5 ATR. Sell 50% at +1R, stop to break-even (next bar). Remaining 50% trails the 10-EMA (never below
        break-even). Bar 15: exit next open if the partial has not filled.
  C     stop = entry - 2 ATR, then a chandelier from day one: stop = max(stop, highest HIGH since entry - 2 ATR), ratchets up only,
        decided at each close for the next bar. No partials. Bar 20: exit next open if the close is below entry + 1R.
  D     stop = 5% below the detector's structural stop (base / swing low), 1-ATR floor. Sell 50% at +2R, stop to break-even (next
        bar). Remaining 50% trails the 20-EMA (never below break-even). Bar 20: exit next open if the partial has not filled and the
        close is below entry + 1R.
  Bh/Ch/Dh  sensitivity only, never used for the deploy rule: the time stop read as a HARD cap (exit at that bar whatever the state).

Deploy rule (owner's): a variant with positive net expectancy on the full sample AND in both halves is the exit logic. Four variants
are read on one sample, so the report also gives week-block bootstrap intervals; a point estimate above zero with an interval
straddling zero is reported as exactly that.

Writes engine_settings.engine2_exits; logs E2EXIT| lines; GET /jobs/engine2-exits.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

import e2_backtest as BT
import e2_manage as MG
from e2_core import Candidate
from e2_det_common import WARMUP
from e2_detectors import detect_all
from e2_features import Bars, compute_features
from e2_manage import (CIRCUIT_MOVE, COST_RT, HALT_GAP_DAYS, MAX_HOLD, MIN_RISK_ATR, SLIP, Trade, _event_between, _locked, initial_stop)

log = logging.getLogger("engine2.exits")
KEY = "engine2_exits"
LOG_PREFIX = "E2EXIT|"
EARLY_BARS = 5


@dataclass(frozen=True)
class ExitSpec:
    name: str
    stop_mode: str = "baseline"            # baseline | atr | structural
    stop_atr: float = 2.0
    struct_buffer: float = 0.05
    be_r: float | None = None              # move to break-even when the HIGH reaches this many R (independent of any partial)
    partial_r: float | None = 2.0
    partial_frac: float = 0.50
    t2_r: float | None = 3.0               # live rules only
    t2_frac: float = 0.25
    trail: str = "base"                    # base | ema10 | ema20 | chand
    trail_from_partial: bool = True        # trail only once the partial has filled (chand ignores this: day one)
    chand_atr: float = 2.0
    time_bars: int = 20
    time_r: float = 1.0
    time_needs_no_partial: bool = True
    time_hard: bool = False


BASE = ExitSpec("BASE")
SPECS = {
    "A": ExitSpec("A", stop_mode="atr", stop_atr=2.0, be_r=1.5, partial_r=2.0, t2_r=None, trail="ema10", time_bars=15, time_r=0.5),
    "B": ExitSpec("B", stop_mode="atr", stop_atr=1.5, partial_r=1.0, t2_r=None, trail="ema10", time_bars=15, time_r=99.0),
    "C": ExitSpec("C", stop_mode="atr", stop_atr=2.0, partial_r=None, t2_r=None, trail="chand", time_bars=20, time_r=1.0, time_needs_no_partial=False),
    "D": ExitSpec("D", stop_mode="structural", partial_r=2.0, t2_r=None, trail="ema20", time_bars=20, time_r=1.0),
}
SENS = {
    "Bh": ExitSpec("Bh", stop_mode="atr", stop_atr=1.5, partial_r=1.0, t2_r=None, trail="ema10", time_bars=15, time_r=1e9, time_needs_no_partial=False, time_hard=True),
    "Ch": ExitSpec("Ch", stop_mode="atr", stop_atr=2.0, partial_r=None, t2_r=None, trail="chand", time_bars=20, time_hard=True, time_needs_no_partial=False),
    "Dh": ExitSpec("Dh", stop_mode="structural", partial_r=2.0, t2_r=None, trail="ema20", time_bars=20, time_hard=True, time_needs_no_partial=False),
}
ALL = {"BASE": BASE, **SPECS, **SENS}


def _ema10(b: Bars) -> np.ndarray:
    a = getattr(b, "_ema10", None)
    if a is None:
        import pandas as pd
        a = pd.Series(b.c).ewm(span=10, adjust=False).mean().to_numpy()
        b._ema10 = a
    return a


def _stop_for(b: Bars, cand: Candidate, entry: float, X: ExitSpec) -> tuple[float, bool]:
    atr = float(b.atr14[cand.idx])
    if X.stop_mode == "baseline":
        return initial_stop(b, cand, entry)
    if X.stop_mode == "atr":
        if np.isnan(atr):
            return initial_stop(b, cand, entry)
        return float(entry - X.stop_atr * atr), False
    # structural: 5% below the detector's own invalidation level; if that level is not below entry, fall back to 2 ATR
    if cand.stop < entry:
        stop = float(cand.stop * (1 - X.struct_buffer))
    elif not np.isnan(atr):
        stop = float(entry - 2.0 * atr)
    else:
        return initial_stop(b, cand, entry)
    floored = False
    if not np.isnan(atr) and entry - stop < MIN_RISK_ATR * atr:
        stop, floored = entry - MIN_RISK_ATR * atr, True
    return float(stop), floored


def simulate_x(b: Bars, cand: Candidate, events: np.ndarray | None, X: ExitSpec) -> Trade:
    """MG.simulate with the exit rulebook as data. With X = BASE it reproduces MG.simulate exactly (tested)."""
    tr = Trade(b.symbol, cand.detector, cand.idx, cand.date, family=cand.family, quality=cand.quality,
               score=cand.score, sector=cand.sector, meta=cand.meta)
    e = cand.idx + 1
    if e >= b.n:
        tr.skipped = "no_next_bar"
        return tr
    ref_open = float(b.o[e])
    entry = ref_open * (1 + SLIP)
    if _locked(b, e):
        tr.skipped = "entry_bar_circuit_locked"
        return tr
    stop, floored = _stop_for(b, cand, entry, X)
    if ref_open <= cand.stop or ref_open <= stop:
        tr.skipped = "open_below_stop"
        return tr
    d_end = b.date[e + 1] if e + 1 < b.n else b.date[e] + np.timedelta64(1, "D")
    if events is not None and len(events) and _event_between(events, b.date[cand.idx], d_end):
        tr.skipped = "results_before_entry"
        return tr
    R = entry - stop
    tr.entry_idx, tr.entry_date, tr.entry, tr.stop0 = e, b.date[e], entry, stop
    tr.risk_frac, tr.floored = R / entry, floored
    t1 = entry + X.partial_r * R if X.partial_r is not None else None
    t2 = entry + X.t2_r * R if X.t2_r is not None else None
    be_lvl = entry + X.be_r * R if X.be_r is not None else None
    ema10 = _ema10(b) if X.trail == "ema10" else None

    stage, rem, stop_now = 0, 1.0, stop
    realised = 0.0
    marks, mfe, mae = [], 0.0, 0.0
    flatten_next = None
    hh_close, hh_high = entry, entry
    be_done = False
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
        pending_stop = None
        if t > e and b.o[t] <= stop_now and not locked:
            leave(t, float(b.o[t]), rem, "stop_gap")
            tr.reason = "stop_gap"
            tr.loss_stop = stop_now < entry
            break
        if not locked:
            if b.l[t] <= stop_now:
                leave(t, stop_now, rem, "stop")
                tr.loss_stop = stop_now < entry
                tr.reason = "stop" if stop_now < entry else ("breakeven_stop" if stage <= 1 else "trail_stop")
                if X.name == "BASE":
                    tr.reason = "stop" if stage == 0 else ("breakeven_stop" if stage == 1 else "trail_stop")
                break
            if stage == 0 and t1 is not None and b.h[t] >= t1:
                leave(t, max(t1, float(b.o[t])) if b.o[t] > t1 else t1, X.partial_frac, "T1")
                stage, tr.be_idx = 1, t + 1
                tr.be_date = b.date[min(t + 1, b.n - 1)]
                pending_stop = entry
            if be_lvl is not None and not be_done and b.h[t] >= be_lvl:
                be_done = True
                pending_stop = entry if pending_stop is None else max(pending_stop, entry)
                if tr.be_idx < 0:
                    tr.be_idx, tr.be_date = t + 1, b.date[min(t + 1, b.n - 1)]
            if stage == 1 and t2 is not None and b.h[t] >= t2:
                leave(t, max(t2, float(b.o[t])) if b.o[t] > t2 else t2, X.t2_frac, "T2")
                stage = 2
        else:
            flatten_next = "circuit_exit"
        r_hi, r_lo = (b.h[t] - entry) / R, (b.l[t] - entry) / R
        mfe, mae = max(mfe, r_hi), min(mae, r_lo)
        marks.append(realised + rem * (b.c[t] / entry - 1.0))
        hh_close = max(hh_close, float(b.c[t]))
        hh_high = max(hh_high, float(b.h[t]))
        if pending_stop is not None:
            stop_now = max(stop_now, pending_stop)
        # trailing stops for the NEXT bar
        if X.trail == "base":
            if stage == 2:
                stop_now = max(stop_now, min(float(b.ema20[t]), hh_close - MG.TRAIL_ATR * float(b.atr14[t])), entry)
        elif X.trail == "none":
            pass
        elif X.trail == "chand":
            a = float(b.atr14[t])
            if not np.isnan(a):
                stop_now = max(stop_now, hh_high - X.chand_atr * a)
        else:
            if stage >= 1 or not X.trail_from_partial:
                ema = ema10[t] if X.trail == "ema10" else b.ema20[t]
                if not np.isnan(ema):
                    stop_now = max(stop_now, float(ema), entry)
        if rem > 1e-9 and flatten_next is None:
            if events is not None and len(events) and t + 1 < b.n and _event_between(events, b.date[t], b.date[t + 1]):
                leave(t, float(b.c[t]), rem, "results_exit")
                tr.reason = "results_exit"
                break
            if k == X.time_bars and (not X.time_needs_no_partial or stage == 0):
                if X.time_hard or b.c[t] < entry + X.time_r * R:
                    flatten_next = "time_stop"
        if t == last and rem > 1e-9:
            leave(t, float(b.c[t]), rem, "open_at_end" if t == b.n - 1 and k < MAX_HOLD else "max_hold")
            tr.reason = tr.exits[-1][3]
            tr.open_at_end = (t == b.n - 1 and k < MAX_HOLD)
        if rem <= 1e-9:
            break

    if rem > 1e-9:
        t = len(marks) + e - 1
        leave(t, float(b.c[t]), rem, "open_at_end")
        tr.reason, tr.open_at_end = "open_at_end", True
    tr.exit_idx = tr.exits[-1][0]
    tr.exit_date = b.date[tr.exit_idx]
    tr.bars = tr.exit_idx - e + 1
    tr.ret = realised - COST_RT
    tr.r = tr.ret / tr.risk_frac if tr.risk_frac > 0 else 0.0
    if not tr.reason:
        tr.reason = tr.exits[-1][3]                                    # a full exit at the partial level = a target hit
    tr.mfe_r, tr.mae_r = float(mfe), float(mae)
    m = np.array(marks, float)
    tr.marks, tr.mark_dates = m, b.date[e:e + len(m)]
    return tr


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def _f(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _live(ts):
    return [t for t in ts if not t.skipped]


def _week_ids(trs: list[Trade]) -> np.ndarray:
    d = np.array([np.datetime64(t.signal_date, "D") for t in trs])
    return ((d - np.datetime64("2000-01-03", "D")).astype(int) // 7)


def boot_mean(trs: list[Trade], vals: np.ndarray | None = None, reps: int = 2000, seed: int = 11) -> tuple[float | None, float | None]:
    """95% interval for the mean R, resampling whole signal weeks (candidates in the same week are correlated)."""
    if len(trs) < 30:
        return None, None
    v = np.array([t.r for t in trs], float) if vals is None else vals
    w = _week_ids(trs)
    u, inv = np.unique(w, return_inverse=True)
    S, N = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(reps, len(u)))
    means = S[idx].sum(1) / N[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def block(trs: list[Trade], mid64, n_cand: int, weeks: float) -> dict:
    live = _live(trs)
    if not live:
        return {"candidates": n_cand, "filled": 0}
    r = np.array([t.r for t in live])
    rf = np.array([t.risk_frac for t in live])
    cost_r = (COST_RT + 2 * SLIP) / np.where(rf > 0, rf, np.nan)
    sig = np.array([np.datetime64(t.signal_date, "D") for t in live])
    h1, h2 = r[sig < mid64], r[sig >= mid64]
    t1, t2 = [t for t, s in zip(live, sig) if s < mid64], [t for t, s in zip(live, sig) if s >= mid64]
    loss = np.array([bool(getattr(t, "loss_stop", False)) for t in live])
    early = np.array([bool(getattr(t, "loss_stop", False)) and t.bars <= EARLY_BARS for t in live])
    mfe = np.array([t.mfe_r for t in live])
    w, l = r[r > 0].sum(), -r[r < 0].sum()
    lo, hi = boot_mean(live)
    lo1, hi1 = boot_mean(t1)
    lo2, hi2 = boot_mean(t2)
    cuts = np.array_split(np.argsort(sig, kind="stable"), 6)
    slices = [float(r[ix].mean()) for ix in cuts if len(ix)]
    return {"candidates": n_cand, "filled": len(live), "per_week_filled": _f(len(live) / weeks, 1),
            "net_r": _f(r.mean()), "net_r_ci95": [_f(lo), _f(hi)], "gross_r": _f(r.mean() + np.nanmean(cost_r)), "cost_r": _f(np.nanmean(cost_r)),
            "first_half": {"n": len(h1), "net_r": _f(h1.mean()) if len(h1) else None, "ci95": [_f(lo1), _f(hi1)]},
            "second_half": {"n": len(h2), "net_r": _f(h2.mean()) if len(h2) else None, "ci95": [_f(lo2), _f(hi2)]},
            "positive_both_halves_and_full": bool(r.mean() > 0 and len(h1) and len(h2) and h1.mean() > 0 and h2.mean() > 0),
            "slices_positive": f"{sum(s > 0 for s in slices)}/{len(slices)}",
            "win_pct": _f(100 * (r > 0).mean(), 1), "median_r": _f(np.median(r)), "pf": _f(w / l, 2) if l > 0 else None,
            "avg_return_pct": _f(100 * np.mean([t.ret for t in live]), 3), "risk_pct_mean": _f(100 * rf.mean(), 2),
            "risk_pct_median": _f(100 * np.median(rf), 2), "risk_pct_p90": _f(100 * np.percentile(rf, 90), 2),
            "stop_out_pct": _f(100 * loss.mean(), 1), "early_stop_out_pct": _f(100 * early.mean(), 1),
            "mfe_ge_1r_pct": _f(100 * (mfe >= 1).mean(), 1), "mfe_ge_2r_pct": _f(100 * (mfe >= 2).mean(), 1), "avg_bars": _f(np.mean([t.bars for t in live]), 1),
            "floored_pct": _f(100 * np.mean([t.floored for t in live]), 1),
            "exit_mix_pct": {k: _f(100 * v / len(live), 1) for k, v in Counter(t.reason for t in live).most_common(7)},
            "skipped": dict(Counter(t.skipped for t in trs if t.skipped))}


def by_detector(res: dict[str, list[Trade]]) -> dict:
    dets = sorted({t.detector for ts in res.values() for t in ts if not t.skipped})
    out = {}
    for d in dets:
        row = {}
        for name, ts in res.items():
            sel = [t for t in _live(ts) if t.detector == d]
            if sel:
                r = np.array([t.r for t in sel])
                row[name] = {"n": len(sel), "net_r": _f(r.mean(), 3)}
        out[d] = row
    return out


def paired_delta(base: list[Trade], other: list[Trade]) -> dict:
    """other minus BASE on candidates filled in both (same signal), week-block bootstrap."""
    bm = {(t.symbol, t.detector, t.signal_idx): t for t in _live(base)}
    pairs = [(bm[(t.symbol, t.detector, t.signal_idx)], t) for t in _live(other) if (t.symbol, t.detector, t.signal_idx) in bm]
    if not pairs:
        return {}
    d = np.array([o.r - a.r for a, o in pairs])
    lo, hi = boot_mean([o for _, o in pairs], vals=d)
    return {"n": len(pairs), "mean_delta_r": _f(d.mean()), "ci95": [_f(lo), _f(hi)]}


# ----------------------------------------------------------------------
# the pass
# ----------------------------------------------------------------------
def collect(data, pn, w0, progress=None) -> tuple[dict[str, list[Trade]], Counter, Counter]:
    res: dict[str, list[Trade]] = {k: [] for k in ALL}
    gated, n_by_det = Counter(), Counter()
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30
    n_sym = len(data.frames)
    for k, (sym, df) in enumerate(sorted(data.frames.items())):
        b = compute_features(sym, df)
        if b is None:
            continue
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        cands, errs = detect_all(b, lo=lo)
        for c in cands:
            g = BT.gate(b, c, data.surveillance.get(sym), surv_active)
            if g:
                gated[g] += 1
                continue
            if pn.row_of.get(b.date[c.idx]) is None:
                gated["date_not_in_calendar"] += 1
                continue
            n_by_det[c.detector] += 1
            ev = data.events.get(sym)
            for name, X in ALL.items():
                if name == "BASE":
                    tr = MG.simulate(b, c, ev)
                    tr.loss_stop = tr.reason == "stop" or (tr.reason == "stop_gap" and len(tr.exits) == 1)     # no partial yet = still the initial stop
                else:
                    tr = simulate_x(b, c, ev, X)
                res[name].append(tr)
        if progress and k % 25 == 0:
            progress(f"symbols {k}/{n_sym}")
    return res, gated, n_by_det


def compute(data, out: dict) -> dict:
    prog = lambda m: log.info("exits: %s", m)             # noqa: E731
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    w1 = pn.cal[-1]
    weeks = len(pn.cal[pn.cal >= w0]) / 5.0
    mid64 = np.datetime64(mid, "D")
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(w1.date()), "weeks": round(weeks, 1)}
    res, gated, n_by_det = collect(data, pn, w0, prog)
    n_cand = sum(n_by_det.values())
    out["candidates"] = {"total_after_gates": n_cand, "by_detector": dict(n_by_det), "gated": dict(gated),
                         "note": "Identical for every variant: only the exit rulebook differs."}
    out["variants"] = {name: block(ts, mid64, n_cand, weeks) for name, ts in res.items()}
    out["paired_vs_base"] = {name: paired_delta(res["BASE"], res[name]) for name in res if name != "BASE"}
    out["by_detector_net_r"] = by_detector(res)
    out["definitions"] = {k: {f: getattr(v, f) for f in v.__dataclass_fields__} for k, v in ALL.items()}
    prim = {k: out["variants"][k] for k in SPECS}
    meeting = [k for k, v in prim.items() if v.get("positive_both_halves_and_full")]
    best = max(prim, key=lambda k: prim[k].get("net_r") if prim[k].get("net_r") is not None else -9)
    out["deploy_rule"] = {"variants_meeting_rule": meeting, "best_by_full_net_r": best,
                          "text": "positive net expectancy on the full sample AND in both halves (primary readings A-D only)",
                          "note": "Four variants were read on one sample; use net_r_ci95 and paired_vs_base to see whether a pass is distinguishable from zero."}
    return out


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running",
           "note": "Measurement only. Detectors, gates, entries and scoring are unchanged; the live exits are unchanged by this run."}
    try:
        data = BT.load_from_db(conn)
        compute(data, out)
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("exit study failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        _put(conn, out)
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the exit study")
    try:
        log_markdown(render_markdown(out))
    finally:
        if own:
            conn.close()
    return out


def _put(conn, value) -> None:
    with conn.cursor() as cur:
        cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (KEY, json.dumps(value, default=str)))
    conn.commit()


def render_markdown(d: dict) -> str:
    L = [f"# Exit redesign study  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    w = d.get("window") or {}
    L.append(f"window {w.get('start')} .. {w.get('end')} ({w.get('weeks')} weeks), halves split at {w.get('split')}")
    c = d.get("candidates") or {}
    L.append(f"candidates after gates (same for every variant): {c.get('total_after_gates')}")
    L.append("## variants (net R per filled trade)")
    for name, v in (d.get("variants") or {}).items():
        if not v.get("filled"):
            L.append(f"{name}: no trades")
            continue
        h1, h2 = v["first_half"], v["second_half"]
        L.append(f"{name}: filled {v['filled']} ({v['per_week_filled']}/wk) netR {v['net_r']} CI{v['net_r_ci95']} grossR {v['gross_r']} | halves {h1['net_r']} / {h2['net_r']} "
                 f"| slices+ {v['slices_positive']} | win {v['win_pct']}% PF {v['pf']} | avg ret {v['avg_return_pct']}% | risk {v['risk_pct_mean']}% (med {v['risk_pct_median']}, p90 {v['risk_pct_p90']}) "
                 f"| stop-out {v['stop_out_pct']}% early {v['early_stop_out_pct']}% | MFE>=1R {v['mfe_ge_1r_pct']}% >=2R {v['mfe_ge_2r_pct']}% | bars {v['avg_bars']} "
                 f"| meets rule: {v['positive_both_halves_and_full']}")
        L.append(f"   exits {v['exit_mix_pct']} skipped {v['skipped']}")
    L.append("## paired vs BASE (same signals)")
    for name, v in (d.get("paired_vs_base") or {}).items():
        L.append(f"{name}: n {v.get('n')} mean delta R {v.get('mean_delta_r')} CI{v.get('ci95')}")
    L.append("## net R by detector (n, net R) per variant")
    for det, row in (d.get("by_detector_net_r") or {}).items():
        L.append(f"{det}: " + " | ".join(f"{k} {v['net_r']} ({v['n']})" for k, v in row.items()))
    L.append("## deploy rule")
    L.append(json.dumps(d.get("deploy_rule"), default=str))
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for i in range(0, max(len(line), 1), 3000):
            log.info("%s%s", LOG_PREFIX, line[i:i + 3000])
            n += 1
    return n
