"""
e2_swingcheck.py — SWING ENGINE, STEP 1b (owner's order of 7 Nov 03:29 IST): is a day-3 entry on a held gap viable (gap_and_hold)?
Measurement only. Nothing here is a detector in the live engine.

Populations (all pass the live gates, enter at the NEXT session's open after the signal bar, exit at the close of the h-th session, h = 1, 3, 5, 10;
market-adjusted by NIFTY close(e-1) -> close(e+h-1); net = adjusted - 0.50; week-block bootstrap; halves split like every other study):
  gap3_day1        every gap-up day (open >= 3% above the previous close). Signal = the gap day, entry the next open (the day-1 entry).
  ep_detector_day1 the live episodic_pivot detector (gap >= 4%, relvol >= 5, close in the top quarter). Signal = the gap day, entry day 1.
  gap_and_hold     gap >= 3% AND the next two sessions HOLD it: no low under the pre-gap close, no close under 97% of the gap-day open (the same
                   hold test gap_and_go uses). Signal = the second holding session, entry the open of the THIRD session after the gap (day 3).
  ep_held_day3     the episodic-style gap days (gap >= 4%, relvol >= 5, close in the top quarter) that then also hold for two sessions; entry day 3.
  gap_and_go       the live gap_and_go detector, for reference.
Control: every 5th bar of every symbol (same as the forward-return test).

Pre-declared decision rule (owner): the 3-session hold from the day-3 entry, net of costs (mean adjusted - 0.50):
    < +0.10%        drop gap_and_hold, build episodic_pivot_swing instead
    +0.10 .. +0.30  shelve, not viable
    >= +0.30        build gap_and_hold as the second swing detector
Both halves are reported next to it; a verdict that rests on one half is called out as such.

Also, for the episodic_pivot_swing proposal (day-1 hold on the live episodic_pivot candidates): open-to-close, open-to-next-open, how far the day went
against and for the entry in ATR units (daily bars: what a tight stop would meet), signals per month.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

import e2_backtest as BT
import e2_det_common as DC
import e2_det_momentum as DM
from e2_det_common import WARMUP
from e2_detectors import detect_all
from e2_features import compute_features
from e2_fwd import Acc, _f, _market, _p, COST_PCT
from e2_manage import _locked

log = logging.getLogger("engine2.swingcheck")
KEY = "engine2_swingcheck"
LOG_PREFIX = "E2SWC|"
H = (1, 3, 5, 10)
GAP_MIN, EP_GAP, EP_VOL, EP_CP = 3.0, 4.0, 5.0, 0.75
HOLD_BARS, HOLD_FADE = 2, 0.97
POPS = ("gap3_day1", "ep_detector_day1", "gap_and_hold", "ep_held_day3", "gap_and_go")


def verdict(net_h3: float | None) -> str:
    if net_h3 is None:
        return "no_data"
    if net_h3 < 0.10:
        return "drop gap_and_hold; build episodic_pivot_swing instead"
    if net_h3 < 0.30:
        return "shelve: not viable"
    return "build gap_and_hold as the second swing detector"


def held(b, g: int) -> bool:
    """the two sessions after gap day g hold the gap (gap_and_go's own hold test)."""
    if g + HOLD_BARS >= b.n:
        return False
    seg = slice(g + 1, g + HOLD_BARS + 1)
    return bool(not (b.l[seg] < b.c_prev[g]).any() and not (b.c[seg] < b.o[g] * HOLD_FADE).any())


def _cand(b, i: int, name: str, g: int):
    c = DC.make(b, "gap_and_go", i, b.h[i], b.l[g], 50.0, gap_pct=round(float(b.gap_pct[g]), 2))
    c.detector = name
    return c


def collect(data, pn, w0):
    acc = {p: Acc() for p in POPS}
    acc["ALL_BARS (control)"] = Acc()
    day1 = []                                                    # episodic_pivot day-1 rows
    gated: dict[str, int] = {}
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30

    def take(name, b, mk, lst):
        a = acc[name]
        for h in H:
            sel = [(c, e) for c, e in lst if e + h - 1 < b.n and e - 1 >= 0 and not np.isnan(mk[e - 1]) and not np.isnan(mk[e + h - 1])]
            if sel:
                es = np.array([e for _, e in sel])
                a.extend(h, [np.datetime64(c.date, "D") for c, _ in sel], (100.0 * (b.c[es + h - 1] / b.o[es] - 1.0)).tolist(),
                         (100.0 * (mk[es + h - 1] / mk[es - 1] - 1.0)).tolist())

    def ok(b, sym, c):
        g = BT.gate(b, c, data.surveillance.get(sym), surv_active)
        if g:
            gated[g] = gated.get(g, 0) + 1
            return None
        if pn.row_of.get(b.date[c.idx]) is None:
            return None
        e = c.idx + 1
        if e >= b.n or _locked(b, e):
            return None
        return e

    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            continue
        mk = _market(data, b)
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        gaps = [g for g in range(max(lo, WARMUP), b.n - 1) if not np.isnan(b.gap_pct[g]) and b.gap_pct[g] >= GAP_MIN]
        lists = {p: [] for p in POPS}
        for g in gaps:
            c = _cand(b, g, "gap3_day1", g)
            e = ok(b, sym, c)
            if e is not None:
                lists["gap3_day1"].append((c, e))
            is_ep = bool(b.gap_pct[g] >= EP_GAP and b.relvol[g] >= EP_VOL and b.close_pos[g] >= EP_CP)
            if held(b, g) and g + HOLD_BARS >= lo:
                s = g + HOLD_BARS
                c2 = _cand(b, s, "gap_and_hold", g)
                e2 = ok(b, sym, c2)
                if e2 is not None:
                    lists["gap_and_hold"].append((c2, e2))
                    if is_ep:
                        c3 = copy.copy(c2)
                        c3.detector = "ep_held_day3"
                        lists["ep_held_day3"].append((c3, e2))
        cands, _ = detect_all(b, lo=lo)
        for c in cands:
            if c.detector not in ("episodic_pivot", "gap_and_go"):
                continue
            e = ok(b, sym, c)
            if e is None:
                continue
            if c.detector == "episodic_pivot":
                lists["ep_detector_day1"].append((c, e))
                atr = float(b.atr14[c.idx]) if not np.isnan(b.atr14[c.idx]) else np.nan
                if e + 1 < b.n and atr > 0 and not np.isnan(mk[e - 1]) and not np.isnan(mk[e]):
                    day1.append((np.datetime64(c.date, "D"), 100.0 * (b.c[e] / b.o[e] - 1), 100.0 * (b.o[e + 1] / b.o[e] - 1),
                                 (b.h[e] - b.o[e]) / atr, (b.o[e] - b.l[e]) / atr, 100.0 * (mk[e] / mk[e - 1] - 1), 100.0 * atr / b.o[e],
                                 100.0 * (b.o[e] / b.c[c.idx] - 1)))
            else:
                lists["gap_and_go"].append((c, e))
        for p, lst in lists.items():
            take(p, b, mk, lst)
        for h in H:                                              # control
            es = np.arange(max(lo, WARMUP) + 1, b.n - h + 1, 5)
            es = es[(es - 1 >= 0) & ~np.isnan(mk[es - 1]) & ~np.isnan(mk[es + h - 1]) & (b.o[es] > 0)]
            if len(es):
                acc["ALL_BARS (control)"].extend(h, b.date[es - 1].astype("datetime64[D]").tolist(),
                                                 (100.0 * (b.c[es + h - 1] / b.o[es] - 1.0)).tolist(),
                                                 (100.0 * (mk[es + h - 1] / mk[es - 1] - 1.0)).tolist())
    return acc, day1, gated


def day1_summary(rows, mid64, months: float) -> dict:
    if not rows:
        return {"n": 0}
    a = np.array([(r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows], float)
    d = np.array([r[0] for r in rows]).astype("datetime64[D]")
    oc, on, up_atr, dn_atr, mkt, atr_pct, gap_in = (a[:, k] for k in range(7))
    adj = oc - mkt

    def pct(x, k):
        return _f(100.0 * float((x >= k).mean()), 1)

    def half(x, m):
        return [_f(x[m].mean()) if m.sum() else None, int(m.sum())]
    m1 = d < mid64
    return {"n": int(len(a)), "per_month": _f(len(a) / months, 2),
            "open_to_close_pct": _f(oc.mean()), "open_to_close_adj_pct": _f(adj.mean()), "adj_net_of_costs_pct": _f(adj.mean() - COST_PCT),
            "halves_adj": [half(adj, m1), half(adj, ~m1)],
            "median_open_to_close_pct": _f(np.median(oc)), "share_up_pct": _f(100 * (oc > 0).mean(), 1),
            "open_to_next_open_pct": _f(on.mean()), "next_open_minus_close_pct": _f((on - oc).mean()),
            "mean_atr_pct": _f(atr_pct.mean()), "mean_open_vs_signal_close_pct": _f(gap_in.mean()),
            "up_move_atr_mean": _f(up_atr.mean()), "down_move_atr_mean": _f(dn_atr.mean()),
            "day_low_reaches_stop_pct": {"0.75atr": pct(dn_atr, 0.75), "1.0atr": pct(dn_atr, 1.0), "1.5atr": pct(dn_atr, 1.5)},
            "day_high_reaches_target_pct": {"0.5atr": pct(up_atr, 0.5), "1.0atr": pct(up_atr, 1.0), "1.5atr": pct(up_atr, 1.5), "2.0atr": pct(up_atr, 2.0)}}


def compute(data, out: dict) -> dict:
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    mid64 = np.datetime64(mid, "D")
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(pn.cal[-1].date())}
    months = max((pn.cal[-1] - w0).days / 30.44, 1.0)
    acc, day1, gated = collect(data, pn, w0)
    stats = {p: {h: a.stats(h, mid64) for h in H} for p, a in acc.items()}
    ctl = stats["ALL_BARS (control)"]
    rows = {}
    for p, per in stats.items():
        ctl_row = p.startswith("ALL_BARS")
        rows[p] = {"horizons": {str(h): per[h] for h in H},
                   "excess_vs_control_pct": None if ctl_row else {str(h): _f((per[h].get("mean_adj_pct") or 0) - (ctl[h].get("mean_adj_pct") or 0))
                                                                for h in H if per[h].get("n")}}
    out["populations"] = rows
    gh = stats["gap_and_hold"][3]
    net3 = gh.get("net_of_costs_pct")
    out["decision"] = {"population": "gap_and_hold, 3-session hold from the day-3 entry", "net_of_costs_pct": net3,
                       "halves_adj_pct": [gh.get("h1", {}).get("mean_adj_pct"), gh.get("h2", {}).get("mean_adj_pct")],
                       "interval_adj_pct": gh.get("ci95"), "n": gh.get("n"), "verdict": verdict(net3),
                       "ep_held_day3_net_3_pct": stats["ep_held_day3"][3].get("net_of_costs_pct")}
    out["episodic_pivot_day1"] = day1_summary(day1, mid64, months)
    out["gated"] = gated
    out["note"] = ("Measurement only. The day-1 stock leg is open-to-close and the index leg close-to-close: read the excess over the control, which carries "
                   "the same mismatch. Costs: %.2f%% round trip incl. slippage." % COST_PCT)
    return out


def render_markdown(d: dict) -> str:
    L = [f"# Swing check: gap_and_hold  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    w = d.get("window") or {}
    L.append(f"window {w.get('start')} .. {w.get('end')}, halves split at {w.get('split')}")
    for p, r in (d.get("populations") or {}).items():
        parts = []
        for h in H:
            s = r["horizons"][str(h)]
            if not s.get("n"):
                parts.append(f"h{h}: n0")
                continue
            parts.append(f"h{h}: n{s['n']} adj {_p(s['mean_adj_pct'])}% CI{s['ci95']} halves {_p(s['h1']['mean_adj_pct'])}/{_p(s['h2']['mean_adj_pct'])} "
                         f"net {_p(s['net_of_costs_pct'])}")
        L.append(f"{p}" + (f" excess vs control {r['excess_vs_control_pct']}" if r.get("excess_vs_control_pct") else "") + ": " + " | ".join(parts))
    L.append("## decision (owner's pre-declared rule)")
    L.append(json.dumps(d.get("decision"), default=str))
    L.append("## episodic_pivot day-1 profile (for the episodic_pivot_swing proposal)")
    L.append(json.dumps(d.get("episodic_pivot_day1"), default=str))
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for i in range(0, max(len(line), 1), 1400):
            log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
            n += 1
    return n


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running", "note": "Measurement only."}
    try:
        data = BT.load_from_db(conn)
        compute(data, out)
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("swing check failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the swing check")
    try:
        log_markdown(render_markdown(out))
    finally:
        if own:
            conn.close()
    return out
