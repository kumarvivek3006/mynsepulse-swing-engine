"""
e2_fwd.py — FORWARD-RETURN TEST (owner's order of 7 Nov 00:17 IST): do the entries have any drift, whatever the exits do?

For every candidate that passes the live gates: enter at the next session's open (the live entry, no stop, no partial, no trailing), hold h
sessions and exit at that session's close. h = 5, 10, 20 (h sessions held, the entry session counted as the first).
    raw      = close[e+h-1] / open[e] - 1
    market   = NIFTY500 (NIFTY50 if that is all there is) close[e+h-1] / close[e-1] - 1   (the index has closes only; its window starts one
               overnight earlier than the stock's, a bias of roughly +0.05..0.1% in the market leg - stated, not corrected)
    adjusted = raw - market                      the headline: market-adjusted forward return, in percent
    net      = adjusted - 0.50                   after the live costs (0.30% round trip + 0.10% slippage per side), in percent
Each row also carries sample size per horizon, median, share of adjusted returns above zero, both halves (split 2 May 2025 like every other study),
and a 95% interval from a week-block bootstrap (candidates in the same week are correlated).
Control row ALL_BARS: the same measure for every 5th bar of every symbol in the window (no detector) - the drift of simply being long a stock,
which is what a detector has to beat.
Rows for pocket_pivot: the live definition (rung 3, shadowed), the strict definition (rung 0) and the original loose one (legacy), so the
definition question can be read next to the class question.

Triage rule (revised 7 Nov 01:25 IST by the owner: the benchmark is the control row, not zero). A horizon PASSES when the detector's mean adjusted
return beats the control's mean at that horizon by at least HALF THE WIDTH of the control's 95% interval, and beats the control's mean in BOTH halves.
  positive  - at least 2 of the 3 horizons pass
  negative  - mean adjusted return <= the control's at all three horizons
  mixed     - everything else (read the per-horizon line: good early and fading, or the reverse)
`triage_vs_zero` keeps the first rule (mean > 0 in the full sample and both halves) for comparison with the 7 Nov 00:50 report.
A "positive" whose interval straddles the control is reported as exactly that. The diagnosis queue is ordered by mean adjusted return at 10 sessions.

Measurement only: no detector, gate, score or exit is touched. Writes engine_settings.engine2_fwd; logs E2FWD| lines; GET /jobs/engine2-fwd.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import e2_backtest as BT
import e2_det_momentum as DM
from e2_det_common import WARMUP
from e2_detectors import dedupe, detect_all
from e2_features import compute_features
from e2_manage import COST_RT, SLIP, _locked

log = logging.getLogger("engine2.fwd")
KEY = "engine2_fwd"
LOG_PREFIX = "E2FWD|"
# 7 Nov (owner): 1 and 3 sessions added to see short-horizon edge the 5/10/20 test cannot. The triage label keeps its original three horizons
# (TRIAGE_H) so the labels stay comparable; every horizon also carries its own beats_control flag (same pre-declared rule, applied per horizon).
HORIZONS = tuple(int(x) for x in os.environ.get("E2_FWD_HORIZONS", "1,3,5,10,20").split(","))
TRIAGE_H = tuple(h for h in (5, 10, 20) if h in HORIZONS)
COST_PCT = 100.0 * (COST_RT + 2 * SLIP)
PP_VARIANTS = {"pocket_pivot_strict_rung0": dict(strict=True, rung=0), "pocket_pivot_legacy": dict(strict=False),
               "pocket_pivot_rung5": dict(strict=True, rung=5)}


def _f(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _boot(weeks: np.ndarray, vals: np.ndarray, reps: int = 2000, seed: int = 17):
    if len(vals) < 30:
        return None, None
    u, inv = np.unique(weeks, return_inverse=True)
    S, N = np.bincount(inv, weights=vals), np.bincount(inv).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(u), size=(reps, len(u)))
    m = S[idx].sum(1) / N[idx].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _week(d64: np.ndarray) -> np.ndarray:
    return (d64.astype("datetime64[D]") - np.datetime64("2000-01-03", "D")).astype(int) // 7


class Acc:
    """Collects (signal date, raw %, market %) per horizon."""
    def __init__(self):
        self.d = {h: ([], [], []) for h in HORIZONS}

    def add(self, h, sig, raw, mkt):
        a = self.d[h]
        a[0].append(sig); a[1].append(raw); a[2].append(mkt)

    def extend(self, h, sig, raw, mkt):
        a = self.d[h]
        a[0].extend(sig); a[1].extend(raw); a[2].extend(mkt)

    def stats(self, h, mid64) -> dict:
        sig, raw, mkt = (np.array(x) for x in self.d[h])
        if len(raw) == 0:
            return {"n": 0}
        sig = sig.astype("datetime64[D]")
        adj = raw - mkt
        lo, hi = _boot(_week(sig), adj)
        h1, h2 = adj[sig < mid64], adj[sig >= mid64]
        return {"n": int(len(adj)), "mean_adj_pct": _f(adj.mean()), "ci95": [_f(lo), _f(hi)], "median_adj_pct": _f(np.median(adj)),
                "pos_pct": _f(100 * (adj > 0).mean(), 1), "mean_raw_pct": _f(raw.mean()), "mean_market_pct": _f(mkt.mean()),
                "net_of_costs_pct": _f(adj.mean() - COST_PCT),
                "h1": {"n": int(len(h1)), "mean_adj_pct": _f(h1.mean()) if len(h1) else None},
                "h2": {"n": int(len(h2)), "mean_adj_pct": _f(h2.mean()) if len(h2) else None},
                "pass": bool(adj.mean() > 0 and len(h1) and len(h2) and h1.mean() > 0 and h2.mean() > 0)}


def beats_control(s: dict, c: dict) -> bool:
    """One horizon, the pre-declared rule: mean adjusted return above the control's by at least half the control's 95% interval width,
    and above the control's mean in BOTH halves."""
    if not s.get("n") or not c.get("n") or s.get("mean_adj_pct") is None or c.get("mean_adj_pct") is None:
        return False
    need = 0.5 * (c["ci95"][1] - c["ci95"][0])
    h1, h2 = s["h1"]["mean_adj_pct"], s["h2"]["mean_adj_pct"]
    return bool(s["mean_adj_pct"] - c["mean_adj_pct"] >= need and h1 is not None and h2 is not None
                and h1 > c["h1"]["mean_adj_pct"] and h2 > c["h2"]["mean_adj_pct"])


def triage(per_h: dict, ctl: dict | None = None) -> str:
    """Control-relative rule when `ctl` (the control row's per-horizon stats) is given; otherwise the original rule against zero."""
    if ctl is not None:
        passes, below, seen = 0, 0, 0
        for h in TRIAGE_H:
            s, c = per_h[h], ctl[h]
            if not s.get("n") or not c.get("n") or s.get("mean_adj_pct") is None:
                continue
            seen += 1
            need = 0.5 * (c["ci95"][1] - c["ci95"][0])
            beats = s["mean_adj_pct"] - c["mean_adj_pct"] >= need
            both = (s["h1"]["mean_adj_pct"] is not None and s["h2"]["mean_adj_pct"] is not None
                    and s["h1"]["mean_adj_pct"] > c["h1"]["mean_adj_pct"] and s["h2"]["mean_adj_pct"] > c["h2"]["mean_adj_pct"])
            passes += bool(beats and both)
            below += bool(s["mean_adj_pct"] <= c["mean_adj_pct"])
        if not seen:
            return "no_data"
        return "positive" if passes >= 2 else ("negative" if below == seen else "mixed")
    return triage_vs_zero(per_h)


def triage_vs_zero(per_h: dict) -> str:
    means = [per_h[h].get("mean_adj_pct") for h in TRIAGE_H if per_h[h].get("n")]
    if not means:
        return "no_data"
    passes = sum(1 for h in TRIAGE_H if per_h[h].get("pass"))
    if passes >= 2:
        return "positive"
    if all(m is not None and m <= 0 for m in means):
        return "negative"
    return "mixed"


def _market(data, b) -> np.ndarray:
    ic = data.index_close
    if ic is None or len(ic) < 250:
        raise RuntimeError("forward-return test needs the index series (NIFTY500/NIFTY50) and it is missing: refusing to substitute a proxy silently")
    return pd.Series(ic).reindex(pd.DatetimeIndex(b.date)).ffill().to_numpy(dtype=float)


def collect(data, pn, w0):
    acc: dict[str, Acc] = {}
    gated: dict[str, int] = {}
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30
    null = Acc()
    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            continue
        mk = _market(data, b)
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        cands, _ = detect_all(b, lo=lo)
        extra = []
        for name, kw in PP_VARIANTS.items():
            for c in dedupe(DM.pocket_pivot(b, {}, lo, b.n, **kw)):
                c2 = copy.copy(c)
                c2.detector = name
                extra.append(c2)
        per: dict[str, list] = {}
        for c in list(cands) + extra:
            g = BT.gate(b, c, data.surveillance.get(sym), surv_active)
            if g:
                gated[g] = gated.get(g, 0) + 1
                continue
            if pn.row_of.get(b.date[c.idx]) is None:
                continue
            e = c.idx + 1
            if e >= b.n or _locked(b, e):
                continue
            per.setdefault(c.detector, []).append((c, e))
        for det, lst in per.items():
            a = acc.setdefault(det, Acc())
            for h in HORIZONS:
                sel = [(c, e) for c, e in lst if e + h - 1 < b.n and e - 1 >= 0 and not np.isnan(mk[e - 1]) and not np.isnan(mk[e + h - 1])]
                if not sel:
                    continue
                es = np.array([e for _, e in sel])
                raw = 100.0 * (b.c[es + h - 1] / b.o[es] - 1.0)
                mkt = 100.0 * (mk[es + h - 1] / mk[es - 1] - 1.0)
                a.extend(h, [np.datetime64(c.date, "D") for c, _ in sel], raw.tolist(), mkt.tolist())
        # control: every 5th bar in the window
        for h in HORIZONS:
            es = np.arange(max(lo, WARMUP) + 1, b.n - h + 1, 5)
            es = es[(es - 1 >= 0) & ~np.isnan(mk[es - 1]) & ~np.isnan(mk[es + h - 1]) & (b.o[es] > 0)]
            if len(es):
                null.extend(h, b.date[es - 1].astype("datetime64[D]").tolist(), (100.0 * (b.c[es + h - 1] / b.o[es] - 1.0)).tolist(),
                            (100.0 * (mk[es + h - 1] / mk[es - 1] - 1.0)).tolist())
    acc["ALL_BARS (control)"] = null
    return acc, gated


def compute(data, out: dict) -> dict:
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    mid64 = np.datetime64(mid, "D")
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(pn.cal[-1].date())}
    acc, gated = collect(data, pn, w0)
    rows = {}
    stats = {det: {h: a.stats(h, mid64) for h in HORIZONS} for det, a in sorted(acc.items())}
    ctl = stats["ALL_BARS (control)"]
    for det, per_int in stats.items():
        is_ctl = det.startswith("ALL_BARS")
        rows[det] = {"horizons": {str(h): per_int[h] for h in HORIZONS},
                     "triage": "control" if is_ctl else triage(per_int, ctl), "triage_vs_zero": "control" if is_ctl else triage_vs_zero(per_int),
                     "beats_control": None if is_ctl else {str(h): beats_control(per_int[h], ctl[h]) for h in HORIZONS if per_int[h].get("n")},
                     "excess_vs_control_pct": None if is_ctl else {str(h): _f((per_int[h].get("mean_adj_pct") or 0) - (ctl[h].get("mean_adj_pct") or 0))
                                                                  for h in HORIZONS if per_int[h].get("n")}}
    out["detectors"] = rows
    out["gated"] = gated
    live = {d: r for d, r in rows.items() if r["triage"] != "control" and not d.startswith("pocket_pivot_")}
    queue = sorted(live, key=lambda d: -(live[d]["horizons"]["10"].get("mean_adj_pct") if live[d]["horizons"]["10"].get("n") else -99))
    out["diagnosis_queue_by_10bar_adj"] = [{"detector": d, "mean_adj_pct_10": live[d]["horizons"]["10"].get("mean_adj_pct"), "n": live[d]["horizons"]["10"].get("n"),
                                            "triage": live[d]["triage"]} for d in queue]
    out["note"] = ("Market-adjusted = stock open-to-close return minus the index's close(e-1)->close(e+h-1) return. "
                   "Compare each detector with ALL_BARS (control): being long a stock at a random time.")
    return out


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running",
           "note": "Measurement only."}
    try:
        data = BT.load_from_db(conn)
        compute(data, out)
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("forward-return test failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the forward-return test")
    try:
        log_markdown(render_markdown(out))
    finally:
        if own:
            conn.close()
    return out


def _p(x, nd=2):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def render_markdown(d: dict) -> str:
    L = [f"# Forward-return test  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    w = d.get("window") or {}
    L.append(f"window {w.get('start')} .. {w.get('end')}, halves split at {w.get('split')}; adjusted = stock minus NIFTY, percent; net = adjusted - {COST_PCT:.2f}")
    for det, r in (d.get("detectors") or {}).items():
        parts = []
        for h in HORIZONS:
            s = r["horizons"][str(h)]
            if not s.get("n"):
                parts.append(f"h{h}: n0")
                continue
            star = "*" if (r.get("beats_control") or {}).get(str(h)) else ""
            parts.append(f"h{h}{star}: n{s['n']} adj {_p(s['mean_adj_pct'])}% CI{s['ci95']} med {_p(s['median_adj_pct'])} pos {s['pos_pct']}% "
                         f"halves {_p(s['h1']['mean_adj_pct'])}/{_p(s['h2']['mean_adj_pct'])} net {_p(s['net_of_costs_pct'])}")
        ex = r.get("excess_vs_control_pct")
        L.append(f"{det} [{r['triage']}; vs zero: {r.get('triage_vs_zero')}]" + (f" excess vs control {ex}" if ex else "") + ": " + " | ".join(parts))
    L.append("## beats the control (mean above control by >= half the control's interval width AND above it in both halves), per horizon; * on a row above")
    for h in HORIZONS:
        names = [det for det, r in (d.get("detectors") or {}).items() if (r.get("beats_control") or {}).get(str(h))]
        L.append(f"h{h}: " + (", ".join(names) if names else "none"))
    L.append("## diagnosis queue (10-session adjusted return, strongest first)")
    for q in d.get("diagnosis_queue_by_10bar_adj") or []:
        L.append(f"{q['detector']}: {q['mean_adj_pct_10']} (n {q['n']}) {q['triage']}")
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for i in range(0, max(len(line), 1), 1400):
            log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
            n += 1
    return n
