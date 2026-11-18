"""
e2_volmatch.py — VOLATILITY-MATCHED CONTROL (owner's order of 18 Nov: gap_and_hold and double_bottom) and the OVERSOLD-BOUNCE ENTRY TEST.

Why: the ALL_BARS control is a random stock on a random day. A detector that fires mostly on volatile stocks can beat it simply because volatile stocks
moved more in this window. So each population is compared with random bars of the SAME volatility:

  matching   ATR% (atr14 / close, percent) at the signal bar, in 5 bins whose edges are the control's own quintiles (10 bins as a sensitivity).
  matched    for every signal, its market-adjusted return minus the mean adjusted return of the control bars in the same ATR bin. The mean of that is the
  excess     matched excess. Halves use the control's own halves (same split, 2 May 2025 like every other study). 95% interval: week-block bootstrap
             over the signals (candidates in a week are correlated); the control means are treated as fixed (the control has tens of thousands of bars).
  rule       (declared before the run) a horizon PASSES when matched excess >= half the width of the control's 95% interval at that horizon, AND the matched
             excess is above zero in BOTH halves, AND its own 95% interval excludes zero. For gap_and_hold / double_bottom a candidate needs the pass at 20
             sessions (the position horizon); for the oversold entry test, a pass at 2 of the 3 short horizons (1, 3, 5), as for every swing entry test.
  also       net of the live costs (0.50%), the ATR mix of the signals against that of their matched control (the match worked if they are close),
             month-by-month and busiest-day concentration checks (a result carried by a few market-wide days is not a pattern).

kind "volmatch": gap_and_hold, gap_and_go, gap_hold_not_confirmed (gap_and_hold rows that gap_and_go never later confirmed - a decomposition that uses
                 hindsight, for interpretation only), double_bottom.   horizons 5, 10, 20
kind "oversold": oversold_dip, oversold_reversal, oversold_dip_any (daily-bar definitions fixed here, before the run).   horizons 1, 3, 5, 10
  oversold_dip       signal day i: close above the 200-day average, 14-day RSI <= 30, close at least 8% below the close 5 sessions earlier; one per episode
                     (nothing within the previous 10 sessions); entry at the next open. The mirror of the day-1 fade the breakout detectors show.
  oversold_reversal  the same oversold condition on the session BEFORE, and the signal day is an up day (close above its open) closing above the previous
                     day's high; entry at the next open. Waits for the turn instead of buying the dip.
  oversold_dip_any   oversold_dip without the 200-day filter (does the trend filter matter?).
Measurement only: no detector, gate, score or exit is touched. Writes engine_settings.engine2_volmatch / engine2_oversold; logs E2VM| lines.
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
from e2_det_common import WARMUP
from e2_detectors import detect_all
from e2_features import compute_features
from e2_fwd import _boot, _f, _market, _week, COST_PCT
from e2_manage import _locked
from e2_swingcheck import GAP_MIN, HOLD_BARS, held

log = logging.getLogger("engine2.volmatch")
LOG_PREFIX = "E2VM|"
KINDS = {
    "volmatch": {"key": "engine2_volmatch", "H": (5, 10, 20), "pops": ("gap_and_hold", "gap_and_go", "gap_hold_not_confirmed", "double_bottom"),
                 "position_h": 20},
    "oversold": {"key": "engine2_oversold", "H": (1, 3, 5, 10), "pops": ("oversold_dip", "oversold_reversal", "oversold_dip_any"), "position_h": None},
}
OS_RSI, OS_DROP5, OS_COOLDOWN = 30.0, -8.0, 10


# ---------------------------------------------------------------------------------------------- matching (pure functions)
def edges_for(atr_ctl: np.ndarray, nbins: int) -> np.ndarray:
    return np.nanpercentile(atr_ctl, np.linspace(0, 100, nbins + 1)[1:-1])


def bin_of(atr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, atr, side="right")


def matched_excess(pop: dict, ctl: dict, mid64, nbins: int = 5):
    """pop / ctl: dicts of aligned arrays sig (datetime64[D]), adj (percent), atr (percent). Returns (per-signal matched excess, per-signal control mean,
    first-half mask) or None when there is nothing to match."""
    ok_c = ~np.isnan(ctl["atr"]) & ~np.isnan(ctl["adj"])
    ok_p = ~np.isnan(pop["atr"]) & ~np.isnan(pop["adj"])
    if ok_c.sum() < 200 or ok_p.sum() < 1:
        return None
    ca, cadj, csig = ctl["atr"][ok_c], ctl["adj"][ok_c], ctl["sig"][ok_c]
    edges = edges_for(ca, nbins)
    cb = bin_of(ca, edges)
    first_c = csig < mid64
    m_all = np.array([cadj[cb == k].mean() if (cb == k).any() else np.nan for k in range(nbins)])
    m_h = [np.array([cadj[(cb == k) & sel].mean() if ((cb == k) & sel).any() else np.nan for k in range(nbins)]) for sel in (first_c, ~first_c)]
    pa, padj, psig = pop["atr"][ok_p], pop["adj"][ok_p], pop["sig"][ok_p]
    pb = bin_of(pa, edges)
    first_p = psig < mid64
    ctl_mean_all = m_all[pb]
    ctl_mean_half = np.where(first_p, m_h[0][pb], m_h[1][pb])
    return {"excess_all": padj - ctl_mean_all, "excess_half": padj - ctl_mean_half, "first": first_p, "sig": psig, "adj": padj, "atr": pa, "bins": pb,
            "ctl_mean": ctl_mean_all, "ctl_atr_by_bin": np.array([ca[cb == k].mean() if (cb == k).any() else np.nan for k in range(nbins)]),
            "ctl_mean_by_bin": m_all}


def month_key(sig: np.ndarray) -> np.ndarray:
    return sig.astype("datetime64[M]")


def analyse(pop: dict, ctl: dict, mid64, ctl_ci_width: float | None) -> dict:
    m = matched_excess(pop, ctl, mid64, 5)
    if m is None:
        return {"n": 0}
    ex_h = m["excess_half"]
    x = m["excess_all"]
    lo, hi = _boot(_week(m["sig"]), x)
    h1, h2 = ex_h[m["first"]], ex_h[~m["first"]]
    need = None if ctl_ci_width is None else 0.5 * ctl_ci_width
    mean = float(x.mean())
    both = bool(len(h1) and len(h2) and h1.mean() > 0 and h2.mean() > 0)
    out = {"n": int(len(x)), "mean_adj_pct": _f(m["adj"].mean()), "raw_excess_vs_control_pct": None,
           "matched_excess_pct": _f(mean), "ci95": [_f(lo), _f(hi)], "halves_matched_excess": [_f(h1.mean()) if len(h1) else None, _f(h2.mean()) if len(h2) else None],
           "halves_n": [int(len(h1)), int(len(h2))], "net_of_costs_pct": _f(m["adj"].mean() - COST_PCT),
           "matched_net_of_costs_pct": _f(mean - COST_PCT), "need_pct": _f(need),
           "atr_mean_pop": _f(m["atr"].mean()), "atr_mean_matched_ctl": _f(float(np.nansum([(m["bins"] == k).mean() * m["ctl_atr_by_bin"][k] for k in range(5)]))),
           "ctl_mean_matched_pct": _f(m["ctl_mean"].mean()),
           "bin_share_pct": [_f(100.0 * (m["bins"] == k).mean(), 1) for k in range(5)],
           "bin_matched_excess": [_f(x[m["bins"] == k].mean()) if (m["bins"] == k).sum() >= 20 else None for k in range(5)]}
    m10 = matched_excess(pop, ctl, mid64, 10)
    out["decile_matched_excess_pct"] = _f(m10["excess_all"].mean()) if m10 is not None else None
    out["passes"] = bool(need is not None and mean >= need and both and lo is not None and lo > 0)
    hm = [v for v in out["halves_matched_excess"] if v is not None]
    out["lopsided"] = bool(len(hm) == 2 and max(hm) > 0 and min(hm) / max(hm) < 0.25)          # a pass carried by one half: reported, never hidden
    out["above_zero_both_halves"] = both
    mo = month_key(m["sig"])
    months = np.unique(mo)
    pos = [float(x[mo == mm].mean()) > 0 for mm in months if (mo == mm).sum() >= 10]
    out["months_positive"] = [int(sum(pos)), int(len(pos))]
    d, cnt = np.unique(m["sig"], return_counts=True)                          # busiest signal days (market-wide event days)
    busy = set(d[np.argsort(-cnt)[:max(1, int(0.05 * len(d)))]].tolist())
    keep = np.array([s not in busy for s in m["sig"].tolist()])
    out["excl_busiest_5pct_days"] = {"n": int(keep.sum()), "matched_excess_pct": _f(x[keep].mean()) if keep.sum() else None}
    return out


# ---------------------------------------------------------------------------------------------- collection
class Rec:
    def __init__(self, H):
        self.H = H
        self.d = {h: ([], [], []) for h in H}

    def extend(self, h, sig, adj, atr):
        a = self.d[h]
        a[0].extend(sig); a[1].extend(adj); a[2].extend(atr)

    def arrays(self, h):
        s, a, v = self.d[h]
        return {"sig": np.array(s, dtype="datetime64[D]"), "adj": np.array(a, float), "atr": np.array(v, float)}


def oversold_signals(b, lo: int) -> dict:
    """signal-bar indices of the three oversold populations; every input is known at the signal bar's close."""
    out = {"oversold_dip": [], "oversold_reversal": [], "oversold_dip_any": []}
    cond_any = np.zeros(b.n, bool)
    cond_trend = np.zeros(b.n, bool)
    for i in range(max(lo, WARMUP, 205), b.n):
        if np.isnan(b.rsi14[i]) or b.c[i - 5] <= 0:
            continue
        if b.rsi14[i] <= OS_RSI and 100.0 * (b.c[i] / b.c[i - 5] - 1.0) <= OS_DROP5:
            cond_any[i] = True
            cond_trend[i] = bool(not np.isnan(b.sma200[i]) and b.c[i] > b.sma200[i])
    last_any = last_tr = -10 ** 9
    for i in range(b.n):
        if cond_any[i] and i - last_any > OS_COOLDOWN:
            out["oversold_dip_any"].append(i)
        if cond_any[i]:
            last_any = i
        if cond_trend[i] and i - last_tr > OS_COOLDOWN:
            out["oversold_dip"].append(i)
        if cond_trend[i]:
            last_tr = i
    for i in range(1, b.n):                                                  # reversal: oversold-in-trend on the session before, up day through its high
        if cond_trend[i - 1] and b.c[i] > b.o[i] and b.c[i] > b.h[i - 1] and not cond_trend[i]:
            out["oversold_reversal"].append(i)
    return out


def _cand(b, i: int, name: str, stop: float):
    c = DC.make(b, "gap_and_go", i, b.h[i], stop, 50.0)
    c.detector = name
    return c


def collect(data, pn, w0, kind: str):
    cfg = KINDS[kind]
    H = cfg["H"]
    recs = {p: Rec(H) for p in cfg["pops"]}
    ctl = Rec(H)
    gated: dict[str, int] = {}
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30

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

    def take(name, b, mk, lst):
        for h in H:
            sel = [(c, e) for c, e in lst if e + h - 1 < b.n and e - 1 >= 0 and not np.isnan(mk[e - 1]) and not np.isnan(mk[e + h - 1])]
            if sel:
                es = np.array([e for _, e in sel])
                idx = np.array([c.idx for c, _ in sel])
                adj = (100.0 * (b.c[es + h - 1] / b.o[es] - 1.0)) - (100.0 * (mk[es + h - 1] / mk[es - 1] - 1.0))
                recs[name].extend(h, [np.datetime64(c.date, "D") for c, _ in sel], adj.tolist(), b.atr_pct[idx].tolist())

    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            continue
        mk = _market(data, b)
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        lists = {p: [] for p in cfg["pops"]}
        if kind == "volmatch":
            cands, _ = detect_all(b, lo=lo)
            go_stops = set()
            for c in cands:
                if c.detector == "gap_and_go":
                    go_stops.add(round(float(c.stop), 6))
                    e = ok(b, sym, c)
                    if e is not None:
                        lists["gap_and_go"].append((c, e))
                elif c.detector == "double_bottom":
                    e = ok(b, sym, c)
                    if e is not None:
                        lists["double_bottom"].append((c, e))
            for g in range(max(lo, WARMUP), b.n - 1):
                if np.isnan(b.gap_pct[g]) or b.gap_pct[g] < GAP_MIN or not held(b, g) or g + HOLD_BARS < lo:
                    continue
                s = g + HOLD_BARS
                c = _cand(b, s, "gap_and_hold", float(b.l[g]))
                e = ok(b, sym, c)
                if e is None:
                    continue
                lists["gap_and_hold"].append((c, e))
                if round(float(b.l[g]), 6) not in go_stops:
                    c2 = copy.copy(c)
                    c2.detector = "gap_hold_not_confirmed"
                    lists["gap_hold_not_confirmed"].append((c2, e))
        else:
            for name, idxs in oversold_signals(b, lo).items():
                for i in idxs:
                    if i < lo:
                        continue
                    c = _cand(b, i, name, float(b.l[i]))
                    e = ok(b, sym, c)
                    if e is not None:
                        lists[name].append((c, e))
        for p, lst in lists.items():
            take(p, b, mk, lst)
        for h in H:                                                           # control: every 5th bar; volatility read at the signal bar (the bar before entry)
            es = np.arange(max(lo, WARMUP) + 1, b.n - h + 1, 5)
            es = es[(es - 1 >= 0) & ~np.isnan(mk[es - 1]) & ~np.isnan(mk[es + h - 1]) & (b.o[es] > 0)]
            if len(es):
                adj = (100.0 * (b.c[es + h - 1] / b.o[es] - 1.0)) - (100.0 * (mk[es + h - 1] / mk[es - 1] - 1.0))
                ctl.extend(h, b.date[es - 1].astype("datetime64[D]").tolist(), adj.tolist(), b.atr_pct[es - 1].tolist())
    return recs, ctl, gated


def compute(data, out: dict, kind: str = "volmatch") -> dict:
    cfg = KINDS[kind]
    H = cfg["H"]
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    mid64 = np.datetime64(mid, "D")
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(pn.cal[-1].date())}
    out["kind"] = kind
    recs, ctl, gated = collect(data, pn, w0, kind)
    rows = {}
    ctl_stats = {}
    for h in H:
        c = ctl.arrays(h)
        lo, hi = _boot(_week(c["sig"]), c["adj"]) if len(c["adj"]) else (None, None)
        ctl_stats[h] = {"n": int(len(c["adj"])), "mean_adj_pct": _f(np.nanmean(c["adj"])) if len(c["adj"]) else None, "ci95": [_f(lo), _f(hi)],
                        "width": None if lo is None else float(hi - lo)}
    out["control"] = {str(h): ctl_stats[h] for h in H}
    for p, r in recs.items():
        per = {}
        for h in H:
            a = r.arrays(h)
            c = ctl.arrays(h)
            res = analyse(a, c, mid64, ctl_stats[h]["width"])
            if res.get("n") and ctl_stats[h]["mean_adj_pct"] is not None:
                res["raw_excess_vs_control_pct"] = _f(res["mean_adj_pct"] - ctl_stats[h]["mean_adj_pct"])
            per[str(h)] = res
        rows[p] = per
    out["populations"] = rows
    if kind == "volmatch":
        ph = str(cfg["position_h"])
        out["decision"] = {p: {"passes_at_20": rows[p][ph].get("passes"), "lopsided": rows[p][ph].get("lopsided"), "matched_excess_20": rows[p][ph].get("matched_excess_pct"),
                               "halves": rows[p][ph].get("halves_matched_excess"), "ci95": rows[p][ph].get("ci95"), "n": rows[p][ph].get("n")}
                           for p in ("gap_and_hold", "gap_and_go", "gap_hold_not_confirmed", "double_bottom") if rows[p][ph].get("n")}
        out["rule"] = "candidate = matched excess >= half the control's interval width, above zero in both halves, interval excludes zero, at 20 sessions"
    else:
        sp = {}
        for p, per in rows.items():
            passes = [h for h in (1, 3, 5) if per[str(h)].get("passes")]
            nets = [h for h in (1, 3, 5, 10) if per[str(h)].get("matched_net_of_costs_pct") is not None and per[str(h)]["matched_net_of_costs_pct"] > 0]
            sp[p] = {"passes_at": passes, "short_pass": len(passes) >= 2, "matched_net_positive_at": nets}
        out["decision"] = sp
        out["rule"] = "short-horizon pass = the matched rule at 2 of the 3 horizons 1, 3, 5"
    out["gated"] = gated
    out["note"] = "Measurement only. matched excess = adjusted return minus the mean of the control bars in the same ATR% bin (control quintiles); percent."
    return out


# ---------------------------------------------------------------------------------------------- run / render
def _p(x, nd=2):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def render_markdown(d: dict) -> str:
    kind = d.get("kind") or "volmatch"
    H = KINDS[kind]["H"]
    L = [f"# Volatility-matched control: {kind}  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    w = d.get("window") or {}
    L.append(f"window {w.get('start')} .. {w.get('end')}, halves split at {w.get('split')}; percent, market-adjusted; net = after {COST_PCT:.2f}")
    L.append("control (every 5th bar): " + "; ".join(f"h{h} n{(d.get('control') or {}).get(str(h), {}).get('n')} mean {_p((d.get('control') or {}).get(str(h), {}).get('mean_adj_pct'))} "
                                                   f"CI{(d.get('control') or {}).get(str(h), {}).get('ci95')}" for h in H))
    for p, per in (d.get("populations") or {}).items():
        for h in H:
            s = per.get(str(h)) or {}
            if not s.get("n"):
                L.append(f"{p} h{h}: n0")
                continue
            L.append(f"{p} h{h}{'*' if s.get('passes') else ''}: n{s['n']} adj {_p(s['mean_adj_pct'])} | raw excess {_p(s.get('raw_excess_vs_control_pct'))} -> MATCHED excess "
                     f"{_p(s['matched_excess_pct'])} CI{s['ci95']} need {_p(s.get('need_pct'))} halves {s['halves_matched_excess']} (n {s['halves_n']}) "
                     f"lopsided {s.get('lopsided')} deciles {_p(s.get('decile_matched_excess_pct'))} | matched net {_p(s.get('matched_net_of_costs_pct'))} | ATR% pop {s.get('atr_mean_pop')} vs matched ctl "
                     f"{s.get('atr_mean_matched_ctl')} bins% {s.get('bin_share_pct')} by-bin {s.get('bin_matched_excess')} | months+ {s.get('months_positive')} | "
                     f"ex busiest 5% days n{(s.get('excl_busiest_5pct_days') or {}).get('n')} {_p((s.get('excl_busiest_5pct_days') or {}).get('matched_excess_pct'))}")
    L.append("## decision (rule declared before the run: " + str(d.get("rule")) + "; a pass with lopsided=true is carried by one half)")
    L.append(json.dumps(d.get("decision"), default=str))
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for i in range(0, max(len(line), 1), 1400):
            log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
            n += 1
    return n


def run(kind: str = "volmatch", conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running", "kind": kind,
           "note": "Measurement only."}
    try:
        data = BT.load_from_db(conn)
        compute(data, out, kind)
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("volatility-matched control failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KINDS[kind]["key"], json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the volatility-matched control")
    try:
        log_markdown(render_markdown(out))
    finally:
        if own:
            conn.close()
    return out
