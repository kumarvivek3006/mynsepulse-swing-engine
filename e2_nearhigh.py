"""
e2_nearhigh.py — 52-WEEK-HIGH AS A STANDALONE FILTER (owner's order of 26 Nov 03:22 IST, finding 2). Measurement only: no detector, gate, score or exit is touched.

Question: five detectors show "the near-52-week-high subset is better". Is that one stock-type effect (any stock near its high does better), in which case the framework's
edge reduces to a simple screen, or is it specific to the detectors?

Population: every stock in the backtest universe (about 500) on every 5th SESSION of the 3-year window (the same calendar day for all symbols, so same-day comparisons are exact).
Signal at the close of day i; entry at the next open; 20-session forward return = next open -> close of the 20th session, MINUS the index over the same stretch (market-adjusted,
no costs) - the same measure as every forward-return study. Horizons 5 and 10 are reported too.
Filter: close within 5%, 10%, 15%, 20% of the 52-week high at the signal close (b.high52). Also the complement (more than 20% below the high).

Reported per filter, at 20 sessions:
  raw          filtered mean, full-universe mean, DIFFERENCE (the owner's gate: > 1.0 percentage point), the same in each half (split 20 May 2025, the mid of the window),
               with a week-block 95% interval on the difference (bars in the same week are correlated).
  same-day     filtered return minus the mean of the whole universe on the SAME signal day: removes "near-high stocks are common in bull phases".
  ATR-matched  filtered return minus random bars of the same ATR% (e2_volmatch): near-high stocks may simply be the higher-volatility ones. Reported next to the raw number.
  gate         the owner's: difference > 1.0 at 20 sessions AND the filtered mean positive in both halves AND the difference positive in both halves
               (an ambiguous clause - both readings are printed).
Comparison with the detector portfolio: all detector signals (same gates, same entry) pooled and per detector, and the SAME detector signals restricted to the near-high set against the
near-high UNIVERSE: if a detector's signals inside the filter beat the filtered universe by nothing, the detector adds nothing the screen does not already give.
Caveats: the universe is today's constituents (survivorship flatters near-high stocks: those that fell and left are absent); 20-session windows overlap, hence the week-block intervals.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np

import e2_backtest as BT
import e2_volmatch as VM
from e2_det_common import WARMUP
from e2_detectors import detect_all
from e2_features import compute_features
from e2_fwd import COST_PCT, _boot, _f, _market, _week
from e2_manage import _locked

log = logging.getLogger("engine2.nearhigh")
KEY = "e2_nearhigh"
LOG_PREFIX = "E2NH|"
H = (5, 10, 20)
POS_H = 20
BANDS = (5, 10, 15, 20)
STEP = 5
BOOT_REPS = 2000
GATE_PP = 1.0


# ---------------------------------------------------------------------------------------------- pure statistics
def boot_diff(weeks: np.ndarray, vals: np.ndarray, mask: np.ndarray, reps: int = BOOT_REPS, seed: int = 31):
    """Week-block bootstrap of mean(vals[mask]) - mean(vals). None where the subset is too small."""
    if mask.sum() < 30 or len(vals) < 100:
        return None, None
    u, inv = np.unique(weeks, return_inverse=True)
    Sa, Na = np.bincount(inv, weights=vals), np.bincount(inv).astype(float)
    Ss, Ns = np.bincount(inv, weights=vals * mask), np.bincount(inv, weights=mask.astype(float))
    idx = np.random.default_rng(seed).integers(0, len(u), size=(reps, len(u)))
    na, ns = Na[idx].sum(1), Ns[idx].sum(1)
    ok = ns > 0
    d = Ss[idx].sum(1)[ok] / ns[ok] - Sa[idx].sum(1)[ok] / na[ok]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def same_day_excess(sig: np.ndarray, adj: np.ndarray, mask: np.ndarray):
    """Per-observation (adj - mean adj of the whole universe on that signal day) for the masked rows."""
    u, inv = np.unique(sig, return_inverse=True)
    day_mean = np.bincount(inv, weights=adj) / np.bincount(inv)
    return (adj - day_mean[inv])[mask]


def filter_stats(uni: dict, mask: np.ndarray, mid64) -> dict:
    """uni: arrays sig (datetime64[D]), adj (%), atr (%), dist (% below the 52w high, <= 0). mask: the filtered rows."""
    sig, adj = uni["sig"], uni["adj"]
    n, N = int(mask.sum()), len(adj)
    if n < 30:
        return {"n": n}
    wk = _week(sig)
    full_m, sel_m = float(adj.mean()), float(adj[mask].mean())
    lo, hi = boot_diff(wk, adj, mask)
    first = sig < mid64
    h_full = [float(adj[first].mean()), float(adj[~first].mean())]
    h_sel = [float(adj[mask & first].mean()) if (mask & first).sum() else None, float(adj[mask & ~first].mean()) if (mask & ~first).sum() else None]
    h_diff = [None if h_sel[k] is None else h_sel[k] - h_full[k] for k in (0, 1)]
    sd = same_day_excess(sig, adj, mask)
    sd_lo, sd_hi = _boot(wk[mask], sd)
    res = {"n": n, "share_of_universe_pct": _f(100.0 * n / N, 1), "mean_adj_pct": _f(sel_m), "full_universe_mean_adj_pct": _f(full_m),
           "raw_difference_pp": _f(sel_m - full_m), "raw_difference_ci95_week_block": [_f(lo), _f(hi)],
           "halves_mean_adj_pct": [_f(x) for x in h_sel], "halves_full_mean_adj_pct": [_f(x) for x in h_full], "halves_difference_pp": [_f(x) for x in h_diff],
           "halves_n": [int((mask & first).sum()), int((mask & ~first).sum())],
           "same_day_excess_pp": _f(sd.mean()), "same_day_excess_ci95": [_f(sd_lo), _f(sd_hi)],
           "net_of_costs_pct": _f(sel_m - COST_PCT), "atr_pct_mean": _f(np.nanmean(uni["atr"][mask])), "atr_pct_mean_full": _f(np.nanmean(uni["atr"])),
           "positive_both_halves": bool(all(x is not None and x > 0 for x in h_sel)),
           "difference_positive_both_halves": bool(all(x is not None and x > 0 for x in h_diff))}
    pop = {"sig": sig[mask], "adj": adj[mask], "atr": uni["atr"][mask]}
    m = VM.matched_excess(pop, uni, mid64, 5)
    if m is not None:
        x = m["excess_all"]
        mlo, mhi = _boot(_week(m["sig"]), x)
        h1, h2 = m["excess_half"][m["first"]], m["excess_half"][~m["first"]]
        res["atr_matched_excess_pp"] = _f(x.mean())
        res["atr_matched_ci95"] = [_f(mlo), _f(mhi)]
        res["atr_matched_halves"] = [_f(h1.mean()) if len(h1) else None, _f(h2.mean()) if len(h2) else None]
    res["gate_difference_gt_1pp"] = bool(res["raw_difference_pp"] is not None and res["raw_difference_pp"] > GATE_PP)
    res["gate_owner_strict"] = bool(res["gate_difference_gt_1pp"] and res["positive_both_halves"] and res["difference_positive_both_halves"])
    res["gate_owner_loose"] = bool(res["gate_difference_gt_1pp"] and res["positive_both_halves"])
    res["gate_survives_atr_matching"] = bool(res.get("atr_matched_excess_pp") is not None and res["atr_matched_excess_pp"] > GATE_PP
                                             and res.get("atr_matched_halves") and all(x is not None and x > 0 for x in res["atr_matched_halves"]))
    return res


def detector_vs_filter(det: dict, uni: dict, band: float, mid64) -> dict:
    """The detector signals inside the band against the universe inside the same band."""
    dm = det["dist"] >= -band
    um = uni["dist"] >= -band
    if dm.sum() < 30 or um.sum() < 100:
        return {"n_near": int(dm.sum())}
    a = {"sig": det["sig"][dm], "adj": det["adj"][dm], "atr": det["atr"][dm]}
    c = {"sig": uni["sig"][um], "adj": uni["adj"][um], "atr": uni["atr"][um]}
    first = a["sig"] < mid64
    cfirst = c["sig"] < mid64
    out = {"n_near": int(dm.sum()), "share_of_signals_near_pct": _f(100.0 * dm.mean(), 1), "mean_adj_pct": _f(a["adj"].mean()), "universe_near_mean_adj_pct": _f(c["adj"].mean()),
           "raw_excess_vs_near_universe_pp": _f(a["adj"].mean() - c["adj"].mean()),
           "halves_excess_pp": [_f(a["adj"][first].mean() - c["adj"][cfirst].mean()) if first.sum() and cfirst.sum() else None,
                                _f(a["adj"][~first].mean() - c["adj"][~cfirst].mean()) if (~first).sum() and (~cfirst).sum() else None]}
    lo, hi = _boot(_week(a["sig"]), a["adj"] - c["adj"].mean())
    out["raw_excess_ci95_week_block"] = [_f(lo), _f(hi)]
    m = VM.matched_excess(a, c, mid64, 5)
    if m is not None:
        out["atr_matched_excess_vs_near_universe_pp"] = _f(m["excess_all"].mean())
    return out


# ---------------------------------------------------------------------------------------------- collection
class Pop:
    def __init__(self):
        self.sig, self.adj, self.atr, self.dist, self.det, self.sym = {h: [] for h in H}, {h: [] for h in H}, {h: [] for h in H}, {h: [] for h in H}, [], []

    def extend(self, h, sig, adj, atr, dist):
        self.sig[h].extend(sig); self.adj[h].extend(adj); self.atr[h].extend(atr); self.dist[h].extend(dist)

    def arrays(self, h):
        return {"sig": np.array(self.sig[h], dtype="datetime64[D]"), "adj": np.array(self.adj[h], float), "atr": np.array(self.atr[h], float),
                "dist": np.array(self.dist[h], float)}


def _dist(b, idx):
    h52 = b.high52[idx]
    return np.where(h52 > 0, 100.0 * (b.c[idx] / np.where(h52 > 0, h52, np.nan) - 1.0), np.nan)


def collect(data, pn, w0, step: int = STEP):
    """Returns (universe Pop, {detector: Pop}, gated counts). Every symbol: sampled calendar days; every detector signal that passes the gates."""
    uni = Pop()
    dets: dict[str, Pop] = {}
    gated: dict[str, int] = {}
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30
    cal = pn.cal.values.astype("datetime64[D]")
    sample_days = cal[::step]
    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            continue
        mk = _market(data, b)
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        dates = b.date.astype("datetime64[D]")
        is_day = np.isin(dates, sample_days)
        cands, _ = detect_all(b, lo=lo)
        for h in H:
            i = np.arange(lo, b.n - h)                                           # signal bars; entry i+1, exit close of bar i+h
            i = i[is_day[i]]
            i = i[(b.o[i + 1] > 0) & ~np.isnan(mk[i]) & ~np.isnan(mk[i + h]) & (b.high52[i] > 0) & ~np.isnan(b.atr_pct[i])]
            if len(i):
                adj = (100.0 * (b.c[i + h] / b.o[i + 1] - 1.0)) - (100.0 * (mk[i + h] / mk[i] - 1.0))
                uni.extend(h, dates[i].tolist(), adj.tolist(), b.atr_pct[i].tolist(), _dist(b, i).tolist())
        per: dict[str, list] = {}
        for c in cands:
            g = BT.gate(b, c, data.surveillance.get(sym), surv_active)
            if g:
                gated[g] = gated.get(g, 0) + 1
                continue
            e = c.idx + 1
            if pn.row_of.get(b.date[c.idx]) is None or e >= b.n or _locked(b, e):
                continue
            per.setdefault(c.detector, []).append(c.idx)
        for det, idxs in per.items():
            p = dets.setdefault(det, Pop())
            for h in H:
                i = np.array([x for x in idxs if x + h < b.n and b.o[x + 1] > 0 and not np.isnan(mk[x]) and not np.isnan(mk[x + h]) and b.high52[x] > 0
                              and not np.isnan(b.atr_pct[x])], dtype=int)
                if len(i):
                    adj = (100.0 * (b.c[i + h] / b.o[i + 1] - 1.0)) - (100.0 * (mk[i + h] / mk[i] - 1.0))
                    p.extend(h, dates[i].tolist(), adj.tolist(), b.atr_pct[i].tolist(), _dist(b, i).tolist())
    return uni, dets, gated


def compute(data, out: dict, step: int = STEP) -> dict:
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    mid64 = np.datetime64(mid, "D")
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(pn.cal[-1].date())}
    uni_p, dets, gated = collect(data, pn, w0, step)
    out["gated"] = gated
    res: dict = {}
    for h in H:
        u = uni_p.arrays(h)
        if len(u["adj"]) < 1000:
            res[str(h)] = {"error": "too few universe observations", "n": int(len(u["adj"]))}
            continue
        per = {"universe_n": int(len(u["adj"])), "universe_mean_adj_pct": _f(u["adj"].mean()), "symbols_days_sampled_every": step}
        for band in BANDS:
            per[f"within_{band}pct"] = filter_stats(u, u["dist"] >= -band, mid64)
        per["beyond_20pct"] = filter_stats(u, u["dist"] < -20, mid64)
        res[str(h)] = per
    out["universe_filter"] = res
    # --- detector portfolio at the position horizon
    u = uni_p.arrays(POS_H)
    pooled = {k: np.concatenate([dets[d].arrays(POS_H)[k] for d in dets]) for k in ("sig", "adj", "atr", "dist")} if dets else None
    port: dict = {}
    if pooled is not None and len(pooled["adj"]) >= 100 and len(u["adj"]):
        first = pooled["sig"] < mid64
        ucm = float(u["adj"].mean())
        port["all_detectors_pooled"] = {"n": int(len(pooled["adj"])), "mean_adj_pct": _f(pooled["adj"].mean()), "excess_vs_full_universe_pp": _f(pooled["adj"].mean() - ucm),
                                        "halves_mean_adj_pct": [_f(pooled["adj"][first].mean()), _f(pooled["adj"][~first].mean())],
                                        "share_within_10pct_of_high": _f(100.0 * (pooled["dist"] >= -10).mean(), 1), "share_within_20pct_of_high": _f(100.0 * (pooled["dist"] >= -20).mean(), 1),
                                        "universe_share_within_10pct": _f(100.0 * (u["dist"] >= -10).mean(), 1), "universe_share_within_20pct": _f(100.0 * (u["dist"] >= -20).mean(), 1)}
        for band in (10, 20):
            port["all_detectors_pooled"][f"inside_{band}pct_vs_near_universe"] = detector_vs_filter(pooled, u, band, mid64)
    for d in sorted(dets):
        a = dets[d].arrays(POS_H)
        if len(a["adj"]) < 30:
            continue
        first = a["sig"] < mid64
        row = {"n": int(len(a["adj"])), "mean_adj_pct": _f(a["adj"].mean()), "halves_mean_adj_pct": [_f(a["adj"][first].mean()) if first.sum() else None, _f(a["adj"][~first].mean()) if (~first).sum() else None],
               "excess_vs_full_universe_pp": _f(a["adj"].mean() - float(u["adj"].mean())), "share_within_10pct_of_high": _f(100.0 * (a["dist"] >= -10).mean(), 1),
               "share_within_20pct_of_high": _f(100.0 * (a["dist"] >= -20).mean(), 1)}
        for band in (10, 20):
            row[f"inside_{band}pct_vs_near_universe"] = detector_vs_filter(a, u, band, mid64)
        port[d] = row
    out["detector_portfolio_h20"] = port
    # --- verdict
    bands = {b: (res.get(str(POS_H)) or {}).get(f"within_{b}pct") or {} for b in BANDS}
    out["decision"] = {"gate_rule": f"filtered set beats the full universe by more than {GATE_PP} pp at 20 sessions, filtered mean positive in both halves (strict also needs the difference positive in both halves)",
                       "by_band": {str(b): {"raw_difference_pp": s.get("raw_difference_pp"), "ci95_week_block": s.get("raw_difference_ci95_week_block"), "halves_mean": s.get("halves_mean_adj_pct"),
                                            "halves_difference": s.get("halves_difference_pp"), "same_day_excess_pp": s.get("same_day_excess_pp"),
                                            "atr_matched_excess_pp": s.get("atr_matched_excess_pp"), "gate_loose": s.get("gate_owner_loose"), "gate_strict": s.get("gate_owner_strict"),
                                            "survives_atr_matching": s.get("gate_survives_atr_matching")} for b, s in bands.items()},
                       "pass_strict_any_band": bool(any(s.get("gate_owner_strict") for s in bands.values())),
                       "pass_loose_any_band": bool(any(s.get("gate_owner_loose") for s in bands.values())),
                       "pass_after_atr_matching_any_band": bool(any(s.get("gate_survives_atr_matching") for s in bands.values()))}
    out["note"] = "Measurement only. Market-adjusted percent, no costs unless labelled. Today's constituents (survivorship). Windows overlap: intervals are week-block."
    return out


# ---------------------------------------------------------------------------------------------- run / render
def render(d: dict) -> str:
    L = ["# E2NH 52-week-high standalone filter (measurement only; nothing live reads this)", json.dumps(d.get("window"), default=str)]
    for h, per in (d.get("universe_filter") or {}).items():
        L.append(f"## universe, {h}-session market-adjusted return")
        L.append(json.dumps({k: v for k, v in per.items() if not k.startswith("within") and k != "beyond_20pct"}, default=str))
        for k, v in per.items():
            if k.startswith("within") or k == "beyond_20pct":
                L.append(f"h{h} {k} " + json.dumps(v, default=str))
    L.append("## decision")
    L.append(json.dumps(d.get("decision"), default=str))
    L.append("## detector portfolio, 20 sessions (inside_Npct_vs_near_universe = the detector's signals within N% of the high against the universe within N%)")
    for k, v in (d.get("detector_portfolio_h20") or {}).items():
        L.append(f"{k} " + json.dumps(v, default=str))
    L.append("gated " + json.dumps(d.get("gated"), default=str))
    if d.get("error"):
        L.append(f"error: {d['error']}")
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
        log.exception("52-week-high standalone study failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        conn.rollback()
        log.exception("could not store the 52-week-high study")
    try:
        log_markdown(render(out))
    finally:
        if own:
            conn.close()
    return out
