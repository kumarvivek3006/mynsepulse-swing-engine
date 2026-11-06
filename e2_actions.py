"""
e2_actions.py — the three actions of 6 Nov, measured on the 3-year backtest BEFORE anything live changes.

  ACTION 1  detector-specific scoring      e2_dscore: priors -> train-half fit -> held-out half
  ACTION 2  tightened pocket_pivot         e2_det_momentum.pocket_pivot(strict=True) vs the current definition
  ACTION 3  vcp port audit                 the old engine's ARMED trigger on Engine 2's unchanged vcp geometry

Nothing here reads, tunes on, or reports the live signals. Measurement only: no detector is removed, the live scan is not
touched. The job writes engine_settings.engine2_actions and logs it as E2ACT| lines (GET /jobs/engine2-actions).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import e2_backtest as BT
import e2_dscore as DS
import e2_portfolio as PF
import e2_scorer as SC
import e2_sector as SE
from e2_det_bases import vcp, vcp_armed
from e2_det_common import WARMUP
from e2_det_momentum import _pp_structure, pocket_pivot
from e2_detectors import DETECTORS, dedupe
from e2_features import compute_features
from e2_manage import Trade, simulate

log = logging.getLogger("engine2.actions")
KEY = "engine2_actions"
LOG_PREFIX = "E2ACT|"
WALK_SLICES = 6


def _f(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# one candidate -> one simulated trade (the same stages per_symbol applies)
# ----------------------------------------------------------------------
class Ctx:
    def __init__(self, data, pn):
        self.data, self.pn = data, pn
        self.surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30


def trade_for(ctx: Ctx, b, c, sym, sec, gated: Counter) -> Trade | None:
    g = BT.gate(b, c, ctx.data.surveillance.get(sym), ctx.surv_active)
    if g:
        gated[g] += 1
        return None
    pn = ctx.pn
    row = pn.row_of.get(b.date[c.idx])
    if row is None:
        gated["date_not_in_calendar"] += 1
        return None
    c.sector = sec
    rs = pn.rs_pct.iat[row, pn.rs_pct.columns.get_loc(sym)] if sym in pn.rs_pct.columns else np.nan
    rk = np.nan
    if sec is not None and sec in pn.sector_rank.columns:
        rk = pn.sector_rank.iat[row, pn.sector_rank.columns.get_loc(sec)]
    n_sec = int(pn.sector_rank.iloc[row].notna().sum())
    state = pn.regime["state"].iat[row]
    frac = SC.components(b, c, rs, SE.sector_points(rk, n_sec), state)
    c.score_parts, c.score = frac, SC.total(frac)
    tr = simulate(b, c, ctx.data.events.get(sym))
    tr.frac, tr.regime_state, tr.score = frac, state, c.score
    return tr


def _without_stop_entry(c):
    import copy
    d = copy.copy(c)
    d.meta = {k: v for k, v in c.meta.items() if k not in ("entry_stop", "entry_expiry")}
    d.detector = c.detector + "_nextopen"
    return d


# ----------------------------------------------------------------------
# variants pass: pocket_pivot (old / strict), vcp (breakout / armed / armed + 60% prior advance / armed at next open)
# ----------------------------------------------------------------------
def variants_pass(ctx: Ctx, w0, progress=None) -> dict:
    data, pn = ctx.data, ctx.pn
    names = ("pp_current", "pp_strict", "vcp_breakout", "vcp_armed", "vcp_armed_p60", "vcp_armed_nextopen")
    out = {k: {"raw": 0, "trades": [], "gated": Counter()} for k in names}
    funnel = Counter()
    n = len(data.frames)
    for k, (sym, df) in enumerate(sorted(data.frames.items())):
        b = compute_features(sym, df)
        if b is None:
            continue
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        sec = pn.sector_of.get(sym)
        sets: dict[str, list] = {}
        try:
            sets["pp_current"] = dedupe(pocket_pivot(b, {}, lo, b.n, strict=False))
            sets["pp_strict"] = dedupe(pocket_pivot(b, {}, lo, b.n, strict=True))
        except Exception:                                                             # noqa: BLE001
            log.exception("pocket_pivot variants failed for %s", sym)
        for c in sets.get("pp_current", []):
            i = c.idx
            has_down = not c.meta.get("no_down_day", False)
            rv = bool(b.relvol[i] >= 1.5)
            up, base = _pp_structure(b, i)
            funnel["current"] += 1
            funnel["has_down_day"] += has_down
            funnel["relvol_ge_1_5"] += rv
            funnel["structure_ok"] += (up or base)
            funnel["uptrend_intact"] += up
            funnel["valid_base_only"] += (base and not up)
            funnel["has_down_and_relvol"] += (has_down and rv)
            funnel["all_three"] += (has_down and rv and (up or base))
        try:
            sets["vcp_breakout"] = dedupe(vcp(b, {}, lo, b.n))
            sets["vcp_armed"] = vcp_armed(b, {}, lo, b.n)
            sets["vcp_armed_p60"] = vcp_armed(b, {}, lo, b.n, prior_uptrend=True)
            sets["vcp_armed_nextopen"] = [_without_stop_entry(c) for c in sets["vcp_armed"]]
        except Exception:                                                             # noqa: BLE001
            log.exception("vcp variants failed for %s", sym)
        for name, cands in sets.items():
            out[name]["raw"] += len(cands)
            for c in cands:
                tr = trade_for(ctx, b, c, sym, sec, out[name]["gated"])
                if tr is not None:
                    out[name]["trades"].append(tr)
        if progress and k % 50 == 0:
            progress(f"variants {k}/{n}")
    out["_funnel"] = dict(funnel)
    return out


def _slices(trs: list[Trade], w0, w1) -> list[dict]:
    live = [t for t in trs if not t.skipped]
    edges = pd.date_range(w0, w1, periods=WALK_SLICES + 1)
    rows = []
    for a, z in zip(edges[:-1], edges[1:]):
        g = [t.r for t in live if a <= pd.Timestamp(t.signal_date) < z or (z == edges[-1] and pd.Timestamp(t.signal_date) == z)]
        rows.append({"from": str(a.date()), "n": len(g), "avg_r": _f(np.mean(g)) if g else None})
    return rows


def variant_block(name: str, v: dict, w0, w1, split64) -> dict:
    trs = v["trades"]
    live = [t for t in trs if not t.skipped]
    skip = Counter(t.skipped for t in trs if t.skipped)
    sl = _slices(trs, w0, w1)
    pos = sum(1 for s in sl if s["n"] >= 5 and s["avg_r"] is not None and s["avg_r"] > 0)
    return {"candidates": v["raw"], "after_gates": len(trs), "gated": dict(v["gated"]), "skipped": dict(skip),
            "filled_trades": len(live), "stats": BT.stats_r(trs),
            "first_half": BT.stats_r([t for t in trs if np.datetime64(t.signal_date, "D") < split64]),
            "second_half": BT.stats_r([t for t in trs if np.datetime64(t.signal_date, "D") >= split64]),
            "walk_forward_slices": sl, "slices_positive_of_with_5plus": f"{pos}/{sum(1 for s in sl if s['n'] >= 5)}"}


# ----------------------------------------------------------------------
# old engine reference (the validated "armed"), straight from its own backtest table
# ----------------------------------------------------------------------
def old_armed_reference(conn) -> dict:
    try:
        with conn.cursor() as cur:
            cur.execute("select run_id from backtest_trades where setup_type like 'armed%%' and r_realised is not null "
                        "group by run_id order by run_id desc limit 1")
            r = cur.fetchone()
            if not r:
                return {"error": "no old-engine run with realised armed trades"}
            cur.execute("select setup_type, pattern, signal_date, r_realised from backtest_trades "
                        "where run_id = %s and setup_type like 'armed%%' and r_realised is not null and exit_reason <> 'never_triggered'", (r[0],))
            df = pd.DataFrame(cur.fetchall(), columns=["setup_type", "pattern", "signal_date", "r"])
        df["r"] = pd.to_numeric(df["r"], errors="coerce")
        df["signal_date"] = pd.to_datetime(df["signal_date"])

        def agg(g):
            return {"n": int(len(g)), "avg_r": _f(g["r"].mean()), "win_rate_pct": _f(100 * (g["r"] > 0).mean(), 1)}
        edges = pd.date_range(df["signal_date"].min(), df["signal_date"].max(), periods=WALK_SLICES + 1)
        sl = []
        for a, z in zip(edges[:-1], edges[1:]):
            g = df[(df["signal_date"] >= a) & (df["signal_date"] <= z if z == edges[-1] else df["signal_date"] < z)]
            sl.append({"from": str(a.date()), **agg(g)} if len(g) else {"from": str(a.date()), "n": 0, "avg_r": None})
        return {"run_id": int(r[0]), "all_armed": agg(df), "by_pattern": {k: agg(g) for k, g in df.groupby("pattern")},
                "by_setup_type": {k: agg(g) for k, g in df.groupby("setup_type")}, "walk_forward_slices": sl,
                "window": [str(df["signal_date"].min().date()), str(df["signal_date"].max().date())]}
    except Exception as e:                                                             # noqa: BLE001
        log.exception("old armed reference failed")
        try:
            conn.rollback()
        except Exception:                                                              # noqa: BLE001
            pass
        return {"error": f"{type(e).__name__}: {e}"[:300]}


# ----------------------------------------------------------------------
# ACTION 1: detector-specific scoring
# ----------------------------------------------------------------------
def scoring_study(ctx: Ctx, trades: list[Trade], w0, mid, w1) -> dict:
    pn = ctx.pn
    live = [t for t in trades if not t.skipped and t.frac is not None]
    split64 = np.datetime64(mid, "D")
    dt = lambda t: np.datetime64(t.signal_date, "D")           # noqa: E731
    train = [t for t in live if dt(t) < split64]
    test = [t for t in live if dt(t) >= split64]
    names = list(DETECTORS)

    model = DS.fit_model([(t.detector, t.frac, t.r, dt(t)) for t in train], names)
    full_model = DS.fit_model([(t.detector, t.frac, t.r, dt(t)) for t in live], names)

    def arr(ts, fn):
        return np.array([fn(t) for t in ts], float)

    schemes = {
        "S0_spec_global_before": lambda t: SC.total(t.frac),
        "S1_priors_only_raw": lambda t: sum(DS.prior_weights(t.detector)[c] * t.frac[c] for c in SC.COMPONENTS),
        "S2_fitted_weights_raw": lambda t: model.score(t.detector, t.frac, mapped=False),
        "S3_fitted_weights_plus_expected_R_mapping": lambda t: model.score(t.detector, t.frac, mapped=True),
    }
    ev = {}
    for sname, fn in schemes.items():
        row = {}
        for label, ts in (("train_half", train), ("test_half_held_out", test), ("full_window", live)):
            y, s = arr(ts, lambda t: t.r), arr(ts, fn)
            row[label] = {"n": len(ts), "spearman_pooled": DS.spearman_pooled(s, y),
                          "spearman_within_detector": DS.spearman_within([t.detector for t in ts], s, y)["weighted_mean"]}
        ev[sname] = row
    detail_test = {sname: DS.spearman_within([t.detector for t in test], arr(test, fn), arr(test, lambda t: t.r))["by_detector"]
                   for sname, fn in schemes.items() if sname in ("S0_spec_global_before", "S3_fitted_weights_plus_expected_R_mapping")}

    # median score per detector, before / after
    med = {}
    for d in names:
        ts = [t for t in live if t.detector == d]
        tt = [t for t in test if t.detector == d]
        if not ts:
            med[d] = {"n": 0}
            continue
        med[d] = {"n": len(ts), "before_spec": _f(np.median([SC.total(t.frac) for t in ts]), 1),
                  "after_raw_weights": _f(np.median([model.score(t.detector, t.frac, mapped=False) for t in ts]), 1),
                  "after_final_score": _f(np.median([model.score(t.detector, t.frac, mapped=True) for t in ts]), 1),
                  "test_half_n": len(tt),
                  "test_half_before": _f(np.median([SC.total(t.frac) for t in tt]), 1) if tt else None,
                  "test_half_after": _f(np.median([model.score(t.detector, t.frac, mapped=True) for t in tt]), 1) if tt else None,
                  "mean_r": _f(np.mean([t.r for t in ts]))}

    # portfolio: which detectors the book takes, before and after (the book is the same; only the ranking changes)
    cal_idx = pn.cal

    def book(score_fn, start, end):
        for t in trades:
            t.score = score_fn(t) if (not t.skipped and t.frac is not None) else t.score
        res = PF.run(trades, pn.regime, pn.returns, cal_idx, PF.Params(), start=start, end=end)
        by = Counter(x["trade"].detector for x in res.taken)
        closed = Counter(x["trade"].detector for x in res.taken if "pnl" in x)
        rr = defaultdict(list)
        for x in res.taken:
            if "pnl" in x:
                rr[x["trade"].detector].append(x["trade"].r)
        return {"metrics": res.metrics, "rejected": dict(res.rejected), "taken_by_detector": dict(by.most_common()),
                "avg_r_by_detector": {k: _f(np.mean(v)) for k, v in rr.items()},
                "episodic_pivot_taken": by.get("episodic_pivot", 0), "gap_and_go_taken": by.get("gap_and_go", 0),
                "pocket_pivot_taken": by.get("pocket_pivot", 0)}

    port = {}
    for label, fn in (("before_spec", schemes["S0_spec_global_before"]), ("after_detector_specific", schemes["S3_fitted_weights_plus_expected_R_mapping"]),
                      ("after_weights_only_raw", schemes["S2_fitted_weights_raw"])):
        port[label] = {"full_window": book(fn, w0, w1), "test_half": book(fn, mid, w1)}
    for t in trades:                                      # leave the trades as the study found them
        if not t.skipped and t.frac is not None:
            t.score = SC.total(t.frac)

    sample = {d: sum(1 for t in train if t.detector == d) for d in names}
    return {"protocol": "priors (instruction) -> train-half ridge fit -> lambda by expanding time-block CV inside train -> held-out half scored once",
            "split": str(mid.date()), "n_train": len(train), "n_test": len(test), "train_trades_by_detector": sample,
            "evaluation": ev, "within_detector_test_half": detail_test, "median_score_by_detector": med, "portfolio": port,
            "weights": {d: {k: v for k, v in e.items() if k in ("weights", "prior", "lam", "n_train", "mu", "raw_mean")} for d, e in model.det.items()},
            "cv": model.meta.get("cv"), "slope_R_per_raw_point": _f(model.slope, 5), "global_mean_r_train": _f(model.global_mean),
            "model_train_only": model.to_json(), "model_full_window_for_deployment": full_model.to_json(),
            "note": "Headline = S3, declared before the run. Pooled Spearman mixes between-detector and within-detector ordering; "
                    "the within-detector figure isolates what the weights themselves add. Full-window numbers include the train half the model was fitted on."}


# ----------------------------------------------------------------------
# job
# ----------------------------------------------------------------------
def compute(data, conn, out: dict) -> dict:
    """Everything after the load, so the whole study can be exercised on synthetic data (conn may be None)."""
    prog = lambda m: log.info("actions: %s", m)           # noqa: E731
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    w1 = pn.cal[-1]
    ctx = Ctx(data, pn)
    weeks = len(pn.cal[pn.cal >= w0]) / 5.0
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(w1.date()), "weeks": round(weeks, 1)}
    split64 = np.datetime64(mid, "D")

    prog("variants pass")
    vp = variants_pass(ctx, w0, prog)
    blocks = {k: variant_block(k, vp[k], w0, w1, split64) for k in vp if not k.startswith("_")}

    pp_cur, pp_str = blocks["pp_current"], blocks["pp_strict"]
    out["action2_pocket_pivot"] = {
        "current": {**pp_cur, "per_week_candidates": _f(pp_cur["candidates"] / weeks, 1), "per_week_after_gates": _f(pp_cur["after_gates"] / weeks, 1)},
        "strict": {**pp_str, "per_week_candidates": _f(pp_str["candidates"] / weeks, 1), "per_week_after_gates": _f(pp_str["after_gates"] / weeks, 1)},
        "reduction_pct_candidates": _f(100 * (1 - pp_str["candidates"] / pp_cur["candidates"]), 1) if pp_cur["candidates"] else None,
        "clause_funnel_on_current_hits": vp["_funnel"], "target_per_week": "5-15",
        "definitions_declared_before_run": "uptrend = close > rising 50d SMA and 50d > 150d; base = close within 10% of the 52-week high and the prior 40 sessions' range <= 20% deep; up day >= 1.5x relvol (50-day average volume)"}
    out["action3_vcp"] = {
        "variants": {k: blocks[k] for k in ("vcp_breakout", "vcp_armed", "vcp_armed_p60", "vcp_armed_nextopen")},
        "old_engine_reference": old_armed_reference(conn) if conn is not None else {"error": "no database connection"},
        "reading_guide": {"vcp_breakout": "Engine 2 as live", "vcp_armed": "same geometry, old armed trigger and buy-stop entry (the port fix)",
                          "vcp_armed_p60": "plus the old base finder's >=60% advance over 120 sessions into the base",
                          "vcp_armed_nextopen": "armed selection but entered at the next open: isolates the buy-stop mechanics"}}
    del vp

    prog("per-symbol scan for the scoring study")
    ps = BT.per_symbol(data, pn, w0, prog)
    out["action1_scoring"] = scoring_study(ctx, ps["trades"], w0, mid, w1)
    return out


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running",
           "note": "Measurement only. No detector is removed; the live scan is unchanged by this run."}
    try:
        data = BT.load_from_db(conn)
        compute(data, conn, out)
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("actions study failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        _put(conn, out)
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the actions study")
    try:
        log_markdown(render_markdown(out))
        log_models(out)
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


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def _st(s: dict) -> str:
    if not s or not s.get("n"):
        return "n=0"
    return f"n={s['n']} win={s['win_rate_pct']}% avgR={s['avg_r']} PF={s['profit_factor_r']}"


def render_markdown(d: dict) -> str:
    L = [f"# Engine 2 actions study ({d.get('generated_at')}, state {d.get('state')}, {d.get('runtime_sec')}s)", ""]
    if d.get("error"):
        L += [f"ERROR: {d['error']}", ""]
    w = d.get("window") or {}
    L += [f"Window {w.get('start')} -> {w.get('end')} ({w.get('weeks')} weeks), train/test split {w.get('split')}", ""]
    a2 = d.get("action2_pocket_pivot")
    if a2:
        L += ["## ACTION 2 pocket_pivot", ""]
        for k in ("current", "strict"):
            x = a2[k]
            L += [f"- {k}: candidates {x['candidates']} ({x['per_week_candidates']}/wk), after gates {x['after_gates']} ({x['per_week_after_gates']}/wk), "
                  f"filled {x['filled_trades']} | {_st(x['stats'])} | 1st half {_st(x['first_half'])} | 2nd half {_st(x['second_half'])}"]
        L += [f"- reduction in candidate count: {a2['reduction_pct_candidates']}%", f"- clause funnel on current hits: {a2['clause_funnel_on_current_hits']}",
              f"- declared: {a2['definitions_declared_before_run']}", ""]
    a3 = d.get("action3_vcp")
    if a3:
        L += ["## ACTION 3 vcp", ""]
        for k, x in a3["variants"].items():
            L += [f"- {k}: candidates {x['candidates']}, after gates {x['after_gates']}, filled {x['filled_trades']}, skipped {x['skipped']} | {_st(x['stats'])} | "
                  f"1st {_st(x['first_half'])} | 2nd {_st(x['second_half'])} | slices>0 {x['slices_positive_of_with_5plus']}"]
            L += [f"    slices: {[(s['from'], s['n'], s['avg_r']) for s in x['walk_forward_slices']]}"]
        L += [f"- old engine armed reference: {json.dumps(a3['old_engine_reference'], default=str)}", ""]
    a1 = d.get("action1_scoring")
    if a1:
        L += ["## ACTION 1 scoring", "", f"split {a1['split']}, n_train {a1['n_train']}, n_test {a1['n_test']}, slope {a1['slope_R_per_raw_point']} R/pt", ""]
        for s, row in a1["evaluation"].items():
            L += [f"- {s}: " + " | ".join(f"{lab}: pooled {v['spearman_pooled']} within {v['spearman_within_detector']} (n={v['n']})" for lab, v in row.items())]
        L += ["", "median score by detector (n, before spec, after raw weights, after final, mean R):"]
        for k, v in a1["median_score_by_detector"].items():
            L += [f"- {k}: " + json.dumps(v)]
        L += ["", "portfolio:"]
        for lab, p in a1["portfolio"].items():
            for per in ("full_window", "test_half"):
                x = p[per]
                L += [f"- {lab} {per}: taken {x['metrics'].get('trades_taken')} ep={x['episodic_pivot_taken']} gap={x['gap_and_go_taken']} pp={x['pocket_pivot_taken']} "
                      f"ret={x['metrics'].get('total_return_pct')}% dd={x['metrics'].get('max_drawdown_pct')}% avgR={x['metrics'].get('avg_r')} by_det={x['taken_by_detector']}"]
        L += ["", "weights (train fit): detector -> lam, n_train, weights"]
        for k, v in a1["weights"].items():
            L += [f"- {k}: lam={v['lam']} n={v['n_train']} mu={_f(v['mu'])} w={v['weights']}"]
        L += ["", f"cv: {json.dumps(a1['cv'])}", "", f"{a1['note']}"]
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        log.info("%s%s", LOG_PREFIX, line[:3500])
        n += 1
    return n


def log_models(d: dict) -> None:
    a1 = d.get("action1_scoring") or {}
    for label in ("model_train_only", "model_full_window_for_deployment"):
        blob = json.dumps(a1.get(label), sort_keys=True, separators=(",", ":"))
        parts = [blob[i:i + 1500] for i in range(0, len(blob), 1500)]
        for k, p in enumerate(parts):
            log.info("%sMODEL|%s|%d/%d|%s", LOG_PREFIX, label, k + 1, len(parts), p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = run(trigger=sys.argv[1] if len(sys.argv) > 1 else "cli")
    print(json.dumps({"state": res.get("state"), "runtime_sec": res.get("runtime_sec")}))
