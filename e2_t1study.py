"""
e2_t1study.py — TIER 1 BACKTEST (owner's order of 26 Nov 00:38 IST). Measurement only: nothing live reads this, nothing is published or gated.

Question: do the four replacement components (e2_tier1: volume tiers, sector tiers, catalyst + earnings date, delivery %) rank the 3-year backtest's trades
better than the current components, when the weights are RE-FITTED on the train half only and scored on the untouched test half?

Same trades as the 6 Nov scoring study (BT.per_symbol, round-1 pocket_pivot universe, the same train/test split, target = realised R under the position engine's
exits). Weights follow the pooled rule already in force (e2_actions S6): keep a component only if its TRAIN Spearman with R is >= 0.05, weight proportional to that
Spearman. So a component that does not rank R on the train half gets weight zero, whatever its tiers look like.

Reported (E2T1| lines, and engine_settings key e2_t1study):
  coverage          how many trades have a delivery %, a results date inside the window, a sector rank (a component that is neutral for most trades cannot help)
  components        train Spearman, old vs new, per component (and delivery at lag 0 and lag 1)
  weights           the old pooled weights and the new ones, re-derived on this sample
  composite         test-half Spearman: spec composite, old pooled, new pooled, with a trade-bootstrap 95% interval for the new one and for new - old
  per detector      test-half Spearman and the mean R of the top-quintile picks, old vs new score
  forward returns   top-quintile picks by old vs new score: mean 5/10/20-session return from the next open (not market-adjusted, no costs) per detector
  day-1 fade        breakout family: mean next-open -> same-day-close return of the top-quintile picks, old vs new score, and the share of picks that closed below the open
Decision gate (owner, ~3 Dec): new composite test-half Spearman > 0.25; at least one breakout detector's day-1 fade below 0.2%; oversold_dip generating shadow rows.
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

import e2_backtest as BT
import e2_scorer as SC
import e2_tier1 as T1
from e2_features import compute_features

log = logging.getLogger("engine2.t1study")
KEY = "e2_t1study"
LOG_PREFIX = "E2T1|"
HORIZONS = (5, 10, 20)
RHO_MIN = 0.05
BOOT_REPS = 400


def _f(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 10:
        return float("nan")
    return SC._spearman(a[m], b[m])


def load_delivery(conn, symbols, start) -> dict:
    """symbol -> (dates datetime64[D], delivery_pct float); empty dict when the table is missing or empty."""
    out: dict = {}
    try:
        with conn.cursor() as cur:
            cur.execute("select symbol, trade_date, delivery_pct from delivery_daily where symbol = any(%s) and trade_date >= %s "
                        "and delivery_pct is not null order by symbol, trade_date", (list(symbols), start))
            rows = cur.fetchall()
    except Exception:                                                  # noqa: BLE001
        conn.rollback()
        return out
    tmp = defaultdict(lambda: ([], []))
    for s, d, p in rows:
        tmp[s][0].append(np.datetime64(d, "D"))
        tmp[s][1].append(float(p))
    for s, (d, p) in tmp.items():
        out[s] = (np.array(d, dtype="datetime64[D]"), np.array(p, float))
    return out


def delivery_at(delivery: dict, sym: str, d64) -> tuple[float | None, float | None]:
    """(delivery % on the signal date, delivery % on the session before it). None where there is no row."""
    e = delivery.get(sym)
    if e is None:
        return None, None
    dates, pct = e
    k = int(np.searchsorted(dates, d64, side="left"))
    lag0 = float(pct[k]) if k < len(dates) and dates[k] == d64 else None
    lag1 = float(pct[k - 1]) if k >= 1 else None
    return lag0, lag1


def enrich(data, pn, delivery: dict, trades, prog=None) -> list[dict]:
    """One record per live trade: old fractions, new fractions, delivery lag-1 fraction, day-1 return and forward returns."""
    by_sym: dict[str, list] = defaultdict(list)
    for t in trades:
        if not t.skipped and t.frac is not None:
            by_sym[t.symbol].append(t)
    recs: list[dict] = []
    for k, (sym, ts) in enumerate(sorted(by_sym.items())):
        df = data.frames.get(sym)
        b = compute_features(sym, df) if df is not None else None
        if b is None:
            continue
        ev = data.events.get(sym)
        for t in ts:
            d64 = np.datetime64(t.signal_date, "D")
            row = pn.row_of.get(d64)
            if row is None:
                continue
            sec = t.sector
            rk, n_sec = np.nan, int(pn.sector_rank.iloc[row].notna().sum())
            if sec is not None and sec in pn.sector_rank.columns:
                rk = pn.sector_rank.iat[row, pn.sector_rank.columns.get_loc(sec)]
            cand = SimpleNamespace(detector=t.detector, meta=t.meta, idx=t.signal_idx)
            d0, d1p = delivery_at(delivery, sym, d64)
            fnew = T1.t1_components(t.frac, b, cand, rk, n_sec, d0, ev)
            e = t.signal_idx + 1
            rec = {"det": t.detector, "family": t.family, "date": d64, "r": float(t.r), "old": {c: float(t.frac[c]) for c in SC.COMPONENTS},
                   "new": {c: float(fnew[c]) for c in T1.NEW_COMPONENTS}, "deliv_lag1": T1.delivery_fraction(d1p),
                   "deliv_missing": bool(fnew["_delivery_missing"]), "earn": fnew["_catalyst_source"] == "earnings",
                   "sector_known": not (rk is None or np.isnan(rk)), "d1": np.nan, **{f"f{h}": np.nan for h in HORIZONS}}
            if e < b.n and b.o[e] > 0:
                rec["d1"] = 100.0 * (b.c[e] / b.o[e] - 1.0)
                for h in HORIZONS:
                    j = e + h - 1
                    if j < b.n:
                        rec[f"f{h}"] = 100.0 * (b.c[j] / b.o[e] - 1.0)
            recs.append(rec)
        if prog and k % 100 == 0:
            prog(f"enrich {k}/{len(by_sym)}")
    return recs


def pooled_weights(rows: list[dict], comps: list[str], key: str) -> tuple[dict, dict]:
    """Train-half Spearman per component and the S6 weights (kept if rho >= 0.05, proportional to rho; all zero if none qualifies)."""
    y = np.array([r["r"] for r in rows], float)
    rho = {c: spearman([r[key][c] for r in rows], y) for c in comps}
    pos = {c: v for c, v in rho.items() if v == v and v >= RHO_MIN}
    w = {c: (100.0 * pos[c] / sum(pos.values()) if c in pos else 0.0) for c in comps} if pos else {c: 0.0 for c in comps}
    return rho, w


def score_of(r: dict, key: str, w: dict) -> float:
    return float(sum(w[c] * r[key][c] for c in w))


def _boot_ci(y, s_old, s_new, reps=BOOT_REPS, seed=11):
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 50:
        return None
    new_l, diff_l = [], []
    for _ in range(reps):
        ix = rng.integers(0, n, n)
        a, b_ = spearman(s_new[ix], y[ix]), spearman(s_old[ix], y[ix])
        new_l.append(a)
        diff_l.append(a - b_)
    return {"new_ci95": [_f(np.nanpercentile(new_l, 2.5)), _f(np.nanpercentile(new_l, 97.5))],
            "diff_ci95": [_f(np.nanpercentile(diff_l, 2.5)), _f(np.nanpercentile(diff_l, 97.5))]}


def _top(rows, scores, frac=0.2, min_n=25):
    n = len(rows)
    if n < min_n:
        return []
    k = max(1, int(np.ceil(frac * n)))
    order = np.argsort(-np.asarray(scores, float), kind="stable")[:k]
    return [rows[i] for i in order]


def _mean(rows, key):
    v = np.array([r[key] for r in rows], float)
    v = v[~np.isnan(v)]
    return _f(v.mean()) if len(v) else None


def study(recs: list[dict], mid) -> dict:
    split = np.datetime64(mid, "D")
    train = [r for r in recs if r["date"] < split]
    test = [r for r in recs if r["date"] >= split]
    out: dict = {"n_trades": len(recs), "n_train": len(train), "n_test": len(test), "split": str(split)}
    out["coverage"] = {
        "delivery_present_pct": _f(100 * np.mean([not r["deliv_missing"] for r in recs])) if recs else None,
        "delivery_present_pct_train": _f(100 * np.mean([not r["deliv_missing"] for r in train])) if train else None,
        "delivery_present_pct_test": _f(100 * np.mean([not r["deliv_missing"] for r in test])) if test else None,
        "earnings_within_5d_pct": _f(100 * np.mean([r["earn"] for r in recs])) if recs else None,
        "sector_known_pct": _f(100 * np.mean([r["sector_known"] for r in recs])) if recs else None,
        "note": "a component that is neutral (0.5) for most trades cannot rank them; delivery_present_pct is the share of trades with a delivery % on the signal date"}
    if len(train) < 200 or len(test) < 100:
        out["error"] = "too few trades for a train/test comparison"
        return out

    old_c, new_c = SC.COMPONENTS, T1.NEW_COMPONENTS
    rho_old, w_old = pooled_weights(train, old_c, "old")
    rho_new, w_new = pooled_weights(train, new_c, "new")
    ytr = np.array([r["r"] for r in train], float)
    dl_rows = [r for r in train if not r["deliv_missing"]]
    out["train_spearman"] = {"old": {c: _f(v) for c, v in rho_old.items()}, "new": {c: _f(v) for c, v in rho_new.items()},
                             "delivery_lag0_on_trades_with_data": _f(spearman([r["new"]["delivery"] for r in dl_rows], [r["r"] for r in dl_rows])),
                             "delivery_lag0_n": len(dl_rows),
                             "delivery_lag1": _f(spearman([r["deliv_lag1"] for r in train], ytr))}
    out["weights"] = {"old_pooled": {c: _f(v, 1) for c, v in w_old.items()}, "new_pooled": {c: _f(v, 1) for c, v in w_new.items()},
                      "rule": f"keep a component if train Spearman with R >= {RHO_MIN}; weight proportional to it; others 0"}

    yte = np.array([r["r"] for r in test], float)
    s_spec = np.array([SC.total(r["old"]) for r in test])
    s_old = np.array([score_of(r, "old", w_old) if sum(w_old.values()) > 0 else SC.total(r["old"]) for r in test])
    s_new = np.array([score_of(r, "new", w_new) if sum(w_new.values()) > 0 else SC.total(r["old"]) for r in test])
    out["composite_test"] = {"spec": _f(spearman(s_spec, yte)), "old_pooled": _f(spearman(s_old, yte)), "new_pooled": _f(spearman(s_new, yte)),
                             **(_boot_ci(yte, s_old, s_new) or {}),
                             "gate_new_gt_0_25": bool(spearman(s_new, yte) > 0.25) if not np.isnan(spearman(s_new, yte)) else False,
                             "note": "weights fitted on the train half only; the test half was not used to choose anything"}
    ytr_s = {"spec": spearman([SC.total(r["old"]) for r in train], ytr),
             "old_pooled": spearman([score_of(r, "old", w_old) for r in train], ytr) if sum(w_old.values()) > 0 else None,
             "new_pooled": spearman([score_of(r, "new", w_new) for r in train], ytr) if sum(w_new.values()) > 0 else None}
    out["composite_train"] = {k: _f(v) for k, v in ytr_s.items()}

    per = {}
    for d in sorted({r["det"] for r in test}):
        ts = [r for r in test if r["det"] == d]
        if len(ts) < 30:
            continue
        so = [score_of(r, "old", w_old) if sum(w_old.values()) > 0 else SC.total(r["old"]) for r in ts]
        sn = [score_of(r, "new", w_new) if sum(w_new.values()) > 0 else SC.total(r["old"]) for r in ts]
        yo = np.array([r["r"] for r in ts], float)
        to, tn = _top(ts, so), _top(ts, sn)
        row = {"n_test": len(ts), "family": ts[0]["family"], "rho_old": _f(spearman(so, yo)), "rho_new": _f(spearman(sn, yo)),
               "mean_R_all": _f(yo.mean()), "top20_n": len(to),
               "top20_R_old": _mean(to, "r"), "top20_R_new": _mean(tn, "r")}
        for h in HORIZONS:
            row[f"all_f{h}"] = _mean(ts, f"f{h}")
            row[f"top20_f{h}_old"] = _mean(to, f"f{h}")
            row[f"top20_f{h}_new"] = _mean(tn, f"f{h}")
        row["d1_all"], row["d1_top20_old"], row["d1_top20_new"] = _mean(ts, "d1"), _mean(to, "d1"), _mean(tn, "d1")
        per[d] = row
    out["per_detector"] = per

    bo = [r for r in test if r["family"] == "breakout"]
    if len(bo) >= 50:
        so = [score_of(r, "old", w_old) if sum(w_old.values()) > 0 else SC.total(r["old"]) for r in bo]
        sn = [score_of(r, "new", w_new) if sum(w_new.values()) > 0 else SC.total(r["old"]) for r in bo]
        to, tn = _top(bo, so), _top(bo, sn)
        neg = lambda rs: _f(100 * np.mean([r["d1"] < 0 for r in rs if not np.isnan(r["d1"])])) if rs else None      # noqa: E731
        out["day1_breakout_pooled"] = {"n": len(bo), "all_mean_d1_pct": _mean(bo, "d1"), "top20_old_mean_d1_pct": _mean(to, "d1"),
                                       "top20_new_mean_d1_pct": _mean(tn, "d1"), "top20_old_share_closed_below_open": neg(to),
                                       "top20_new_share_closed_below_open": neg(tn),
                                       "definition": "next-open entry to that session's close, % (no costs); negative = faded. Gate: a breakout detector whose top-quintile mean is above -0.2 under the new score"}
    return out


def render(d: dict) -> str:
    L = ["# E2T1 tier 1 replacement-component backtest (measurement only; nothing live reads this)", json.dumps(d.get("window"), default=str),
         f"trades {d.get('n_trades')} (train {d.get('n_train')}, test {d.get('n_test')}), split {d.get('split')}"]
    for k in ("coverage", "train_spearman", "weights", "composite_train", "composite_test", "day1_breakout_pooled"):
        L.append(f"## {k}")
        L.append(json.dumps(d.get(k), default=str))
    L.append("## per detector (test half; top20 = top-quintile picks by that score)")
    for det, row in (d.get("per_detector") or {}).items():
        L.append(f"{det} " + json.dumps(row, default=str))
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


def compute(data, delivery: dict, out: dict, prog=None) -> dict:
    import e2_det_momentum as DM
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(pn.cal[-1].date())}
    keep = DM.PP_STRICT
    DM.PP_STRICT = False                      # the round-1 universe, as in the 6 Nov scoring study
    try:
        ps = BT.per_symbol(data, pn, w0, prog)
    finally:
        DM.PP_STRICT = keep
    recs = enrich(data, pn, delivery, ps["trades"], prog)
    out.update(study(recs, mid))
    return out


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running",
           "note": "Measurement only. The live scan is unchanged by this run."}
    try:
        data = BT.load_from_db(conn)
        start = min(df["trade_date"].iloc[0] for df in data.frames.values() if len(df))
        delivery = load_delivery(conn, list(data.frames), start)
        out["delivery_symbols"] = len(delivery)
        compute(data, delivery, out, lambda m: log.info("t1study: %s", m))
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("tier 1 study failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        conn.rollback()
        log.exception("could not store the tier 1 study")
    try:
        log_markdown(render(out))
    finally:
        if own:
            conn.close()
    return out
