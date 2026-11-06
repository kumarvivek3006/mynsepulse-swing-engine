"""
e2_fwddiag.py — ONE-DETECTOR FORWARD-RETURN DIAGNOSIS (owner's order of 7 Nov: triage the non-publishing detectors, one per day).

Question: where, inside one detector, does the forward return live? Same measurement as e2_fwd (live next-open entry, no stop, hold h sessions,
market-adjusted vs the index, percent), now split by the context at the signal bar:
  features   relvol, close position in the day's range, ATR% , distance to the 52-week high, extension over the 20-EMA, distance over the
             200-day, RSI14, 60-session relative strength vs the index, index trend (index over its 50-day average), index 20-session return,
             the entry gap (next open vs signal close), plus whatever the detector itself records (meta: depth, bars down, touch, ...)
  for each  terciles (cut points from this detector's own candidates), n and mean adjusted return at 10 and 20 sessions, the excess over the
            no-detector control, Spearman(feature, adjusted return) with a week-block bootstrap interval, and whether the top-minus-bottom
            tercile difference has the same sign in both halves.
Hypotheses only. A feature is carried forward as a candidate definition change ONLY if it is monotone with a bootstrap interval that excludes zero
and keeps its sign in both halves. The held-out check then fixes the cut at the FIRST-half median (no look at the second half), applies it to the
SECOND half and compares with the unfiltered second half. Nothing here changes a detector, gate, score or exit.

Writes engine_settings.engine2_fwddiag_<detector>; logs E2FWDD| lines; GET /jobs/engine2-fwddiag?detector=...
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import e2_backtest as BT
from e2_det_common import WARMUP
from e2_detectors import detect_all
from e2_features import compute_features
from e2_fwd import _boot, _market, _week
from e2_manage import _locked

log = logging.getLogger("engine2.fwddiag")
LOG_PREFIX = "E2FWDD|"
HS = (10, 20)


def key_for(detector: str) -> str:
    return f"engine2_fwddiag_{detector}"


def _f(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 30:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _rho_ci(weeks: np.ndarray, x: np.ndarray, y: np.ndarray, reps: int = 300, seed: int = 5):
    ok = ~(np.isnan(x) | np.isnan(y))
    weeks, x, y = weeks[ok], x[ok], y[ok]
    if len(x) < 60:
        return None, None
    u, inv = np.unique(weeks, return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(u))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, len(u), size=len(u))
        ix = np.concatenate([groups[k] for k in pick])
        vals.append(_spearman(x[ix], y[ix]))
    vals = [v for v in vals if v == v]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (None, None)


def features_at(b, i: int, mk: np.ndarray, mk_sma50: np.ndarray, meta: dict) -> dict:
    e = i + 1
    f = {
        "relvol": float(b.relvol[i]), "close_pos": float(b.close_pos[i]), "atr_pct": float(b.atr_pct[i]),
        "dist_52w_high_pct": float((b.c[i] / b.high52[i] - 1.0) * 100.0) if b.high52[i] > 0 else np.nan,
        "ext_over_ema20_pct": float((b.c[i] / b.ema20[i] - 1.0) * 100.0) if b.ema20[i] > 0 else np.nan,
        "over_sma200_pct": float((b.c[i] / b.sma200[i] - 1.0) * 100.0) if b.sma200[i] > 0 else np.nan,
        "rsi14": float(b.rsi14[i]),
        "rs60_vs_index_pct": float(((b.c[i] / b.c[i - 60]) - (mk[i] / mk[i - 60])) * 100.0) if i >= 60 and mk[i - 60] > 0 else np.nan,
        "index_over_sma50_pct": float((mk[i] / mk_sma50[i] - 1.0) * 100.0) if mk_sma50[i] > 0 else np.nan,
        "index_ret20_pct": float((mk[i] / mk[i - 20] - 1.0) * 100.0) if i >= 20 and mk[i - 20] > 0 else np.nan,
        "entry_gap_pct": float((b.o[e] / b.c[i] - 1.0) * 100.0),
        "hv_pct": float(b.hv_pct[i]),
    }
    for k, v in (meta or {}).items():
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            f[f"meta_{k}"] = float(v)
    return f


def collect(data, pn, w0, detector: str):
    rows, ctrl = [], {h: ([], []) for h in HS}
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30
    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            continue
        mk = _market(data, b)
        mk_sma50 = pd.Series(mk).rolling(50, min_periods=30).mean().to_numpy()
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        cands, _ = detect_all(b, lo=lo)
        for c in cands:
            if c.detector != detector:
                continue
            if BT.gate(b, c, data.surveillance.get(sym), surv_active) or pn.row_of.get(b.date[c.idx]) is None:
                continue
            e = c.idx + 1
            if e >= b.n or _locked(b, e) or e - 1 < 0 or np.isnan(mk[e - 1]):
                continue
            r = {"sig": np.datetime64(c.date, "D"), "symbol": sym, "idx": int(c.idx)}
            for h in HS:
                j = e + h - 1
                r[f"adj{h}"] = (100.0 * (b.c[j] / b.o[e] - 1.0) - 100.0 * (mk[j] / mk[e - 1] - 1.0)) if j < b.n and not np.isnan(mk[j]) else np.nan
            r.update(features_at(b, c.idx, mk, mk_sma50, c.meta))
            rows.append(r)
        for h in HS:                                             # control: every 5th bar
            es = np.arange(lo + 1, b.n - h + 1, 5)
            es = es[(es - 1 >= 0) & ~np.isnan(mk[es - 1]) & ~np.isnan(mk[es + h - 1]) & (b.o[es] > 0)]
            if len(es):
                a = 100.0 * (b.c[es + h - 1] / b.o[es] - 1.0) - 100.0 * (mk[es + h - 1] / mk[es - 1] - 1.0)
                ctrl[h][0].extend(a.tolist()); ctrl[h][1].extend(b.date[es - 1].astype("datetime64[D]").tolist())
    return pd.DataFrame(rows), {h: (np.array(v[0]), np.array(v[1])) for h, v in ctrl.items()}


def analyse(df: pd.DataFrame, ctrl: dict, mid64) -> dict:
    out = {"n": int(len(df))}
    ctl = {h: float(np.mean(ctrl[h][0])) for h in HS}
    out["control_mean_adj_pct"] = {str(h): _f(ctl[h]) for h in HS}
    wk = _week(df["sig"].to_numpy())
    first = (df["sig"] < mid64).to_numpy()
    base = {}
    for h in HS:
        y = df[f"adj{h}"].to_numpy()
        ok = ~np.isnan(y)
        lo, hi = _boot(wk[ok], y[ok])
        base[str(h)] = {"n": int(ok.sum()), "mean": _f(np.nanmean(y)), "ci95": [_f(lo), _f(hi)], "excess_vs_control": _f(np.nanmean(y) - ctl[h]),
                        "h1": _f(np.nanmean(y[first & ok])) if (first & ok).any() else None, "h2": _f(np.nanmean(y[~first & ok])) if (~first & ok).any() else None}
    out["overall"] = base
    feats = [c for c in df.columns if c not in ("sig", "symbol", "idx", "adj10", "adj20") and df[c].nunique() > 3]
    table = []
    for f in feats:
        x = df[f].to_numpy(float)
        if np.isnan(x).mean() > 0.5:
            continue
        q1, q2 = np.nanpercentile(x, [33.3, 66.7])
        bucket = np.where(x <= q1, 0, np.where(x <= q2, 1, 2))
        bucket = np.where(np.isnan(x), -1, bucket)
        row = {"feature": f, "cuts": [_f(q1), _f(q2)]}
        for h in HS:
            y = df[f"adj{h}"].to_numpy()
            ok = ~np.isnan(y)
            per = []
            for k in (0, 1, 2):
                m = (bucket == k) & ok
                per.append({"n": int(m.sum()), "mean": _f(np.mean(y[m])) if m.any() else None})
            rho = _spearman(x, y)
            lo, hi = _rho_ci(wk, x, y)
            diffs = []
            for half in (first, ~first):
                lo_m, hi_m = (bucket == 0) & ok & half, (bucket == 2) & ok & half
                diffs.append(float(np.mean(y[hi_m]) - np.mean(y[lo_m])) if lo_m.sum() >= 10 and hi_m.sum() >= 10 else np.nan)
            row[f"h{h}"] = {"terciles": per, "rho": _f(rho), "rho_ci95": [_f(lo), _f(hi)], "top_minus_bottom_by_half": [_f(d) for d in diffs],
                            "same_sign_both_halves": bool(all(d == d for d in diffs) and diffs[0] * diffs[1] > 0),
                            "interval_excludes_zero": bool(lo is not None and (lo > 0 or hi < 0))}
        table.append(row)
    out["features"] = table
    # held-out check on features that are monotone with an interval that excludes zero at 20 sessions and keep sign in both halves
    checks = []
    for row in table:
        r20 = row["h20"]
        if not (r20["interval_excludes_zero"] and r20["same_sign_both_halves"] and r20["rho"] is not None):
            continue
        x = df[row["feature"]].to_numpy(float)
        cut = float(np.nanmedian(x[first]))                        # fixed on the FIRST half only
        favour_high = r20["rho"] > 0
        keep2 = (~first) & ((x >= cut) if favour_high else (x <= cut))
        y = df["adj20"].to_numpy()
        m2 = (~first) & ~np.isnan(y)
        mk2 = keep2 & ~np.isnan(y)
        if mk2.sum() < 20:
            continue
        checks.append({"feature": row["feature"], "rule": f"{'>=' if favour_high else '<='} {cut:.3f} (first-half median)", "second_half_all": {"n": int(m2.sum()), "mean20": _f(y[m2].mean())},
                       "second_half_kept": {"n": int(mk2.sum()), "mean20": _f(y[mk2].mean())}, "second_half_dropped": {"n": int((m2 & ~mk2).sum()), "mean20": _f(y[m2 & ~mk2].mean())},
                       "second_half_control_mean20": _f(float(np.mean(ctrl[20][0][ctrl[20][1] >= mid64])))})
    out["held_out_checks"] = checks
    return out


def compute(data, detector: str, out: dict) -> dict:
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    out["detector"], out["window"] = detector, {"start": str(w0.date()), "split": str(mid.date()), "end": str(pn.cal[-1].date())}
    df, ctrl = collect(data, pn, w0, detector)
    if df.empty:
        out["error"] = "no candidates"
        return out
    out.update(analyse(df, ctrl, np.datetime64(mid, "D")))
    return out


def run(detector: str, conn=None, trigger: str = "manual") -> dict:
    """`detector` may be a comma-separated list: the history is loaded once and each detector is diagnosed in turn (one result per detector)."""
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    names = [d.strip() for d in detector.split(",") if d.strip()]
    data, load_err = None, None
    try:
        data = BT.load_from_db(conn)
    except Exception as e:                                                             # noqa: BLE001
        log.exception("forward-return diagnosis: loading failed")
        load_err = f"{type(e).__name__}: {e}"[:300]
    last = {}
    try:
        for det in names:
            t0 = time.time()
            out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running", "note": "Measurement only."}
            if load_err:
                out["state"], out["error"], out["detector"] = "failed", load_err, det
            else:
                try:
                    compute(data, det, out)
                    out["state"] = "done"
                except Exception as e:                                                 # noqa: BLE001
                    log.exception("forward-return diagnosis failed for %s", det)
                    out["state"], out["error"], out["detector"] = "failed", f"{type(e).__name__}: {e}"[:300], det
            out["runtime_sec"] = round(time.time() - t0)
            try:
                with conn.cursor() as cur:
                    cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                                "on conflict (key) do update set value = excluded.value, updated_at = now()", (key_for(det), json.dumps(out, default=str)))
                conn.commit()
            except Exception:                                                          # noqa: BLE001
                log.exception("could not store the diagnosis")
            log_markdown(render_markdown(out))
            last = out
    finally:
        if own:
            conn.close()
    return last


def render_markdown(d: dict) -> str:
    L = [f"# Forward-return diagnosis: {d.get('detector')}  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    L.append(f"n {d.get('n')}  control mean adj {d.get('control_mean_adj_pct')}")
    for h, s in (d.get("overall") or {}).items():
        L.append(f"overall h{h}: n {s['n']} mean {s['mean']} CI{s['ci95']} excess vs control {s['excess_vs_control']} halves {s['h1']}/{s['h2']}")
    L.append("## features (terciles low/mid/high: n, mean adj%)")
    for r in d.get("features") or []:
        parts = []
        for h in HS:
            x = r[f"h{h}"]
            t = x["terciles"]
            parts.append(f"h{h} " + " / ".join(f"{(c['mean'] if c['mean'] is not None else float('nan')):+.2f}({c['n']})" for c in t)
                         + f" rho {x['rho']} CI{x['rho_ci95']} tb/halves {x['top_minus_bottom_by_half']}{' *' if x['interval_excludes_zero'] and x['same_sign_both_halves'] else ''}")
        L.append(f"{r['feature']} cuts{r['cuts']}: " + " | ".join(parts))
    L.append("## held-out checks (cut fixed on the first half, applied to the second)")
    for c in d.get("held_out_checks") or []:
        L.append(json.dumps(c))
    if not d.get("held_out_checks"):
        L.append("none: no feature was monotone with an interval excluding zero and the same sign in both halves")
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for i in range(0, max(len(line), 1), 1400):
            log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
            n += 1
    return n
