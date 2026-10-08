"""
e2_diagnosis.py — why did four patterns lose in the old engine, and what does the corrected Engine 2
definition do with the very same setups?

Patterns: cup_handle, ascending_base, flag_pennant, asc_triangle (all four were negative in the old
3-year run and are corrected in e2_det_bases.py).

It runs ON THE SERVER, against the database. Nobody runs SQL. It starts itself ahead of every
validation run (see e2_service.run_validation_blocking), stores its result in
engine_settings['engine2_diagnosis'] and is read at GET /jobs/engine2-diagnosis and inside
GET /jobs/engine2-report.

MEASUREMENT ONLY. Nothing here removes, disables or filters a pattern, and no threshold is touched.

Per pattern:
  old_engine   from the latest stored backtest run that has trades of that pattern: n, avg R, win %,
               exits by reason, MFE/MAE, bars held, how many reached +1R, how many stopped within 2 bars,
               fill vs published trigger, real risk %, first-half / second-half avg R, and the shape
               columns the old engine recorded (handle slope, cup shape).
  replay       every old trade re-examined by the CORRECTED Engine 2 detector on the same symbol and the
               same dates (signal date - 3 bars .. entry date + 3 bars): `kept` = the corrected definition
               still recognises a breakout there, `rejected` = it does not. Both groups carry the old
               engine's realised R, so the question "would the new definition have avoided the losers, and
               at what cost to the winners" is answered by numbers. Kept setups also carry the R they
               would have made under Engine 2's own trade management.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

log = logging.getLogger("engine2.diagnosis")

KEY = "engine2_diagnosis"
PATTERNS = ("cup_handle", "ascending_base", "flag_pennant", "asc_triangle")
PRE_BARS, POST_BARS = 3, 3
HISTORY_DAYS = 900                      # calendar days of bars before the earliest signal (features need ~250)


def _f(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _put(conn, value) -> None:
    with conn.cursor() as cur:
        cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (KEY, json.dumps(value, default=str)))
    conn.commit()


def _table_has(conn, col: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("select 1 from information_schema.columns where table_name = 'backtest_trades' "
                    "and table_schema = current_schema() and column_name = %s", (col,))
        return cur.fetchone() is not None


# ----------------------------------------------------------------------
# old-engine statistics
# ----------------------------------------------------------------------
def load_old_trades(conn, pattern: str) -> tuple[int | None, pd.DataFrame]:
    """The trades of `pattern` from the latest run that has any with a realised result."""
    with conn.cursor() as cur:
        cur.execute("select run_id from backtest_trades where pattern = %s and r_realised is not null "
                    "group by run_id order by run_id desc limit 1", (pattern,))
        r = cur.fetchone()
    if not r:
        return None, pd.DataFrame()
    shape = [c for c in ("handle_slope", "handle_slope_pct", "handle_depth_pct", "cup_shape", "cup_rounding_bars", "base_duration")
             if _table_has(conn, c)]
    cols = ["symbol", "signal_date", "entry_date", "exit_date", "entry_trigger", "entry_price", "stop_loss", "t1",
            "r_planned", "r_realised", "exit_reason", "max_favourable_r", "max_adverse_r", "bars_held", "regime", "band"] + shape
    with conn.cursor() as cur:
        cur.execute(f"select {', '.join(cols)} from backtest_trades where run_id = %s and pattern = %s and r_realised is not null "
                    "order by signal_date", (r[0], pattern))
        df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in ("entry_trigger", "entry_price", "stop_loss", "t1", "r_planned", "r_realised", "max_favourable_r", "max_adverse_r"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return int(r[0]), df


def old_engine_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if not n:
        return {"n": 0}
    r = df["r_realised"]
    fill = (df["entry_price"] / df["entry_trigger"] - 1) * 100
    risk = (df["entry_price"] - df["stop_loss"]) / df["entry_price"] * 100
    ordered = df.sort_values("signal_date")
    half = n // 2
    out = {
        "n": n, "avg_r": _f(r.mean()), "median_r": _f(r.median()), "total_r": _f(r.sum(), 2),
        "win_rate_pct": _f((r > 0).mean() * 100, 1), "loss_pct_at_or_below_minus_0_9r": _f((r <= -0.9).mean() * 100, 1),
        "exits": {k: {"n": int(len(g)), "avg_r": _f(g["r_realised"].mean())} for k, g in df.groupby("exit_reason", dropna=False)},
        "stop_exit_pct": _f((df["exit_reason"].astype(str).str.contains("stop", case=False)).mean() * 100, 1),
        "avg_mfe_r": _f(df["max_favourable_r"].mean()), "best_mfe_r": _f(df["max_favourable_r"].max()),
        "reached_1r": int((df["max_favourable_r"] >= 1.0).sum()), "avg_mae_r": _f(df["max_adverse_r"].mean()),
        "avg_bars_held": _f(df["bars_held"].mean(), 1), "stopped_within_2_bars": int((df["bars_held"] <= 2).sum()),
        "avg_fill_vs_trigger_pct": _f(fill.mean(), 2), "avg_real_risk_pct": _f(risk.mean(), 2),
        "first_half_avg_r": _f(ordered["r_realised"].iloc[:half].mean()) if half else None,
        "second_half_avg_r": _f(ordered["r_realised"].iloc[half:].mean()) if n - half else None,
        "by_regime": {str(k): {"n": int(len(g)), "avg_r": _f(g["r_realised"].mean())} for k, g in df.groupby("regime", dropna=False)},
    }
    for col in ("handle_slope", "cup_shape"):
        if col in df and df[col].notna().any():
            out[f"by_{col}"] = {str(k): {"n": int(len(g)), "avg_r": _f(g["r_realised"].mean()),
                                         "win_rate_pct": _f((g["r_realised"] > 0).mean() * 100, 1)}
                                for k, g in df.groupby(col, dropna=True)}
    for col in ("handle_depth_pct", "cup_rounding_bars", "base_duration"):
        if col in df and df[col].notna().any():
            v = pd.to_numeric(df[col], errors="coerce")
            ok = v.notna()
            out.setdefault("shape_means", {})[col] = {"all": _f(v.mean(), 2),
                                                      "winners": _f(v[ok & (df["r_realised"] > 0)].mean(), 2),
                                                      "losers": _f(v[ok & (df["r_realised"] <= 0)].mean(), 2)}
    return out


# ----------------------------------------------------------------------
# replay through the corrected detector
# ----------------------------------------------------------------------
def _bars_for(conn, symbol: str, earliest):
    from e2_features import compute_features
    start = (pd.Timestamp(earliest) - pd.Timedelta(days=HISTORY_DAYS)).date()
    with conn.cursor() as cur:
        cur.execute("select trade_date, adj_open, adj_high, adj_low, adj_close, volume from ohlcv_daily "
                    "where symbol = %s and trade_date >= %s order by trade_date", (symbol, start))
        rows = cur.fetchall()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["trade_date", "open", "high", "low", "close", "volume"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return compute_features(symbol, df)


def replay(conn, pattern: str, df: pd.DataFrame) -> dict:
    from e2_detectors import detect_all
    from e2_det_common import WARMUP
    import e2_manage as MG

    if df.empty:
        return {"n": 0}
    earliest = df["signal_date"].min()
    rows, skipped = [], Counter()
    cache: dict[str, object] = {}
    for t in df.itertuples():
        b = cache.get(t.symbol)
        if t.symbol not in cache:
            b = cache[t.symbol] = _bars_for(conn, t.symbol, earliest)
        if b is None:
            skipped["no_or_short_price_history"] += 1
            continue
        i_sig = int(np.searchsorted(b.date, np.datetime64(t.signal_date, "D")))
        end_d = t.entry_date if t.entry_date is not None and not pd.isna(t.entry_date) else t.signal_date
        i_end = int(np.searchsorted(b.date, np.datetime64(end_d, "D")))
        if i_sig >= b.n or i_sig < WARMUP:
            skipped["signal_outside_computable_range"] += 1
            continue
        lo, hi = max(i_sig - PRE_BARS, WARMUP), min(i_end + POST_BARS + 1, b.n)
        cands, errs = detect_all(b, lo=lo, hi=hi, only=[pattern])
        if errs:
            skipped["detector_error"] += 1
        hit = [c for c in cands if lo <= c.idx < hi]
        rec = {"symbol": t.symbol, "signal_date": str(t.signal_date), "old_r": _f(t.r_realised), "old_exit": t.exit_reason,
               "kept": bool(hit)}
        if hit:
            c = min(hit, key=lambda c: abs(c.idx - i_end))
            tr = MG.simulate(b, c, None)
            rec.update(new_detection_date=str(c.date)[:10], new_quality=_f(c.quality, 1),
                       engine2_r=None if tr.skipped or tr.open_at_end else _f(tr.r), engine2_exit=None if tr.skipped else tr.reason)
        rows.append(rec)

    def group(sel):
        g = [r for r in rows if sel(r)]
        old = [r["old_r"] for r in g if r["old_r"] is not None]
        e2 = [r["engine2_r"] for r in g if r.get("engine2_r") is not None]
        return {"n": len(g), "old_avg_r": _f(np.mean(old)) if old else None,
                "old_winners": int(sum(1 for x in old if x > 0)), "old_losers": int(sum(1 for x in old if x <= 0)),
                "engine2_avg_r": _f(np.mean(e2)) if e2 else None, "engine2_n": len(e2)}
    kept, rej = group(lambda r: r["kept"]), group(lambda r: not r["kept"])
    tot = kept["n"] + rej["n"]
    return {"window": f"signal date - {PRE_BARS} bars .. entry date + {POST_BARS} bars", "n_replayed": tot, "not_replayed": dict(skipped),
            "kept_by_corrected_definition": kept, "rejected_by_corrected_definition": rej,
            "rejected_pct": _f(100 * rej["n"] / tot, 1) if tot else None,
            "trades": rows}


def reading(p: str, old: dict, rp: dict) -> list[str]:
    """Plain statements of what the numbers say. No recommendation to drop anything."""
    if not old.get("n"):
        return ["No stored backtest trades of this pattern, so there is no old-engine record to diagnose."]
    L = [f"Old engine: {old['n']} trades, avg {old['avg_r']}R, {old['win_rate_pct']}% winners, {old['stop_exit_pct']}% ended on the stop; "
         f"average best excursion {old['avg_mfe_r']}R, {old['reached_1r']} ever reached +1R, {old['stopped_within_2_bars']} were out within 2 bars."]
    if old.get("avg_mfe_r") is not None and old["avg_mfe_r"] < 0.5 and old["stop_exit_pct"] and old["stop_exit_pct"] >= 70:
        L.append("They did not work and then fail: price rarely went in favour at all before the stop (failed-breakout profile, "
                 "which points at what was being called a breakout, not at exits).")
    if old.get("avg_fill_vs_trigger_pct") is not None and old["avg_fill_vs_trigger_pct"] > 1.0:
        L.append(f"Fills averaged {old['avg_fill_vs_trigger_pct']}% above the published trigger (chasing / gap fills).")
    if rp.get("n_replayed"):
        k, r = rp["kept_by_corrected_definition"], rp["rejected_by_corrected_definition"]
        L.append(f"Corrected definition on the same setups: rejects {r['n']} of {rp['n_replayed']} ({rp['rejected_pct']}%). "
                 f"Rejected group: old avg {r['old_avg_r']}R ({r['old_winners']} winners / {r['old_losers']} losers). "
                 f"Kept group: old avg {k['old_avg_r']}R ({k['old_winners']} / {k['old_losers']}), Engine 2 management avg {k['engine2_avg_r']}R on {k['engine2_n']}.")
    return L


# ----------------------------------------------------------------------
def run(conn=None, trigger: str = "manual") -> dict:
    """Never raises: a failure is stored as the result (a diagnosis that cannot run must say why)."""
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running",
           "note": "Measurement only. No pattern is removed, disabled or filtered by this diagnosis.", "patterns": {}}
    try:
        for p in PATTERNS:
            try:
                run_id, df = load_old_trades(conn, p)
                old = old_engine_stats(df)
                rp = replay(conn, p, df)
                summary = {"backtest_run_id": run_id, "old_engine": old, "replay": rp}
                summary["reading"] = reading(p, old, rp)
                out["patterns"][p] = summary
            except Exception as e:                                  # noqa: BLE001 — one pattern failing must not hide the others
                log.exception("diagnosis of %s failed", p)
                try:
                    conn.rollback()
                except Exception:
                    pass
                out["patterns"][p] = {"error": f"{type(e).__name__}: {e}"[:300]}
        out["state"] = "done" if not any("error" in v for v in out["patterns"].values()) else "done_with_errors"
    except Exception as e:                                          # noqa: BLE001
        log.exception("diagnosis failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    try:
        _put(conn, out)
    except Exception:                                               # noqa: BLE001
        log.exception("could not store the diagnosis")
    finally:
        if own:
            conn.close()
    return out


def render_markdown(d: dict) -> str:
    L = ["## Disabled-pattern diagnosis (old engine vs corrected definition)", "", d.get("note", ""), ""]
    for p, v in (d.get("patterns") or {}).items():
        L.append(f"### {p}")
        if "error" in v:
            L += [f"Could not be diagnosed: {v['error']}", ""]
            continue
        L += [f"- {line}" for line in v.get("reading", [])]
        L.append("")
    return "\n".join(L)


def brief(d: dict | None) -> dict | None:
    if not d:
        return None
    out = {"generated_at": d.get("generated_at"), "state": d.get("state"), "patterns": {}}
    for p, v in (d.get("patterns") or {}).items():
        if "error" in v:
            out["patterns"][p] = {"error": v["error"]}
            continue
        o, r = v["old_engine"], v["replay"]
        out["patterns"][p] = {"old_n": o.get("n"), "old_avg_r": o.get("avg_r"), "old_stop_exit_pct": o.get("stop_exit_pct"),
                              "replayed": r.get("n_replayed"), "rejected_by_corrected_pct": r.get("rejected_pct")}
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = run(trigger=sys.argv[1] if len(sys.argv) > 1 else "cli")
    print(json.dumps(brief(res), default=str))
