"""
e2_defdiag.py — DEFINITION diagnosis. Measurement only: nothing here removes, disables, filters or re-tunes a detector.

Two questions from the week-1 review of the validation report:

  1. pocket_pivot fired ~18,700 times (next most frequent trade-maker: ~1,900 for breakout_retest). Is the
     definition too loose, and where do its trades come from?
       The detector implements "up day, volume above the largest down-day volume of the prior 10 bars, close in
       the top half" and nothing else (e2_det_momentum.py). An independent recognizer agrees with it
       (detected == expected), so it is faithful to that text; the open question is whether that text is
       the whole definition. This study measures the trades by CONTEXT the text does not constrain:
       trend, extension above the 50-day, distance from the 20-EMA, relative strength, how far the volume
       exceeded the down-day reference, whether it is a repeat hit in the same symbol, and the market regime.
       Every bucket edge below is FIXED IN CODE before any result is seen; every bucket is reported with its
       first-half / second-half result so a pattern that does not hold in both halves can be seen as such.
       `textbook_context` is declared here, before the first run, as the one composite to compare against
       the unrestricted definition. It is a measurement, not an adoption: nothing changes unless the
       result holds in both halves of the window and the owner decides.

  2. vcp is -0.37R in Engine 2. The old engine's VCP trades are in backtest_trades. For every one of them the
     Engine 2 definition is evaluated clause by clause on the same symbol and dates, and the clause that
     first stops it is recorded. That says whether the port changed the definition (a clause the old engine
     never applied) or the setups simply fail the same spec. `vcp_trace` re-states the detector's clauses in
     the detector's own order and is checked against the detector itself (test_defdiag in the delivery tests).

Runs ON THE SERVER; stores engine_settings['engine2_defdiag']; GET /jobs/engine2-defdiag.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger("engine2.defdiag")
KEY = "engine2_defdiag"
LOG_PREFIX = "E2DEFDIAG|"

# ---- fixed bucket edges (declared before any result; never adjusted to a result) -----------------------
EXT50_EDGES = (0.0, 3.0, 8.0, 15.0)            # % close is above the 50-day SMA
EXT20_EDGES = (0.0, 2.0, 5.0, 10.0)            # % close is above the 20-EMA
RS_EDGES = (50.0, 80.0)                        # relative-strength percentile
VOLRATIO_EDGES = (1.2, 2.0)                    # breakout volume / reference (largest down-day volume)
REPEAT_WINDOW = 10                             # a hit within this many bars of the previous hit in the symbol = repeat
TEXTBOOK_MAX_EXT20 = 5.0                       # textbook_context: uptrend AND no more than 5% above the 20-EMA
MIN_BUCKET_N = 30                              # smaller buckets are shown but flagged thin


def _f(x, nd=3):
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _bucket(x, edges, unit="") -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if x < edges[0]:
        return f"< {edges[0]:g}{unit}"
    for lo, hi in zip(edges, edges[1:]):
        if x < hi:
            return f"{lo:g} to {hi:g}{unit}"
    return f">= {edges[-1]:g}{unit}"


def _stats(rs: list[float], first: list[float], second: list[float]) -> dict:
    if not rs:
        return {"n": 0}
    a = np.asarray(rs, float)
    w, l = a[a > 0].sum(), -a[a < 0].sum()
    return {"n": len(rs), "avg_r": _f(a.mean()), "win_pct": _f(100 * (a > 0).mean(), 1), "median_r": _f(np.median(a)),
            "pf": _f(w / l, 2) if l > 0 else None,
            "first_half": {"n": len(first), "avg_r": _f(np.mean(first)) if first else None},
            "second_half": {"n": len(second), "avg_r": _f(np.mean(second)) if second else None},
            "thin": len(rs) < MIN_BUCKET_N}


def _order(label: str):
    """'< 0%' first, ranges by lower edge, '>= x' last, anything else (n/a, flags) alphabetical after the ranges."""
    import re
    m = re.search(r"-?\d+(\.\d+)?", label)
    if label.startswith("<"):
        return (0, float(m.group()) if m else 0.0, label)
    if label.startswith(">="):
        return (2, float(m.group()) if m else 0.0, label)
    if m and label[0].isdigit():
        return (1, float(m.group()), label)
    return (3, 0.0, label)


def _table(rows: list[dict], key: str) -> dict:
    g: dict[str, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
    for r in rows:
        a, f, s = g[r[key]]
        a.append(r["r"])
        (f if r["half"] == 1 else s).append(r["r"])
    return {k: _stats(*v) for k, v in sorted(g.items(), key=lambda kv: _order(kv[0]))}


# ======================================================================
# 1. pocket_pivot
# ======================================================================
def pocket_pivot_study(data, pn, window_start, mid) -> dict:
    import e2_backtest as BT
    import e2_det_momentum as MOM
    from e2_det_common import WARMUP
    from e2_detectors import detect_all
    from e2_features import compute_features
    from e2_manage import simulate

    surv_days = data.coverage.get("surveillance", {}).get("snapshot_days", 0)
    surv_active = surv_days >= 30
    split = np.datetime64(mid, "D")
    rows: list[dict] = []
    symbol_days = raw_hits = gated = 0
    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            continue
        lo = max(int(np.searchsorted(b.date, np.datetime64(window_start, "D"))), WARMUP)
        symbol_days += max(0, b.n - lo)
        raw = MOM.pocket_pivot(b, {}, lo, b.n)                       # every hit, before the 3-bar cooldown
        raw_hits += len(raw)
        raw_idx = np.array([c.idx for c in raw], int)
        cands, _ = detect_all(b, lo=lo, only=["pocket_pivot"])
        for c in cands:
            if BT.gate(b, c, data.surveillance.get(sym), surv_active):
                gated += 1
                continue
            row_i = pn.row_of.get(b.date[c.idx])
            tr = simulate(b, c, data.events.get(sym))
            if tr.skipped:
                continue
            i = c.idx
            prev = raw_idx[raw_idx < i]
            gap = int(i - prev.max()) if prev.size else None
            ext50 = (b.c[i] / b.sma50[i] - 1) * 100 if b.sma50[i] > 0 else np.nan
            ext20 = (b.c[i] / b.ema20[i] - 1) * 100 if b.ema20[i] > 0 else np.nan
            trend = bool(b.c[i] > b.sma50[i] and b.sma50_up[i])
            rs = np.nan
            state = None
            if row_i is not None:
                if sym in pn.rs_pct.columns:
                    rs = pn.rs_pct.iat[row_i, pn.rs_pct.columns.get_loc(sym)]
                state = pn.regime["state"].iat[row_i]
            meta = c.meta or {}
            rows.append({
                "r": float(tr.r), "half": 1 if np.datetime64(tr.signal_date, "D") < split else 2,
                "trend": "uptrend (close>50d, 50d rising)" if trend else "not in uptrend",
                "ext50": _bucket(ext50, EXT50_EDGES, "%"), "ext20": _bucket(ext20, EXT20_EDGES, "%"),
                "rs": _bucket(float(rs) if rs == rs else None, RS_EDGES), "regime": str(state),
                "vol_vs_down": _bucket(meta.get("vol_vs_down"), VOLRATIO_EDGES, "x"),
                "no_down_day": "no down day in the 10 bars (20-bar avg used)" if meta.get("no_down_day") else "down day present",
                "repeat": "repeat hit (<= %d bars after the previous)" % REPEAT_WINDOW if gap is not None and gap <= REPEAT_WINDOW else "first hit",
                "textbook": "textbook_context" if (trend and ext20 == ext20 and ext20 <= TEXTBOOK_MAX_EXT20) else "outside textbook_context",
                "trend_b": trend,
            })
    out = {
        "definition": "up day, volume > largest down-day volume of the prior 10 bars (20-bar average if none), close in the top half. "
                      "No trend, location or extension condition.",
        "eligible_symbol_days": symbol_days, "raw_hits_before_cooldown": raw_hits,
        "raw_hit_rate_pct_of_symbol_days": _f(100 * raw_hits / symbol_days, 2) if symbol_days else None,
        "trades_after_cooldown_and_gates": len(rows), "gated": gated,
        "overall": _stats([r["r"] for r in rows], [r["r"] for r in rows if r["half"] == 1], [r["r"] for r in rows if r["half"] == 2]),
        "declared_before_first_run": {"textbook_context": f"close > 50-day SMA AND 50-day SMA rising AND close <= {TEXTBOOK_MAX_EXT20:g}% above the 20-EMA",
                                      "edges": {"ext50": EXT50_EDGES, "ext20": EXT20_EDGES, "rs": RS_EDGES, "vol_vs_down": VOLRATIO_EDGES},
                                      "adoption_rule": "nothing is adopted from this table; a context is only a candidate for a definition change if "
                                                       f"its avg R is positive in BOTH halves with n >= {MIN_BUCKET_N} in each"},
        "by": {k: _table(rows, k) for k in ("textbook", "trend", "ext50", "ext20", "rs", "vol_vs_down", "no_down_day", "repeat", "regime")},
    }
    return out


# ======================================================================
# 2. vcp: clause trace of the Engine 2 definition
# ======================================================================
CLAUSES = ("close_not_above_prior_5_bar_high", "breakout_volume_below_1.5x", "no_52_week_high_or_history",
           "not_in_uptrend_(close>50d>150d)", "no_record_high_pivot_in_200_bars", "close_not_above_pivot",
           "base_shorter_than_25_bars", "pivot_below_85pct_of_52wk_high", "fewer_than_2_contractions",
           "first_contraction_deeper_than_40pct", "contractions_not_each_below_0.7x_the_prior", "final_contraction_8pct_or_more", "PASS")
_RANK = {c: k for k, c in enumerate(CLAUSES)}


def vcp_trace(b, i: int) -> str:
    """The first clause of the Engine 2 vcp definition that stops bar i (or PASS). Mirrors e2_det_bases.vcp clause for clause."""
    import e2_det_bases as BS
    from e2_det_common import record_highs, zigzag

    if i < 160 or i >= b.n:
        return "no_52_week_high_or_history"
    if not b.c[i] > b.h[i - 5:i].max():
        return "close_not_above_prior_5_bar_high"
    if not b.relvol[i] >= BS.VOL15:
        return "breakout_volume_below_1.5x"
    if np.isnan(b.high52[i]):
        return "no_52_week_high_or_history"
    if not BS._trend_ctx(b, i):
        return "not_in_uptrend_(close>50d>150d)"
    pivots = record_highs(b.h, i, 200)
    if not pivots:
        return "no_record_high_pivot_in_200_bars"
    best = "close_not_above_pivot"
    for p in pivots:                                                  # the detector accepts the first pivot that passes everything
        pivot = float(b.h[p])
        if b.c[i] <= pivot:
            why = "close_not_above_pivot"
        elif i - p < 25:
            why = "base_shorter_than_25_bars"
        elif pivot < 0.85 * b.high52[i]:
            why = "pivot_below_85pct_of_52wk_high"
        else:
            zz = zigzag(b.h, b.l, p, i - 1, max(BS.VCP_ZZ, BS.VCP_ZZ_ATR * float(b.atr_pct[i]) / 100.0))
            depths = [(zz[k][2] - zz[k + 1][2]) / zz[k][2] * 100 for k in range(len(zz) - 1)
                      if zz[k][1] == "H" and zz[k + 1][1] == "L"]
            if len(depths) < 2:
                why = "fewer_than_2_contractions"
            elif depths[0] > BS.VCP_MAX_DEPTH:
                why = "first_contraction_deeper_than_40pct"
            elif any(nx > pv * BS.VCP_RATIO for pv, nx in zip(depths, depths[1:])):
                why = "contractions_not_each_below_0.7x_the_prior"
            elif depths[-1] >= BS.VCP_FINAL:
                why = "final_contraction_8pct_or_more"
            else:
                return "PASS"
        if _RANK[why] > _RANK[best]:
            best = why                                                # report how far the closest pivot got
    return best


def trace_window(b, lo: int, hi: int) -> str:
    """Best (furthest-reaching) outcome over the bars lo..hi-1: PASS if any bar passes, else the deepest clause reached."""
    best, best_rank = "close_not_above_prior_5_bar_high", -1
    for i in range(max(lo, 0), min(hi, b.n)):
        t = vcp_trace(b, i)
        if t == "PASS":
            return "PASS"
        if _RANK[t] > best_rank:
            best, best_rank = t, _RANK[t]
    return best


# ======================================================================
def render_markdown(d: dict) -> str:
    L = ["## Definition diagnosis (measurement only)", "", d.get("note", ""), ""]
    pp = d.get("pocket_pivot")
    if pp and "error" in pp:
        L += ["### pocket_pivot", f"Could not be measured: {pp['error']}", ""]
    elif pp:
        o = pp["overall"]
        L += ["### pocket_pivot", f"- Definition as implemented: {pp['definition']}",
              f"- {pp['raw_hits_before_cooldown']} hits over {pp['eligible_symbol_days']} eligible symbol-days = {pp['raw_hit_rate_pct_of_symbol_days']}% of all symbol-days; "
              f"{pp['trades_after_cooldown_and_gates']} trades after the 3-bar cooldown and gates. Overall avg {o.get('avg_r')}R, {o.get('win_pct')}% winners "
              f"(first half {o['first_half']['avg_r']}R, second half {o['second_half']['avg_r']}R).",
              f"- Declared before the first run: {pp['declared_before_first_run']['textbook_context']}. Rule: {pp['declared_before_first_run']['adoption_rule']}.", ""]
        for dim, tab in pp["by"].items():
            L += [f"**{dim}**", "", "| bucket | n | avg R | win % | PF | 1st half R (n) | 2nd half R (n) |", "|---|---|---|---|---|---|---|"]
            for k, s in tab.items():
                if not s.get("n"):
                    continue
                L.append(f"| {k}{' (thin)' if s['thin'] else ''} | {s['n']} | {s['avg_r']} | {s['win_pct']} | {s['pf']} | "
                         f"{s['first_half']['avg_r']} ({s['first_half']['n']}) | {s['second_half']['avg_r']} ({s['second_half']['n']}) |")
            L.append("")
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for k in range(0, max(len(line), 1), 1500):
            log.info("%s%s", LOG_PREFIX, line[k:k + 1500])
            n += 1
    return n


def _put(conn, value) -> None:
    with conn.cursor() as cur:
        cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (KEY, json.dumps(value, default=str)))
    conn.commit()


def run(conn=None, trigger: str = "manual") -> dict:
    """Never raises: a failure is stored as the result (a diagnosis that cannot run must say why)."""
    import e2_backtest as BT
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running",
           "note": "Measurement only. No detector is removed, disabled, filtered or re-tuned by this diagnosis."}
    try:
        data = BT.load_from_db(conn)
        pn = BT.build_panels(data)
        w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
        mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
        try:
            out["pocket_pivot"] = pocket_pivot_study(data, pn, w0, mid)
        except Exception as e:                                       # noqa: BLE001
            log.exception("pocket_pivot study failed")
            out["pocket_pivot"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        out["state"] = "done" if "error" not in out["pocket_pivot"] else "done_with_errors"
    except Exception as e:                                           # noqa: BLE001
        log.exception("definition diagnosis failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        _put(conn, out)
    except Exception:                                                # noqa: BLE001
        log.exception("could not store the definition diagnosis")
    try:
        log_markdown(render_markdown(out))
    finally:
        if own:
            conn.close()
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = run(trigger=sys.argv[1] if len(sys.argv) > 1 else "cli")
    print(json.dumps({"state": res.get("state"), "runtime_sec": res.get("runtime_sec")}))
