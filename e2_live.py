"""
e2_live.py — the nightly SHADOW scan: the same features, the same 20 detectors, the same gates,
scoring and trade rules as the validation run, applied to the latest completed session.

SHADOW means it records and does not publish: nothing here writes to `signals` and the live
strategy engine does not read any of it. Every candidate is appended to a forward log, and each
night the older entries are played forward with the same trade manager, so the forward record
accumulates per detector on real, unseen sessions.

Stored (engine_settings, no schema change needed):
  engine2_shadow_latest   the latest COMPLETED session's candidates, plans and hypothetical book
                          (written by the premarket and postclose scans)
  engine2_intraday_latest the latest forming-bar scan (provisional: never enters the forward log)
  engine2_shadow_log      every candidate of the last ~90 sessions with its outcome once known

scan(conn, mode) is the one implementation behind all three scheduled scans (premarket,
intraday, postclose) and the manual shadow route; shadow_scan(conn) is kept as the postclose alias.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import e2_backtest as BT
import e2_manage as MG
import e2_portfolio as PF
import e2_scorer as SC
import e2_sector as SE
from e2_core import Candidate
from e2_detectors import DETECTORS as DETECTOR_NAMES, detect_all
from e2_features import compute_features

log = logging.getLogger("engine2.live")
LATEST_KEY, LOG_KEY = "engine2_shadow_latest", "engine2_shadow_log"
INTRADAY_KEY = "engine2_intraday_latest"
LOG_KEEP_DAYS, LOG_MAX = 150, 8000
EQUITY = float(os.environ.get("E2_SHADOW_EQUITY", "1000000"))
LIVE_LOOKBACK = 40


def _get(conn, key):
    with conn.cursor() as cur:
        cur.execute("select value, updated_at from engine_settings where key = %s", (key,))
        r = cur.fetchone()
    return (r[0], r[1]) if r else (None, None)


def put(conn, key, value) -> None:
    with conn.cursor() as cur:
        cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (key, json.dumps(value, default=str)))
    conn.commit()


def plan_for(b, c: Candidate, regime_mult: float) -> dict | None:
    """Entry is a market order on the next open; the reference price is the signal close."""
    entry = float(b.c[c.idx]) * (1 + MG.SLIP)
    if c.stop >= entry:
        return None
    stop, floored = MG.initial_stop(b, c, entry)
    R = entry - stop
    if R <= 0:
        return None
    risk_frac = R / entry
    risk_amt = EQUITY * 0.01 * regime_mult
    value = min(risk_amt / risk_frac, 0.20 * EQUITY)
    return {"entry_ref": round(entry, 2), "stop": round(stop, 2), "risk_pct": round(100 * risk_frac, 2),
            "t1": round(entry + MG.T1_R * R, 2), "t1_exit_pct": int(100 * MG.T1_FRAC),
            "t2": round(entry + MG.T2_R * R, 2), "t2_exit_pct": int(100 * MG.T2_FRAC),
            "trail": "last 25%: looser of 20EMA and highest close - 2xATR, never below break-even",
            "after_t1": "stop to break-even", "time_stop": f"bar {MG.TIME_STOP_BARS}: exit if T1 not hit and close < entry + 1R",
            "shares": int(math.floor(value / entry)), "position_value": round(value, 0), "stop_raised_to_1atr": floored}


def shadow_scan(conn) -> dict:
    return scan(conn, "postclose")


def scan(conn, mode: str = "postclose", data=None, extra: dict | None = None) -> dict:
    """
    One scan of the whole universe: every detector on every symbol, hard gates, scoring, the
    hypothetical book. `data` may be supplied by the caller (the intraday scan passes data that
    already carries today's forming bar); otherwise it is loaded from the database.
    mode: 'premarket' | 'postclose' (completed session -> forward log) | 'intraday' (provisional).
    """
    provisional = mode == "intraday"
    data = data if data is not None else BT.load_from_db(conn, window_bars=60)
    pn = BT.build_panels(data)
    last = pn.cal[-1]
    last64 = np.datetime64(last, "D")
    row = len(pn.cal) - 1
    regime = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)) for k, v in pn.regime.iloc[-1].items()
              if k in ("score", "state", "size_mult", "max_positions", "index", "breadth", "vix", "sector", "ad", "dispersion")}
    mult, max_pos = float(pn.regime["size_mult"].iat[-1]), int(pn.regime["max_positions"].iat[-1])
    state = str(pn.regime["state"].iat[-1])
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30

    rows, gated, stale, short = [], Counter(), 0, 0
    by_det = Counter()
    det_errors, det_error_msgs = Counter(), {}                  # a detector that throws is a visible line, never a hole
    bars_cache = {}
    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            short += 1
            continue
        bars_cache[sym] = b
        if b.date[-1] != last64:
            stale += 1
            continue
        cs, errs = detect_all(b, lo=b.n - LIVE_LOOKBACK)          # a window, so the 3-bar cooldown sees yesterday's duplicates
        for name, msg in errs:
            det_errors[name] += 1
            det_error_msgs.setdefault(name, msg[:200])
        cs = [c for c in cs if c.idx == b.n - 1]
        for c in cs:
            by_det[c.detector] += 1
            g = BT.gate(b, c, data.surveillance.get(sym), surv_active)
            if g:
                gated[g] += 1
                continue
            sec = pn.sector_of.get(sym)
            c.sector = sec
            rs = pn.rs_pct.iat[row, pn.rs_pct.columns.get_loc(sym)] if sym in pn.rs_pct.columns else np.nan
            rk = pn.sector_rank.iat[row, pn.sector_rank.columns.get_loc(sec)] if sec in pn.sector_rank.columns else np.nan
            frac = SC.components(b, c, rs, SE.sector_points(rk, int(pn.sector_rank.iloc[row].notna().sum())), state)
            c.score_parts, c.score = frac, SC.total(frac)
            plan = plan_for(b, c, mult)
            if plan is None:
                gated["no_valid_stop"] += 1
                continue
            ev = data.events.get(sym)
            nxt = None
            if ev is not None and len(ev):
                k = np.searchsorted(ev, last64, side="left")
                nxt = str(ev[k]) if k < len(ev) else None
            rows.append({"symbol": sym, "detector": c.detector, "family": c.family, "signal_date": str(last.date()),
                         "score": round(c.score, 1), "quality": round(c.quality, 1), "pivot": round(c.pivot, 2),
                         "sector": sec, "rs_percentile": None if np.isnan(rs) else round(float(rs), 1),
                         "components": {k: round(float(v), 3) for k, v in frac.items() if not k.startswith("_")},
                         "catalyst_source": frac["_catalyst_source"], "next_results": nxt, "plan": plan, "meta": c.meta})
    rows.sort(key=lambda r: -r["score"])

    # hypothetical book from an empty portfolio, using the same caps as the portfolio layer
    sel, rej, heat, sec_n = [], Counter(), 0.0, Counter()
    ret = pn.returns
    for r in rows:
        if len(sel) >= max_pos:
            rej["max_positions"] += 1; r["selected"] = False; r["why_not"] = "max_positions"; continue
        if heat + mult > 6.0 + 1e-9:
            rej["heat_cap"] += 1; r["selected"] = False; r["why_not"] = "heat_cap"; continue
        if r["sector"] is not None and sec_n[r["sector"]] >= 2:
            rej["sector_cap"] += 1; r["selected"] = False; r["why_not"] = "sector_cap"; continue
        if r["symbol"] in {s["symbol"] for s in sel}:
            rej["already_picked"] += 1; r["selected"] = False; r["why_not"] = "already_picked"; continue
        bad = False
        for s in sel:
            a, o = ret[r["symbol"]].iloc[-60:], ret[s["symbol"]].iloc[-60:]
            m = a.notna() & o.notna()
            if m.sum() >= 30 and np.corrcoef(a[m], o[m])[0, 1] > 0.7:
                bad = True
                break
        if bad:
            rej["correlation"] += 1; r["selected"] = False; r["why_not"] = "correlation"; continue
        r["selected"] = True
        sel.append(r); heat += mult
        if r["sector"] is not None:
            sec_n[r["sector"]] += 1

    # ---- forward log: add today's, resolve older ----------------------------------------
    # A forming bar is provisional (partial volume, partial range): it must never enter the forward
    # record that is later scored against real outcomes.
    log_val, _ = _get(conn, LOG_KEY)
    flog = list(log_val or [])
    if provisional:
        flog = []
    have = {(e["s"], e["d"], e["dt"]) for e in flog}
    for r in ([] if provisional else rows):
        key = (r["symbol"], r["detector"], r["signal_date"])
        if key not in have:
            flog.append({"s": r["symbol"], "d": r["detector"], "dt": r["signal_date"], "sc": r["score"], "q": r["quality"],
                         "pv": r["pivot"], "st": r["plan"]["stop"], "sec": r["sector"], "sel": bool(r["selected"]), "fam": r["family"]})
    resolved = 0
    cutoff = (last - pd.Timedelta(days=LOG_KEEP_DAYS)).strftime("%Y-%m-%d")
    for e in flog:
        if "r" in e or e["dt"] >= str(last.date()):
            continue
        b = bars_cache.get(e["s"])
        if b is None:
            continue
        i = int(np.searchsorted(b.date, np.datetime64(e["dt"], "D")))
        if i >= b.n or str(b.date[i]) != e["dt"]:
            continue
        cand = Candidate(e["s"], e["d"], i, b.date[i], e["pv"], e["st"], e["q"], e["fam"])
        tr = MG.simulate(b, cand, data.events.get(e["s"]))
        if tr.skipped:
            e["r"], e["skip"] = None, tr.skipped
            resolved += 1
        elif not tr.open_at_end:
            e["r"], e["bars"], e["reason"] = round(tr.r, 3), tr.bars, tr.reason
            resolved += 1
    flog = [e for e in flog if e["dt"] >= cutoff][-LOG_MAX:]
    if not provisional:
        put(conn, LOG_KEY, flog)

    lag = (pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None) - last.normalize()).days
    out = {"as_of": str(last.date()), "as_of_lag_days": int(lag), "generated_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
           "mode": "shadow", "scan_mode": mode, "provisional": provisional,
           "regime": regime, "universe_symbols": len(data.frames), "universe": (data.coverage or {}).get("universe"),
           "stale_symbols": stale, "symbols_too_short": short, "detectors_run": len(DETECTOR_NAMES),
           "detector_errors": dict(det_errors), "detector_error_samples": det_error_msgs,
           "candidates_raw_by_detector": dict(by_det), "gated": dict(gated),
           "candidates": len(rows), "selected": len(sel), "not_selected_reasons": dict(rej),
           "picks": [r for r in rows if r["selected"]], "all_candidates": rows[:150],
           "forward_log": forward_summary(flog) if not provisional else None, "log_resolved_this_run": resolved,
           "note": "Shadow mode: recorded only. Nothing here is published to signals."}
    if extra:
        out.update(extra)
    put(conn, INTRADAY_KEY if provisional else LATEST_KEY, out)
    return out


def forward_summary(flog: list[dict]) -> dict:
    closed = [e for e in flog if e.get("r") is not None]
    by = defaultdict(list)
    for e in closed:
        by[e["d"]].append(e["r"])
    def agg(v):
        a = np.array(v)
        return {"n": len(a), "avg_r": round(float(a.mean()), 3), "win_rate_pct": round(100 * float((a > 0).mean()), 1)}
    return {"entries": len(flog), "closed": len(closed), "open_or_unresolved": len(flog) - len(closed),
            "overall": agg([e["r"] for e in closed]) if closed else None,
            "selected_only": agg([e["r"] for e in closed if e.get("sel")]) if any(e.get("sel") for e in closed) else None,
            "by_detector": {k: agg(v) for k, v in sorted(by.items())}}
