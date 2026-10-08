"""
e2_backtest.py — the validation run: 3 years x the Nifty 500, every detector on every stock.

Pipeline (each stage reports what it dropped and why; nothing is silent):
  1. load        adjusted OHLCV for the scan universe, index/VIX, industries, results dates
  2. panels      close panel -> relative-strength percentiles, sector composites + rank, regime
  3. per symbol  features -> all 20 detectors -> HARD GATES (liquidity, surveillance, data
                 quality; nothing else rejects) -> component scores -> trade simulation
                 -> independent recognizers for the exhaustiveness count
  4. calibrate   score weights on the first half, judged on the second
  5. portfolio   full window (spec weights, nothing fitted) and held-out half (calibrated),
                 plus the same book without regime modulation
  6. report      one JSON document

Honest limits are written into the report (`caveats`), not left for the reader to discover.
"""
from __future__ import annotations

import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import e2_diagnostics as DG
import e2_portfolio as PF
import e2_regime as RG
import e2_scorer as SC
import e2_sector as SE
from e2_core import Candidate
from e2_detectors import DETECTORS, detect_all
from e2_det_common import WARMUP
from e2_features import MIN_BARS, Bars, compute_features
from e2_manage import Trade, simulate

log = logging.getLogger("engine2")

CODE_VERSION = "e2.1.1"      # 1.1: + portfolio_attribution (measurement only; no detector, score or portfolio rule changed)
MIN_TURNOVER_CR = float(os.environ.get("MIN_TURNOVER_CR", "5"))      # same floor as the live engine's gate 0
WINDOW_BARS = int(os.environ.get("E2_WINDOW_BARS", "756"))           # ~3 years of sessions
EXTRA_DAYS = 620                                                      # calendar days of history before the window


@dataclass
class Data:
    frames: dict[str, pd.DataFrame]
    index_close: pd.Series | None
    index_symbol: str | None
    vix: pd.Series | None
    industry: dict[str, str | None]
    sector_col: dict[str, str | None]
    events: dict[str, np.ndarray]
    surveillance: dict[str, np.ndarray]          # symbol -> sorted datetime64[D] of flagged snapshots
    coverage: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# 1. load
# ----------------------------------------------------------------------
def load_from_db(conn, window_bars: int = WINDOW_BARS) -> Data:
    from universe import load_scan_universe
    uni = load_scan_universe(conn)
    syms = uni.symbols
    cov: dict = {"universe": uni.detail}
    with conn.cursor() as cur:
        cur.execute("select max(trade_date) from ohlcv_daily where symbol = 'NIFTY50'")
        end = cur.fetchone()[0]
        start = pd.Timestamp(end) - pd.Timedelta(days=int(window_bars * 7 / 5) + EXTRA_DAYS)
        cur.execute("""select symbol, trade_date, adj_open, adj_high, adj_low, adj_close, volume
                       from ohlcv_daily where symbol = any(%s) and trade_date >= %s
                       order by symbol, trade_date""", (syms, start.date()))
        rows = cur.fetchall()
        idx_close, idx_sym = None, None
        for cand in ("NIFTY500", "NIFTY50"):
            cur.execute("select trade_date, adj_close from ohlcv_daily where symbol = %s and trade_date >= %s order by trade_date",
                        (cand, start.date()))
            r = cur.fetchall()
            if len(r) > 250:
                idx_close = pd.Series([float(x[1]) for x in r], index=pd.to_datetime([x[0] for x in r]))
                idx_sym = cand
                break
        cur.execute("select trade_date, adj_close from ohlcv_daily where symbol = 'INDIAVIX' and trade_date >= %s order by trade_date",
                    (start.date(),))
        r = cur.fetchall()
        vix = pd.Series([float(x[1]) for x in r], index=pd.to_datetime([x[0] for x in r])) if len(r) > 100 else None
        cur.execute("select symbol, industry, sector from symbols where symbol = any(%s)", (syms,))
        ind, sec = {}, {}
        for s_, i_, c_ in cur.fetchall():
            ind[s_], sec[s_] = i_, c_
        events: dict[str, np.ndarray] = {}
        cov["results_calendar"] = {"rows": 0}
        try:
            cur.execute("select symbol, event_date from events_calendar where event_type = 'results' and symbol = any(%s) order by event_date", (syms,))
            ev = defaultdict(list)
            for s_, d_ in cur.fetchall():
                ev[s_].append(np.datetime64(d_, "D"))
            events = {k: np.array(v, dtype="datetime64[D]") for k, v in ev.items()}
            cov["results_calendar"] = {"rows": sum(len(v) for v in events.values()), "symbols": len(events),
                                       "earliest": str(min(v[0] for v in events.values())) if events else None,
                                       "latest": str(max(v[-1] for v in events.values())) if events else None}
        except Exception as e:                       # table missing/empty is a coverage fact, not a crash
            conn.rollback()
            cov["results_calendar"] = {"error": str(e)[:160]}
        surv: dict[str, np.ndarray] = {}
        cov["surveillance"] = {"rows": 0}
        try:
            cur.execute("select symbol, as_of from surveillance where symbol = any(%s) and list_type in ('ASM','GSM') order by as_of", (syms,))
            sv = defaultdict(list)
            n = 0
            for s_, d_ in cur.fetchall():
                sv[s_].append(np.datetime64(d_, "D")); n += 1
            surv = {k: np.array(v, dtype="datetime64[D]") for k, v in sv.items()}
            cur.execute("select count(distinct as_of), min(as_of), max(as_of) from surveillance")
            nd, d0, d1 = cur.fetchone()
            cov["surveillance"] = {"flag_rows": n, "snapshot_days": int(nd or 0), "earliest": str(d0), "latest": str(d1)}
        except Exception as e:
            conn.rollback()
            cov["surveillance"] = {"error": str(e)[:160]}
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r[0]].append(r[1:])
    frames = {}
    for s_, v in by.items():
        df = pd.DataFrame(v, columns=["trade_date", "open", "high", "low", "close", "volume"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        frames[s_] = df
    cov["rows_loaded"] = len(rows)
    cov["symbols_with_prices"] = len(frames)
    cov["universe_symbols_without_prices"] = sorted(set(syms) - set(frames))[:50]
    return Data(frames, idx_close, idx_sym, vix, ind, sec, events, surv, cov)


# ----------------------------------------------------------------------
# 2. panels
# ----------------------------------------------------------------------
@dataclass
class Panels:
    cal: pd.DatetimeIndex
    close: pd.DataFrame
    returns: pd.DataFrame
    rs_pct: pd.DataFrame
    sector_of: dict[str, str | None]
    sector_comp: pd.DataFrame
    sector_rank: pd.DataFrame
    regime: pd.DataFrame
    row_of: dict


def build_panels(data: Data) -> Panels:
    cols = {s: df.set_index("trade_date")["close"] for s, df in data.frames.items() if len(df) >= 60}
    close = pd.DataFrame(cols).sort_index()
    if data.index_close is not None:
        close = close.reindex(close.index.union(data.index_close.index)).sort_index()
        close = close[close.index.isin(data.index_close.index)] if len(data.index_close) > 250 else close
    cal = close.index
    ret = close.pct_change(fill_method=None)
    rs_raw = (0.4 * (close / close.shift(63) - 1) + 0.3 * (close / close.shift(126) - 1)
              + 0.3 * (close / close.shift(252) - 1)).where(close.shift(252).notna(),
                                                           0.57 * (close / close.shift(63) - 1) + 0.43 * (close / close.shift(126) - 1))
    rs_pct = rs_raw.rank(axis=1, pct=True) * 100.0
    sector_of = {s: SE.sector_of(data.industry.get(s), data.sector_col.get(s)) for s in close.columns}
    comp = SE.composites(close, sector_of)
    rank = SE.rs_rank(comp)
    ix = data.index_close if data.index_close is not None else close.mean(axis=1)
    regime = RG.regime_frame(ix, data.vix, close, comp)
    return Panels(cal, close, ret, rs_pct, sector_of, comp, rank, regime, {np.datetime64(d, "D"): k for k, d in enumerate(cal)})


# ----------------------------------------------------------------------
# 3. per-symbol pass
# ----------------------------------------------------------------------
def _bar_ok(b: Bars, i: int) -> bool:
    for j in (i, i + 1):
        if j >= b.n:
            continue
        if not (b.c[j] > 0 and b.o[j] > 0 and b.h[j] >= b.l[j] and b.l[j] > 0 and b.v[j] >= 0) or np.isnan(b.c[j]):
            return False
    return b.v[i] > 0


def gate(b: Bars, cand: Candidate, surv: np.ndarray | None, surv_active: bool) -> str | None:
    i = cand.idx
    t = b.turnover20_cr[i]
    if np.isnan(t) or t < MIN_TURNOVER_CR:
        return "liquidity"
    if not _bar_ok(b, i):
        return "data_quality"
    if surv_active and surv is not None and len(surv):
        d = b.date[i]
        k = np.searchsorted(surv, d, side="right") - 1
        if k >= 0 and (d - surv[k]).astype(int) <= 7:
            return "surveillance"
    return None


def per_symbol(data: Data, pn: Panels, window_start, progress=None) -> dict:
    out = dict(trades=[], cand_counts=Counter(), gated=Counter(), gated_by_det=defaultdict(Counter),
               errors=defaultdict(Counter), short=[], compare={}, catalyst_src=Counter(), rs_missing=0,
               n_candidates=0, floored=0, skip_reasons=Counter(), unknown_sector=0, symbols_run=0)
    surv_days = data.coverage.get("surveillance", {}).get("snapshot_days", 0)
    surv_active = surv_days >= 30
    n_sym = len(data.frames)
    for k, (sym, df) in enumerate(sorted(data.frames.items())):
        b = compute_features(sym, df)
        if b is None:
            out["short"].append(sym)
            continue
        lo = int(np.searchsorted(b.date, np.datetime64(window_start, "D")))
        lo = max(lo, WARMUP)
        cands, errs = detect_all(b, lo=lo)
        out["symbols_run"] += 1
        for name, msg in errs:
            out["errors"][name][msg] += 1
        sec = pn.sector_of.get(sym)
        if sec is None:
            out["unknown_sector"] += 1
        det_idx: dict[str, list[int]] = defaultdict(list)
        for c in cands:
            out["cand_counts"][c.detector] += 1
            det_idx[c.detector].append(c.idx)
            g = gate(b, c, data.surveillance.get(sym), surv_active)
            if g:
                out["gated"][g] += 1
                out["gated_by_det"][c.detector][g] += 1
                continue
            out["n_candidates"] += 1
            c.sector = sec
            row = pn.row_of.get(b.date[c.idx])
            if row is None:
                out["gated"]["date_not_in_calendar"] += 1
                continue
            rs = pn.rs_pct.iat[row, pn.rs_pct.columns.get_loc(sym)] if sym in pn.rs_pct.columns else np.nan
            rk = np.nan
            if sec is not None and sec in pn.sector_rank.columns:
                rk = pn.sector_rank.iat[row, pn.sector_rank.columns.get_loc(sec)]
            n_sec = int(pn.sector_rank.iloc[row].notna().sum())
            state = pn.regime["state"].iat[row]
            frac = SC.components(b, c, rs, SE.sector_points(rk, n_sec), state)
            out["catalyst_src"][frac["_catalyst_source"]] += 1
            out["rs_missing"] += int(frac["_rs_missing"])
            c.score_parts = frac
            c.score = SC.total(frac)
            tr = simulate(b, c, data.events.get(sym))
            tr.frac, tr.regime_state, tr.score = frac, state, c.score
            if tr.skipped:
                out["skip_reasons"][tr.skipped] += 1
            elif tr.floored:
                out["floored"] += 1
            out["trades"].append(tr)
        hi = b.n
        out["compare"][sym] = {name: DG.compare_symbol(b, name, det_idx.get(name, []), lo, hi) for name in DETECTORS}
        if progress and k % 25 == 0:
            progress(f"symbols {k}/{n_sym}")
    return out


# ----------------------------------------------------------------------
# 4-5. calibrate and book
# ----------------------------------------------------------------------
def rescore(trades: list[Trade], weights: dict) -> None:
    for t in trades:
        t.score = SC.total(t.frac, weights)


def stats_r(trs: list[Trade]) -> dict:
    live = [t for t in trs if not t.skipped]
    if not live:
        return {"n": 0}
    r = np.array([t.r for t in live])
    w, l = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": len(live), "win_rate_pct": round(100 * float((r > 0).mean()), 1), "avg_r": round(float(r.mean()), 3),
            "median_r": round(float(np.median(r)), 3), "profit_factor_r": round(float(w / l), 2) if l > 0 else None,
            "avg_bars": round(float(np.mean([t.bars for t in live])), 1),
            "avg_mfe_r": round(float(np.mean([t.mfe_r for t in live])), 2), "avg_mae_r": round(float(np.mean([t.mae_r for t in live])), 2),
            "stopped_within_5_bars_pct": round(100 * float(np.mean([(t.reason in DG.STOP_REASONS and t.bars <= DG.FAIL_BARS) for t in live])), 1),
            "open_at_end": int(sum(t.open_at_end for t in live))}


def run_study(data: Data, progress=None, window_bars: int = WINDOW_BARS) -> dict:
    t0 = time.time()
    report: dict = {"code_version": CODE_VERSION, "started": pd.Timestamp.utcnow().isoformat(timespec="seconds")}
    prog = progress or (lambda m: None)
    prog("building panels")
    pn = build_panels(data)
    if len(pn.cal) < 300:
        raise RuntimeError(f"calendar too short for a study: {len(pn.cal)} sessions")
    w0 = pn.cal[max(0, len(pn.cal) - window_bars)]
    w1 = pn.cal[-1]
    mid = pn.cal[max(0, len(pn.cal) - window_bars // 2)]
    report["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(w1.date()), "sessions": int(len(pn.cal[pn.cal >= w0]))}
    prog("scanning symbols")
    ps = per_symbol(data, pn, w0, prog)
    trades: list[Trade] = ps["trades"]
    live = [t for t in trades if not t.skipped]
    split64 = np.datetime64(mid, "D")

    # ---- calibration (train = first half, test = second half) -------------------------
    prog("calibrating scorer")
    tr_ = [(t.frac, t.r) for t in live if np.datetime64(t.signal_date, "D") < split64]
    te_ = [(t.frac, t.r) for t in live if np.datetime64(t.signal_date, "D") >= split64]
    cal = SC.calibrate(tr_, te_)
    rescore(trades, SC.SPEC_WEIGHTS)
    spec_scores = {id(t): t.score for t in trades}

    # ---- portfolio runs ----------------------------------------------------------------
    prog("portfolio simulation")
    cal_idx = pn.cal
    full = PF.run(trades, pn.regime, pn.returns, cal_idx, PF.Params(), start=w0, end=w1)
    no_regime = PF.run(trades, pn.regime, pn.returns, cal_idx, PF.Params(use_regime=False), start=w0, end=w1)
    test_spec = PF.run(trades, pn.regime, pn.returns, cal_idx, PF.Params(), start=mid, end=w1)
    rescore(trades, cal["weights"])
    test_cal = PF.run(trades, pn.regime, pn.returns, cal_idx, PF.Params(), start=mid, end=w1)
    rescore(trades, SC.SPEC_WEIGHTS)

    def book(res: PF.Result) -> dict:
        eq = res.equity
        mo = eq.resample("ME").last() if len(eq) else eq
        return {"metrics": res.metrics, "rejected": dict(res.rejected), "drawdown_halts": res.halts, "halted_sessions": res.halted_sessions,
                "avg_positions": round(float(res.daily["positions"].mean()), 2) if len(res.daily) else None,
                "avg_exposure_pct": round(100 * float(res.daily["exposure"].mean()), 1) if len(res.daily) else None,
                "equity_monthly": [[str(d.date()), round(float(v), 0)] for d, v in mo.items()]}

    def attribution(res: PF.Result) -> dict:
        """Which detectors / regimes the portfolio's trades actually came from, and what each contributed. Measurement only."""
        def agg(rs: list[float]) -> dict:
            a = np.asarray(rs, float)
            return {"n": len(rs), "avg_r": round(float(a.mean()), 3), "total_r": round(float(a.sum()), 2),
                    "win_rate_pct": round(100 * float((a > 0).mean()), 1)}
        det, reg = defaultdict(list), defaultdict(list)
        for x in res.taken:
            if "pnl" not in x:
                continue
            t = x["trade"]
            det[t.detector].append(t.r)
            reg[str(t.regime_state)].append(t.r)
        return {"by_detector": {k: agg(v) for k, v in sorted(det.items(), key=lambda kv: -len(kv[1]))},
                "by_regime": {k: agg(v) for k, v in sorted(reg.items())},
                "closed_trades": sum(len(v) for v in det.values())}

    ix_ret = None
    if data.index_close is not None:
        ic = data.index_close.reindex(pn.cal).ffill()
        ic = ic[ic.index >= w0]
        ix_ret = round(100 * float(ic.iloc[-1] / ic.iloc[0] - 1), 1)

    # ---- per-detector exhaustiveness + performance -------------------------------------
    prog("detector diagnostics")
    by_det: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_det[t.detector].append(t)
    diag = DG.aggregate(ps["compare"], by_det)
    det_rows = {}
    for name in DETECTORS:
        trs = by_det.get(name, [])
        st = stats_r(trs)
        half = {"first": stats_r([t for t in trs if np.datetime64(t.signal_date, "D") < split64]),
                "second": stats_r([t for t in trs if np.datetime64(t.signal_date, "D") >= split64])}
        reg = {s: stats_r([t for t in trs if t.regime_state == s]) for s in ("risk_off", "neutral", "risk_on")}
        det_rows[name] = {"candidates_raw": ps["cand_counts"].get(name, 0), "gated_out": dict(ps["gated_by_det"].get(name, {})),
                          "exhaustiveness": diag[name], "performance_all_candidates": st, "by_half": half, "by_regime": reg,
                          "errors": dict(ps["errors"].get(name, {}))}
    # ---- score ranks outcomes? -------------------------------------------------------
    sc_rows = []
    if live:
        arr = np.array([t.score for t in live]); rr = np.array([t.r for t in live])
        qs = np.quantile(arr, [0, .2, .4, .6, .8, 1.0])
        for a, b_ in zip(qs[:-1], qs[1:]):
            m = (arr >= a) & (arr <= b_)
            sc_rows.append({"score_from": round(float(a), 1), "score_to": round(float(b_), 1), "n": int(m.sum()),
                            "avg_r": round(float(rr[m].mean()), 3), "win_rate_pct": round(100 * float((rr[m] > 0).mean()), 1)})
    # ---- signals per day (an OUTPUT) ----------------------------------------------------
    per_day = Counter(np.datetime64(t.signal_date, "D") for t in trades)
    days = len(pn.cal[pn.cal >= w0])
    vals = np.array([per_day.get(np.datetime64(d, "D"), 0) for d in pn.cal[pn.cal >= w0]])
    rg = pn.regime.loc[pn.regime.index >= w0]

    report.update({
        "universe": data.coverage.get("universe"),
        "data_coverage": {k: v for k, v in data.coverage.items() if k != "universe"},
        "symbols_run": ps["symbols_run"], "symbols_too_short": ps["short"][:50], "n_symbols_too_short": len(ps["short"]),
        "index_used": data.index_symbol, "index_return_over_window_pct": ix_ret,
        "gates": {"liquidity_floor_cr": MIN_TURNOVER_CR, "rejected": dict(ps["gated"]),
                  "surveillance_applied": data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30,
                  "candidates_after_gates": ps["n_candidates"]},
        "trade_skips": dict(ps["skip_reasons"]), "stops_raised_to_1atr_floor": ps["floored"],
        "detectors": det_rows,
        "regime": {"state_share_pct": {k: round(100 * float((rg["state"] == k).mean()), 1) for k in ("risk_off", "neutral", "risk_on")},
                   "avg_score": round(float(rg["score"].mean()), 1), "vix_known": bool(rg["vix_known"].iloc[-1]),
                   "last": {k: (round(float(v), 1) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in rg.iloc[-1].items()
                            if k in ("score", "state", "index", "breadth", "vix", "sector", "ad", "dispersion", "size_mult", "max_positions")}},
        "sector": {"source": "equal_weight_composite_of_universe", "sectors_found": list(pn.sector_comp.columns),
                   "stocks_without_sector": ps["unknown_sector"],
                   "unmapped_industries": sorted({str(data.industry.get(s)) for s, v in pn.sector_of.items() if v is None})[:30],
                   "last_rank": {k: int(v) for k, v in pn.sector_rank.iloc[-1].dropna().items()}},
        "scoring": {"calibration": cal, "r_by_score_quintile": sc_rows, "catalyst_source": dict(ps["catalyst_src"]),
                    "rs_missing": ps["rs_missing"]},
        "signals": {"total_candidates_all_detectors": len(trades), "per_day_mean": round(float(vals.mean()), 1),
                    "per_day_median": float(np.median(vals)), "per_day_p90": float(np.quantile(vals, 0.9)),
                    "per_day_max": int(vals.max()), "days": int(days),
                    "by_regime_mean": {k: round(float(vals[(rg["state"] == k).values].mean()), 1) if (rg["state"] == k).any() else None
                                       for k in ("risk_off", "neutral", "risk_on")},
                    "taken_by_portfolio_full_window": full.metrics.get("trades_taken")},
        "portfolio": {"full_window_spec_weights": book(full), "full_window_no_regime_modulation": book(no_regime),
                      "test_half_spec_weights": book(test_spec),
                      "test_half_calibrated_weights": book(test_cal),
                      "attribution": {"full_window_spec_weights": attribution(full), "test_half_spec_weights": attribution(test_spec)},
                      "note": "full_window uses the spec's weights: nothing in it was fitted. test_half_* start at the split date with a fresh book."},
        "caveats": CAVEATS,
        "runtime_sec": round(time.time() - t0, 1),
    })
    return report


CAVEATS = [
    "SURVIVORSHIP: the universe is today's Nifty 500 constituents with their full history. Stocks that were delisted, merged or dropped from the index are absent, which flatters every number here.",
    "UNIVERSE LOOK-AHEAD: membership is as of today, not as of each signal date.",
    "COSTS are an assumption (0.30% round trip + 0.10% slippage per side) applied uniformly; real impact on thin names is larger.",
    "INTRABAR order is unknown from daily bars: a bar touching both stop and target counts as a STOP, and a stop is assumed to fill at the stop price unless the bar opened beyond it.",
    "SECTOR strength comes from equal-weighted composites of the universe because NSE sector indices are not in the database.",
    "CATALYST history (news/results) does not exist for the window: episodic pivots and the catalyst score use a gap-and-volume proxy and say so (`catalyst_source`).",
    "SURVEILLANCE (ASM/GSM) is applied only if the table holds >= 30 daily snapshots; `gates.surveillance_applied` says whether it was.",
    "RESULTS-DATE exits apply only where events_calendar has rows (`data_coverage.results_calendar`).",
    "RECOGNIZERS are a second implementation by the same author from the same written spec: they catch implementation errors, not a shared misreading of the spec.",
    "Long-only: no borrow is modelled; descending triangles are traded on the upside break only.",
    "Score calibration uses the first half only and is adopted only if it ranks the second half at least as well as the spec weights.",
    "Detector thresholds were fixed from the written definitions before any result existed; no threshold was tuned toward a count.",
]
