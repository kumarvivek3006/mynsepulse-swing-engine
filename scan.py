"""
Scan runner — walks the universe through the sequential gates.

Loads every symbol's bars once, computes the market regime, then applies
Gate 0, Gate 2 (skipped, see below) and Gate 3 in order. Both passes and
rejections are written to gate_log: the discard pile is the only way to
answer "why didn't my stock show up", and the only way to tell an
over-tight filter from a genuinely quiet market.

Gate 2 is logged as skipped on every symbol rather than omitted. A missing
gate that leaves no trace looks identical to a gate that passed everything.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta

import pandas as pd

from gates import (
    add_indicators,
    evaluate_regime,
    gate0_tradability,
    gate3_trend_structure,
    relative_strength,
)
from ingest import connect
from setups import Rejected, build_setup

log = logging.getLogger(__name__)

INDEX_SYMBOLS = ("NIFTY50", "NIFTY500", "INDIAVIX")
MIN_BARS = 210          # enough for a 200 DMA plus slope
FUNDAMENTALS_READY = False   # flip when NSE XBRL ingestion exists
CAPITAL = float(os.environ.get("SWING_CAPITAL", "0") or 0)
RISK_PCT = float(os.environ.get("RISK_PCT", "1.0"))
SIGNAL_EXPIRY_SESSIONS = int(os.environ.get("SIGNAL_EXPIRY_SESSIONS", "5"))


def _load_frames(conn) -> dict[str, pd.DataFrame]:
    """One query for the whole universe; split in pandas rather than 500 round trips."""
    query = """
        select symbol, trade_date, adj_open, adj_high, adj_low, adj_close, volume
        from ohlcv_daily
        order by symbol, trade_date
    """
    df = pd.read_sql(query, conn)
    df.columns = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return {sym: g.reset_index(drop=True) for sym, g in df.groupby("symbol")}


def _breadth(frames: dict[str, pd.DataFrame], universe: list[str]) -> float:
    above = total = 0
    for sym in universe:
        df = frames.get(sym)
        if df is None or len(df) < 60:
            continue
        sma50 = df["close"].rolling(50).mean().iloc[-1]
        if pd.isna(sma50):
            continue
        total += 1
        above += int(df["close"].iloc[-1] > sma50)
    return (above / total * 100) if total else 0.0


def run_scan(as_of: date | None = None) -> dict:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    as_of = as_of or date.today()
    conn = connect()

    try:
        with conn.cursor() as cur:
            cur.execute("""
                select s.symbol,
                       coalesce(bool_or(v.symbol is not null), false) as flagged
                from symbols s
                left join surveillance v
                       on v.symbol = s.symbol
                      and v.as_of >= current_date - 5
                where s.is_active
                  and coalesce(s.series, '') <> 'INDEX'
                group by s.symbol
                order by s.symbol
            """)
            rows = cur.fetchall()

        universe = [r[0] for r in rows]
        flagged = {r[0] for r in rows if r[1]}
        log.info("Universe: %d symbols, %d under surveillance", len(universe), len(flagged))

        frames = _load_frames(conn)

        # --- Gate 1: regime, once for the whole run -------------------
        nifty = frames.get("NIFTY50")
        vix = frames.get("INDIAVIX")
        if nifty is None or len(nifty) < 60:
            raise RuntimeError("NIFTY50 series missing or too short — cannot assess regime")

        breadth = _breadth(frames, universe)
        regime = evaluate_regime(nifty, vix if vix is not None else pd.DataFrame(), breadth)
        log.info("Regime: %s (breadth %.1f%%, vix %s, distribution %d)",
                 regime["state"], breadth, regime["vix"], regime["distribution_days"])

        with conn.cursor() as cur:
            cur.execute("""
                insert into market_regime
                    (as_of, state, nifty_close, nifty_vs_20dma, nifty_vs_50dma,
                     breadth_above_50dma, vix, vix_10d_change, distribution_days, notes)
                values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                on conflict (as_of) do update set
                    state = excluded.state,
                    nifty_close = excluded.nifty_close,
                    breadth_above_50dma = excluded.breadth_above_50dma,
                    vix = excluded.vix,
                    distribution_days = excluded.distribution_days,
                    notes = excluded.notes
            """, (as_of, regime["state"], regime["nifty_close"], regime["nifty_vs_20dma"],
                  regime["nifty_vs_50dma"], regime["breadth_above_50dma"], regime["vix"],
                  regime["vix_10d_change"], regime["distribution_days"],
                  json.dumps(regime["notes"])))
            cur.execute("delete from gate_log where as_of_date = %s", (as_of,))
        conn.commit()

        # --- Gates 0 / 2 / 3 per symbol -------------------------------
        counts: dict[str, int] = {}
        signals: list[dict] = []
        log_rows: list[tuple] = []

        for sym in universe:
            df = frames.get(sym)
            if df is None or len(df) < MIN_BARS:
                counts["no_data"] = counts.get("no_data", 0) + 1
                log_rows.append((as_of, sym, "gate0", "insufficient_history",
                                 json.dumps({"bars": 0 if df is None else len(df)})))
                continue

            df = add_indicators(df)

            r0 = gate0_tradability(sym, df, sym in flagged)
            if not r0.passed:
                counts[r0.reason] = counts.get(r0.reason, 0) + 1
                log_rows.append((as_of, sym, "gate0", r0.reason, json.dumps(r0.detail)))
                continue

            if not FUNDAMENTALS_READY:
                # Recorded, not enforced. See module docstring.
                log_rows.append((as_of, sym, None, "gate2_skipped",
                                 json.dumps({"reason": "fundamentals not ingested"})))

            r3 = gate3_trend_structure(sym, df)
            if not r3.passed:
                counts[r3.reason] = counts.get(r3.reason, 0) + 1
                log_rows.append((as_of, sym, "gate3", r3.reason, json.dumps(r3.detail)))
                continue

            rs63 = relative_strength(df, nifty, 63)
            rs126 = relative_strength(df, nifty, 126)
            last = df.iloc[-1]
            counts["passed_gates_0_3"] = counts.get("passed_gates_0_3", 0) + 1

            # Gates 4-7. In a risk-off regime the correct output is nothing,
            # so setups are not even constructed — an empty list is the
            # answer, not a failure.
            if regime["state"] == "risk_off":
                log_rows.append((as_of, sym, "gate1", "regime_risk_off", "{}"))
                continue

            try:
                setup = build_setup(sym, df, rs63, rs126, FUNDAMENTALS_READY)
            except Rejected as rej:
                counts[rej.reason] = counts.get(rej.reason, 0) + 1
                log_rows.append((as_of, sym, rej.gate, rej.reason,
                                 json.dumps(rej.detail)))
                continue

            risk_per_share = setup.entry - setup.stop
            qty = risk_amount = None
            if CAPITAL > 0 and risk_per_share > 0:
                budget = CAPITAL * (RISK_PCT / 100.0)
                if regime["state"] == "neutral":
                    budget /= 2          # half size in a mixed tape
                qty = int(budget // risk_per_share)
                cap_qty = int((CAPITAL * 0.10) // setup.entry)
                qty = max(0, min(qty, cap_qty))
                risk_amount = round(qty * risk_per_share, 2)

            signals.append({
                "symbol": sym,
                "setup_type": setup.setup_type,
                "pattern": setup.pattern,
                "entry_trigger": setup.entry,
                "stop_loss": setup.stop,
                "t1": setup.t1,
                "t2": setup.t2,
                "r_multiple_t1": setup.r_multiple_t1,
                "qty_suggested": qty,
                "risk_amount": risk_amount,
                "score_total": setup.score_total,
                "score_breakdown": setup.score_breakdown,
                "pivot_bar_date": str(df["trade_date"].iloc[setup.base.pivot_idx])[:10],
                "base_start_date": str(df["trade_date"].iloc[setup.base.start_idx])[:10],
                "base_low": setup.base.base_low,
                "close": round(float(last["close"]), 2),
                "atr14": round(float(last["atr14"]), 2),
                "rs63": rs63,
                "rs126": rs126,
                "stop_basis": setup.stop_basis,
                "t1_basis": setup.t1_basis,
                "notes": setup.notes,
            })
            counts["signal"] = counts.get("signal", 0) + 1
            log_rows.append((as_of, sym, None, "signal_generated",
                             json.dumps({"score": setup.score_total,
                                         "rr": setup.r_multiple_t1})))

        signals.sort(key=lambda s: -s["score_total"])

        with conn.cursor() as cur:
            cur.executemany(
                "insert into gate_log (as_of_date, symbol, failed_gate, reason_code, detail) "
                "values (%s,%s,%s,%s,%s)", log_rows)

            # Supersede today's previous run rather than accumulating
            # duplicates; untriggered signals from earlier days keep running
            # to their own expiry.
            cur.execute("delete from signals where as_of_date = %s and status = 'pending'",
                        (as_of,))

            for s_ in signals:
                cur.execute("""
                    insert into signals
                        (symbol, as_of_date, setup_type, pattern, entry_trigger,
                         stop_loss, t1, t2, r_multiple_t1, qty_suggested, risk_amount,
                         score_total, score_breakdown, pivot_bar_date, base_start_date,
                         base_low, regime_state, notes, status, expires_on)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                            'pending', %s)
                """, (s_["symbol"], as_of, s_["setup_type"], s_["pattern"],
                      s_["entry_trigger"], s_["stop_loss"], s_["t1"], s_["t2"],
                      s_["r_multiple_t1"], s_["qty_suggested"], s_["risk_amount"],
                      s_["score_total"], json.dumps(s_["score_breakdown"]),
                      s_["pivot_bar_date"], s_["base_start_date"], s_["base_low"],
                      regime["state"],
                      json.dumps({"stop_basis": s_["stop_basis"],
                                  "t1_basis": s_["t1_basis"],
                                  "rs63": s_["rs63"], "rs126": s_["rs126"],
                                  "close": s_["close"], "atr14": s_["atr14"],
                                  "notes": s_["notes"]}),
                      as_of + timedelta(days=SIGNAL_EXPIRY_SESSIONS * 2)))

            # Age out anything past its window.
            cur.execute("update signals set status = 'expired' "
                        "where status = 'pending' and expires_on < %s", (as_of,))
        conn.commit()

        summary = {
            "as_of": str(as_of),
            "regime": regime["state"],
            "breadth_above_50dma": regime["breadth_above_50dma"],
            "vix": regime["vix"],
            "distribution_days": regime["distribution_days"],
            "universe": len(universe),
            "passed_structure": counts.get("passed_gates_0_3", 0),
            "signals": len(signals),
            "rejections": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "gate2_enforced": FUNDAMENTALS_READY,
        }
        log.info("Scan complete: %d signals from %d structural candidates (%s regime)",
                 len(signals), counts.get("passed_gates_0_3", 0), regime["state"])
        return summary

    finally:
        conn.close()


if __name__ == "__main__":
    import pprint
    pprint.pprint(run_scan())
