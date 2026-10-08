"""
e2_publish.py — Engine 2's publishing path: scan output -> the live `signals` table the UI reads.

Used ONLY when ENGINE_MODE=engine2 (see engine_mode.py). In legacy mode nothing here runs.

The UI needs no change: the Swing Desk reads the Railway /signals endpoint, which serves rows of
`signals` (pending / triggered) with a band derived from score_total, plus market_regime and the
engine_settings.last_scan_summary the /jobs/status route returns. Engine 2 writes exactly those
shapes. What differs is stated in each row's notes (engine = engine2, detector, plan, is_provisional).

What a row means (Engine 2 trades the CONFIRMED bar, not an armed level):
  entry_trigger  reference price = signal-bar close plus slippage; the plan is a market order at the
                 NEXT open, so this is where the entry is expected, not a stop-entry level
  stop_loss      structural stop, floored at 1 ATR (never looser than the setup's invalidation)
  t1 / t2        2R (scale 50%, stop to break-even) / 3R (scale 25%), last 25% on the trail
  expires_on     E2_SIGNAL_EXPIRY_DAYS after the signal date (default 4): it is a next-open entry

Provisional (hourly intraday) rows: a forming bar has partial volume. They are published with
notes.is_provisional = true so the UI flags them, and the definitive scan resolves each one:
confirmed -> same row updated in place with the final levels, is_provisional false;
not confirmed at the close -> status 'expired', notes.withdrawn says why. A row the user has TAKEN
(status triggered) is never modified.

Everything is one transaction: a publish either lands whole or not at all. The kill switch is
re-checked immediately before writing. A row that would break the UI (non-finite numbers, entry not
above stop, t1 not above entry, unknown symbol) is NOT written; it is counted in `malformed` and
logged at ERROR, and the rest of the scan still publishes.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import date, datetime, timedelta

log = logging.getLogger("engine2.publish")

EXPIRY_DAYS = int(os.environ.get("E2_SIGNAL_EXPIRY_DAYS", "4"))
MAX_POSITION_FRAC = 0.20
T1_NOTE = "2R: book 50%, stop to break-even"
T2_NOTE = "3R: book 25%, trail the last 25%"


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _settings(conn) -> tuple[float | None, float | None]:
    """Capital and risk % per trade exactly as the user set them in the UI (engine_settings)."""
    with conn.cursor() as cur:
        cur.execute("select key, value from engine_settings where key in ('capital', 'risk_pct')")
        d = {k: _num(v) for k, v in cur.fetchall()}
    return d.get("capital"), d.get("risk_pct")


def setup_type_for(family: str, detector: str) -> str:
    """The UI/legacy vocabulary for setup_type; the exact detector is in `pattern`, the family in notes."""
    return "pullback" if family == "continuation" else "breakout"


def build_row(pick: dict, *, as_of: date, provisional: bool, regime_state: str, size_mult: float,
              capital: float | None, risk_pct: float | None, last_close: float | None) -> tuple[dict | None, str | None]:
    """(row, None) or (None, why_malformed)."""
    plan = pick.get("plan") or {}
    entry, stop = _num(plan.get("entry_ref")), _num(plan.get("stop"))
    t1, t2 = _num(plan.get("t1")), _num(plan.get("t2"))
    score = _num(pick.get("score"))
    if None in (entry, stop, t1, score) or entry <= 0 or stop <= 0:
        return None, "non_finite_or_missing_level"
    if not stop < entry < t1:
        return None, f"levels_out_of_order stop={stop} entry={entry} t1={t1}"
    if t2 is not None and t2 <= t1:
        t2 = None
    if not 0 <= score <= 100:
        return None, f"score_out_of_range {score}"
    risk_ps = entry - stop
    if capital and risk_pct:
        risk_amt = capital * risk_pct / 100.0 * (size_mult if size_mult and size_mult > 0 else 1.0)
        qty = int(math.floor(min(risk_amt / risk_ps, MAX_POSITION_FRAC * capital / entry)))
    else:
        qty = int(plan.get("shares") or 0)
    qty = max(qty, 0)
    det = pick["detector"]
    notes_list = [f"Engine 2 {det} confirmed on {pick.get('signal_date')} (quality {pick.get('quality')}).",
                  f"Entry: market order at the next open; reference {entry}.",
                  f"Stop {stop} ({plan.get('risk_pct')}% risk{'; raised to 1 ATR floor' if plan.get('stop_raised_to_1atr') else ''}).",
                  f"T1 {t1} = {T1_NOTE}." + (f" T2 {t2} = {T2_NOTE}." if t2 else ""),
                  f"Trail: {plan.get('trail')}.", f"Time stop: {plan.get('time_stop')}."]
    if pick.get("next_results"):
        notes_list.append(f"Next results date: {pick['next_results']}.")
    if provisional:
        notes_list.insert(0, "PROVISIONAL: built on a forming intraday bar (partial volume); confirmed or withdrawn at the close.")
    notes = {"engine": "engine2", "detector": det, "family": pick.get("family"), "stop_basis": "structural stop (1 ATR floor)",
             "t1_basis": T1_NOTE, "t2_basis": T2_NOTE if t2 else None, "rs_rank_pct": pick.get("rs_percentile"),
             "is_add_on": False, "is_transition": False, "is_provisional": provisional, "is_new_opportunity": False,
             "close": last_close, "sector": pick.get("sector"), "pivot": pick.get("pivot"), "plan": plan,
             "catalyst_source": pick.get("catalyst_source"), "next_results": pick.get("next_results"),
             "detector_meta": pick.get("meta"), "notes": notes_list}
    return {
        "symbol": pick["symbol"], "as_of": as_of, "setup_type": setup_type_for(pick.get("family") or "", det), "pattern": det,
        "entry_trigger": round(entry, 2), "stop_loss": round(stop, 2), "t1": round(t1, 2), "t2": round(t2, 2) if t2 else None,
        "r_multiple_t1": round((t1 - entry) / risk_ps, 2), "qty_suggested": qty, "risk_amount": round(qty * risk_ps, 2),
        "score_total": round(score, 1),
        "score_breakdown": {"components": pick.get("components"), "quality": pick.get("quality"),
                            "family": pick.get("family"), "catalyst_source": pick.get("catalyst_source")},
        "pivot_bar_date": as_of, "regime_state": regime_state, "notes": notes,
        "expires_on": as_of + timedelta(days=EXPIRY_DAYS),
    }, None


def write_regime(conn, out: dict, as_of: date) -> None:
    """The UI's regime panel reads market_regime (the legacy scan wrote it; in engine2 mode that scan is paused)."""
    reg, mk = out.get("regime") or {}, out.get("market") or {}
    state = str(reg.get("state") or "")
    if state not in ("risk_on", "neutral", "risk_off"):
        raise ValueError(f"engine-2 regime state {state!r} is not one the UI knows")
    notes = {"engine": "engine2", "regime_score": reg.get("score"), "size_mult": reg.get("size_mult"),
             "max_positions": reg.get("max_positions"), "components": {k: reg.get(k) for k in ("index", "breadth", "vix", "sector", "ad", "dispersion")}}
    with conn.cursor() as cur:
        cur.execute("""
            insert into market_regime (as_of, state, nifty_close, breadth_above_50dma, vix, notes)
            values (%s,%s,%s,%s,%s,%s)
            on conflict (as_of) do update set state = excluded.state, nifty_close = excluded.nifty_close,
                breadth_above_50dma = excluded.breadth_above_50dma, vix = excluded.vix, notes = excluded.notes
        """, (as_of, state, mk.get("nifty_close"), mk.get("breadth_above_50dma"), mk.get("vix"), json.dumps(notes)))


def lifecycle(conn, today: date) -> dict:
    """What the legacy scan did to every pending signal each run, and which is paused in engine2 mode."""
    with conn.cursor() as cur:
        cur.execute("""
            with latest as (select distinct on (symbol) symbol, adj_close from ohlcv_daily order by symbol, trade_date desc)
            update signals s set status = 'invalidated' from latest l
             where l.symbol = s.symbol and s.status = 'pending' and l.adj_close <= s.stop_loss
        """)
        invalidated = cur.rowcount
        cur.execute("update signals set status = 'expired' where status = 'pending' and expires_on < %s", (today,))
        expired = cur.rowcount
    return {"invalidated": invalidated, "expired": expired}


def publish(conn, out: dict, mode: str, today: date | None = None) -> dict:
    """
    Publish the portfolio-layer picks of one scan. `out` is e2_live.scan's return value.
    Raises on a failure (nothing is written); returns the counts otherwise.
    """
    from ingest import kill_switch_active
    provisional = bool(out.get("provisional"))
    as_of = date.fromisoformat(out["as_of"])
    from datetime import timezone
    today = today or datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
    picks = list(out.get("picks") or [])
    reg = out.get("regime") or {}
    res = {"as_of": str(as_of), "mode": mode, "provisional": provisional, "picks_in": len(picks), "inserted": 0, "updated": 0,
           "unchanged": 0, "skipped_already_taken": 0, "skipped_symbol_has_open_position": 0, "withdrawn": 0, "malformed": 0,
           "malformed_detail": [], "invalidated": 0, "expired": 0, "symbols": []}

    if kill_switch_active(conn):
        log.warning("Kill switch active: Engine 2 publish aborted, nothing written")
        res["aborted"] = "aborted_kill_switch"
        return res

    capital, risk_pct = _settings(conn)
    res["sizing_source"] = "user settings (capital, risk_pct)" if capital and risk_pct else "engine-2 default plan size"
    try:
        with conn.cursor() as cur:                                    # serialise with the legacy shadow scan (see e2_legacy_shadow)
            cur.execute("select pg_advisory_xact_lock(7340001)")
        write_regime(conn, out, as_of)

        with conn.cursor() as cur:
            cur.execute("select id, symbol, notes->>'detector', status, coalesce((notes->>'is_provisional')::boolean, false) "
                        "from signals where as_of_date = %s and notes->>'engine' = 'engine2'", (as_of,))
            existing = {(r[1], r[2]): {"id": r[0], "status": r[3], "provisional": r[4]} for r in cur.fetchall()}
            cur.execute("select distinct s.symbol from signals s left join signal_outcomes o on o.signal_id = s.id "
                        "where s.status = 'triggered' and o.exit_date is null")
            open_syms = {r[0] for r in cur.fetchall()}
            syms = [p["symbol"] for p in picks]
            cur.execute("select symbol, adj_close from (select distinct on (symbol) symbol, adj_close from ohlcv_daily "
                        "where symbol = any(%s) order by symbol, trade_date desc) t", (syms,)) if syms else None
            last_close = {r[0]: _num(r[1]) for r in cur.fetchall()} if syms else {}
            cur.execute("select symbol from symbols where symbol = any(%s)", (syms,)) if syms else None
            known = {r[0] for r in cur.fetchall()} if syms else set()

        keep: set[tuple] = set()
        for p in picks:
            key = (p["symbol"], p["detector"])
            if p["symbol"] not in known:
                res["malformed"] += 1
                res["malformed_detail"].append({"symbol": p["symbol"], "why": "unknown_symbol"})
                continue
            row, why = build_row(p, as_of=as_of, provisional=provisional, regime_state=str(reg.get("state")),
                                 size_mult=_num(reg.get("size_mult")) or 1.0, capital=capital, risk_pct=risk_pct,
                                 last_close=last_close.get(p["symbol"]))
            if row is None:
                res["malformed"] += 1
                res["malformed_detail"].append({"symbol": p["symbol"], "detector": p["detector"], "why": why})
                log.error("Engine 2 publish: %s/%s not written, malformed: %s", p["symbol"], p["detector"], why)
                continue
            ex = existing.get(key)
            keep.add(key)
            if ex and ex["status"] != "pending":
                res["skipped_already_taken"] += 1                     # triggered / closed: the user's trade, never modified
                continue
            if not ex and p["symbol"] in open_syms:
                res["skipped_symbol_has_open_position"] += 1          # you are already in this stock
                continue
            if ex and provisional and not ex["provisional"]:
                res["unchanged"] += 1                                 # a definitive row is never replaced by a forming-bar one
                continue
            args = (row["setup_type"], row["pattern"], row["entry_trigger"], row["stop_loss"], row["t1"], row["t2"], row["r_multiple_t1"],
                    row["qty_suggested"], row["risk_amount"], row["score_total"], json.dumps(row["score_breakdown"], default=str),
                    row["pivot_bar_date"], row["regime_state"], json.dumps(row["notes"], default=str), row["expires_on"])
            with conn.cursor() as cur:
                if ex:
                    cur.execute("""update signals set setup_type=%s, pattern=%s, entry_trigger=%s, stop_loss=%s, t1=%s, t2=%s,
                                   r_multiple_t1=%s, qty_suggested=%s, risk_amount=%s, score_total=%s, score_breakdown=%s::jsonb,
                                   pivot_bar_date=%s, regime_state=%s, notes=%s::jsonb, expires_on=%s where id = %s""", args + (ex["id"],))
                    res["updated"] += 1
                else:
                    cur.execute("""insert into signals (setup_type, pattern, entry_trigger, stop_loss, t1, t2, r_multiple_t1, qty_suggested,
                                   risk_amount, score_total, score_breakdown, pivot_bar_date, regime_state, notes, expires_on,
                                   symbol, as_of_date, status)
                                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,%s,%s,%s,'pending')""",
                                args + (row["symbol"], as_of))
                    res["inserted"] += 1
            res["symbols"].append(f"{p['symbol']}:{p['detector']}")

        if not provisional:                                           # the definitive scan settles every provisional row of that date
            for key, ex in existing.items():
                if ex["provisional"] and ex["status"] == "pending" and key not in keep:
                    with conn.cursor() as cur:
                        cur.execute("""update signals set status = 'expired',
                                       notes = jsonb_set(notes, '{withdrawn}', to_jsonb(%s::text)) where id = %s""",
                                    (f"provisional intraday signal not confirmed by the {mode} scan", ex["id"]))
                    res["withdrawn"] += 1

        res.update(lifecycle(conn, today))
        if kill_switch_active(conn):                                  # STOP pressed while this was running: write nothing
            conn.rollback()
            res.update(aborted="aborted_kill_switch", inserted=0, updated=0, withdrawn=0)
            log.warning("Kill switch fired during Engine 2 publish: transaction rolled back")
            return res
        conn.commit()
    except Exception:
        conn.rollback()
        log.exception("Engine 2 publish failed; nothing was written")
        raise
    log.info("Engine 2 published [%s %s]: %d new, %d updated, %d withdrawn, %d malformed", mode, as_of, res["inserted"],
             res["updated"], res["withdrawn"], res["malformed"])
    return res


def ui_summary(summary: dict, out: dict, pub: dict) -> dict:
    """The shape the UI's /jobs/status reader expects for last_scan_summary, plus the full Engine 2 diagnostics under `engine2`."""
    reg, mk = out.get("regime") or {}, out.get("market") or {}
    rej = dict(out.get("gated") or {})
    for k, v in (out.get("not_selected_reasons") or {}).items():
        rej[f"portfolio_{k}"] = v
    fr = (summary.get("freshness") or {}).get("final") or (summary.get("freshness") or {}).get("initial") or {}
    uni = out.get("universe") or {}
    return {
        "engine": "engine2", "as_of": out.get("as_of"), "mode": summary.get("mode"),
        "regime": reg.get("state"), "breadth_above_50dma": mk.get("breadth_above_50dma"), "vix": mk.get("vix"),
        "distribution_days": None,
        "universe": uni.get("scanned") if isinstance(uni, dict) else out.get("universe_symbols"), "universe_detail": uni,
        "passed_structure": out.get("candidates"), "signals": pub.get("inserted", 0) + pub.get("updated", 0) + pub.get("unchanged", 0),
        "rejections": rej,
        "gate2_enforced": False, "gate2_coverage": None, "gate2_blocked_by": "not_used_by_engine2", "min_score_pct": None,
        "freshness": {"refreshed": bool((summary.get("freshness") or {}).get("self_refresh")), "latest_bar": fr.get("index_latest"),
                      "expected": fr.get("expected_session"), "reason": "ok" if fr.get("ok") else "see engine2.freshness", "bars_added": None},
        "invalidated": pub.get("invalidated"), "new_opportunities": pub.get("inserted"), "add_ons": 0,
        "suppressed_already_taken": pub.get("skipped_already_taken", 0) + pub.get("skipped_symbol_has_open_position", 0),
        "provisional_signals": (pub.get("inserted", 0) + pub.get("updated", 0)) if out.get("provisional") else 0,
        "engine2": {k: v for k, v in summary.items() if k != "watchlist"}, "publish": pub,
        "written_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
