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
from datetime import date, datetime, timedelta

import gc

import pandas as pd

from gates import (
    gate2_fundamentals,
    gate3_stage1_transition,
    add_indicators,
    evaluate_regime,
    gate0_tradability,
    gate3_trend_structure,
    relative_strength,
)
from ingest import connect
from upstox_client import IST
from fundamentals import load_snapshots
from setups import Rejected, build_setup

log = logging.getLogger(__name__)

INDEX_SYMBOLS = ("NIFTY50", "NIFTY500", "INDIAVIX")
MIN_BARS = 210          # enough for a 200 DMA plus slope
FUNDAMENTALS_READY = False   # flip when NSE XBRL ingestion exists
CAPITAL = float(os.environ.get("SWING_CAPITAL", "0") or 0)
RISK_PCT = float(os.environ.get("RISK_PCT", "1.0"))
SIGNAL_EXPIRY_SESSIONS = int(os.environ.get("SIGNAL_EXPIRY_SESSIONS", "5"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "65"))
MIN_SCORE_NEUTRAL = float(os.environ.get("MIN_SCORE_NEUTRAL", "72"))
MIN_SCORE_RISK_OFF = float(os.environ.get("MIN_SCORE_RISK_OFF", "80"))
# An intraday run that cannot see the market must not conclude the market
# is empty. On a holiday, or with a dead token, the forming-bar fetch
# returns nothing for every symbol — and writing that result would erase
# the morning's armed watchlist.
INTRADAY_MIN_COVERAGE = float(os.environ.get("INTRADAY_MIN_COVERAGE", "0.5"))
# How far an entry must move before a regenerated setup counts as a NEW
# opportunity rather than the same one refreshed. Below this it is the same
# base being re-measured; above it the engine has found a different pivot.
MATERIAL_CHANGE_PCT = float(os.environ.get("MATERIAL_CHANGE_PCT", "2.0"))
# Stage 1 -> 2 transition path.
#
# "auto" (the default) means the engine decides for itself: the path stays
# dormant until enough delivery history exists to support it, then switches
# on without anyone remembering to do anything. Set "true" or "false" to
# override, but nothing requires you to.
STAGE1_TRANSITION_MODE = os.environ.get("STAGE1_TRANSITION", "auto").lower()
# Delivery history needed before the path is trusted: this many sessions,
# across this share of the universe.
TRANSITION_MIN_SESSIONS = int(os.environ.get("TRANSITION_MIN_SESSIONS", "20"))
TRANSITION_MIN_COVERAGE = float(os.environ.get("TRANSITION_MIN_COVERAGE", "0.4"))


def transition_readiness(conn, universe_size: int) -> dict:
    """
    Decide whether the transition path has the evidence it needs.

    The path leans on delivery expansion as its confirmation. Running it
    without delivery history would mean relaxing the trend filter and
    replacing it with nothing — which is how a second route becomes a back
    door. So it stays off until the data is actually there, and turns
    itself on when it is.
    """
    if STAGE1_TRANSITION_MODE == "false":
        return {"enabled": False, "reason": "disabled_by_config"}

    with conn.cursor() as cur:
        cur.execute("""
            select count(*) from (
                select symbol from delivery_daily
                where delivery_pct is not null
                group by symbol having count(*) >= %s
            ) t
        """, (TRANSITION_MIN_SESSIONS,))
        covered = cur.fetchone()[0]

    coverage = (covered / universe_size) if universe_size else 0.0

    if STAGE1_TRANSITION_MODE == "true":
        return {"enabled": True, "reason": "forced_on",
                "delivery_coverage": round(coverage, 3), "symbols": covered}

    ready = coverage >= TRANSITION_MIN_COVERAGE
    return {
        "enabled": ready,
        "reason": "auto_ready" if ready else "auto_waiting_for_delivery_history",
        "delivery_coverage": round(coverage, 3),
        "symbols": covered,
        "needed_sessions": TRANSITION_MIN_SESSIONS,
        "needed_coverage": TRANSITION_MIN_COVERAGE,
    }


class ScanAborted(RuntimeError):
    """Refused to write. The inputs were not trustworthy."""


# Remembers a refresh that fetched nothing, so an exchange holiday does not
# trigger a fruitless 500-symbol crawl on every scan for the rest of the day.
_last_refresh_attempt: dict = {"for_session": None, "at": None, "gained": 0}


def expected_last_session(now: datetime | None = None) -> date:
    """
    The most recent session whose CLOSING bar should exist.

    Today only after the close settles; before that, the previous weekday.
    Exchange holidays are not enumerated — on a holiday this returns a date
    with no data, the refresh finds nothing, and _last_refresh_attempt stops
    it retrying.
    """
    now = now or datetime.now(IST)
    day = now.date()
    # 18:00, not 16:00. Expecting today's bar before the broker has published
    # it made the freshness gate conclude the data was current and skip the
    # refresh — hiding the very staleness it exists to catch.
    if now.weekday() < 5 and (now.hour * 60 + now.minute) >= 18 * 60:
        return day
    while True:
        day -= timedelta(days=1)
        if day.weekday() < 5:
            return day


def ensure_bars_current(conn) -> dict:
    """
    Refresh prices only when they are actually behind.

    One query decides. If the newest stored bar already covers the expected
    session, nothing else runs — so this costs nothing on the common path.
    When bars ARE stale, backfill_prices is itself incremental and only
    fetches the symbols that are behind.

    This is what stops a manual scan silently analysing yesterday.
    """
    with conn.cursor() as cur:
        cur.execute("""
            select max(o.trade_date) from ohlcv_daily o
            join symbols s on s.symbol = o.symbol
            where coalesce(s.series, '') <> 'INDEX'
        """)
        latest = cur.fetchone()[0]

    expected = expected_last_session()
    if latest is not None and latest >= expected:
        return {"refreshed": False, "latest_bar": str(latest),
                "expected": str(expected), "reason": "already_current"}

    # Already tried for this session and got nothing — holiday, or the
    # broker has not published yet. Do not crawl again.
    if (_last_refresh_attempt["for_session"] == expected
            and _last_refresh_attempt["gained"] == 0
            and _last_refresh_attempt["at"]
            and (datetime.now(IST) - _last_refresh_attempt["at"]).seconds < 3600):
        return {"refreshed": False, "latest_bar": str(latest) if latest else None,
                "expected": str(expected), "reason": "no_data_for_session"}

    from ingest import backfill_prices, sync_indices
    from upstox_client import InstrumentMaster, TokenStore, UpstoxClient

    # Check the token BEFORE attempting. backfill_prices catches per-symbol
    # errors and continues, so an expired token fails all 500 silently and
    # returns zero bars — which is indistinguishable from a holiday. Reporting
    # that as "no_data_for_session" hid the real cause, and the hour-long memo
    # then suppressed retries even after a fresh login.
    if TokenStore().valid_token() is None:
        return {"refreshed": False, "latest_bar": str(latest) if latest else None,
                "expected": str(expected),
                "reason": "upstox_token_expired",
                "action": "Log in to Upstox, then re-run the scan."}

    log.info("Bars stale (latest %s, expected %s) — refreshing", latest, expected)
    client, master = UpstoxClient(), InstrumentMaster()
    gained = 0
    try:
        sync_indices(conn, client, master)
        gained = backfill_prices(conn, client)

        # History does not carry today's candle. If the expected session is
        # today and history did not supply it, build it from intraday.
        if expected == date.today():
            from ingest import sync_today_from_intraday
            gained += sync_today_from_intraday(conn, client)
    except Exception as exc:
        log.warning("Price refresh failed: %s", exc)
        return {"refreshed": False, "latest_bar": str(latest) if latest else None,
                "expected": str(expected), "reason": f"failed: {str(exc)[:150]}"}

    # Only memoise a genuine empty result. An auth failure returns earlier and
    # never reaches this point, so logging in makes the next scan try again.
    _last_refresh_attempt.update(for_session=expected, at=datetime.now(IST),
                                 gained=gained)

    # Re-read the newest bar so the caller can see what the refresh achieved.
    # Omitting this left latest_bar null on a successful refresh, which the UI
    # rendered as a dash — indistinguishable from having no data at all.
    with conn.cursor() as cur:
        cur.execute("""
            select max(o.trade_date) from ohlcv_daily o
            join symbols s on s.symbol = o.symbol
            where coalesce(s.series, '') <> 'INDEX'
        """)
        latest_after = cur.fetchone()[0]

    return {"refreshed": True, "bars_added": gained,
            "latest_bar": str(latest_after) if latest_after else None,
            "expected": str(expected),
            "reason": "up_to_date" if latest_after and latest_after >= expected
                      else "broker_has_no_bar_yet"}


def _load_symbol(conn, symbol: str, limit: int = 800) -> pd.DataFrame | None:
    """
    One symbol's bars.

    Fetched per symbol rather than pulling the whole table. Measuring the
    old approach showed the cost was not the per-symbol frames — pandas
    groupby returns cheap views — but materialising 350k rows as Python
    tuples via fetchall() and copying them into a DataFrame, which cost
    ~166 MB before any indicator was computed. That is what exhausted the
    container. 800 bars is well beyond the 250 any gate needs.
    """
    with conn.cursor() as cur:
        cur.execute("""
            select trade_date, adj_open, adj_high, adj_low, adj_close, volume
            from ohlcv_daily where symbol = %s
            order by trade_date desc limit %s
        """, (symbol, limit))
        rows = cur.fetchall()
    if not rows:
        return None

    df = pd.DataFrame(rows[::-1], columns=["trade_date", "open", "high",
                                           "low", "close", "volume"])
    # Postgres returns datetime.date; the intraday path appends today's
    # provisional bar as a pandas Timestamp. Mixing the two makes the column
    # unsortable — "Cannot compare Timestamp with datetime.date" — which
    # failed every intraday scan. Normalise here, once, at the only place
    # bars enter the system.
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df


def _breadth(conn, universe: list[str]) -> float:
    """
    Percentage of the universe above its own 50 DMA, computed in Postgres.

    Doing this in SQL returns one row per symbol instead of shipping every
    bar to Python for a single boolean per stock.
    """
    with conn.cursor() as cur:
        cur.execute("""
            with recent as (
                select symbol, adj_close,
                       row_number() over (partition by symbol
                                          order by trade_date desc) as rn
                from ohlcv_daily
                where symbol = any(%s)
            ),
            agg as (
                select symbol,
                       max(adj_close) filter (where rn = 1) as last_close,
                       avg(adj_close) filter (where rn <= 50) as sma50,
                       count(*) filter (where rn <= 50) as bars
                from recent where rn <= 50 group by symbol
            )
            select count(*) filter (where last_close > sma50), count(*)
            from agg where bars = 50
        """, (universe,))
        above, total = cur.fetchone()
    return (above / total * 100) if total else 0.0


def _forming_bar(client, conn, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Append today's in-progress session as a provisional daily bar.

    Only for the intraday slot, and only for symbols that already cleared
    the structural gates — 67 calls, not 500. The bar is real data, just
    incomplete: no volume projection is applied anywhere. A breakout whose
    volume only arrives in the closing surge will not confirm at 15:00,
    and that is the correct trade-off. Inventing the missing volume would
    manufacture a confirmation that has not happened.
    """
    out = {}
    today = date.today()
    for sym in symbols:
        df = _load_symbol(conn, sym)
        if df is None or len(df) == 0:
            continue
        key = _instrument_keys.get(sym)
        if not key:
            continue
        try:
            candles = client.intraday_today(key)
        except Exception as exc:
            log.debug("intraday fetch failed for %s: %s", sym, exc)
            continue
        if not candles:
            continue

        o = float(candles[-1][1]); h = max(float(c[2]) for c in candles)
        l = min(float(c[3]) for c in candles); c_ = float(candles[0][4])
        v = sum(int(c[5] or 0) for c in candles)
        # Upstox returns intraday candles newest-first.
        o = float(candles[-1][1])

        if pd.Timestamp(df["trade_date"].iloc[-1]).date() == today:
            df = df.iloc[:-1]      # replace a partial bar from an earlier slot
        bar = pd.DataFrame([{"trade_date": pd.Timestamp(today),
                             "open": o, "high": h, "low": l, "close": c_, "volume": v}])
        out[sym] = pd.concat([df, bar], ignore_index=True)
    return out


_instrument_keys: dict[str, str] = {}


def run_scan(as_of: date | None = None, mode: str = "postclose") -> dict:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    as_of = as_of or date.today()
    conn = connect()

    try:
        # Intraday runs fetch their own forming bars, so they neither need
        # nor want the end-of-day refresh.
        freshness = ({"refreshed": False, "reason": "intraday_mode"}
                     if mode == "intraday" else ensure_bars_current(conn))

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

        # Symbols already held. A stock you are in should not reappear as a
        # fresh recommendation — the position is managed on My Trades, and
        # a duplicate card invites doubling into the same risk.
        with conn.cursor() as cur:
            cur.execute("""
                select distinct s.symbol from signals s
                join signal_outcomes o on o.signal_id = s.id
                where o.entry_price is not null and o.exit_price is null
            """)
            open_positions = {r[0] for r in cur.fetchall()}
        if open_positions:
            log.info("%d symbols have open positions; new setups on them will be "
                     "flagged as add-ons", len(open_positions))

        with conn.cursor() as cur:
            cur.execute("select symbol, upstox_instrument_key from symbols "
                        "where upstox_instrument_key is not null")
            _instrument_keys.clear()
            _instrument_keys.update(dict(cur.fetchall()))
        log.info("Universe: %d symbols, %d under surveillance", len(universe), len(flagged))

        # Gate 2 inputs. Absence is recorded per symbol, never assumed to
        # be a pass — a stock we hold no fundamentals for is flagged as
        # unchecked rather than quietly treated as clean.
        # Delivery expansion, used ONLY by the transition path. The main
        # pipeline never reads it.
        transition_state = transition_readiness(conn, len(universe))
        transition_on = transition_state["enabled"]
        log.info("Transition path: %s (%s)",
                 "on" if transition_on else "off", transition_state["reason"])

        delivery_expanding: set[str] = set()
        if transition_on:
            with conn.cursor() as cur:
                cur.execute("""
                    with recent as (
                        select symbol, delivery_pct,
                               row_number() over (partition by symbol
                                                  order by trade_date desc) as rn
                        from delivery_daily where delivery_pct is not null
                    )
                    select symbol
                    from recent where rn <= 40
                    group by symbol
                    having count(*) >= 20
                       and avg(delivery_pct) filter (where rn <= 10) >
                           avg(delivery_pct) filter (where rn > 10) * 1.1
                """)
                delivery_expanding = {r[0] for r in cur.fetchall()}
            log.info("Transition path enabled: %d symbols show delivery expansion",
                     len(delivery_expanding))

        snapshots = load_snapshots(conn)
        fundamentals_ready = len(snapshots) >= len(universe) * 0.5
        log.info("Fundamentals: %d symbols with data (gate 2 %s)",
                 len(snapshots), "enforced" if fundamentals_ready else "advisory only")

        # --- Gate 1: regime, once for the whole run -------------------
        nifty = _load_symbol(conn, "NIFTY50")
        vix = _load_symbol(conn, "INDIAVIX")
        if nifty is None or len(nifty) < 60:
            raise RuntimeError("NIFTY50 series missing or too short — cannot assess regime")

        breadth = _breadth(conn, universe)
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
        intraday_frames: dict[str, pd.DataFrame] = {}

        if mode == "intraday":
            from upstox_client import UpstoxClient
            structural = []
            for sym in universe:
                d = _load_symbol(conn, sym)
                if d is None or len(d) < MIN_BARS:
                    continue
                d = add_indicators(d)
                passed = (gate0_tradability(sym, d, sym in flagged).passed
                          and gate3_trend_structure(sym, d).passed)
                del d                       # one symbol's indicators at a time
                if passed:
                    structural.append(sym)
            gc.collect()
            log.info("Intraday slot: fetching forming bars for %d candidates",
                     len(structural))
            intraday_frames = _forming_bar(UpstoxClient(), conn, structural)

            coverage = (len(intraday_frames) / len(structural)) if structural else 0.0
            if coverage < INTRADAY_MIN_COVERAGE:
                raise ScanAborted(
                    f"Intraday coverage {coverage:.0%} of {len(structural)} candidates "
                    f"is below {INTRADAY_MIN_COVERAGE:.0%}. Exchange holiday, stale "
                    "token, or a broker outage. Existing signals left untouched."
                )

        for processed, sym in enumerate(universe, 1):
            if processed % 100 == 0:
                gc.collect()
            df = _load_symbol(conn, sym)
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

            r2 = gate2_fundamentals(sym, snapshots.get(sym))
            if not r2.passed:
                counts[r2.reason] = counts.get(r2.reason, 0) + 1
                log_rows.append((as_of, sym, "gate2", r2.reason, json.dumps(r2.detail)))
                continue
            if r2.reason == "gate2_no_data":
                counts["gate2_no_data"] = counts.get("gate2_no_data", 0) + 1
                log_rows.append((as_of, sym, None, "gate2_no_data", "{}"))

            transition = False
            r3 = gate3_trend_structure(sym, df)
            if not r3.passed:
                # Second path. Only reached by candidates Gate 3 already
                # rejected, so nothing that passes today can be lost.
                # Requires delivery evidence — without it, fail closed.
                if transition_on and sym in delivery_expanding:
                    r3b = gate3_stage1_transition(sym, df)
                    if r3b.passed:
                        transition = True
                        r3 = r3b
                        counts["transition_candidate"] = counts.get(
                            "transition_candidate", 0) + 1
                if not transition:
                    counts[r3.reason] = counts.get(r3.reason, 0) + 1
                    log_rows.append((as_of, sym, "gate3", r3.reason,
                                     json.dumps(r3.detail)))
                    continue

            rs63 = relative_strength(df, nifty, 63)
            rs126 = relative_strength(df, nifty, 126)
            last = df.iloc[-1]
            counts["passed_gates_0_3"] = counts.get("passed_gates_0_3", 0) + 1

            # Gates 4-7. In a risk-off regime the correct output is nothing,
            # so setups are not even constructed — an empty list is the
            # answer, not a failure.
            if mode == "intraday":
                live = intraday_frames.get(sym)
                if live is None:
                    log_rows.append((as_of, sym, "gate5", "no_intraday_bar", "{}"))
                    counts["no_intraday_bar"] = counts.get("no_intraday_bar", 0) + 1
                    continue
                df = add_indicators(live)

            try:
                setup = build_setup(sym, df, rs63, rs126,
                                    snapshots.get(sym) if fundamentals_ready else None,
                                    transition=transition,
                                    last_bar_incomplete=(mode == "intraday"))
            except Rejected as rej:
                counts[rej.reason] = counts.get(rej.reason, 0) + 1
                log_rows.append((as_of, sym, rej.gate, rej.reason,
                                 json.dumps(rej.detail)))
                continue

            # Only setups worth acting on are published. Everything else is
            # logged and stays out of the recommendation list — the desk
            # shows trades, not a research feed.
            # The floors are expressed on a 100-point scale, but the
            # achievable max is lower while fundamentals and news score
            # zero. Comparing a 72 threshold against an 85-point ceiling
            # silently applies an 85th-percentile bar instead of the 72%
            # the spec intends, so scale the floor to what is actually
            # scoreable.
            max_possible = float(setup.score_breakdown.get("max_possible", 100)) or 100.0
            floor_pct = {"risk_on": MIN_SCORE,
                         "neutral": MIN_SCORE_NEUTRAL,
                         "risk_off": MIN_SCORE_RISK_OFF}[regime["state"]]
            floor = round(floor_pct * max_possible / 100.0, 1)
            if setup.score_total < floor:
                counts["below_min_score"] = counts.get("below_min_score", 0) + 1
                log_rows.append((as_of, sym, "score", "below_min_score",
                                 json.dumps({"score": setup.score_total,
                                             "floor": floor,
                                             "floor_pct": floor_pct,
                                             "max_possible": max_possible,
                                             "regime": regime["state"]})))
                continue

            # Position sizing is no longer stored at scan time. It is derived
            # on read from the capital setting, so editing capital updates
            # every card immediately rather than waiting for the next scan.
            qty = risk_amount = None

            signals.append({
                "symbol": sym,
                "is_add_on": sym in open_positions,
                "is_transition": transition,
                # True only when the unfinished bar actually contributed to a
                # level. An armed setup found intraday is NOT provisional: its
                # pivot, stop and target all come from closed bars, which is
                # precisely what makes it useful hours before the close.
                "is_provisional": setup.provisional,
                "setup_type": setup.setup_type,
                "pattern": setup.pattern,
                "contracting": setup.base.contracting,
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
                "extension": setup.extension,
                "stop_basis": setup.stop_basis,
                "t1_basis": setup.t1_basis,
                "notes": setup.notes,
            })
            counts["signal"] = counts.get("signal", 0) + 1
            log_rows.append((as_of, sym, None, "signal_generated",
                             json.dumps({"score": setup.score_total,
                                         "rr": setup.r_multiple_t1})))
            del df

        intraday_frames.clear()
        gc.collect()

        signals.sort(key=lambda s: -s["score_total"])

        with conn.cursor() as cur:
            cur.executemany(
                "insert into gate_log (as_of_date, symbol, failed_gate, reason_code, detail) "
                "values (%s,%s,%s,%s,%s)", log_rows)

            # Supersede today's previous run rather than accumulating
            # duplicates; untriggered signals from earlier days keep running
            # to their own expiry.
            # Preserve when each setup was FIRST seen today. A card armed
            # since 10:00 and still armed at 14:00 is a different thing from
            # one that appeared this hour, and that distinction is lost if
            # every re-scan resets the timestamp.
            # Pending AND triggered. Loading only pending was the bug behind
            # duplicate cards: once a signal is marked taken it becomes
            # 'triggered', dropped out of the comparison, and the next scan
            # re-issued the identical setup as a fresh pending signal beside
            # the open trade.
            cur.execute("""
                select id, symbol, setup_type, entry_trigger, generated_at, status
                from signals where status in ('pending', 'triggered')
            """)
            existing: dict[str, list] = {}
            for sig_id, sym_, stype, entry_, gen_at, status_ in cur.fetchall():
                existing.setdefault(sym_, []).append(
                    {"id": sig_id, "setup_type": stype, "entry": float(entry_),
                     "generated_at": gen_at, "status": status_})

            # Decide, per setup, whether this is the SAME opportunity being
            # re-measured or a genuinely different one.
            #
            # Same  -> replace in place, keeping the original generated_at so
            #          persistence is still visible. This is what stops one
            #          card per day for a stock that keeps qualifying.
            # New   -> insert alongside. A different pivot, or armed becoming
            #          a confirmed breakout, is a fresh entry opportunity with
            #          its own levels, not a duplicate.
            first_seen: dict = {}
            superseded: list = []
            new_opportunities = 0

            # Remove pending signals that duplicate a position already open.
            # The matching logic above stops NEW duplicates being written, but
            # rows created before that fix are still in the table and no scan
            # would ever clear them — they are not regenerated, so nothing
            # touches them. This cleans them up regardless.
            cur.execute("""
                delete from signals p
                 using signals t
                  join signal_outcomes o on o.signal_id = t.id
                 where p.status = 'pending'
                   and t.status = 'triggered'
                   and o.entry_price is not null
                   and o.exit_price is null
                   and p.symbol = t.symbol
                   and p.setup_type = t.setup_type
                   and t.entry_trigger > 0
                   and abs(p.entry_trigger / t.entry_trigger - 1) * 100 <= %s
            """, (MATERIAL_CHANGE_PCT,))
            stale_duplicates = cur.rowcount

            # Two pending rows for the same symbol and setup at effectively the
            # same entry are the same opportunity written twice. Keep the
            # oldest — it carries the original generated_at, so persistence is
            # preserved — and drop the rest.
            cur.execute("""
                delete from signals p
                 using signals q
                 where p.status = 'pending' and q.status = 'pending'
                   and p.symbol = q.symbol
                   and p.setup_type = q.setup_type
                   and q.entry_trigger > 0
                   and abs(p.entry_trigger / q.entry_trigger - 1) * 100 <= %s
                   and (p.generated_at > q.generated_at
                        or (p.generated_at = q.generated_at and p.id > q.id))
            """, (MATERIAL_CHANGE_PCT,))
            stale_duplicates += cur.rowcount

            if stale_duplicates:
                log.info("Cleared %d pending signals duplicating open positions",
                         stale_duplicates)

            to_write, already_taken, frozen = [], 0, 0

            # Levels come from COMPLETED daily bars, and exactly one new bar
            # exists per trading day. So a card's entry, stop and target may
            # change once a day at most, at the 15:45 post-close run.
            #
            # The 08:15 premarket slot sees the same bars the previous
            # evening's post-close already used — no bar forms overnight — so
            # recomputing there produces the same answer at best, and a
            # spurious change at worst. Intraday runs see a forming bar and
            # must never rewrite a level an order is already resting against.
            #
            # The exception is recovery: if this run actually fetched missing
            # bars, the post-close that should have computed them did not run,
            # and this slot has to do it instead.
            may_revise = (mode == "postclose") or bool(freshness.get("refreshed"))

            # Armed and breakout are the SAME opportunity at two moments.
            # Both anchor entry to pivot x 1.0025 and take the same stop and
            # targets from the same base — the trigger type changes no level.
            # So an armed setup breaking out is a status change on the card
            # you are already watching, not a second trade. Requiring
            # setup_type equality here created a duplicate card with
            # identical numbers every time a setup triggered.
            def _pivot_anchored(kind: str) -> bool:
                return kind.startswith("armed") or kind.startswith("breakout")

            trigger_transitions = []

            for s_ in signals:
                matches = [
                    e for e in existing.get(s_["symbol"], [])
                    if (e["setup_type"] == s_["setup_type"]
                        or (_pivot_anchored(e["setup_type"])
                            and _pivot_anchored(s_["setup_type"])))
                    and e["entry"] > 0
                    and abs(s_["entry_trigger"] / e["entry"] - 1) * 100 <= MATERIAL_CHANGE_PCT
                ]

                # Matches an OPEN trade: this is the position you are already
                # in, not a re-entry. Issuing it would put an identical card
                # beside the trade on the dashboard.
                if any(m["status"] == "triggered" for m in matches):
                    already_taken += 1
                    log_rows.append((as_of, s_["symbol"], None,
                                     "same_setup_already_taken", "{}"))
                    continue

                if matches:
                    if not may_revise:
                        # Levels stay frozen, but if the trigger has changed
                        # -- armed becoming a live breakout -- record that on
                        # the existing card. It is new information about price,
                        # not a revision of any level.
                        changed = [m for m in matches
                                   if m["setup_type"] != s_["setup_type"]
                                   and m["status"] == "pending"]
                        if changed:
                            trigger_transitions.append(
                                (s_["setup_type"], [m["id"] for m in changed]))
                        frozen += 1
                        continue

                    superseded.extend(m["id"] for m in matches)
                    first_seen[s_["symbol"]] = min(m["generated_at"] for m in matches)
                    s_["is_new_opportunity"] = False
                else:
                    # Nothing comparable. A different pivot, or a different
                    # trigger type, is a genuine second entry.
                    s_["is_new_opportunity"] = bool(existing.get(s_["symbol"]))
                    if s_["is_new_opportunity"]:
                        new_opportunities += 1

                to_write.append(s_)

            signals = to_write

            for new_type, ids in trigger_transitions:
                cur.execute("update signals set setup_type = %s where id = any(%s)",
                            (new_type, ids))

            if superseded:
                cur.execute("delete from signals where id = any(%s)", (superseded,))

            for s_ in signals:
                cur.execute("""
                    insert into signals
                        (symbol, as_of_date, setup_type, pattern, entry_trigger,
                         stop_loss, t1, t2, r_multiple_t1, qty_suggested, risk_amount,
                         score_total, score_breakdown, pivot_bar_date, base_start_date,
                         base_low, regime_state, notes, status, expires_on,
                         generated_at)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                            'pending', %s, coalesce(%s, now()))
                """, (s_["symbol"], as_of, s_["setup_type"], s_["pattern"],
                      s_["entry_trigger"], s_["stop_loss"], s_["t1"], s_["t2"],
                      s_["r_multiple_t1"], s_["qty_suggested"], s_["risk_amount"],
                      s_["score_total"], json.dumps(s_["score_breakdown"]),
                      s_["pivot_bar_date"], s_["base_start_date"], s_["base_low"],
                      regime["state"],
                      json.dumps({"stop_basis": s_["stop_basis"],
                                  "t1_basis": s_["t1_basis"],
                                  "is_add_on": s_["is_add_on"],
                                  "is_transition": s_["is_transition"],
                                  "is_provisional": s_["is_provisional"],
                                  "is_new_opportunity": s_.get("is_new_opportunity", False),
                                  "rs63": s_["rs63"], "rs126": s_["rs126"],
                                  "close": s_["close"], "atr14": s_["atr14"],
                                  "extension": s_["extension"],
                                  "contracting": s_["contracting"],
                                  "notes": s_["notes"]}),
                      as_of + timedelta(days=SIGNAL_EXPIRY_SESSIONS * 2),
                      first_seen.get(s_["symbol"])))

            # A pending signal is removed only when it is INVALIDATED: price
            # closed at or below the stop before ever triggering. The base is
            # gone and the level no longer means anything.
            #
            # Not being regenerated is deliberately NOT a reason to drop it.
            # An armed setup can fail the score floor or the coiling test on a
            # single intraday bar while the base and pivot remain perfectly
            # intact — dropping it there would churn cards off the dashboard
            # for no real change in the setup.
            cur.execute("""
                with latest as (
                    select distinct on (symbol) symbol, adj_close
                    from ohlcv_daily order by symbol, trade_date desc
                )
                update signals s
                   set status = 'invalidated'
                  from latest l
                 where l.symbol = s.symbol
                   and s.status = 'pending'
                   and l.adj_close <= s.stop_loss
            """)
            invalidated = cur.rowcount

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
            "gate2_enforced": fundamentals_ready,
            "gate2_coverage": len(snapshots),
            "freshness": freshness,
            "invalidated": invalidated,
            "new_opportunities": new_opportunities,
            "suppressed_already_taken": already_taken,
            "stale_duplicates_cleared": stale_duplicates,
            "levels_frozen": frozen,
            "levels_revisable": may_revise,
            "provisional_signals": sum(1 for s_ in signals if s_["is_provisional"]),
            "trigger_transitions": len(trigger_transitions),
            "add_ons": sum(1 for s_ in signals if s_["is_add_on"]),
            "transition": transition_state,
            "transition_signals": sum(1 for s_ in signals if s_["is_transition"]),
            "min_score_pct": {"risk_on": MIN_SCORE, "neutral": MIN_SCORE_NEUTRAL,
                              "risk_off": MIN_SCORE_RISK_OFF}[regime["state"]],
            "regime_detail": regime["notes"],
            "mode": mode,
        }
        log.info("Scan complete [%s]: %d signals from %d structural candidates (%s regime)",
                 mode, len(signals), counts.get("passed_gates_0_3", 0), regime["state"])
        return summary

    finally:
        conn.close()


if __name__ == "__main__":
    import pprint
    pprint.pprint(run_scan())
