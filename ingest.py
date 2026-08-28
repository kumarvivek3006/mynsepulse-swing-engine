"""
Ingestion jobs for the swing engine.

Writes over a direct Postgres connection rather than PostgREST: the initial
backfill is roughly 375k rows and COPY is the only sane way to move that.
It also sidesteps having to expose the `swing` schema to the REST API.

Job order matters on a cold start:

    sync_universe            -> symbols
    sync_corporate_actions   -> corporate_actions
    backfill_prices          -> ohlcv_daily (raw, unadjusted)
    apply_adjustments        -> ohlcv_daily.adj_factor
    sync_surveillance        -> surveillance

Adjustments run after prices because a factor is only meaningful once the
bars it applies to exist. Re-running any job is safe; all writes upsert.
"""

from __future__ import annotations

import io
import logging
import os
from datetime import date, datetime, timedelta

import psycopg
from psycopg import sql

from nse_client import NSEClient, NSEUnavailable
from upstox_client import InstrumentMaster, UpstoxClient

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
UNIVERSE = os.environ.get("UNIVERSE", "nifty500")
BACKFILL_YEARS = int(os.environ.get("BACKFILL_YEARS", "3"))


def connect() -> psycopg.Connection:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("set search_path to swing, public")
    return conn


def _run_log(conn, job: str, status: str, rows: int = 0, error: str | None = None):
    with conn.cursor() as cur:
        cur.execute(
            "insert into ingestion_runs (job, finished_at, status, rows_written, error) "
            "values (%s, now(), %s, %s, %s)",
            (job, status, rows, error),
        )
    conn.commit()


# ---------------------------------------------------------------------
# 1. Universe
# ---------------------------------------------------------------------
def sync_universe(conn, nse: NSEClient, master: InstrumentMaster) -> int:
    """
    NSE index constituents joined to Upstox instrument keys.

    Resolution is attempted by ISIN first. NSE tickers and Upstox trading
    symbols do occasionally disagree, but ISINs do not, and the whole
    reason for storing instrument keys is that they survive renames.
    """
    constituents = nse.index_constituents(UNIVERSE)
    master.load()

    by_isin = {i.isin: i for i in master.equities() if i.isin}
    rows, unresolved = [], []

    for c in constituents:
        inst = by_isin.get(c.isin) if c.isin else None
        if inst is None:
            inst = master.resolve(c.symbol)
        if inst is None:
            unresolved.append(c.symbol)
            continue
        rows.append((c.symbol, inst.instrument_key, c.isin or inst.isin,
                     c.company_name, inst.instrument_type, c.industry, True))

    if unresolved:
        # Not fatal, but never silent: a constituent we cannot price is a
        # hole in the universe and needs to be seen.
        log.warning("Unresolved constituents (%d): %s",
                    len(unresolved), ", ".join(sorted(unresolved)[:25]))

    if len(rows) < len(constituents) * 0.9:
        raise NSEUnavailable(
            f"Only resolved {len(rows)}/{len(constituents)} constituents to "
            "instrument keys. Refusing to write a degraded universe."
        )

    with conn.cursor() as cur:
        cur.execute("update symbols set in_nifty500 = false")
        cur.executemany(
            """
            insert into symbols
                (symbol, upstox_instrument_key, isin, company_name, series,
                 industry, in_nifty500, is_active, updated_at)
            values (%s, %s, %s, %s, %s, %s, %s, true, now())
            on conflict (symbol) do update set
                upstox_instrument_key = excluded.upstox_instrument_key,
                isin         = coalesce(excluded.isin, symbols.isin),
                company_name = excluded.company_name,
                industry     = coalesce(excluded.industry, symbols.industry),
                in_nifty500  = excluded.in_nifty500,
                is_active    = true,
                updated_at   = now()
            """,
            rows,
        )
    conn.commit()
    log.info("Universe synced: %d symbols (%d unresolved)", len(rows), len(unresolved))
    return len(rows)


# ---------------------------------------------------------------------
# 2. Corporate actions
# ---------------------------------------------------------------------
def sync_corporate_actions(conn, nse: NSEClient, years: int = BACKFILL_YEARS) -> int:
    """Fetched in 90-day windows — NSE truncates wide date ranges."""
    end = date.today()
    start = end - timedelta(days=365 * years + 30)
    written, cursor = 0, start

    with conn.cursor() as cur:
        while cursor < end:
            window_end = min(cursor + timedelta(days=90), end)
            for ca in nse.corporate_actions(cursor, window_end):
                cur.execute(
                    """
                    insert into corporate_actions
                        (symbol, ex_date, action_type, ratio_from, ratio_to, raw_purpose)
                    select %s, %s, %s, %s, %s, %s
                    where exists (select 1 from symbols where symbol = %s)
                    on conflict (symbol, ex_date, action_type, raw_purpose) do nothing
                    """,
                    (ca.symbol, ca.ex_date, ca.action_type, ca.ratio_from,
                     ca.ratio_to, ca.purpose, ca.symbol),
                )
                written += cur.rowcount
            cursor = window_end + timedelta(days=1)
    conn.commit()
    log.info("Corporate actions written: %d", written)
    return written


# ---------------------------------------------------------------------
# 3. Prices
# ---------------------------------------------------------------------
def backfill_prices(conn, client: UpstoxClient, years: int = BACKFILL_YEARS,
                    only_missing: bool = True) -> int:
    """
    Resumable price backfill.

    Each symbol is fetched from the day after its latest stored bar, so an
    interrupted run picks up where it stopped instead of re-downloading
    three years. COPY is used per symbol via a staging table so a failure
    mid-symbol cannot leave partial bars committed.
    """
    end = date.today()
    default_start = end - timedelta(days=365 * years)

    with conn.cursor() as cur:
        cur.execute("""
            select s.symbol, s.upstox_instrument_key, max(o.trade_date)
            from symbols s
            left join ohlcv_daily o on o.symbol = s.symbol
            where s.is_active and s.upstox_instrument_key is not null
            group by s.symbol, s.upstox_instrument_key
            order by s.symbol
        """)
        targets = cur.fetchall()

    total = 0
    for symbol, instrument_key, latest in targets:
        start = (latest + timedelta(days=1)) if latest else default_start
        if only_missing and start > end:
            continue

        try:
            candles = client.historical_daily(instrument_key, start, end)
        except Exception as exc:
            log.error("Backfill failed for %s: %s", symbol, exc)
            continue

        if not candles:
            continue

        buf = io.StringIO()
        for c in candles:
            # [ts, open, high, low, close, volume, oi]
            ts = c[0][:10] if isinstance(c[0], str) else datetime.fromtimestamp(c[0] / 1000).date()
            buf.write(f"{symbol}\t{ts}\t{c[1]}\t{c[2]}\t{c[3]}\t{c[4]}\t{int(c[5])}\n")
        buf.seek(0)

        with conn.cursor() as cur:
            cur.execute("""
                create temp table if not exists _stage_ohlcv (
                    symbol text, trade_date date, open numeric, high numeric,
                    low numeric, close numeric, volume bigint
                ) on commit drop
            """)
            cur.execute("truncate _stage_ohlcv")
            with cur.copy("copy _stage_ohlcv from stdin") as copy:
                copy.write(buf.read())
            cur.execute("""
                insert into ohlcv_daily (symbol, trade_date, open, high, low, close, volume)
                select symbol, trade_date, open, high, low, close, volume from _stage_ohlcv
                on conflict (symbol, trade_date) do update set
                    open = excluded.open, high = excluded.high, low = excluded.low,
                    close = excluded.close, volume = excluded.volume
            """)
            total += cur.rowcount
        conn.commit()
        log.debug("%s: %d bars", symbol, len(candles))

    log.info("Backfill complete: %d bars across %d symbols", total, len(targets))
    return total


# ---------------------------------------------------------------------
# 4. Adjustment factors
# ---------------------------------------------------------------------
def apply_adjustments(conn) -> int:
    """
    Verify that prices are corporate-action adjusted. Do not adjust them.

    Upstox returns history already adjusted for splits and bonuses. An
    earlier version of this job applied its own back-adjustment factors on
    top, which double-adjusted the series and inserted a 10x cliff into
    NESTLEIND across its 1:10 split — precisely the kind of artefact that
    base detection would read as a breakout.

    So this step now checks rather than mutates. For every split/bonus we
    know about, it measures the close-to-close move across the ex-date. On
    an already-adjusted series that move is ordinary. A move close to the
    action's ratio means the source stopped adjusting, and that must fail
    loudly rather than quietly feeding wrong levels into every stop loss.
    """
    with conn.cursor() as cur:
        # Any leftover factors from the old behaviour must go.
        cur.execute("update ohlcv_daily set adj_factor = 1.0 where adj_factor <> 1.0")
        reset = cur.rowcount
        if reset:
            log.warning("Reset %d bars carrying stale adjustment factors", reset)

        cur.execute("""
            select symbol, ex_date, action_type, ratio_from, ratio_to
            from corporate_actions
            where action_type in ('split','bonus')
              and ratio_from is not null and ratio_to is not null
            order by symbol, ex_date
        """)
        actions = cur.fetchall()

        suspect = []
        for symbol, ex_date, kind, a, b in actions:
            expected = (a / b) if kind == "split" else (a / (a + b))
            if not expected or not (0 < expected < 1):
                continue

            cur.execute("""
                with bars as (
                    select trade_date, close,
                           lag(close) over (order by trade_date) as prev
                    from ohlcv_daily
                    where symbol = %s
                      and trade_date between %s::date - 10 and %s::date + 10
                )
                select trade_date, prev, close
                from bars
                where trade_date >= %s and prev is not null
                order by trade_date
                limit 1
            """, (symbol, ex_date, ex_date, ex_date))
            row = cur.fetchone()
            if not row or not row[1]:
                continue

            observed = row[2] / row[1]
            # If the series were unadjusted, observed would land near the
            # ratio itself (0.1 for a 1:10 split). Allow generous slack for
            # genuine price movement on the day.
            if observed < expected * 1.5:
                suspect.append((symbol, ex_date, kind, expected, round(observed, 4)))

    conn.commit()

    if suspect:
        detail = "; ".join(f"{s} {d} {k} expected~{e:.3f} observed {o}"
                           for s, d, k, e, o in suspect[:10])
        raise RuntimeError(
            f"{len(suspect)} corporate action(s) appear UNADJUSTED in the price "
            f"series. Prices cannot be trusted for level derivation. {detail}"
        )

    log.info("Adjustment check passed: %d split/bonus events, none unadjusted", len(actions))
    return len(actions)


# ---------------------------------------------------------------------
# 4b. Indices
#
# Gate 1 (market regime) needs the index series themselves, which are not
# in the constituent universe. Upstox exposes them under the NSE_INDEX
# segment. Breadth is computed from our own 500 symbols, so only the
# headline index and VIX need fetching.
# ---------------------------------------------------------------------
INDEX_NAMES = {
    "NIFTY50": ["nifty 50", "nifty50"],
    "NIFTY500": ["nifty 500", "nifty500"],
    "INDIAVIX": ["india vix", "indiavix"],
}


def sync_indices(conn, client: UpstoxClient, master: InstrumentMaster,
                 years: int = BACKFILL_YEARS) -> int:
    master.load()
    index_instruments = [
        i for i in master._by_key.values() if i.segment == "NSE_INDEX"
    ]

    resolved: dict[str, str] = {}
    for code, aliases in INDEX_NAMES.items():
        for inst in index_instruments:
            label = (inst.name or inst.trading_symbol or "").strip().lower()
            if label in aliases:
                resolved[code] = inst.instrument_key
                break

    missing = set(INDEX_NAMES) - set(resolved)
    if "NIFTY50" in missing or "INDIAVIX" in missing:
        raise RuntimeError(
            f"Could not resolve required indices {sorted(missing)} in the Upstox "
            "instrument master. Market regime cannot be computed without them."
        )
    if missing:
        log.warning("Optional indices unresolved: %s", sorted(missing))

    end = date.today()
    default_start = end - timedelta(days=365 * years)
    total = 0

    with conn.cursor() as cur:
        for code, key in resolved.items():
            cur.execute("""
                insert into symbols (symbol, upstox_instrument_key, company_name,
                                     series, is_active, updated_at)
                values (%s, %s, %s, 'INDEX', true, now())
                on conflict (symbol) do update set
                    upstox_instrument_key = excluded.upstox_instrument_key,
                    series = 'INDEX', updated_at = now()
            """, (code, key, code))

            cur.execute("select max(trade_date) from ohlcv_daily where symbol = %s", (code,))
            latest = cur.fetchone()[0]
            start = (latest + timedelta(days=1)) if latest else default_start
            if start > end:
                continue

            try:
                candles = client.historical_daily(key, start, end)
            except Exception as exc:
                log.error("Index backfill failed for %s: %s", code, exc)
                continue

            for c in candles:
                ts = c[0][:10] if isinstance(c[0], str) else \
                    datetime.fromtimestamp(c[0] / 1000).date()
                cur.execute("""
                    insert into ohlcv_daily
                        (symbol, trade_date, open, high, low, close, volume)
                    values (%s, %s, %s, %s, %s, %s, %s)
                    on conflict (symbol, trade_date) do update set
                        open = excluded.open, high = excluded.high,
                        low = excluded.low, close = excluded.close
                """, (code, ts, c[1], c[2], c[3], c[4], int(c[5] or 0)))
                total += 1
            log.info("%s: %d bars", code, len(candles))

    conn.commit()
    log.info("Index backfill complete: %d bars", total)
    return total


# ---------------------------------------------------------------------
# 5. Surveillance
# ---------------------------------------------------------------------
def sync_surveillance(conn, nse: NSEClient) -> int:
    today = date.today()
    written = 0
    with conn.cursor() as cur:
        for list_type, rows in nse.surveillance().items():
            for r in rows:
                sym = (r.get("symbol") or r.get("Symbol") or "").strip().upper()
                if not sym:
                    continue
                cur.execute(
                    """
                    insert into surveillance (symbol, as_of, list_type, stage)
                    select %s, %s, %s, %s
                    where exists (select 1 from symbols where symbol = %s)
                    on conflict (symbol, as_of, list_type) do update
                        set stage = excluded.stage
                    """,
                    (sym, today, list_type.upper(),
                     str(r.get("longTermStage") or r.get("stage") or ""), sym),
                )
                written += cur.rowcount
    conn.commit()
    log.info("Surveillance rows written: %d", written)
    return written


# ---------------------------------------------------------------------
def cold_start() -> None:
    """Full first run. Safe to re-run; every step is idempotent."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    nse, master, client = NSEClient(), InstrumentMaster(), UpstoxClient()
    conn = connect()
    try:
        for name, fn in [
            ("sync_universe", lambda: sync_universe(conn, nse, master)),
            ("sync_corporate_actions", lambda: sync_corporate_actions(conn, nse)),
            ("backfill_prices", lambda: backfill_prices(conn, client)),
            ("sync_indices", lambda: sync_indices(conn, client, master)),
            ("verify_adjustments", lambda: apply_adjustments(conn)),
            ("sync_surveillance", lambda: sync_surveillance(conn, nse)),
        ]:
            log.info("=== %s ===", name)
            try:
                _run_log(conn, name, "success", fn())
            except Exception as exc:
                conn.rollback()
                _run_log(conn, name, "failed", 0, str(exc))
                log.exception("%s failed", name)
                raise
    finally:
        conn.close()


if __name__ == "__main__":
    cold_start()
