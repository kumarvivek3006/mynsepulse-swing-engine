"""
Indian market holiday calendar.

The engine previously had no concept of a market holiday. On a holiday
ensure_bars_current() waits for a bar that will never arrive, triggers a
500-symbol backfill, and the post-close scan then runs against stale data.

Source of truth is the Upstox Market Information API (public, CDN-cached,
no auth). A bundled JSON is the fallback. Which one was used is always
recorded — a silent fallback that drifts out of date is worse than a
visible failure.

MUHURAT is deliberately not a holiday. Upstox classifies it as
SPECIAL_TIMING, and its session is a ~1 hour evening window. The date moves
every year with the lunar calendar, so it is fetched, never hardcoded.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path

import requests

log = logging.getLogger(__name__)

UPSTOX_MARKET_BASE = os.environ.get(
    "UPSTOX_MARKET_BASE", "https://api.upstox.com/v2/market")
HOLIDAY_FETCH_TIMEOUT = float(os.environ.get("HOLIDAY_FETCH_TIMEOUT", "10"))
FALLBACK_PATH = Path(__file__).with_name("holidays_fallback.json")

# Normal NSE equity session.
NORMAL_OPEN, NORMAL_CLOSE = time(9, 15), time(15, 30)

# A SPECIAL_TIMING session starting this late is a Muhurat window rather
# than, say, a shortened morning session. Kept as a threshold rather than a
# date because the date moves every year.
MUHURAT_MIN_OPEN_HOUR = int(os.environ.get("MUHURAT_MIN_OPEN_HOUR", "17"))


# ---------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------
def _get(path: str) -> dict | None:
    url = f"{UPSTOX_MARKET_BASE}{path}"
    try:
        r = requests.get(url, headers={"Accept": "application/json"},
                         timeout=HOLIDAY_FETCH_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("market calendar fetch failed %s: %s", url, str(exc)[:200])
        return None


def _parse_date(raw) -> date | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(str(raw).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _parse_time(raw) -> time | None:
    """
    Accepts "09:15", "09:15:00", and epoch milliseconds — Upstox has used
    more than one representation across its market endpoints.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)) or str(raw).isdigit():
        try:
            ms = int(raw)
            # Heuristic: anything this large is an epoch timestamp.
            if ms > 10_000_000:
                return datetime.fromtimestamp(ms / 1000).time()
        except (TypeError, ValueError, OSError):
            return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(str(raw).strip(), fmt).time()
        except ValueError:
            continue
    return None


def fetch_upstox_holidays() -> list[dict]:
    """
    GET /v2/market/holidays — CURRENT YEAR ONLY.

    That limitation is why the year-boundary check exists: on 1 January the
    table holds last year's list and nothing else.

    Returns [] on failure rather than raising, so a fetch problem degrades
    to the fallback instead of stopping a scan.
    """
    payload = _get("/holidays")
    if not payload:
        return []

    data = payload.get("data", payload)
    if not isinstance(data, list):
        log.warning("holidays: unexpected payload shape %s",
                    type(data).__name__)
        return []

    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        d = _parse_date(item.get("date") or item.get("holiday_date"))
        if d is None:
            continue
        # Only NSE equity matters here. An MCX-only closure is not a
        # holiday for this engine.
        segments = item.get("closed_exchanges") or item.get("exchanges") or []
        if segments and not any("NSE" in str(s).upper() for s in segments):
            continue
        out.append({
            "trade_date": d,
            "description": item.get("description") or item.get("holiday_name"),
            "holiday_type": (item.get("holiday_type")
                             or item.get("type") or "TRADING_HOLIDAY"),
        })
    return out


def fetch_upstox_timings(d: date) -> dict | None:
    """
    GET /v2/market/timings/{date}. Returns {segment: {open, close}} or None.

    None means "could not determine", which callers must treat as a full
    holiday — see sync_holidays. Guessing a session window is how the
    engine would end up trading a day the exchange was shut.
    """
    payload = _get(f"/timings/{d.isoformat()}")
    if not payload:
        return None

    data = payload.get("data", payload)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return None

    out: dict = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        seg = (item.get("exchange") or item.get("exchange_segment")
               or item.get("segment") or "NSE")
        o = _parse_time(item.get("start_time") or item.get("open"))
        c = _parse_time(item.get("end_time") or item.get("close"))
        if o and c:
            out[str(seg).upper()] = {"open": o, "close": c}
    return out or None


def _nse_window(timings: dict | None) -> tuple[time, time] | None:
    if not timings:
        return None
    for key in ("NSE_EQ", "NSE", "NSE_EQUITY", "NSE_INDEX"):
        if key in timings:
            return timings[key]["open"], timings[key]["close"]
    first = next(iter(timings.values()), None)
    return (first["open"], first["close"]) if first else None


def classify_special_timing(d: date) -> tuple[bool, tuple[time, time] | None]:
    """
    Is this SPECIAL_TIMING date a Muhurat session?

    Returns (is_muhurat, window). FAILS CLOSED: if timings cannot be
    fetched, returns (False, None) so the caller records a full holiday.
    Assuming a session exists when the API is unreachable would have the
    engine scan on a day the market never opened.
    """
    window = _nse_window(fetch_upstox_timings(d))
    if window is None:
        log.warning("timings unavailable for SPECIAL_TIMING %s — "
                    "treating as full holiday (fail closed)", d)
        return False, None
    return (window[0].hour >= MUHURAT_MIN_OPEN_HOUR), window


# ---------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------
def load_fallback(year: int | None = None) -> list[dict]:
    """Bundled JSON. Returns [] if absent or unparseable."""
    if not FALLBACK_PATH.exists():
        log.warning("holidays_fallback.json not found at %s", FALLBACK_PATH)
        return []
    try:
        blob = json.loads(FALLBACK_PATH.read_text())
    except Exception as exc:
        log.error("holidays_fallback.json unreadable: %s", exc)
        return []

    years = blob.get("years", {})
    keys = [str(year)] if year else list(years)
    out = []
    for k in keys:
        for item in years.get(k, []):
            d = _parse_date(item.get("date"))
            if d is None:
                continue
            out.append({
                "trade_date": d,
                "description": item.get("description"),
                "holiday_type": item.get("type", "TRADING_HOLIDAY"),
                "open": _parse_time(item.get("open")),
                "close": _parse_time(item.get("close")),
            })
    return out


# ---------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------
def sync_holidays(conn, year: int | None = None) -> dict:
    """
    Fetch, classify and upsert. Falls back to the bundled file on failure.
    """
    year = year or date.today().year
    rows = fetch_upstox_holidays()
    source = "upstox_api"

    if not rows:
        rows = load_fallback(year)
        source = "fallback" if rows else "failed"
        log.warning("holiday sync using %s for %s", source, year)

    muhurat_detected, written = False, 0
    for row in rows:
        d = row["trade_date"]
        if d.year != year:
            continue
        htype = str(row.get("holiday_type") or "TRADING_HOLIDAY").upper()
        is_muhurat_day = False

        if "SPECIAL" in htype:
            if source == "upstox_api":
                is_muhurat_day, window = classify_special_timing(d)
                if window is None:
                    # Fail closed: no timings means treat as a full holiday.
                    htype = "TRADING_HOLIDAY"
            else:
                o = row.get("open")
                is_muhurat_day = bool(o and o.hour >= MUHURAT_MIN_OPEN_HOUR)
            muhurat_detected = muhurat_detected or is_muhurat_day

        with conn.cursor() as cur:
            cur.execute("""
                insert into trading_holidays
                    (trade_date, description, year, holiday_type,
                     is_muhurat, source, fetched_at)
                values (%s,%s,%s,%s,%s,%s, now())
                on conflict (trade_date) do update set
                    description  = excluded.description,
                    holiday_type = excluded.holiday_type,
                    is_muhurat   = excluded.is_muhurat,
                    source       = excluded.source,
                    fetched_at   = now()
            """, (d, row.get("description"), d.year, htype,
                  is_muhurat_day, source))
            written += cur.rowcount
    conn.commit()

    log.info("Holiday sync %s: %d rows from %s (muhurat=%s)",
             year, written, source, muhurat_detected)
    return {"fetched": written, "source": source, "year": year,
            "muhurat_detected": muhurat_detected}


# ---------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------
def is_trading_holiday(conn, d: date) -> bool:
    """True only for a FULL closure. A Muhurat day returns False."""
    with conn.cursor() as cur:
        cur.execute("""
            select 1 from trading_holidays
            where trade_date = %s and coalesce(is_muhurat, false) = false
        """, (d,))
        return cur.fetchone() is not None


def is_muhurat(conn, d: date) -> bool:
    with conn.cursor() as cur:
        cur.execute("select coalesce(is_muhurat, false) "
                    "from trading_holidays where trade_date = %s", (d,))
        row = cur.fetchone()
        return bool(row and row[0])


def holiday_description(conn, d: date) -> str | None:
    with conn.cursor() as cur:
        cur.execute("select description from trading_holidays "
                    "where trade_date = %s", (d,))
        row = cur.fetchone()
        return row[0] if row else None


def session_window(conn, d: date) -> tuple[time, time] | None:
    """
    (open, close) for a tradeable day, None for a holiday or weekend.
    """
    if d.weekday() >= 5:
        return None
    if is_muhurat(conn, d):
        window = _nse_window(fetch_upstox_timings(d))
        if window:
            return window
        # Known Muhurat day but timings unreachable: the typical window is
        # returned rather than None, since the table already records the
        # session as existing. Logged so the assumption is visible.
        log.warning("Muhurat %s: timings unavailable, assuming 18:15-19:15", d)
        return time(18, 15), time(19, 15)
    if is_trading_holiday(conn, d):
        return None
    return NORMAL_OPEN, NORMAL_CLOSE


def expected_last_session(conn, now: datetime | None = None) -> date:
    """
    Most recent date whose CLOSING bar should exist.

    Walks backwards skipping weekends and full holidays. A Muhurat day IS a
    session and is not skipped.

    Replaces a version that only skipped weekends — which is why a holiday
    made the engine wait for a bar that would never arrive and then
    backfill 500 symbols looking for it.
    """
    now = now or datetime.now()
    d = now.date()

    # Today only counts once its close has passed.
    window = session_window(conn, d)
    if window is None or now.time() < window[1]:
        d -= timedelta(days=1)

    for _ in range(30):                       # generous: covers any cluster
        if d.weekday() < 5 and not is_trading_holiday(conn, d):
            return d
        d -= timedelta(days=1)
    return d


def next_holiday(conn, after: date | None = None) -> dict | None:
    after = after or date.today()
    with conn.cursor() as cur:
        cur.execute("""
            select trade_date, description, is_muhurat
            from trading_holidays
            where trade_date > %s order by trade_date limit 1
        """, (after,))
        row = cur.fetchone()
    if not row:
        return None
    return {"date": str(row[0]), "description": row[1],
            "is_muhurat": bool(row[2]), "days_away": (row[0] - after).days}


def calendar_health(conn, year: int | None = None) -> dict:
    """Fields for the daily summary, plus the alert conditions."""
    year = year or date.today().year
    with conn.cursor() as cur:
        cur.execute("""
            select count(*), max(fetched_at),
                   max(source) filter (where source is not null),
                   min(trade_date) filter (where is_muhurat)
            from trading_holidays where year = %s
        """, (year,))
        count, fetched_at, source, muhurat = cur.fetchone()

    days_since = ((date.today() - fetched_at.date()).days
                  if fetched_at else None)
    alerts = []
    if (count or 0) < 10:
        alerts.append(f"only {count} holidays loaded for {year}")
    if days_since is not None and days_since > 35:
        alerts.append(f"last sync {days_since} days ago")
    if source == "fallback":
        alerts.append("using bundled fallback, not the API")
    if muhurat is None and date.today().month >= 12:
        alerts.append("Muhurat not detected and it is December")

    return {
        "holidays_loaded_current_year": count or 0,
        "muhurat_date": str(muhurat) if muhurat else None,
        "last_holiday_sync": str(fetched_at) if fetched_at else None,
        "holiday_source": source,
        "next_holiday": next_holiday(conn),
        "alerts": alerts,
    }
