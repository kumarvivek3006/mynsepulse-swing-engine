"""
data_probe.py — Step 1.5: is historical INTRADAY data available, and how far back?

The engine stores daily bars only. UpstoxClient has intraday_today() (the
current session) and historical_daily(); nothing fetches or stores past
intraday bars. Whether Upstox SERVES them for past dates, and for how far back
and at which intervals, was an open question that the Episodic Pivot detector
depends on (gap + opening-range behaviour). This answers it from the live API
rather than from a recollection of its documentation: one call per
(date, interval), coverage judged against a full session.

Read-only. Writes nothing. Needs a valid Upstox token (it will say so).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

SESSION_MINUTES = 375            # 09:15-15:30 IST on a normal day
FULL_COVERAGE = 0.95             # >= 95% of the bars a full session would have

# Sampled lookbacks, in days. Dense near today, sparse far back; 1095 is the
# 3-year backtest window and 1825 is 5 years.
DEFAULT_LOOKBACKS = (1, 7, 30, 90, 180, 365, 730, 1095, 1825)
DEFAULT_INTERVALS = (1, 5, 15)


def intraday_path(instrument_key: str, interval_min: int, d: date) -> str:
    """Same URL shape historical_daily() uses, with the minutes unit."""
    return (f"/v3/historical-candle/{instrument_key}/minutes/{interval_min}/"
            f"{d.isoformat()}/{d.isoformat()}")


def summarise_candles(payload: dict, interval_min: int) -> dict:
    """One probe cell: what came back for one (date, interval)."""
    expected = SESSION_MINUTES // interval_min
    if not isinstance(payload, dict):
        return {"ok": False, "candles": 0, "expected": expected, "error": "non-dict response"}
    if payload.get("status") != "success":
        errs = payload.get("errors") or payload.get("message") or payload
        return {"ok": False, "candles": 0, "expected": expected,
                "error": str(errs)[:160]}
    candles = (payload.get("data") or {}).get("candles") or []
    n = len(candles)
    cell = {"candles": n, "expected": expected,
            "coverage": round(n / expected, 3) if expected else None}
    if n:
        stamps = sorted(c[0] for c in candles)
        cell["first"], cell["last"] = str(stamps[0])[11:16], str(stamps[-1])[11:16]
    cell["ok"] = bool(n and n / expected >= FULL_COVERAGE)
    return cell


def pick_probe_dates(conn, today: date, lookbacks=DEFAULT_LOOKBACKS) -> list[date]:
    """Nearest earlier trading day for each lookback (weekday, not a full holiday)."""
    from market_calendar import is_trading_holiday
    out = []
    for lb in lookbacks:
        d = today - timedelta(days=lb)
        for _ in range(10):
            if d.weekday() < 5 and not is_trading_holiday(conn, d):
                break
            d -= timedelta(days=1)
        if d not in out:
            out.append(d)
    return out


def verdict(matrix: dict) -> dict:
    """
    matrix: {interval_min: [(date, cell), ...]} ordered newest to oldest.
    For each interval: the oldest probed date with FULL coverage that has no
    failing date NEWER than it (a gap in the middle would otherwise be hidden).
    """
    out = {}
    for interval, rows in matrix.items():
        usable_back_to = None
        first_failure = None
        for d, cell in rows:                    # newest -> oldest
            if cell.get("ok"):
                if first_failure is None:
                    usable_back_to = d
            elif first_failure is None:
                first_failure = d
        out[str(interval)] = {
            "full_sessions_back_to": str(usable_back_to) if usable_back_to else None,
            "first_probed_date_that_failed": str(first_failure) if first_failure else None,
            "usable_for_3y_backtest": bool(usable_back_to and
                                           (rows[0][0] - usable_back_to).days >= 1090),
        }
    return out


def probe_intraday(client, conn, symbol: str = "RELIANCE",
                   intervals=DEFAULT_INTERVALS, today: date | None = None,
                   lookbacks=DEFAULT_LOOKBACKS) -> dict:
    today = today or date.today()
    with conn.cursor() as cur:
        cur.execute("select upstox_instrument_key from symbols where symbol = %s", (symbol,))
        row = cur.fetchone()
    if not row or not row[0]:
        return {"error": f"no instrument key stored for {symbol}"}
    key = row[0]

    dates = pick_probe_dates(conn, today, lookbacks)
    matrix: dict[int, list] = {i: [] for i in intervals}
    for d in dates:
        for i in intervals:
            try:
                payload = client._get(intraday_path(key, i, d))
                cell = summarise_candles(payload, i)
            except Exception as exc:                 # a dead token must be loud
                if type(exc).__name__ == "TokenExpired":
                    raise
                cell = {"ok": False, "candles": 0, "error": str(exc)[:160]}
            matrix[i].append((d, cell))

    return {
        "symbol": symbol, "instrument_key": key, "probed_at": datetime.now().isoformat(timespec="seconds"),
        "dates_probed": [str(d) for d in dates],
        "verdict": verdict(matrix),
        "cells": {str(i): [{"date": str(d), **c} for d, c in rows] for i, rows in matrix.items()},
        "stored_intraday_history": False,
        "note": ("The engine stores NO intraday history; this probe only shows what the API "
                 "could supply. Using it would mean a new fetch-and-store path."),
    }
