"""
Scan scheduler.

Runs in-process rather than as Railway cron jobs, because a cron service
would need its own container and Railway volumes attach to exactly one
service — and the Upstox token lives on this one's volume.

Three slots, each with a different job:

  08:15  premarket   Refresh data, then publish the ARMED WATCHLIST — the
                     setups coiling under a pivot, with entry triggers to
                     place as resting stop orders before the open. This is
                     the slot that matters. A stop order above the pivot
                     cannot miss the breakout; a scan cannot be fast
                     enough to catch it.

  15:00  intraday    Evaluate today's forming bar so a breakout confirming
                     now can be acted on in the last half hour, rather
                     than a day late.

  15:45  postclose   Definitive scan on the completed daily bar. Supersedes
                     the day's earlier runs and arms tomorrow.

Weekends are skipped. Exchange holidays are not enumerated — on a holiday
there is no fresh bar, and the scan simply reproduces the previous
session rather than inventing anything.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from upstox_client import IST

log = logging.getLogger(__name__)

SCHEDULER_ENABLED = os.environ.get("SCHEDULER_ENABLED", "true").lower() == "true"
PREMARKET_TIME = os.environ.get("PREMARKET_TIME_IST", "08:15")
# Hourly through the session. The first is at 10:00, not 09:15 — a bar
# forty-five minutes old carries enough volume for the confirmation test
# to mean something, whereas one five minutes old does not.
INTRADAY_HOURS = os.environ.get("INTRADAY_SCAN_HOURS", "10,11,12,13,14,15")
# Upstox does not publish the completed daily candle at 15:30. It appears a
# few hours later. Running the definitive scan at 15:45 fetched nothing, so
# the newest bar stayed a day behind and the scan silently analysed the
# PREVIOUS session — which is exactly what was observed.
POSTCLOSE_TIME = os.environ.get("POSTCLOSE_SCAN_IST", "18:30")

# Monthly maintenance jobs. Each is independently switchable.
UNIVERSE_SYNC_ENABLED = os.environ.get("UNIVERSE_SYNC_ENABLED", "true").lower() == "true"
HOLIDAY_SYNC_ENABLED = os.environ.get("HOLIDAY_SYNC_ENABLED", "true").lower() == "true"

_scheduler: BackgroundScheduler | None = None
_last_runs: dict[str, dict] = {}


def _record(slot: str, status: str, detail: dict | None = None) -> None:
    _last_runs[slot] = {
        "status": status,
        "at": datetime.now(IST).isoformat(),
        "detail": detail or {},
    }


def _guarded(slot: str, fn) -> None:
    """A failing slot must never kill the scheduler thread."""
    log.info("Scheduled slot starting: %s", slot)
    try:
        result = fn()
        _record(slot, "success", result if isinstance(result, dict) else None)
        log.info("Slot %s finished", slot)
    except Exception as exc:
        _record(slot, "failed", {"error": str(exc)[:400]})
        log.exception("Slot %s failed", slot)
        # _last_runs is memory-only — a restart clears it, and this was
        # the only record of the failure existing anywhere. A short-lived
        # connection opened only on this (rare) path persists it to
        # ingestion_runs, visible in the DB rather than stderr alone.
        try:
            from ingest import connect, _run_log
            conn = connect()
            try:
                _run_log(conn, slot, "failed", 0, str(exc)[:500])
            finally:
                conn.close()
        except Exception:
            log.exception("Also failed to persist %s failure to ingestion_runs", slot)


# ---------------------------------------------------------------------
def _premarket() -> dict:
    """
    Refresh the inputs that change overnight, then arm the watchlist.

    Surveillance and corporate actions move daily and both can disqualify
    a setup, so they are refreshed before anything is published. Prices
    are not re-fetched — yesterday's close is already stored.
    """
    from nse_client import NSEClient
    from ingest import connect, sync_corporate_actions, sync_surveillance
    from scan import run_scan

    nse = NSEClient()
    conn = connect()
    try:
        try:
            sync_surveillance(conn, nse)
        except Exception as exc:
            log.warning("Premarket surveillance refresh failed: %s", exc)
        try:
            sync_corporate_actions(conn, nse, years=1)
        except Exception as exc:
            log.warning("Premarket corporate actions refresh failed: %s", exc)
    finally:
        conn.close()

    return run_scan(mode="premarket")


def _skip_today(mode: str) -> dict | None:
    """
    A result dict if this slot should not run today, else None.

    Exists because the token checks below run BEFORE run_scan(), and
    run_scan() is where the holiday guard lives. On a weekday market
    holiday with an expired token — which is every weekday holiday, since
    nobody logs in to Upstox for a closed market — the token check won and
    reported failure. 2 Oct 2026 (Gandhi Jayanti): the 08:21 IST premarket
    slot logged "Holiday ... no scan", then six intraday slots and the
    postclose each raised token_invalid. Seven false failures, and a
    warning that cries wolf on ~14 weekdays a year trains the operator to
    ignore it on the day it is real.

    The result shape matches run_scan's own skip results, so a skip looks
    identical in _last_runs whichever layer made it.

    Deliberately NOT swallowed: if the calendar lookup itself fails, the
    exception propagates and _guarded records the slot as failed. Skipping
    silently on a lookup error would be the same silent-failure class this
    check exists to close.
    """
    from ingest import connect
    from market_calendar import holiday_description, is_muhurat, is_trading_holiday

    today = datetime.now(IST).date()
    conn = connect()
    try:
        if is_trading_holiday(conn, today):
            return {"as_of": str(today), "mode": mode,
                    "status": "skipped_holiday",
                    "reason": holiday_description(conn, today) or "market holiday",
                    "signals": 0, "universe": 0}
        # Muhurat is a real ~1 hour evening session. run_scan() skips the
        # intraday slot on it; that skip is behind the token check too.
        if mode == "intraday" and is_muhurat(conn, today):
            return {"as_of": str(today), "mode": mode,
                    "status": "skipped_muhurat_intraday",
                    "signals": 0, "universe": 0}
        return None
    finally:
        conn.close()


def _intraday() -> dict:
    from scan import run_scan
    from upstox_client import TokenStore

    # Holiday / Muhurat guard FIRST — before the token check. See _skip_today.
    skipped = _skip_today("intraday")
    if skipped is not None:
        log.info("Intraday slot skipped: %s", skipped.get("reason") or skipped["status"])
        return skipped

    # Intraday's Upstox dependency is entirely inside _forming_bar()
    # (scan.py) via client.intraday_today() — a path ensure_bars_current()
    # never touches, since intraday explicitly skips it. Without this,
    # an expired token here failed per-symbol inside the scan with no
    # upfront signal, and _guarded() recorded the slot "success" regardless.
    #
    # RAISES rather than degrading gracefully: unlike premarket, intraday
    # has no meaningful fallback — its entire purpose is today's forming
    # bar. Caught by _guarded(), which correctly marks the slot "failed".
    if TokenStore().valid_token() is None:
        raise RuntimeError("token_invalid")

    return run_scan(mode="intraday")


def _postclose() -> dict:
    """
    Fetch the completed session's bars and delivery data, then run the
    definitive scan.

    Delivery is refreshed here rather than as a separate job the user has
    to remember. A failure is logged and swallowed — delivery feeds a
    diagnostic and an opt-in path, so it must never stop the scan that
    produces the actual signals.
    """
    from ingest import (backfill_prices, connect, sync_delivery, sync_indices,
                        sync_today_from_intraday)
    from nse_client import NSEClient
    from upstox_client import InstrumentMaster, TokenStore, UpstoxClient
    from scan import run_scan

    # Holiday guard FIRST — before the token check, and before any Upstox
    # call. See _skip_today. Skipping the whole slot on a holiday also stops
    # sync_indices/backfill_prices hunting for bars a closed market never
    # produced, which is what this slot did on holidays before the token
    # check existed.
    skipped = _skip_today("postclose")
    if skipped is not None:
        log.info("Postclose slot skipped: %s", skipped.get("reason") or skipped["status"])
        return skipped

    # Checked BEFORE sync_indices/backfill_prices — both run ahead of
    # run_scan() and are entirely Upstox-dependent. Without this, a dead
    # token meant backfill_prices caught each of 500 per-symbol failures
    # and continued (documented in ensure_bars_current's own comment),
    # returning silently with zero bars gained — indistinguishable from a
    # quiet day. RAISES: postclose has no meaningful fallback either.
    if TokenStore().valid_token() is None:
        raise RuntimeError("token_invalid")

    client, master = UpstoxClient(), InstrumentMaster()
    conn = connect()
    try:
        sync_indices(conn, client, master)
        backfill_prices(conn, client)
        # Today's completed bar comes from the intraday endpoint; the
        # historical endpoint will not have it until tomorrow.
        try:
            sync_today_from_intraday(conn, client)
        except Exception as exc:
            log.warning("Today's bar from intraday failed: %s", exc)
        try:
            sync_delivery(conn, NSEClient())
        except Exception as exc:
            log.warning("Delivery refresh failed (non-fatal): %s", exc)
    finally:
        conn.close()

    return run_scan(mode="postclose")


def _sync_universe_job() -> dict:
    """
    Monthly refresh of the Nifty 500 universe, first Saturday 06:00 IST.

    sync_universe was only ever reachable through cold_start (a manual
    route), so the universe silently stopped refreshing — last run 18 Sep —
    and listings, delistings and ticker renames accumulated unseen between
    index rebalances.

    Needs NO Upstox token, which matters: this runs on a Saturday, hours
    after the 03:30 IST expiry, with no weekend login. NSE supplies the
    constituents; the Upstox instrument master is a public file.

    A refusal (more than MAX_UNIVERSE_RETIREMENTS keys changing owner, or
    under 90% of constituents resolving) is NOT caught here. It propagates
    to _guarded, which records the slot as failed in _last_runs and
    persists it to ingestion_runs. A sync that skipped its writes and
    reported success would be the silent-failure class this build has spent
    its whole length closing. Nothing is written before the refusal.

    New symbols have no bars until Monday's postclose, where backfill_prices
    fetches their full history by instrument key; until then they appear as
    insufficient_history, which is harmless.
    """
    from ingest import _run_log, connect, sync_universe
    from nse_client import NSEClient
    from upstox_client import InstrumentMaster

    conn = connect()
    try:
        n = sync_universe(conn, NSEClient(), InstrumentMaster())
        _run_log(conn, "sync_universe", "success", n)
        return {"symbols": n}
    except Exception:
        conn.rollback()
        raise                      # _guarded records and persists the failure
    finally:
        conn.close()


def _sync_holidays_job() -> dict:
    """
    Monthly market-holiday refresh, first Monday 06:00 IST.

    Upstox's holidays endpoint returns the CURRENT YEAR ONLY, so without this
    the calendar cannot learn next year's holidays, and the holiday guard
    would let a scan run against a closed market on 1 Jan onward.

    An empty result, or one that came from the bundled fallback instead of
    the API, is raised rather than reported as success. The fallback holds
    only the five fixed-date holidays, so a run that "succeeded" on it would
    look healthy while the lunar dates (Holi, Diwali, Eid ...) went stale.
    The rows are already committed when this raises; only the run record is
    affected.
    """
    from ingest import _run_log, connect
    from market_calendar import sync_holidays

    conn = connect()
    try:
        result = sync_holidays(conn)
        fetched, source = result.get("fetched", 0), result.get("source")
        if not fetched:
            raise RuntimeError(f"holiday_sync_empty: source={source}")
        if source != "upstox_api":
            raise RuntimeError(f"holiday_sync_degraded: source={source}, "
                               f"{fetched} rows (fixed-date holidays only)")
        _run_log(conn, "sync_holidays", "success", fetched)
        return result
    finally:
        conn.close()


SLOTS = {
    "premarket": (PREMARKET_TIME, _premarket),
    "postclose": (POSTCLOSE_TIME, _postclose),
}


# ---------------------------------------------------------------------
# Kill switch
#
# Persistent via a row in engine_settings, not a flag file. engine_settings
# is already the established pattern in this codebase for exactly this
# kind of small persistent engine state (last_scan_summary uses it the
# same way) — queryable through the same Supabase interface already used
# for everything else here, and introduces no new dependency: the kill
# endpoint already needs DB access to do anything.
#
# DB read failure at start() defaults to STARTING, not staying killed.
# The DB being briefly unreachable at boot must not silently leave the
# whole engine dark with no scan running and no visible reason — that is
# a worse failure mode than the rare case of an intended kill not
# surviving a startup race with a DB outage.
# ---------------------------------------------------------------------
def _kill_switch_active(conn=None) -> bool:
    own_conn = conn is None
    try:
        # Connection acquisition moved INSIDE the try. It was outside on
        # the first pass — tested directly, and a connect() failure
        # propagated straight out uncaught instead of degrading to "not
        # killed", which is the entire point of this function on a DB
        # outage.
        if own_conn:
            from ingest import connect
            conn = connect()
        with conn.cursor() as cur:
            cur.execute("select value from engine_settings "
                       "where key = 'kill_switch'")
            row = cur.fetchone()
        return bool(row and row[0] and row[0].get("active"))
    except Exception:
        log.exception("Kill-switch check failed — defaulting to NOT killed "
                      "(starting the scheduler rather than staying dark)")
        return False
    finally:
        if own_conn and conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def start() -> BackgroundScheduler | None:
    global _scheduler
    if not SCHEDULER_ENABLED:
        log.info("Scheduler disabled by SCHEDULER_ENABLED")
        return None
    if _scheduler is not None:
        return _scheduler
    if _kill_switch_active():
        log.warning("Kill switch active — scheduler NOT started. "
                   "POST /jobs/kill-switch/reset to clear.")
        return None

    _scheduler = BackgroundScheduler(timezone=IST)
    for slot, (hhmm, fn) in SLOTS.items():
        hour, minute = (int(x) for x in hhmm.split(":"))
        _scheduler.add_job(
            _guarded, CronTrigger(day_of_week="mon-fri", hour=hour, minute=minute,
                                  timezone=IST),
            args=[slot, fn], id=slot, replace_existing=True,
            misfire_grace_time=900,   # a restart near the slot still runs it
            coalesce=True,            # never run a backlog twice
            max_instances=1,
        )
        log.info("Scheduled %s at %s IST (Mon-Fri)", slot, hhmm)

    # Hourly intraday scans. A run that cannot see the market raises
    # ScanAborted and leaves existing signals alone, so a holiday costs
    # nothing.
    hours = [h.strip() for h in INTRADAY_HOURS.split(",") if h.strip()]
    if hours:
        _scheduler.add_job(
            _guarded,
            CronTrigger(day_of_week="mon-fri", hour=",".join(hours), minute=0,
                        timezone=IST),
            args=["intraday", _intraday], id="intraday", replace_existing=True,
            misfire_grace_time=600, coalesce=True, max_instances=1,
        )
        log.info("Scheduled intraday hourly at %s:00 IST (Mon-Fri)",
                 ", ".join(hours))

    # Monthly maintenance. APScheduler AND-s day with day_of_week (Unix cron
    # ORs them), so "day 1-7 + Saturday" is exactly the first Saturday and
    # /jobs/schedule reports the true next run instead of a misleading weekly
    # one. misfire_grace_time is 6h: a deploy landing on the 06:00 slot must
    # not push a monthly job out by a whole month.
    if UNIVERSE_SYNC_ENABLED:
        _scheduler.add_job(
            _guarded,
            CronTrigger(day="1-7", day_of_week="sat", hour=6, minute=0,
                        timezone=IST),
            args=["sync_universe", _sync_universe_job], id="sync_universe",
            replace_existing=True, misfire_grace_time=21600,
            coalesce=True, max_instances=1,
        )
        log.info("Scheduled sync_universe first Saturday 06:00 IST")

    if HOLIDAY_SYNC_ENABLED:
        _scheduler.add_job(
            _guarded,
            CronTrigger(day="1-7", day_of_week="mon", hour=6, minute=0,
                        timezone=IST),
            args=["sync_holidays", _sync_holidays_job], id="sync_holidays",
            replace_existing=True, misfire_grace_time=21600,
            coalesce=True, max_instances=1,
        )
        log.info("Scheduled sync_holidays first Monday 06:00 IST")

    _scheduler.start()
    return _scheduler


def status() -> dict:
    jobs = []
    if _scheduler:
        for job in _scheduler.get_jobs():
            jobs.append({
                "slot": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })
    return {
        "enabled": SCHEDULER_ENABLED,
        "now_ist": datetime.now(IST).isoformat(),
        "slots": {**{k: v[0] for k, v in SLOTS.items()},
                  "intraday": f"hourly {INTRADAY_HOURS}"},
        "jobs": jobs,
        "last_runs": _last_runs,
    }
