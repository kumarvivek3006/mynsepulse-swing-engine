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


def _intraday() -> dict:
    from scan import run_scan
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
    from ingest import backfill_prices, connect, sync_delivery, sync_indices
    from nse_client import NSEClient
    from upstox_client import InstrumentMaster, UpstoxClient
    from scan import run_scan

    client, master = UpstoxClient(), InstrumentMaster()
    conn = connect()
    try:
        sync_indices(conn, client, master)
        backfill_prices(conn, client)
        try:
            sync_delivery(conn, NSEClient())
        except Exception as exc:
            log.warning("Delivery refresh failed (non-fatal): %s", exc)
    finally:
        conn.close()

    return run_scan(mode="postclose")


SLOTS = {
    "premarket": (PREMARKET_TIME, _premarket),
    "postclose": (POSTCLOSE_TIME, _postclose),
}


def start() -> BackgroundScheduler | None:
    global _scheduler
    if not SCHEDULER_ENABLED:
        log.info("Scheduler disabled by SCHEDULER_ENABLED")
        return None
    if _scheduler is not None:
        return _scheduler

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
