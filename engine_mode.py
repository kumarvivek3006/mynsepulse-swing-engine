"""
engine_mode.py — which engine publishes signals. ONE switch.

  ENGINE_MODE=legacy   (default)  the original scan engine publishes to `signals`.
                                  Engine 2 scans on its schedule in SHADOW (records, never publishes).
  ENGINE_MODE=engine2             Engine 2 publishes to `signals` and to the UI-facing
                                  last_scan_summary. The original run_scan is PAUSED: its scheduled
                                  slots keep loading data but run the old scan in SHADOW (inside a
                                  transaction that is rolled back, results recorded for comparison).

ROLLBACK: set ENGINE_MODE=legacy (or delete the variable) in Railway and let it redeploy (~1 minute).
The next scheduled slot is the original engine again. Nothing is deleted in either direction: the old
engine's files are untouched, Engine 2's signals already in the table stay until they expire.

Anything other than the two values above is treated as legacy and logged at ERROR: a typo must fail
SAFE (the validated-in-production engine), never silently publish from the new one.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("engine.mode")
LEGACY, ENGINE2 = "legacy", "engine2"


def current() -> str:
    raw = os.environ.get("ENGINE_MODE", LEGACY).strip().lower() or LEGACY
    if raw not in (LEGACY, ENGINE2):
        log.error("ENGINE_MODE=%r is not 'legacy' or 'engine2'; treating it as legacy (the safe engine)", raw)
        return LEGACY
    return raw


def is_engine2() -> bool:
    return current() == ENGINE2
