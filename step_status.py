"""
step_status.py — ONE read-only answer to "is Step 1 done, and if not, why not".

GET /jobs/step-status returns everything in one JSON so nobody has to run SQL
blocks or compare several endpoints. Step 1 closes on three facts only:

  1. the last COMPLETED scan scanned the corrected universe (its count equals
     what the current rule yields from the symbols table, 497 today)
  2. that scan's summary carries universe_detail (i.e. the new scan.py ran)
  3. the scheduled fundamentals refresh ran on the new code (its own summary
     row in ingestion_runs; a manual /jobs/fundamentals run cannot satisfy it)

When a fact is not yet true the response says WHY and WHEN it will resolve on
its own (the next scheduled slot), so waiting needs no action from the user.
Read-only: it writes nothing.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
BRIEF_TARGET_UNIVERSE = 497       # 500 in the index less the three REITs (informational)
_STARTED = datetime.now(IST)       # import time of this module == process start


def _setting(conn, key):
    with conn.cursor() as cur:
        cur.execute("select value, updated_at from engine_settings where key = %s", (key,))
        row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def _last_run(conn, job):
    with conn.cursor() as cur:
        cur.execute("select finished_at, status, rows_written, error from ingestion_runs "
                    "where job = %s order by finished_at desc limit 1", (job,))
        row = cur.fetchone()
    if not row:
        return None
    return {"finished_at": row[0].isoformat() if row[0] else None, "status": row[1],
            "rows_written": row[2], "error": row[3]}


def _iso(x):
    return x.isoformat() if hasattr(x, "isoformat") else x


def build(conn, *, next_runs: dict | None = None, token_valid: bool | None = None,
          env: dict | None = None, now: datetime | None = None,
          started_at: datetime | None = None) -> dict:
    from universe import load_scan_universe
    import scan as _scan

    next_runs = next_runs or {}
    env = env or {}
    now = now or datetime.now(IST)
    started_at = started_at or _STARTED

    live = load_scan_universe(conn).detail

    # --- scans -------------------------------------------------------
    completed, completed_at = _setting(conn, "last_completed_scan")
    latest, latest_at = _setting(conn, "last_scan_summary")
    summary, summary_at = (completed, completed_at) if completed else (latest, latest_at)
    summary = summary or {}
    detail = summary.get("universe_detail")
    predates = bool(summary_at and summary_at < started_at)
    scan = {
        "last_completed_at": _iso(summary_at), "mode": summary.get("mode"),
        "as_of": summary.get("as_of"), "universe": summary.get("universe"),
        "universe_detail": detail, "signals": summary.get("signals"),
        "regime": summary.get("regime"), "breadth_above_50dma": summary.get("breadth_above_50dma"),
        "predates_this_deploy": predates,
        "from_last_completed_key": bool(completed),
        "latest_event": ({"status": latest.get("status") or "completed", "reason": latest.get("reason"),
                          "at": _iso(latest_at)} if latest else None),
        "next_scheduled": {k: next_runs.get(k) for k in ("premarket", "intraday", "postclose")},
    }
    ran_new = isinstance(detail, dict) and "scanned" in detail

    # --- fundamentals refresh ---------------------------------------
    fr = _last_run(conn, "refresh_fundamentals")
    fund = {"last_run": fr, "next_scheduled": next_runs.get("refresh_fundamentals"),
            "ran_on_new_code": bool(fr and fr["status"] == "success")}

    # --- data quality ------------------------------------------------
    dq, dq_at = _setting(conn, "data_quality_latest")
    dqs = ({"last_run_at": _iso(dq_at), "severity": dq.get("severity"),
            "stale_bars": (dq.get("stale") or {}).get("count"),
            "stale_pct": (dq.get("stale") or {}).get("pct"),
            "critical": dq.get("critical"), "warnings": dq.get("warnings"),
            "expected_last_session": (dq.get("stale") or {}).get("expected_last_session")}
           if dq else {"last_run_at": None, "status": "pending"})
    dqs["next_scheduled"] = next_runs.get("data_quality")

    # --- intraday probe ----------------------------------------------
    pr, pr_at = _setting(conn, "intraday_probe_latest")
    probe = ({"status": "done", "at": _iso(pr_at), "verdict": pr.get("verdict"),
              "error": pr.get("error")} if pr else
             {"status": "pending",
              "runs_automatically": "with the next 19:00 IST data-quality slot that has a valid Upstox token",
              "next_attempt": next_runs.get("data_quality")})

    # --- the three facts ----------------------------------------------
    expected = live["scanned"]
    c1 = bool(summary.get("universe") == expected and ran_new)
    c2 = ran_new
    c3 = fund["ran_on_new_code"]
    blocking = []
    if not (c1 and c2):
        first_scan = min([v for v in scan["next_scheduled"].values() if v], default=None)
        if not summary:
            why = "no completed scan has been recorded yet"
        elif predates and not ran_new:
            why = "the last recorded scan predates this deployment (old code)"
        elif not ran_new:
            why = "a scan ran on this deployment but its summary has no universe_detail (unexpected: investigate)"
        else:
            why = f"last scan universe {summary.get('universe')} != corrected universe {expected}"
        blocking.append({"fact": "scan on the new code", "why": why,
                         "resolves": f"automatically at the next scheduled scan ({first_scan})"
                                     if first_scan else "no scan is scheduled (scheduler off or kill switch active)"})
    if not c3:
        why = ("last refresh FAILED: " + str(fr["error"])[:200]) if fr and fr["status"] != "success" \
            else "the scheduled refresh has not run on this code yet"
        res = f"automatically at {fund['next_scheduled']}" if fund["next_scheduled"] else "not scheduled (disabled, or no Friday left in a results month)"
        if token_valid is False:
            res += "; NOTE: no valid Upstox token right now, so log in before then"
        blocking.append({"fact": "fundamentals refresh on the new code", "why": why, "resolves": res})

    return {
        "generated_at": now.isoformat(),
        "step1": {
            "closed": c1 and c2 and c3,
            "facts": {
                "scanned_universe_is_corrected": {"ok": c1, "last_scan_universe": summary.get("universe"),
                                                  "corrected_universe_now": expected,
                                                  "brief_target": BRIEF_TARGET_UNIVERSE,
                                                  "matches_brief": expected == BRIEF_TARGET_UNIVERSE},
                "universe_detail_populated": {"ok": c2},
                "fundamentals_refresh_ran_on_new_code": {"ok": c3},
            },
            "blocking": blocking,
        },
        "universe": {"symbols_table_now": live,
                     "last_scan": {"universe": summary.get("universe"), "universe_detail": detail}},
        "scan": scan,
        "deploy": {"process_started_at": started_at.isoformat(),
                   "commit": (env.get("RAILWAY_GIT_COMMIT_SHA") or "")[:7] or None,
                   "deployment_id": env.get("RAILWAY_DEPLOYMENT_ID"),
                   "running_code": {"universe_filter": hasattr(_scan, "load_scan_universe"),
                                    "persists_scan_summary": hasattr(_scan, "store_summary"),
                                    "scheduled_slots": sorted(next_runs)},
                   "upstox_token_valid": token_valid},
        "fundamentals_refresh": fund,
        "data_quality": dqs,
        "intraday_probe": probe,
    }
