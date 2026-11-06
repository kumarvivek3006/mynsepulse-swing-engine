"""
e2_service.py — runs engine 2 on the server and exposes what it found. Nothing here touches the
live strategy engine, the signals table or the live scan.

  Validation   3-year study of all 20 detectors on the scan universe. Runs in a SEPARATE PROCESS
               (a crash or an out-of-memory kill cannot take the web service down), starts by
               itself the first night after a deploy of new engine-2 code (22:30 IST), and
               retries on later nights (up to 3 attempts) if it fails.
  Auto-scan    unattended, Mon-Fri, no manual trigger (see e2_scan.py for what each slot does):
                 engine2_premarket   08:00 IST   prior completed session -> armed watchlist
                 engine2_intraday    10:10..15:10 IST hourly, today's forming bar (provisional)
                 engine2_postclose   18:30 IST   definitive scan on the completed bar -> tomorrow's watchlist
               Each runs in its own process, is recorded in engine2_last_scan_summary / engine2_scan_runs with
               full diagnostics, and a failed slot raises so the scheduler's _last_runs shows a specific reason.
               Shadow: records, never publishes to `signals`.

  GET  /jobs/engine2-status          progress + report headline + auto-scan health (last runs, next runs)
  GET  /jobs/engine2-report          the full report (JSON);  ?format=md for a readable version
  GET  /jobs/engine2-shadow          the latest completed-session scan (watchlist and hypothetical book)
  GET  /jobs/engine2-diagnosis       cup_handle / ascending_base / flag_pennant / asc_triangle: old engine vs corrected definition
  GET  /jobs/engine2-scan            the latest scan summary and the recent runs;  ?mode=intraday for the forming-bar scan
  GET  /jobs/engine2-candidates      per-candidate score, rank, selected / why not, vs the day's selected scores and cut-off;  ?detector=a,b&since=YYYY-MM-DD
  GET  /jobs/engine2-components      the seven score components (points) of every ranked candidate, per detector means, selected vs not;  ?detector=a,b&since=&until=
  POST /jobs/engine2/validate        start a validation run now (returns immediately)
  POST /jobs/engine2/scan?mode=      run one scan now (premarket | intraday | postclose); returns immediately
  POST /jobs/engine2/shadow          same as scan?mode=postclose
  GET  /jobs/engine2-defdiag         pocket_pivot context breakdown (?format=md); logged as E2DEFDIAG| lines
  POST /jobs/engine2/defdiag         run the definition diagnosis now
  GET  /jobs/engine2-actions         the 6 Nov actions study: detector-specific scoring, strict pocket_pivot, vcp armed port (?format=md)
  POST /jobs/engine2/actions         run it now (also runs ~20 minutes after a deploy while E2_ACTIONS_BOOT=1)
  GET  /jobs/engine2-exits           the 6 Nov exit redesign study: variants A-D vs the live exits (?format=md); logged as E2EXIT| lines
  GET  /jobs/engine2-fwd             the 7 Nov forward-return test: market-adjusted 5/10/20-session returns by detector, no stops (?format=md); E2FWD| lines
  POST /jobs/engine2/fwd             run it now (also runs ~24 minutes after a deploy while E2_FWD_BOOT=1)
  GET  /jobs/engine2-fwddiag         forward returns inside ONE detector split by signal-bar context (?detector=pullback_ema&format=md); E2FWDD| lines
  POST /jobs/engine2/fwddiag         run it now (?detector=); also ~28 minutes after a deploy for E2_FWDDIAG_DETECTOR (default pullback_ema) while E2_FWDDIAG_BOOT=1
  GET  /jobs/engine2-exits-wl        the 7 Nov exit study on the publishing detectors only + the 5-/20-session mode rulebooks (?format=md); E2EXW| lines
  POST /jobs/engine2/exits-wl        run it now (also ~32 minutes after a deploy while E2_EXITS_WL_BOOT=1)
  POST /jobs/engine2/exits           run it now (also runs ~22 minutes after a deploy while E2_EXITS_BOOT=1)
  POST /jobs/engine2/diagnose        re-run the disabled-pattern diagnosis now (it also runs ahead of every validation)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request
from fastapi.responses import PlainTextResponse

log = logging.getLogger("engine2.service")

REPORT_KEY, PROGRESS_KEY = "engine2_report", "engine2_progress"
ENABLED = os.environ.get("ENGINE2_ENABLED", "true").lower() == "true"
VALIDATE_TIME = os.environ.get("ENGINE2_VALIDATE_TIME_IST", "22:30")
PREMARKET_TIME = os.environ.get("ENGINE2_PREMARKET_TIME_IST", "08:00")
POSTCLOSE_TIME = os.environ.get("ENGINE2_POSTCLOSE_TIME_IST", "18:30")
INTRADAY_HOURS = os.environ.get("ENGINE2_INTRADAY_HOURS_IST", "10-15")
INTRADAY_MINUTE = int(os.environ.get("ENGINE2_INTRADAY_MINUTE_IST", "10"))     # staggered off HH:00, where the live intraday runs
SCAN_TIMEOUT_MIN = {"premarket": 45, "intraday": 25, "postclose": 75}
MAX_ATTEMPTS = 3
STALE_MIN = 45                 # a 'running' record not touched for this long belongs to a dead process
_lock = threading.Lock()


# ----------------------------------------------------------------------
# storage
# ----------------------------------------------------------------------
def _get(conn, key):
    with conn.cursor() as cur:
        cur.execute("select value, updated_at from engine_settings where key = %s", (key,))
        r = cur.fetchone()
    return (r[0], r[1]) if r else (None, None)


def _put(conn, key, value) -> None:
    with conn.cursor() as cur:
        cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (key, json.dumps(value, default=str)))
    conn.commit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_min(at) -> float | None:
    """Minutes since a timestamptz read from the DB, whatever the session time zone is."""
    if at is None:
        return None
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - at).total_seconds() / 60.0


def _progress(conn, **kw) -> dict:
    cur, _ = _get(conn, PROGRESS_KEY)
    cur = dict(cur or {})
    cur.update(kw)
    cur["updated"] = _now_iso()
    _put(conn, PROGRESS_KEY, cur)
    return cur


# ----------------------------------------------------------------------
# the work (runs in the child process)
# ----------------------------------------------------------------------
def run_validation_blocking(trigger: str = "manual") -> dict:
    import e2_backtest as BT
    from ingest import connect

    conn = connect()
    try:
        prev, _ = _get(conn, PROGRESS_KEY)
        same = (prev or {}).get("code_version") == BT.CODE_VERSION
        attempts = 1 if (trigger == "manual" or not same) else int((prev or {}).get("attempts", 0)) + 1
        _progress(conn, state="running", stage="loading data", code_version=BT.CODE_VERSION, trigger=trigger,
                  attempts=attempts, started=_now_iso(), error=None, pid=os.getpid())
        try:
            _progress(conn, stage="pattern diagnosis")
            try:                                                    # cheap, server-side, and never able to stop the study
                import e2_diagnosis
                e2_diagnosis.run(conn, f"validation:{trigger}")
            except Exception:                                       # noqa: BLE001
                log.exception("pattern diagnosis failed (validation continues)")
                conn.rollback()
            _progress(conn, stage="loading data")
            data = BT.load_from_db(conn)
            if len(data.frames) < 50:
                raise RuntimeError(f"only {len(data.frames)} symbols have prices; refusing to report on that")
            rep = BT.run_study(data, progress=lambda m: _progress(conn, stage=m))
            _put(conn, REPORT_KEY, rep)
            _progress(conn, state="done", stage="finished", error=None, runtime_sec=rep.get("runtime_sec"))
            try:                                                    # the finished report goes into the service log (E2REPORT| lines)
                log_report()
            except Exception:                                       # noqa: BLE001
                log.exception("could not log the finished report")
            return {"state": "done", "runtime_sec": rep.get("runtime_sec")}
        except Exception as e:                                    # noqa: BLE001
            log.exception("engine2 validation failed")
            try:
                conn.rollback()
                _progress(conn, state="failed", error=f"{type(e).__name__}: {e}"[:500])
            except Exception:
                pass
            raise
    finally:
        conn.close()


def _spawn(kind: str, trigger: str, extra_env: dict | None = None) -> dict:
    """Start the work in its own interpreter. Returns at once. kind: 'validate' | 'scan:<mode>'."""
    with _lock:
        if kind.startswith("scan:"):
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_scan.py"), kind.split(":", 1)[1], trigger]
        else:
            cmd = [sys.executable, os.path.abspath(__file__), kind, trigger]
        env = dict(os.environ)
        env.update(extra_env or {})
        p = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=env)   # inherits stdout/stderr -> Railway logs
        threading.Thread(target=p.wait, daemon=True).start()                                  # reap it, no zombie
        return {"started": True, "kind": kind, "pid": p.pid, "trigger": trigger}


def _running_here() -> bool:
    try:
        from ingest import connect
        conn = connect()
        try:
            pr, at = _get(conn, PROGRESS_KEY)
        finally:
            conn.close()
        if not pr or pr.get("state") != "running":
            return False
        age = _age_min(at)
        return bool(age is not None and age < STALE_MIN)
    except Exception:
        return False


# ----------------------------------------------------------------------
# scheduling
# ----------------------------------------------------------------------
def _defdiag_if_idle() -> dict:
    """Boot job: the definition diagnosis loads the full price history (~0.6 GB), so it never overlaps a validation run."""
    if _running_here():
        return {"skipped": "a validation run is in progress; POST /jobs/engine2/defdiag when it has finished"}
    return _spawn("defdiag", "boot")


def _actions_if_idle() -> dict:
    """Boot job for the 6 Nov actions study (loads the full history twice over: never alongside a validation run)."""
    if os.environ.get("E2_ACTIONS_BOOT", "0") != "1":              # the 6 Nov study is finished: it no longer reruns on every deploy
        return {"skipped": "E2_ACTIONS_BOOT is not 1"}
    if _running_here():
        return {"skipped": "a validation run is in progress; POST /jobs/engine2/actions when it has finished"}
    return _spawn("actions", "boot")


def _exits_if_idle() -> dict:
    """Boot job for the exit redesign study (loads the full history: never alongside a validation run)."""
    if os.environ.get("E2_EXITS_BOOT", "0") != "1":                # finished 6 Nov: no variant passed
        return {"skipped": "E2_EXITS_BOOT is not 1"}
    if _running_here():
        return {"skipped": "a validation run is in progress; POST /jobs/engine2/exits when it has finished"}
    return _spawn("exits", "boot")


def _fwd_if_idle() -> dict:
    """Boot job for the forward-return test (loads the full history: never alongside a validation run)."""
    if os.environ.get("E2_FWD_BOOT", "1") != "1":                  # re-run 7 Nov for the control-relative labels
        return {"skipped": "E2_FWD_BOOT is not 1"}
    if _running_here():
        return {"skipped": "a validation run is in progress; POST /jobs/engine2/fwd when it has finished"}
    return _spawn("fwd", "boot")


def _fwddiag_if_idle() -> dict:
    """Boot job for the one-detector forward-return diagnosis (the detector under triage today)."""
    if os.environ.get("E2_FWDDIAG_BOOT", "1") != "1":
        return {"skipped": "E2_FWDDIAG_BOOT is not 1"}
    if _running_here():
        return {"skipped": "a validation run is in progress; POST /jobs/engine2/fwddiag when it has finished"}
    return _spawn("fwddiag", "boot", {"E2_FWDDIAG_DETECTOR": os.environ.get("E2_FWDDIAG_DETECTOR", "pullback_ema,ascending_base")})


def _exits_wl_if_idle() -> dict:
    if os.environ.get("E2_EXITS_WL_BOOT", "1") != "1":
        return {"skipped": "E2_EXITS_WL_BOOT is not 1"}
    if _running_here():
        return {"skipped": "a validation run is in progress; POST /jobs/engine2/exits-wl when it has finished"}
    return _spawn("exits_wl", "boot")


def _validate_if_needed() -> dict:
    """Daily 22:30 IST: run the validation if this code version has no report yet."""
    import e2_backtest as BT
    from ingest import connect

    conn = connect()
    try:
        rep, _ = _get(conn, REPORT_KEY)
        pr, _ = _get(conn, PROGRESS_KEY)
    finally:
        conn.close()
    if rep and rep.get("code_version") == BT.CODE_VERSION:
        return {"skipped": "report exists for " + BT.CODE_VERSION}
    if _running_here():
        return {"skipped": "already running"}
    # attempts counts STARTS, so a run killed without a trace (out of memory) is counted too
    if pr and pr.get("code_version") == BT.CODE_VERSION and pr.get("state") != "done" and int(pr.get("attempts", 0)) >= MAX_ATTEMPTS:
        return {"skipped": f"gave up after {MAX_ATTEMPTS} attempts (last state {pr.get('state')}): {pr.get('error')}. "
                           "POST /jobs/engine2/validate starts a fresh run."}
    return _spawn("validate", "scheduled")


def _skip_today(mode: str) -> dict | None:
    """Holiday / Muhurat guard, run BEFORE any token check or child process (the 2 Oct false-token_invalid lesson)."""
    from ingest import connect
    from market_calendar import holiday_description, is_muhurat, is_trading_holiday
    from upstox_client import IST as _IST

    today = datetime.now(_IST).date()
    conn = connect()
    try:
        if is_trading_holiday(conn, today):
            return {"as_of": str(today), "mode": mode, "status": "skipped_holiday",
                    "reason": holiday_description(conn, today) or "market holiday"}
        if mode == "intraday" and is_muhurat(conn, today):
            return {"as_of": str(today), "mode": mode, "status": "skipped_muhurat_intraday",
                    "reason": "muhurat session: a forming bar minutes old is not scored against 50-day averages"}
        return None                                       # a lookup failure propagates: the slot is recorded as failed
    finally:
        conn.close()


def run_scan_slot(mode: str, trigger: str = "scheduled") -> dict:
    """
    One scheduled scan. BLOCKS until the child finishes, then returns its summary (so the scheduler's
    _last_runs shows the diagnostics) or RAISES with the specific reason (so it shows 'failed' and the log
    carries an ERROR with a traceback). The child records the run itself; this wrapper records only what the
    child could not (a kill, a timeout, a failure before it started).
    """
    import e2_scan

    skipped = _skip_today(mode)
    if skipped is not None:
        log.info("Engine 2 %s slot skipped: %s", mode, skipped["reason"])
        e2_scan.record_skip(mode, skipped, trigger)
        return skipped

    if mode == "intraday":                                # no child, no import cost, no Upstox call
        from upstox_client import TokenStore
        if TokenStore().valid_token() is None:
            e2_scan.record_failure(mode, "token_invalid", {"action": "Log in to Upstox (the token expires 03:30 IST daily)."}, trigger)
            raise RuntimeError("token_invalid")

    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_scan.py"), mode, trigger]
    try:
        p = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=dict(os.environ),
                           stdout=subprocess.PIPE, text=True, timeout=SCAN_TIMEOUT_MIN[mode] * 60)   # stderr -> Railway logs
    except subprocess.TimeoutExpired:
        e2_scan.record_failure(mode, f"timeout_after_{SCAN_TIMEOUT_MIN[mode]}_min", trigger=trigger)
        raise RuntimeError(f"timeout_after_{SCAN_TIMEOUT_MIN[mode]}_min")

    result = None
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith("E2SCAN_RESULT "):
            try:
                result = json.loads(line[len("E2SCAN_RESULT "):])
            except ValueError:
                pass
            break
    if result is None:                                    # killed (out of memory) or crashed before it could report
        reason = f"child_died_without_result rc={p.returncode}"
        e2_scan.record_failure(mode, reason, trigger=trigger)
        raise RuntimeError(reason)
    if p.returncode != 0 or result.get("status") == "failed":
        raise RuntimeError(str(result.get("reason") or f"scan failed rc={p.returncode}"))
    return result


def log_candidates() -> dict:
    """
    Write the per-candidate record for the detectors that have had no slot (default episodic_pivot and gap_and_go, from the
    cutover) into the service log, one E2CAND| line per day, so it is readable from the Railway log as well as from
    GET /jobs/engine2-candidates. Runs once after each start. Read-only.
    """
    import json as _json
    from ingest import connect
    dets = os.environ.get("E2_CANDIDATE_LOG_DETECTORS", "episodic_pivot,gap_and_go")
    since = os.environ.get("E2_CANDIDATE_LOG_SINCE", "2026-10-09")
    conn = connect()
    try:
        rep = candidates_report(conn, dets, since, None)
    finally:
        conn.close()
    for d in rep["days"]:
        log.info("E2CAND| %s", _json.dumps(d, separators=(",", ":"), default=str))
    log.info("E2CAND| %d day(s) with candidates from %s for %s", len(rep["days"]), since, dets)
    return {"days": len(rep["days"])}


def log_components_latest() -> dict:
    """
    Write the component breakdown of the latest completed scan (engine2_shadow_latest) into the service log as E2COMP| lines.
    Runs once after each start, so the most recent session's components can be read before the next scan replaces them. Read-only.
    """
    import e2_scan
    from ingest import connect
    conn = connect()
    try:
        latest, _ = _get(conn, "engine2_shadow_latest")
    finally:
        conn.close()
    if not latest or not latest.get("all_candidates"):
        log.info("E2COMP| no stored scan with candidates to replay")
        return {"lines": 0}
    lines = e2_scan.component_lines("latest", latest, tag=f"replay-{latest.get('scan_mode')}")
    for ln in lines:
        log.info("%s", ln)
    return {"lines": len(lines), "as_of": latest.get("as_of")}


def log_report() -> dict:
    """
    Write the stored validation report and the pattern diagnosis into the service log, one line per row
    (prefix E2REPORT|). The report lives in the database and is served at /jobs/engine2-report; this makes
    the same text readable from the Railway log as well, for whoever cannot reach the endpoint. Read-only.
    """
    from ingest import connect
    conn = connect()
    try:
        rep, _ = _get(conn, REPORT_KEY)
        diag, _ = _get(conn, "engine2_diagnosis")
    finally:
        conn.close()
    if not rep:
        log.error("E2REPORT| no engine-2 report is stored yet")
        return {"logged": 0}
    text = render_markdown(rep)
    if diag:
        import e2_diagnosis
        text += "\n\n" + e2_diagnosis.render_markdown(diag)
    n = 0
    for line in text.splitlines():
        for i in range(0, max(len(line), 1), 1500):
            log.info("E2REPORT| %s", line[i:i + 1500])
            n += 1
    return {"logged": n, "code_version": rep.get("code_version")}


def _scan_job(mode: str):
    return lambda: run_scan_slot(mode, "scheduled")


def schedule(scheduler, guarded, ist) -> None:
    """Called from scheduler.start(). Never raises: engine 2 must not be able to stop the live scheduler."""
    if not ENABLED:
        log.info("Engine 2 disabled by ENGINE2_ENABLED")
        return
    try:
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.date import DateTrigger

        vh, vm = (int(x) for x in VALIDATE_TIME.split(":"))
        ph, pm = (int(x) for x in PREMARKET_TIME.split(":"))
        ch, cm = (int(x) for x in POSTCLOSE_TIME.split(":"))
        scheduler.add_job(guarded, CronTrigger(hour=vh, minute=vm, timezone=ist), args=["engine2_validate", _validate_if_needed],
                          id="engine2_validate", replace_existing=True, misfire_grace_time=3600, coalesce=True, max_instances=1)
        scheduler.add_job(guarded, CronTrigger(day_of_week="mon-fri", hour=ph, minute=pm, timezone=ist),
                          args=["engine2_premarket", _scan_job("premarket")], id="engine2_premarket", replace_existing=True,
                          misfire_grace_time=1800, coalesce=True, max_instances=1)
        scheduler.add_job(guarded, CronTrigger(day_of_week="mon-fri", hour=INTRADAY_HOURS, minute=INTRADAY_MINUTE, timezone=ist),
                          args=["engine2_intraday", _scan_job("intraday")], id="engine2_intraday", replace_existing=True,
                          misfire_grace_time=600, coalesce=True, max_instances=1)
        scheduler.add_job(guarded, CronTrigger(day_of_week="mon-fri", hour=ch, minute=cm, timezone=ist),
                          args=["engine2_postclose", _scan_job("postclose")], id="engine2_postclose", replace_existing=True,
                          misfire_grace_time=3600, coalesce=True, max_instances=1)
        try:                                              # the retired 20:00 shadow job must not survive a persisted job store
            scheduler.remove_job("engine2_shadow")
        except Exception:
            pass
        now = datetime.now(ist)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=2), timezone=ist),
                          args=["engine2_diagnosis", lambda: _spawn("diagnose", "boot")], id="engine2_diagnosis_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=4), timezone=ist),
                          args=["engine2_report_log", log_report], id="engine2_report_log_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=10), timezone=ist),   # the diagnosis may still have been running at +4
                          args=["engine2_report_log", log_report], id="engine2_report_log_late", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=15), timezone=ist),
                          args=["engine2_defdiag", _defdiag_if_idle], id="engine2_defdiag_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=22), timezone=ist),
                          args=["engine2_actions", _actions_if_idle], id="engine2_actions_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=24), timezone=ist),
                          args=["engine2_exits", _exits_if_idle], id="engine2_exits_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=26), timezone=ist),
                          args=["engine2_fwd", _fwd_if_idle], id="engine2_fwd_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=28), timezone=ist),
                          args=["engine2_fwddiag", _fwddiag_if_idle], id="engine2_fwddiag_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=32), timezone=ist),
                          args=["engine2_exits_wl", _exits_wl_if_idle], id="engine2_exits_wl_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=6), timezone=ist),
                          args=["engine2_candidates_log", log_candidates], id="engine2_candidates_log_boot", replace_existing=True)
        scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=8), timezone=ist),
                          args=["engine2_components_log", log_components_latest], id="engine2_components_log_boot", replace_existing=True)
        if now.hour >= 22 or now.hour < 5:                      # deployed overnight: do not wait a day
            scheduler.add_job(guarded, DateTrigger(run_date=now + timedelta(minutes=3), timezone=ist),
                              args=["engine2_validate", _validate_if_needed], id="engine2_validate_boot", replace_existing=True)
        log.info("Scheduled engine2_premarket Mon-Fri %s, engine2_intraday Mon-Fri hours %s at :%02d, engine2_postclose Mon-Fri %s, "
                 "engine2_validate daily %s (all IST)", PREMARKET_TIME, INTRADAY_HOURS, INTRADAY_MINUTE, POSTCLOSE_TIME, VALIDATE_TIME)
    except Exception:
        log.exception("Engine 2 scheduling failed (live scheduler unaffected)")


# ----------------------------------------------------------------------
# read side
# ----------------------------------------------------------------------
def headline(rep: dict) -> dict:
    det = rep.get("detectors", {})
    pf = rep.get("portfolio", {})
    full = (pf.get("full_window_spec_weights") or {})
    return {
        "code_version": rep.get("code_version"), "window": rep.get("window"), "runtime_sec": rep.get("runtime_sec"),
        "symbols_run": rep.get("symbols_run"), "candidates_after_gates": (rep.get("gates") or {}).get("candidates_after_gates"),
        "detector_status": {k: v["exhaustiveness"]["status"] for k, v in det.items()},
        "detectors_with_errors": [k for k, v in det.items() if v.get("errors")],
        "signals_per_day": {k: (rep.get("signals") or {}).get(k) for k in ("per_day_mean", "per_day_median", "per_day_p90", "per_day_max")},
        "portfolio_full_window": full.get("metrics"),
        "portfolio_test_half_calibrated": ((pf.get("test_half_calibrated_weights") or {}).get("metrics")),
        "calibration_adopted": ((rep.get("scoring") or {}).get("calibration") or {}).get("adopted"),
    }


def _next_runs() -> dict:
    """When each Engine 2 slot fires next, read from the live scheduler (the same source as /jobs/schedule)."""
    try:
        import scheduler
        return {j.id: (j.next_run_time.isoformat() if j.next_run_time else None)
                for j in (scheduler._scheduler.get_jobs() if scheduler._scheduler else []) if j.id.startswith("engine2_")}
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _last_slot_results() -> dict:
    try:
        import scheduler
        return {k: v for k, v in scheduler._last_runs.items() if k.startswith("engine2_")}
    except Exception:                                            # noqa: BLE001
        return {}


def _diag_brief(conn):
    try:
        import e2_diagnosis
        d, _ = _get(conn, e2_diagnosis.KEY)
        return e2_diagnosis.brief(d)
    except Exception as e:                                       # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def auto_scan_brief(conn) -> dict:
    last, last_at = _get(conn, "engine2_last_scan_summary")
    runs, _ = _get(conn, "engine2_scan_runs")
    runs = list(runs or [])
    cols = ("mode", "as_of", "status", "reason", "trigger", "finished_at", "duration_sec", "universe", "detectors_firing",
            "candidates", "signals", "degraded")
    failed = [r for r in runs if r.get("status") == "failed"]
    import engine_mode
    return {
        "enabled": ENABLED, "engine_mode": engine_mode.current(),
        "publishing": ("Engine 2 publishes to the signals table; the original scan runs in shadow" if engine_mode.is_engine2()
                       else "the original engine publishes; Engine 2 records in shadow"),
        "slots": {"engine2_premarket": f"Mon-Fri {PREMARKET_TIME} IST", "engine2_intraday": f"Mon-Fri hours {INTRADAY_HOURS} at :{INTRADAY_MINUTE:02d} IST",
                  "engine2_postclose": f"Mon-Fri {POSTCLOSE_TIME} IST", "engine2_validate": f"daily {VALIDATE_TIME} IST"},
        "next_runs": _next_runs(),
        "last_scan": ({k: last.get(k) for k in (*cols, "regime", "detector_errors", "publishing")} if last else None),
        "recent_runs": [{k: r.get(k) for k in cols} for r in runs[-12:]],
        "failed_runs_recent": [{k: r.get(k) for k in ("mode", "reason", "finished_at")} for r in failed[-5:]],
        "scheduler_last_results": {k: {"status": v.get("status"), "at": v.get("at"),
                                       "reason": (v.get("detail") or {}).get("error") or (v.get("detail") or {}).get("reason")}
                                   for k, v in _last_slot_results().items()},
    }


def status_brief(conn) -> dict:
    pr, pr_at = _get(conn, PROGRESS_KEY)
    rep, rep_at = _get(conn, REPORT_KEY)
    sh, sh_at = _get(conn, "engine2_shadow_latest")
    state = (pr or {}).get("state", "never_run")
    age = _age_min(pr_at)
    if state == "running" and age is not None and age > STALE_MIN:
        state = "interrupted"
    try:
        import e2_backtest as BT
        cv = BT.CODE_VERSION
    except Exception as e:                                       # an import error in engine 2 is itself a finding
        cv, state = f"IMPORT FAILED: {e}"[:200], "import_failed"
    return {
        "code_version": cv, "enabled": ENABLED, "state": state,
        "progress": {k: (pr or {}).get(k) for k in ("stage", "attempts", "trigger", "started", "updated", "error")},
        "next_validation_check": f"daily {VALIDATE_TIME} IST (runs only if this code version has no report)",
        "report": ({"generated_at": rep_at.isoformat() if rep_at else None, "matches_running_code": rep.get("code_version") == cv,
                    "headline": headline(rep)} if rep else None),
        "auto_scan": auto_scan_brief(conn),
        "diagnosis": _diag_brief(conn),
        "shadow": ({"as_of": sh.get("as_of"), "candidates": sh.get("candidates"), "selected": sh.get("selected"),
                    "regime": (sh.get("regime") or {}).get("state"), "forward_log": sh.get("forward_log")} if sh else None),
    }


def render_markdown(rep: dict) -> str:
    L = [f"# Engine 2 validation — {rep.get('code_version')}", ""]
    w = rep.get("window", {})
    L += [f"Window {w.get('start')} → {w.get('end')} ({w.get('sessions')} sessions); train/test split at {w.get('split')}.",
          f"Universe scanned: {(rep.get('universe') or {}).get('scanned')}; symbols run: {rep.get('symbols_run')}; "
          f"index return over the window: {rep.get('index_return_over_window_pct')}%.", ""]
    s = rep.get("signals", {})
    L += ["## Signal count (an OUTPUT)", f"All-detector candidates after hard gates: {s.get('total_candidates_all_detectors')} over {s.get('days')} sessions — "
          f"mean {s.get('per_day_mean')}/day, median {s.get('per_day_median')}, p90 {s.get('per_day_p90')}, max {s.get('per_day_max')}. "
          f"By regime (mean/day): {s.get('by_regime_mean')}. Taken by the portfolio (full window): {s.get('taken_by_portfolio_full_window')}.", ""]
    L += ["## Per-detector exhaustiveness", "", "| detector | detected | expected | miss rate | FP rate | status | trades | avg R | win % |", "|---|---|---|---|---|---|---|---|---|"]
    for k, v in rep.get("detectors", {}).items():
        e, p = v["exhaustiveness"], v["performance_all_candidates"]
        L.append(f"| {k} | {e['setups_detected']} | {e['setups_expected']} | {e['miss_rate']} | {e['false_positive_rate']} | {e['status']} | "
                 f"{p.get('n')} | {p.get('avg_r')} | {p.get('win_rate_pct')} |")
    L += ["", "## Portfolio", ""]
    for k, v in (rep.get("portfolio") or {}).items():
        if isinstance(v, dict) and "metrics" in v:
            L.append(f"- **{k}**: {v['metrics']}  (rejected: {v.get('rejected')}; halts: {v.get('drawdown_halts')})")
    att = (rep.get("portfolio") or {}).get("attribution") or {}
    for label, a in att.items():
        L += ["", f"## Portfolio attribution — {label} ({a.get('closed_trades')} closed trades)", "",
              "| detector | trades | avg R | total R | win % |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v['n']} | {v['avg_r']} | {v['total_r']} | {v['win_rate_pct']} |" for k, v in a.get("by_detector", {}).items()]
        L += ["", "By regime at signal: " + "; ".join(f"{k}: {v['n']} trades, {v['avg_r']}R avg, {v['total_r']}R total" for k, v in a.get("by_regime", {}).items())]
    cal = (rep.get("scoring") or {}).get("calibration") or {}
    L += ["", "## Scoring", f"Calibration adopted: {cal.get('adopted')} — {cal.get('reason')}",
          f"Spearman(score, R) test half: spec {cal.get('spearman_test_spec')}, calibrated {cal.get('spearman_test_cal')}"]
    L += ["", "## Regime", str(rep.get("regime")), "", "## Gates", str(rep.get("gates")), "", "## Caveats"]
    L += [f"- {c}" for c in rep.get("caveats", [])]
    return "\n".join(L)


def candidates_report(conn, detector: str | None = None, since: str | None = None, until: str | None = None) -> dict:
    """Per-candidate record from the forward log; see GET /jobs/engine2-candidates. Read-only."""
    from collections import defaultdict
    flog, _ = _get(conn, "engine2_shadow_log")
    allruns, _ = _get(conn, "engine2_scan_runs")
    want = {d.strip() for d in detector.split(",") if d.strip()} if detector else None
    raw_by_day = {r["as_of"]: r.get("detector_candidates") or {} for r in (allruns or [])
                  if r.get("mode") == "postclose" and r.get("status") == "success" and r.get("as_of")}
    by_day = defaultdict(list)
    for e in (flog or []):
        if (since and e["dt"] < since) or (until and e["dt"] > until):
            continue
        by_day[e["dt"]].append(e)
    days = []
    for dt in sorted(set(by_day) | {d for d in raw_by_day if (not since or d >= since) and (not until or d <= until)}):
        day = sorted(by_day.get(dt, []), key=lambda e: -e["sc"])
        rank = {(e["s"], e["d"]): i + 1 for i, e in enumerate(day)}
        chosen = [e for e in day if e.get("sel")]
        cutoff = min((e["sc"] for e in chosen), default=None)
        mine = [e for e in day if want is None or e["d"] in want]
        raw = raw_by_day.get(dt)
        if want and not mine and not (raw and any(raw.get(d) for d in want)):
            continue
        row = {"date": dt, "ranked_total": len(day), "slots_filled": len(chosen), "cutoff_score": cutoff,
               "selected": [{"symbol": e["s"], "detector": e["d"], "score": e["sc"], "rank": rank[(e["s"], e["d"])]} for e in chosen],
               "candidates": [{"symbol": e["s"], "detector": e["d"], "score": e["sc"], "quality": e.get("q"),
                               "rank": rank[(e["s"], e["d"])], "selected": bool(e.get("sel")),
                               "why_not": None if e.get("sel") else (e.get("wn") or "not_recorded"),
                               "points_below_cutoff": None if (e.get("sel") or cutoff is None) else round(cutoff - e["sc"], 1)}
                              for e in mine]}
        if raw is not None:
            dets = want or {d for d, n in raw.items() if n}
            row["raw_vs_ranked"] = {d: {"raw": raw.get(d, 0), "ranked": sum(1 for e in day if e["d"] == d),
                                        "removed_before_ranking": raw.get(d, 0) - sum(1 for e in day if e["d"] == d)}
                                    for d in sorted(dets)}
        days.append(row)
    return {"detector_filter": sorted(want) if want else None, "days": days,
            "note": "score = composite used for ranking; selection takes the highest scores subject to max_positions, heat, sector and correlation caps. "
                    "raw_vs_ranked needs the day's postclose run record (the last 80 runs are kept)."}


def components_report(conn, detector: str | None = None, since: str | None = None, until: str | None = None,
                      with_rows: bool = True) -> dict:
    """Score components (points) per ranked candidate from the forward log; see GET /jobs/engine2-components. Read-only.
    Entries logged before 2026-11-06 carry no components and are counted in `entries_without_components`."""
    import e2_scorer as SC
    from collections import defaultdict
    flog, _ = _get(conn, "engine2_shadow_log")
    want = {d.strip() for d in detector.split(",") if d.strip()} if detector else None
    inwin = [e for e in (flog or []) if not ((since and e["dt"] < since) or (until and e["dt"] > until))]
    by_day = defaultdict(list)
    for e in inwin:
        by_day[e["dt"]].append(e)
    cols = SC.COMPONENTS
    rows, groups = [], defaultdict(list)
    for dt in sorted(by_day):
        day = sorted(by_day[dt], key=lambda e: -e["sc"])
        rank = {(e["s"], e["d"]): e.get("rk") or (i + 1) for i, e in enumerate(day)}
        cutoff = min((e["sc"] for e in day if e.get("sel")), default=None)
        for e in day:
            if not e.get("cp"):
                continue
            pts = dict(zip(cols, e["cp"]))
            sel = bool(e.get("sel"))
            r = {"date": dt, "symbol": e["s"], "detector": e["d"], "rank": rank[(e["s"], e["d"])], "score": e["sc"], "selected": sel,
                 "why_not": None if sel else (e.get("wn") or "not_recorded"),
                 "points_below_cutoff": None if (sel or cutoff is None) else round(cutoff - e["sc"], 1),
                 "cutoff": cutoff, "catalyst_source": {"c": "confirmed", "p": "proxy"}.get(e.get("cs"), e.get("cs")),
                 **{f"{k}_score": pts[k] for k in cols}}
            groups[e["d"]].append((pts, sel, e["sc"]))
            groups["ALL_SELECTED" if sel else "ALL_NOT_SELECTED"].append((pts, sel, e["sc"]))
            if sel:
                groups[e["d"] + ":selected"].append((pts, sel, e["sc"]))
            if want is None or e["d"] in want:
                rows.append(r)

    def agg(items):
        n = len(items)
        return {"n": n, "score_mean": round(sum(i[2] for i in items) / n, 1),
                **{f"{k}_mean": round(sum(i[0][k] for i in items) / n, 2) for k in cols}}
    summary = {g: agg(v) for g, v in sorted(groups.items())}
    ref = summary.get("ALL_SELECTED")
    if ref:
        for g, a in summary.items():
            a["minus_selected_mean"] = {k: round(a[f"{k}_mean"] - ref[f"{k}_mean"], 2) for k in cols}
    return {"detector_filter": sorted(want) if want else None, "components": cols, "max_points": SC.SPEC_WEIGHTS,
            "entries_in_window": len(inwin), "entries_without_components": sum(1 for e in inwin if not e.get("cp")),
            "rows": rows if with_rows else None, "summary": summary,
            "note": "points = component fraction x spec weight; the seven add up to the score. summary has one group per detector, "
                    "'<detector>:selected', ALL_SELECTED and ALL_NOT_SELECTED; minus_selected_mean is a group's mean points minus the mean "
                    "of every selected row, per component (negative = lower than what gets selected). Only completed-session (postclose) "
                    "candidates are kept here; every scan's candidates (incl. intraday) are also written to the log as E2COMP| lines."}


def register(app, *, require_internal_key, ist) -> None:
    from ingest import connect

    @app.get("/jobs/engine2-status")
    def e2_status(request: Request):
        require_internal_key(request)
        conn = connect()
        try:
            return status_brief(conn)
        finally:
            conn.close()

    @app.get("/jobs/engine2-report")
    def e2_report(request: Request, format: str = "json", section: str | None = None):
        require_internal_key(request)
        conn = connect()
        try:
            rep, at = _get(conn, REPORT_KEY)
            diag, _ = _get(conn, "engine2_diagnosis")
        finally:
            conn.close()
        if not rep:
            raise HTTPException(404, "no engine-2 report yet — see /jobs/engine2-status")
        full = {**rep, "disabled_pattern_diagnosis": diag} if diag else rep
        if format == "md":
            md = render_markdown(rep)
            if diag:
                import e2_diagnosis
                md += "\n\n" + e2_diagnosis.render_markdown(diag)
            return PlainTextResponse(md)
        if section:
            if section not in full:
                raise HTTPException(404, f"no section {section!r}; sections: {sorted(full)}")
            return {section: full[section]}
        return full

    @app.get("/jobs/engine2-diagnosis")
    def e2_diagnosis_view(request: Request, format: str = "json", pattern: str | None = None, trades: bool = False):
        """Old engine vs corrected definition for the four disabled patterns. ?trades=true adds the per-trade replay."""
        require_internal_key(request)
        import e2_diagnosis
        conn = connect()
        try:
            d, _ = _get(conn, e2_diagnosis.KEY)
        finally:
            conn.close()
        if not d:
            raise HTTPException(404, "the diagnosis has not run yet (it starts ~2 minutes after a deploy and ahead of every validation)")
        if format == "md":
            return PlainTextResponse(e2_diagnosis.render_markdown(d))
        if pattern:
            if pattern not in d.get("patterns", {}):
                raise HTTPException(404, f"no pattern {pattern!r}; patterns: {list(d.get('patterns', {}))}")
            d = {**d, "patterns": {pattern: d["patterns"][pattern]}}
        if not trades:
            d = {**d, "patterns": {k: ({**v, "replay": {rk: rv for rk, rv in v["replay"].items() if rk != "trades"}} if "replay" in v else v)
                                   for k, v in d["patterns"].items()}}
        return d

    @app.post("/jobs/engine2/diagnose")
    def e2_diagnose_now(request: Request):
        require_internal_key(request)
        return _spawn("diagnose", "manual")

    @app.get("/jobs/engine2-defdiag")
    def e2_defdiag_view(request: Request, format: str = "json"):
        """Definition diagnosis: pocket_pivot context breakdown (the vcp clause trace is inside /jobs/engine2-diagnosis)."""
        require_internal_key(request)
        import e2_defdiag
        conn = connect()
        try:
            d, _ = _get(conn, e2_defdiag.KEY)
        finally:
            conn.close()
        if not d:
            raise HTTPException(404, "the definition diagnosis has not run yet (it starts ~15 minutes after a deploy; POST /jobs/engine2/defdiag to start it now)")
        return PlainTextResponse(e2_defdiag.render_markdown(d)) if format == "md" else d

    @app.post("/jobs/engine2/defdiag")
    def e2_defdiag_now(request: Request):
        require_internal_key(request)
        if _running_here():
            return {"started": False, "reason": "a validation run is in progress"}
        return _spawn("defdiag", "manual")

    @app.get("/jobs/engine2-actions")
    def e2_actions_view(request: Request, format: str = "json"):
        """The 6 Nov actions study (measurement only)."""
        require_internal_key(request)
        import e2_actions
        conn = connect()
        try:
            d, _ = _get(conn, e2_actions.KEY)
        finally:
            conn.close()
        if not d:
            raise HTTPException(404, "the actions study has not run yet (POST /jobs/engine2/actions to start it)")
        return PlainTextResponse(e2_actions.render_markdown(d)) if format == "md" else d

    @app.post("/jobs/engine2/actions")
    def e2_actions_now(request: Request):
        require_internal_key(request)
        if _running_here():
            return {"started": False, "reason": "a validation run is in progress"}
        return _spawn("actions", "manual")

    @app.get("/jobs/engine2-exits")
    def e2_exits_view(request: Request, format: str = "json"):
        """The exit redesign study (measurement only)."""
        require_internal_key(request)
        import e2_exits
        conn = connect()
        try:
            d, _ = _get(conn, e2_exits.KEY)
        finally:
            conn.close()
        if not d:
            raise HTTPException(404, "the exit study has not run yet (POST /jobs/engine2/exits to start it)")
        return PlainTextResponse(e2_exits.render_markdown(d)) if format == "md" else d

    @app.post("/jobs/engine2/exits")
    def e2_exits_now(request: Request):
        require_internal_key(request)
        if _running_here():
            return {"started": False, "reason": "a validation run is in progress"}
        return _spawn("exits", "manual")

    @app.get("/jobs/engine2-fwd")
    def e2_fwd_view(request: Request, format: str = "json"):
        """The forward-return test (measurement only)."""
        require_internal_key(request)
        import e2_fwd
        conn = connect()
        try:
            d, _ = _get(conn, e2_fwd.KEY)
        finally:
            conn.close()
        if not d:
            raise HTTPException(404, "the forward-return test has not run yet (POST /jobs/engine2/fwd to start it)")
        return PlainTextResponse(e2_fwd.render_markdown(d)) if format == "md" else d

    @app.post("/jobs/engine2/fwd")
    def e2_fwd_now(request: Request):
        require_internal_key(request)
        if _running_here():
            return {"started": False, "reason": "a validation run is in progress"}
        return _spawn("fwd", "manual")

    @app.get("/jobs/engine2-fwddiag")
    def e2_fwddiag_view(request: Request, detector: str = "pullback_ema", format: str = "json"):
        """Forward returns inside one detector, split by signal-bar context (measurement only)."""
        require_internal_key(request)
        import e2_fwddiag
        conn = connect()
        try:
            d, _ = _get(conn, e2_fwddiag.key_for(detector))
        finally:
            conn.close()
        if not d:
            raise HTTPException(404, f"no diagnosis stored for {detector} (POST /jobs/engine2/fwddiag?detector={detector})")
        return PlainTextResponse(e2_fwddiag.render_markdown(d)) if format == "md" else d

    @app.post("/jobs/engine2/fwddiag")
    def e2_fwddiag_now(request: Request, detector: str = "pullback_ema"):
        require_internal_key(request)
        if _running_here():
            return {"started": False, "reason": "a validation run is in progress"}
        return _spawn("fwddiag", "manual", {"E2_FWDDIAG_DETECTOR": detector})

    @app.get("/jobs/engine2-exits-wl")
    def e2_exits_wl_view(request: Request, format: str = "json"):
        require_internal_key(request)
        import e2_exits_wl
        conn = connect()
        try:
            d, _ = _get(conn, e2_exits_wl.KEY)
        finally:
            conn.close()
        if not d:
            raise HTTPException(404, "the whitelist exit study has not run yet (POST /jobs/engine2/exits-wl)")
        return PlainTextResponse(e2_exits_wl.render_markdown(d)) if format == "md" else d

    @app.post("/jobs/engine2/exits-wl")
    def e2_exits_wl_now(request: Request):
        require_internal_key(request)
        if _running_here():
            return {"started": False, "reason": "a validation run is in progress"}
        return _spawn("exits_wl", "manual")

    @app.get("/jobs/engine2-shadow")
    def e2_shadow(request: Request):
        require_internal_key(request)
        conn = connect()
        try:
            sh, _ = _get(conn, "engine2_shadow_latest")
        finally:
            conn.close()
        if not sh:
            raise HTTPException(404, "no shadow scan yet")
        return sh

    @app.post("/jobs/engine2/validate")
    def e2_validate(request: Request):
        require_internal_key(request)
        if _running_here():
            return {"started": False, "reason": "a run is already in progress"}
        return _spawn("validate", "manual")

    @app.get("/jobs/engine2-scan")
    def e2_scan_view(request: Request, mode: str | None = None, runs: int = 20):
        """The latest scan (full diagnostics + watchlist) and the recent runs. ?mode=intraday for the forming-bar scan."""
        require_internal_key(request)
        conn = connect()
        try:
            last, _ = _get(conn, "engine2_last_scan_summary")
            allruns, _ = _get(conn, "engine2_scan_runs")
            intr, _ = _get(conn, "engine2_intraday_latest")
        finally:
            conn.close()
        if mode == "intraday":
            if not intr:
                raise HTTPException(404, "no intraday scan has completed yet")
            return intr
        return {"last_scan": last, "recent_runs": list(allruns or [])[-max(1, min(runs, 80)):],
                "next_runs": _next_runs()}

    @app.get("/jobs/engine2-candidates")
    def e2_candidates(request: Request, detector: str | None = None, since: str | None = None, until: str | None = None):
        """
        Per-candidate record from the forward log: every candidate that reached the score ranking at a non-provisional scan, with
        its composite score, rank, whether the portfolio selected it and (from 2026-11-06 on) the rule that kept it out - next to
        the scores of what WAS selected that day and the cut-off. ?detector=episodic_pivot,gap_and_go  &since=2026-10-09.

        Candidates that fired but are missing here were removed by a hard gate (liquidity, data_quality, surveillance) or had no
        valid stop BEFORE ranking; raw_vs_ranked shows how many, per detector, wherever the day's postclose run record is still kept.
        Read-only. Nothing here is used by any decision.
        """
        require_internal_key(request)
        conn = connect()
        try:
            return candidates_report(conn, detector, since, until)
        finally:
            conn.close()

    @app.get("/jobs/engine2-components")
    def e2_components(request: Request, detector: str | None = None, since: str | None = None, until: str | None = None,
                      rows: bool = True):
        """
        The seven components of the score (trend, rs, volume, base, catalyst, sector, regime - in points) for every candidate that
        reached the ranking at a completed-session scan, with total score, rank, selected / why not and the day's cut-off, plus
        per-detector mean points and the same for selected rows, so low components can be read off directly.
        ?detector=episodic_pivot,gap_and_go &since=2026-11-06 &rows=false for the summary only. Recorded from 2026-11-06.
        Read-only. Nothing here is used by any decision.
        """
        require_internal_key(request)
        conn = connect()
        try:
            return components_report(conn, detector, since, until, rows)
        finally:
            conn.close()

    @app.post("/jobs/engine2/scan")
    def e2_scan_now(request: Request, mode: str = "postclose"):
        require_internal_key(request)
        if mode not in ("premarket", "intraday", "postclose"):
            raise HTTPException(400, "mode must be premarket | intraday | postclose")
        return _spawn(f"scan:{mode}", "manual")

    @app.post("/jobs/engine2/shadow")
    def e2_shadow_now(request: Request):
        require_internal_key(request)
        return _spawn("scan:postclose", "manual")


if __name__ == "__main__":                                    # the child process
    logging.basicConfig(level=logging.INFO)
    kind = sys.argv[1] if len(sys.argv) > 1 else "validate"
    trig = sys.argv[2] if len(sys.argv) > 2 else "cli"
    if kind == "defdiag":
        import e2_defdiag
        print(json.dumps({k: v for k, v in e2_defdiag.run(trigger=trig).items() if k in ("state", "runtime_sec", "error")}, default=str))
    elif kind == "exits_wl":
        import e2_exits_wl
        print(json.dumps({k: v for k, v in e2_exits_wl.run(trigger=trig).items() if k in ("state", "runtime_sec", "error")}, default=str))
    elif kind == "fwddiag":
        import e2_fwddiag
        print(json.dumps({k: v for k, v in e2_fwddiag.run(os.environ.get("E2_FWDDIAG_DETECTOR", "pullback_ema,ascending_base"), trigger=trig).items()
                          if k in ("state", "runtime_sec", "error")}, default=str))
    elif kind == "fwd":
        import e2_fwd
        print(json.dumps({k: v for k, v in e2_fwd.run(trigger=trig).items() if k in ("state", "runtime_sec", "error")}, default=str))
    elif kind == "exits":
        import e2_exits
        print(json.dumps({k: v for k, v in e2_exits.run(trigger=trig).items() if k in ("state", "runtime_sec", "error")}, default=str))
    elif kind == "actions":
        import e2_actions
        print(json.dumps({k: v for k, v in e2_actions.run(trigger=trig).items() if k in ("state", "runtime_sec", "error")}, default=str))
    elif kind == "diagnose":
        import e2_diagnosis
        print(json.dumps(e2_diagnosis.brief(e2_diagnosis.run(trigger=trig)), default=str))
    else:
        print(json.dumps(run_validation_blocking(trig), default=str))
