"""
e2_legacy_shadow.py — the ORIGINAL scan engine, run in SHADOW for the week-1 comparison.

Used only when ENGINE_MODE=engine2 (engine_mode.py). The original run_scan is not edited. It is run in a
child process with its database connection replaced by a wrapper whose commit() and close() do nothing, so
EVERYTHING it writes (signals, gate_log, market_regime, last_scan_summary, the invalidate/expire updates)
lives in one open transaction. Before that transaction is rolled back the signals the scan WOULD have
published are read out of it (rows carrying this transaction's id), and the run is recorded:

  legacy_shadow_latest   the latest run, with the signals it would have issued
  legacy_shadow_runs     the last RUNS_KEEP runs (compact)

Each record also carries the comparison with what Engine 2 actually published for the same session:
symbols both found, only the original found, only Engine 2 found.

Nothing the original engine does here reaches the signals table, the UI, or last_scan_summary.

Ordering: both engines touch market_regime and the invalidate/expire updates. The shadow takes the same
advisory lock Engine 2's publisher takes, at the start of its transaction, so the two never wait on each
other's row locks in opposite orders (a deadlock). In the postclose slot the shadow also waits (bounded)
for Engine 2's own postclose scan to finish, so Engine 2 is never the one kept waiting.

Run:  python e2_legacy_shadow.py <premarket|intraday|postclose>      (the child; prints LEGACYSHADOW_RESULT <json>)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from unittest import mock

log = logging.getLogger("engine2.legacy_shadow")
IST = timezone(timedelta(hours=5, minutes=30))
LATEST_KEY, RUNS_KEY = "legacy_shadow_latest", "legacy_shadow_runs"
RUNS_KEEP = 80
ADVISORY_LOCK = 7340001
WAIT_FOR_ENGINE2_MIN = float(os.environ.get("E2_SHADOW_WAIT_FOR_ENGINE2_MIN", "25"))
TIMEOUT_MIN = {"premarket": 40, "intraday": 25, "postclose": 60}


class _TxnConn:
    """A connection that cannot commit or close: the scan's whole write set stays in one transaction."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def commit(self):
        pass

    def close(self):
        pass


def _put(conn, key, value) -> None:
    with conn.cursor() as cur:
        cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (key, json.dumps(value, default=str)))
    conn.commit()


def _get(conn, key):
    with conn.cursor() as cur:
        cur.execute("select value from engine_settings where key = %s", (key,))
        r = cur.fetchone()
    return r[0] if r else None


def _wait_for_engine2(mode: str) -> dict:
    """Let Engine 2's slot for this mode finish first (it is the publishing engine). Bounded; then proceed anyway."""
    from ingest import connect
    t0 = time.time()
    since = datetime.now(IST) - timedelta(minutes=90)
    while True:
        conn = connect()
        try:
            runs = _get(conn, "engine2_scan_runs") or []
        finally:
            conn.close()
        for r in reversed(runs):
            try:
                done = datetime.fromisoformat(r.get("finished_at", ""))
            except ValueError:
                continue
            if r.get("mode") == mode and done >= since and done.date() == datetime.now(IST).date():
                return {"engine2_finished": True, "waited_sec": round(time.time() - t0)}
        if mode != "postclose" or time.time() - t0 > WAIT_FOR_ENGINE2_MIN * 60:
            return {"engine2_finished": False, "waited_sec": round(time.time() - t0)}
        time.sleep(30)


def run(mode: str) -> dict:
    from ingest import connect
    started = datetime.now(IST)
    wait = _wait_for_engine2(mode) if mode == "postclose" else {"engine2_finished": None}
    import scan as live_scan

    real = connect()
    shadow_conn = _TxnConn(real)
    result: dict = {"engine": "legacy", "shadow": True, "mode": mode, "started_at": started.isoformat(timespec="seconds"), "wait": wait}
    try:
        with real.cursor() as cur:
            cur.execute("select pg_advisory_xact_lock(%s)", (ADVISORY_LOCK,))        # same lock Engine 2's publisher takes
        with mock.patch.object(live_scan, "connect", lambda: shadow_conn), \
             mock.patch.object(live_scan, "ensure_bars_current", lambda c: {"refreshed": False, "reason": "legacy_shadow"}):
            summary = live_scan.run_scan(mode=mode)
        as_of = summary.get("as_of")
        rows = []
        if summary.get("status") is None and as_of:
            with real.cursor() as cur:                                                 # rows written by THIS transaction
                cur.execute("""select symbol, pattern, setup_type, entry_trigger, stop_loss, t1, t2, score_total, status,
                                      coalesce((notes->>'is_provisional')::boolean, false)
                               from signals
                               where as_of_date = %s and coalesce(notes->>'engine','') <> 'engine2'
                                 and xmin::text = (txid_current() %% 4294967296)::text
                               order by score_total desc""", (as_of,))
                rows = [dict(zip(("symbol", "pattern", "setup_type", "entry", "stop", "t1", "t2", "score", "status", "provisional"),
                                 [float(v) if hasattr(v, "as_integer_ratio") and not isinstance(v, (int, bool)) else v for v in r]))
                        for r in cur.fetchall()]
        real.rollback()                                                              # discard EVERYTHING the original scan wrote
        result.update(status=summary.get("status", "success"), as_of=as_of,
                      regime=summary.get("regime"), universe=summary.get("universe"),
                      passed_structure=summary.get("passed_structure"), signals=len(rows),
                      would_have_published=rows, rejections_top=dict(list((summary.get("rejections") or {}).items())[:8]))
    except Exception as e:                                                           # noqa: BLE001
        try:
            real.rollback()
        except Exception:
            pass
        log.exception("legacy shadow scan failed")
        result.update(status="failed", reason=f"{type(e).__name__}: {e}"[:300])
    finally:
        try:
            real.close()
        except Exception:
            pass

    conn = connect()
    try:
        if result.get("as_of"):
            with conn.cursor() as cur:
                cur.execute("select symbol, pattern from signals where as_of_date = %s and notes->>'engine' = 'engine2'", (result["as_of"],))
                e2 = {r[0] for r in cur.fetchall()}
            lg = {r["symbol"] for r in result.get("would_have_published", [])}
            result["comparison_with_engine2"] = {"both": sorted(lg & e2), "only_original": sorted(lg - e2), "only_engine2": sorted(e2 - lg),
                                                 "n_original": len(lg), "n_engine2": len(e2)}
        result["finished_at"] = datetime.now(IST).isoformat(timespec="seconds")
        _put(conn, LATEST_KEY, result)
        runs = list(_get(conn, RUNS_KEY) or [])
        runs.append({k: v for k, v in result.items() if k != "would_have_published"})
        _put(conn, RUNS_KEY, runs[-RUNS_KEEP:])
    finally:
        conn.close()
    return result


def run_slot(mode: str) -> dict:
    """Called by the original engine's scheduled slots (scheduler._scan_or_shadow). Blocks; raises on failure."""
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_legacy_shadow.py"), mode]
    try:
        p = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), env=dict(os.environ),
                           stdout=subprocess.PIPE, text=True, timeout=TIMEOUT_MIN.get(mode, 40) * 60)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"legacy_shadow_timeout_{mode}")
    res = None
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith("LEGACYSHADOW_RESULT "):
            res = json.loads(line[len("LEGACYSHADOW_RESULT "):])
            break
    if res is None:
        raise RuntimeError(f"legacy_shadow_died rc={p.returncode}")
    if res.get("status") == "failed":
        raise RuntimeError(f"legacy_shadow_failed: {res.get('reason')}")
    return {"as_of": res.get("as_of"), "mode": mode, "status": "shadow_" + str(res.get("status")), "signals": res.get("signals"),
            "universe": res.get("universe"), "note": "ENGINE_MODE=engine2: the original scan ran in SHADOW (nothing published)."}


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = run(sys.argv[1] if len(sys.argv) > 1 else "postclose")
    print("LEGACYSHADOW_RESULT " + json.dumps({k: v for k, v in out.items() if k != "would_have_published"}, default=str), flush=True)
    sys.exit(2 if out.get("status") == "failed" else 0)
