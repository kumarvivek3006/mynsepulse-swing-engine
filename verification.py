"""
Signal verification: for every signal the engine published, did the user TAKE it or SKIP it, and if skipped, why.

Why this exists. Outcome attribution per detector (taken -> won/lost/avg R) and skip-reason clusters per detector
(bad_geometry / pattern_absent / bad_rr ...) were recommended at cutover and never built, so weeks 3 and 4 of the
review could not be produced. This module is the whole feature:

  signals.verification_outcome   'taken' | 'skipped' | NULL (NULL = not yet reviewed by the user)
  signals.verification_reason    skipped only: one of REASONS
  signals.verification_note      optional free text
  signals.verification_at        when the user marked it

  POST /signals/verify           set / change / clear the outcome (the UI's "Mark taken" / "Mark skipped")
  POST /signals/take             (existing) now also sets outcome='taken' - the fill price flow is unchanged
  GET  /jobs/verification-status unreviewed counts, the >48h list, and per-detector taken/skipped/reason tallies
  daily 19:10 IST                a log line: how many signals are unreviewed after 48 hours (healthy = 0)

The schema change is applied by the service itself at startup (idempotent ALTER ... IF NOT EXISTS, mirrored in
005_verification.sql), so nobody has to run SQL. If that cannot be applied the failure is logged at ERROR, every
verification endpoint answers 503 with the reason, and the rest of the service is unaffected.

Nothing here reads or changes a detector, a score, a gate or the selection.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from fastapi import Body, HTTPException, Request

log = logging.getLogger("verification")

OUTCOMES = ("taken", "skipped")
REASONS = ("bad_geometry", "pattern_absent", "bad_rr", "stale_data", "regime_mismatch", "other")
UNREVIEWED_AFTER_HOURS = 48
# Signals before the Engine 2 cutover were never part of this workflow; counting them would make "0 = healthy" unreachable.
START_DATE = os.environ.get("VERIFICATION_START_DATE", "2026-10-09")
RECON_TIME_IST = os.environ.get("VERIFICATION_RECON_TIME_IST", "19:10")

DDL = (
    "alter table signals add column if not exists verification_outcome text check (verification_outcome in ('taken','skipped'))",
    "alter table signals add column if not exists verification_reason text",
    "alter table signals add column if not exists verification_note text",
    "alter table signals add column if not exists verification_at timestamptz",
    "create index if not exists signals_unreviewed_idx on signals (as_of_date) where verification_outcome is null",
    # A signal with a recorded fill IS a taken signal; mark the ones that predate this column. Idempotent.
    """update signals s
          set verification_outcome = 'taken', verification_at = coalesce(o.entry_date::timestamptz, now())
         from signal_outcomes o
        where o.signal_id = s.id and o.entry_price is not null and s.verification_outcome is null""",
)

_state = {"ready": False, "error": None, "checked": 0.0}


def ensure_schema() -> bool:
    """Apply the columns. Safe to call on every start. Never raises."""
    from ingest import connect
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("set local lock_timeout = '10s'")
                for stmt in DDL:
                    cur.execute(stmt)
            conn.commit()
        finally:
            conn.close()
        _state.update(ready=True, error=None, checked=time.time())
        log.info("Verification schema ready (signals.verification_outcome / _reason / _note / _at)")
        return True
    except Exception as exc:                                         # noqa: BLE001
        _state.update(ready=False, error=str(exc)[:300], checked=time.time())
        log.error("Verification schema NOT applied - signal verification is unavailable until it is: %s", exc)
        return False


def is_ready(conn=None) -> bool:
    """True when the columns exist. A positive answer is cached for the process; a negative one is re-checked every 60 s."""
    if _state["ready"]:
        return True
    if time.time() - _state["checked"] < 60:
        return False
    from ingest import connect
    own = conn is None
    try:
        conn = conn or connect()
        with conn.cursor() as cur:
            cur.execute("select 1 from information_schema.columns where table_schema = 'swing' and table_name = 'signals' "
                        "and column_name = 'verification_outcome'")
            ok = cur.fetchone() is not None
        _state.update(ready=ok, checked=time.time(), error=None if ok else (_state["error"] or "columns not present"))
        if not ok and own:
            return ensure_schema()               # self-heal: the startup attempt may have lost a lock race with a publish
        return ok
    except Exception as exc:                                         # noqa: BLE001
        _state.update(ready=False, checked=time.time(), error=str(exc)[:300])
        return False
    finally:
        if own and conn is not None:
            conn.close()


def annotate(rows: list[dict], id_key: str = "id") -> None:
    """Add the four verification fields to rows an endpoint already built. Degrades to nothing if the columns are absent."""
    if not rows or not is_ready():
        return
    from ingest import connect
    ids = [str(r[id_key]) for r in rows if r.get(id_key)]
    if not ids:
        return
    try:
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("select id::text, verification_outcome, verification_reason, verification_note, verification_at "
                            "from signals where id::text = any(%s)", (ids,))
                got = {r[0]: r for r in cur.fetchall()}
        finally:
            conn.close()
    except Exception:                                                # noqa: BLE001
        log.exception("Could not read verification fields (response unaffected)")
        return
    for r in rows:
        g = got.get(str(r.get(id_key)))
        r["verification_outcome"] = g[1] if g else None
        r["verification_reason"] = g[2] if g else None
        r["verification_note"] = g[3] if g else None
        r["verification_at"] = g[4].isoformat() if g and g[4] else None


def mark_taken(cur, signal_id) -> None:
    """Called inside /signals/take's own transaction. A no-op if the columns are not there."""
    if not is_ready():
        return
    cur.execute("update signals set verification_outcome = 'taken', verification_reason = null, verification_at = now() "
                "where id = %s", (signal_id,))


def set_outcome(signal_id: str, outcome: str | None, reason: str | None, note: str | None) -> dict:
    if outcome not in (None, *OUTCOMES):
        raise HTTPException(400, f"outcome must be one of {list(OUTCOMES)} or null (to clear)")
    if outcome == "skipped" and reason not in REASONS:
        raise HTTPException(400, f"a skipped signal needs a reason, one of {list(REASONS)}")
    if outcome != "skipped" and reason:
        raise HTTPException(400, "reason is only valid with outcome 'skipped'")
    if not is_ready():
        raise HTTPException(503, f"signal verification is not available: {_state['error'] or 'columns not present yet'}")
    from ingest import connect
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("select status, verification_outcome from signals where id::text = %s", (str(signal_id),))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "signal not found")
            if outcome == "skipped" and row[0] == "triggered":
                raise HTTPException(409, "this signal has an open trade; close the trade instead of marking it skipped")
            cur.execute("update signals set verification_outcome = %s, verification_reason = %s, verification_note = %s, "
                        "verification_at = case when %s::text is null then null else now() end "
                        "where id::text = %s",
                        (outcome, reason if outcome == "skipped" else None, (note or None), outcome, str(signal_id)))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "signal_id": str(signal_id), "verification_outcome": outcome,
            "verification_reason": reason if outcome == "skipped" else None, "previous": row[1]}


def reconcile(conn, hours: int = UNREVIEWED_AFTER_HOURS, listing: int = 50) -> dict:
    """Signals still unreviewed after `hours`. Provisional rows the postclose withdrew were never confirmed signals and are excluded."""
    base = ("from signals where verification_outcome is null and as_of_date >= %s::date "
            "and generated_at < now() - make_interval(hours => %s) and coalesce(notes->>'withdrawn', '') = ''")
    with conn.cursor() as cur:
        cur.execute(f"select count(*), min(as_of_date) {base}", (START_DATE, hours))
        n, oldest = cur.fetchone()
        cur.execute(f"select status, count(*) {base} group by 1 order by 2 desc", (START_DATE, hours))
        by_status = {r[0]: r[1] for r in cur.fetchall()}
        cur.execute(f"select id::text, symbol, pattern, as_of_date, status, score_total {base} "
                    f"order by as_of_date, score_total desc limit %s", (START_DATE, hours, listing))
        items = [{"id": r[0], "symbol": r[1], "detector": r[2], "as_of": str(r[3]), "status": r[4],
                  "score": float(r[5]) if r[5] is not None else None} for r in cur.fetchall()]
    return {"unreviewed_over_hours": hours, "count": n, "oldest_as_of": str(oldest) if oldest else None,
            "by_status": by_status, "items": items, "counted_from": START_DATE, "healthy": n == 0}


def status_report(conn) -> dict:
    """What the review needs: totals, the >48h list, and taken / skipped / reasons per detector."""
    out = {"schema_ready": is_ready(conn), "schema_error": _state["error"], "reasons_allowed": list(REASONS),
           "counted_from": START_DATE}
    if not out["schema_ready"]:
        return out
    with conn.cursor() as cur:
        cur.execute("""
            select pattern,
                   count(*) filter (where verification_outcome = 'taken'),
                   count(*) filter (where verification_outcome = 'skipped'),
                   count(*) filter (where verification_outcome is null),
                   count(*)
              from signals
             where as_of_date >= %s::date and coalesce(notes->>'withdrawn', '') = ''
             group by 1 order by 5 desc""", (START_DATE,))
        out["by_detector"] = [{"detector": r[0], "taken": r[1], "skipped": r[2], "unreviewed": r[3], "total": r[4]}
                              for r in cur.fetchall()]
        cur.execute("""
            select pattern, verification_reason, count(*)
              from signals
             where verification_outcome = 'skipped' and as_of_date >= %s::date
             group by 1, 2 order by 1, 3 desc""", (START_DATE,))
        out["skip_reasons"] = [{"detector": r[0], "reason": r[1], "count": r[2]} for r in cur.fetchall()]
    out["reconciliation"] = reconcile(conn)
    return out


def run_reconciliation() -> dict:
    """The daily slot. Logs one line; WARNING when anything is unreviewed after 48 hours."""
    from ingest import connect
    if not is_ready():
        ensure_schema()
    if not is_ready():
        log.error("Verification reconciliation skipped: schema not available (%s)", _state["error"])
        return {"status": "failed", "error": _state["error"]}
    conn = connect()
    try:
        rec = reconcile(conn, listing=20)
    finally:
        conn.close()
    line = ("Verification reconciliation: %d signal(s) unreviewed for more than %d h (oldest as_of %s) by status %s%s"
            % (rec["count"], rec["unreviewed_over_hours"], rec["oldest_as_of"], rec["by_status"],
               "" if rec["count"] == 0 else " | " + ", ".join(f"{i['symbol']}:{i['detector']}@{i['as_of']}" for i in rec["items"][:10])))
    (log.info if rec["count"] == 0 else log.warning)("%s", line)
    return {"count": rec["count"], "oldest_as_of": rec["oldest_as_of"], "by_status": rec["by_status"]}


def schedule(scheduler, guarded, ist) -> None:
    from apscheduler.triggers.cron import CronTrigger
    h, m = (int(x) for x in RECON_TIME_IST.split(":"))
    scheduler.add_job(guarded, CronTrigger(hour=h, minute=m, timezone=ist), args=["verification_recon", run_reconciliation],
                      id="verification_recon", replace_existing=True, misfire_grace_time=3600, coalesce=True, max_instances=1)
    log.info("Scheduled verification_recon daily %s IST", RECON_TIME_IST)


def register(app, *, require_internal_key) -> None:
    from ingest import connect

    @app.post("/signals/verify")
    def verify_signal(request: Request, body: dict = Body(...)):
        """Mark a signal taken / skipped (with a reason), or clear the mark with outcome null."""
        require_internal_key(request)
        sid = body.get("signal_id")
        if not sid:
            raise HTTPException(400, "signal_id required")
        return set_outcome(sid, body.get("outcome"), body.get("reason"), body.get("note"))

    @app.get("/jobs/verification-status")
    def verification_status(request: Request):
        require_internal_key(request)
        conn = connect()
        try:
            return status_report(conn)
        finally:
            conn.close()
