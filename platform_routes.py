"""
platform_routes.py — read-only diagnostics added in Step 1 of the rebuild.

Registered from server.py with one call. Nothing here writes to a market table;
the only write is the data-quality report into engine_settings. None of it is
imported by scan.py or the scheduler's scan path.

  GET  /jobs/universe-report     what the scan universe is now vs the old
                                 is_active-only rule (the before/after numbers)
  GET  /jobs/data-quality        latest stored report
  POST /jobs/data-quality        run the checks now (synchronous, a few seconds)
  GET  /jobs/intraday-probe      is historical intraday data available, how far back
  GET  /jobs/step-status         ONE json: is Step 1 closed, and if not why and when
"""
from __future__ import annotations

import json
from datetime import date, datetime

from fastapi import HTTPException, Request


def _int_list(raw: str, default: tuple) -> tuple:
    try:
        return tuple(int(x) for x in raw.split(",") if x.strip()) or default
    except ValueError:
        raise HTTPException(400, f"not a comma-separated list of integers: {raw!r}")


def register(app, *, require_internal_key, ist):

    @app.get("/jobs/universe-report")
    def universe_report(request: Request):
        """
        BEFORE (the old rule: every active non-index symbol) vs AFTER (active,
        in the Nifty 500, not excluded), computed from the same database at the
        same moment, so the comparison needs no pre-deploy snapshot. Includes
        breadth both ways, since breadth feeds the regime, and the names that
        reach Gate 2 with no fundamentals under each rule.
        """
        require_internal_key(request)
        from ingest import connect
        from scan import _breadth
        from universe import load_scan_universe, trust_lookalikes

        conn = connect()
        try:
            after = load_scan_universe(conn)
            with conn.cursor() as cur:
                cur.execute("select symbol from symbols where is_active "
                            "and coalesce(series,'') <> 'INDEX' order by symbol")
                before = [r[0] for r in cur.fetchall()]
                cur.execute("select distinct symbol from fundamentals_quarterly")
                have = {r[0] for r in cur.fetchall()}
            b_breadth, a_breadth = _breadth(conn, before), _breadth(conn, after.symbols)
            removed = sorted(set(before) - set(after.symbols))
            no_f_before = sorted(s for s in before if s not in have)
            no_f_after = sorted(s for s in after.symbols if s not in have)
            return {
                "as_of": datetime.now(ist).isoformat(timespec="seconds"),
                "before": {"rule": "is_active and not INDEX", "universe": len(before),
                           "breadth_above_50dma": round(b_breadth, 2),
                           "no_fundamentals": len(no_f_before)},
                "after": {"rule": "is_active and in_nifty500 and not excluded",
                          "universe": len(after.symbols),
                          "breadth_above_50dma": round(a_breadth, 2),
                          "no_fundamentals": len(no_f_after), "detail": after.detail},
                "delta": {"universe": len(after.symbols) - len(before),
                          "breadth_pts": round(a_breadth - b_breadth, 2)},
                "removed_symbols": removed,
                "no_fundamentals_before": no_f_before,
                "no_fundamentals_after": no_f_after,
                "no_fundamentals_removed_by_filter": sorted(set(no_f_before) - set(no_f_after)),
                "trust_lookalikes_not_excluded": trust_lookalikes(conn),
            }
        finally:
            conn.close()

    @app.get("/jobs/data-quality")
    def data_quality_latest(request: Request):
        require_internal_key(request)
        from ingest import connect
        conn = connect()
        try:
            with conn.cursor() as cur:
                cur.execute("select value, updated_at from engine_settings "
                            "where key = 'data_quality_latest'")
                row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return {"report": None, "note": "no report stored yet; POST /jobs/data-quality to run one"}
        return {"stored_at": str(row[1]), "report": row[0]}

    @app.post("/jobs/data-quality")
    def data_quality_run(request: Request):
        require_internal_key(request)
        import data_quality
        from ingest import connect
        conn = connect()
        try:
            rep = data_quality.run_checks(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "insert into engine_settings (key, value, updated_at) values "
                    "('data_quality_latest', %s::jsonb, now()) "
                    "on conflict (key) do update set value = excluded.value, updated_at = now()",
                    (json.dumps(rep, default=str),))
            conn.commit()
            return rep
        finally:
            conn.close()

    @app.get("/jobs/intraday-probe")
    def intraday_probe(request: Request):
        """
        GET /jobs/intraday-probe?symbol=RELIANCE&intervals=1,5,15
        One Upstox call per (sampled date, interval). Read-only. Needs a valid token.
        """
        require_internal_key(request)
        from ingest import connect
        from upstox_client import TokenStore, UpstoxClient
        import data_probe

        if TokenStore().valid_token() is None:
            raise HTTPException(409, "No valid Upstox token — log in first")
        symbol = request.query_params.get("symbol", "RELIANCE").upper()
        intervals = _int_list(request.query_params.get("intervals", ""), data_probe.DEFAULT_INTERVALS)
        conn = connect()
        try:
            result = data_probe.probe_intraday(UpstoxClient(), conn, symbol, intervals)
            if "verdict" in result:
                data_probe.store_result(conn, result)     # readable later via /jobs/step-status
            return result
        finally:
            conn.close()

    @app.get("/jobs/step-status")
    def step_status(request: Request):
        """
        The single status call for the rebuild's Step 1. Read-only. Same
        x-internal-key header as /jobs/status.
        """
        require_internal_key(request)
        import os
        import scheduler
        import step_status as SS
        from ingest import connect

        next_runs = {}
        if scheduler._scheduler:
            for job in scheduler._scheduler.get_jobs():
                next_runs[job.id] = job.next_run_time.isoformat() if job.next_run_time else None
        try:
            from upstox_client import TokenStore
            token_valid = TokenStore().valid_token() is not None
        except Exception:
            token_valid = None
        conn = connect()
        try:
            return SS.build(conn, next_runs=next_runs, token_valid=token_valid, env=dict(os.environ))
        finally:
            conn.close()
