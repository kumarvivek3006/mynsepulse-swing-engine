"""
e2_probe30.py — is 30-minute (and 15-minute) HISTORY served by Upstox, and how far back? (owner's order of 7 Nov 12:38 IST: open_range_breakout needs
the first 30-minute range; if Upstox provides it, that detector can be built; if not, the candidate is blocked.)

Two things, both logged with the prefix E2PRB|:
  1. the verdict the engine's own intraday probe stored earlier (engine_settings.intraday_probe_latest), if there is one;
  2. a fresh probe at 15 and 30 minutes over lookbacks out to ~4 years, when a valid Upstox token is present (the token is valid only after the owner's
     daily login; without one this says so and nothing is guessed).
Read-only apart from storing the fresh verdict under engine_settings.engine2_probe30.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

log = logging.getLogger("engine2.probe30")
KEY = "engine2_probe30"
LOG_PREFIX = "E2PRB|"
INTERVALS = (15, 30)
LOOKBACKS = (1, 7, 30, 90, 180, 365, 540, 730, 1095, 1460)


def stored_verdict(conn) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("select value, updated_at from engine_settings where key = 'intraday_probe_latest'")
        row = cur.fetchone()
    if not row or not isinstance(row[0], dict):
        return None
    v = row[0]
    return {"stored_at": str(row[1]), "symbol": v.get("symbol"), "probed_at": v.get("probed_at"), "verdict": v.get("verdict"),
            "error": v.get("error"), "dates_probed": v.get("dates_probed")}


def run(conn=None, trigger: str = "manual", client=None) -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running"}
    try:
        out["stored_probe"] = stored_verdict(conn)
        try:
            if client is None:
                from upstox_client import TokenStore, UpstoxClient
                if TokenStore().valid_token() is None:
                    out["fresh_probe"] = {"ran": False, "reason": "no valid Upstox token (the daily login has not happened since the 03:30 IST expiry)"}
                    client = None
                else:
                    client = UpstoxClient()
            if client is not None:
                import data_probe
                res = data_probe.probe_intraday(client, conn, "RELIANCE", INTERVALS, lookbacks=LOOKBACKS)
                out["fresh_probe"] = {"ran": True, "symbol": res.get("symbol"), "error": res.get("error"), "verdict": res.get("verdict"),
                                      "cells": res.get("cells")}
        except Exception as e:                                                         # noqa: BLE001
            log.exception("fresh intraday probe failed")
            out["fresh_probe"] = {"ran": False, "reason": f"{type(e).__name__}: {e}"[:300]}
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("probe30 failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the probe30 result")
    try:
        for line in render(out).splitlines():
            for i in range(0, max(len(line), 1), 1400):
                log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
    finally:
        if own:
            conn.close()
    return out


def render(d: dict) -> str:
    L = [f"# Intraday history probe  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    L.append("stored probe (engine_settings.intraday_probe_latest): " + json.dumps(d.get("stored_probe"), default=str))
    fp = d.get("fresh_probe") or {}
    L.append(f"fresh probe ran: {fp.get('ran')}" + (f" reason: {fp.get('reason')}" if fp.get("reason") else ""))
    if fp.get("ran"):
        L.append("verdict: " + json.dumps(fp.get("verdict"), default=str))
        for itv, rows in (fp.get("cells") or {}).items():
            L.append(f"{itv}-minute: " + "; ".join(f"{r['date']} n{r.get('candles')}/{r.get('expected')}" + ("" if r.get("ok") else f" ({r.get('error') or 'partial'})") for r in rows))
    return "\n".join(L)
