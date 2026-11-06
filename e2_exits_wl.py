"""
e2_exits_wl.py — EXIT STUDY ON THE PUBLISHING DETECTORS ONLY (owner's order of 7 Nov 01:25 IST), plus the horizon-specific exit test.

Part 1 - variants A/B/C/D exactly as defined in e2_exits (and the live exits as BASE), on episodic_pivot and gap_and_go only, per detector,
          full sample and both halves. Deploy rule (owner's): a variant that is positive (net R) on BOTH detectors AND in both halves of each.
Part 2 - horizon-specific rulebooks, defined here before any result:
  5-session mode   S1 stop 1.0 ATR, no target, exit at the next open after bar 5
                   S2 stop 1.5 ATR, no target, exit at the next open after bar 5
                   S3 stop 1.25 ATR, whole position sold at +1.5R, otherwise exit after bar 5
                   S4 stop 1.25 ATR, whole position sold at +2.0R, otherwise exit after bar 5
  20-session mode  L1 stop 2 ATR   | L2 stop 3 ATR | L3 stop 5% below the structural level;
                   all three: sell 50% at +3R (stop to break-even), sell 25% at +5R, the last 25% trails (looser of 20-EMA and highest close - 2 ATR),
                   no time stop until bar 20, then everything left exits at the next open.
Selection caveat that stays with every number here: the two detectors were chosen BECAUSE they were the positive ones in this same 3-year sample,
so their results under any exit are optimistic by construction. The forward-return test (which does not use stops) is the independent check on them,
and the live forward log from 9 Nov is the only out-of-sample evidence there will be.

Writes engine_settings.engine2_exits_wl; logs E2EXW| lines; GET /jobs/engine2-exits-wl.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

import e2_backtest as BT
import e2_manage as MG
from e2_det_common import WARMUP
from e2_detectors import detect_all
from e2_exits import BASE, SPECS, ExitSpec, block, boot_mean, paired_delta, simulate_x
from e2_features import compute_features

log = logging.getLogger("engine2.exits_wl")
KEY = "engine2_exits_wl"
LOG_PREFIX = "E2EXW|"
WHITELIST = tuple(d.strip() for d in os.environ.get("E2_PUBLISH_DETECTORS", "episodic_pivot,gap_and_go").split(",") if d.strip()) or ("episodic_pivot", "gap_and_go")

H5 = {
    "S1": ExitSpec("S1", stop_mode="atr", stop_atr=1.0, partial_r=None, t2_r=None, trail="none", time_bars=5, time_hard=True, time_needs_no_partial=False),
    "S2": ExitSpec("S2", stop_mode="atr", stop_atr=1.5, partial_r=None, t2_r=None, trail="none", time_bars=5, time_hard=True, time_needs_no_partial=False),
    "S3": ExitSpec("S3", stop_mode="atr", stop_atr=1.25, partial_r=1.5, partial_frac=1.0, t2_r=None, trail="none", time_bars=5, time_hard=True, time_needs_no_partial=False),
    "S4": ExitSpec("S4", stop_mode="atr", stop_atr=1.25, partial_r=2.0, partial_frac=1.0, t2_r=None, trail="none", time_bars=5, time_hard=True, time_needs_no_partial=False),
}
_L = dict(partial_r=3.0, partial_frac=0.50, t2_r=5.0, t2_frac=0.25, trail="base", time_bars=20, time_hard=True, time_needs_no_partial=False)
H20 = {
    "L1": ExitSpec("L1", stop_mode="atr", stop_atr=2.0, **_L),
    "L2": ExitSpec("L2", stop_mode="atr", stop_atr=3.0, **_L),
    "L3": ExitSpec("L3", stop_mode="structural", **_L),
}
PRIMARY = dict(SPECS)
ALL = {"BASE": BASE, **PRIMARY, **H5, **H20}


def collect(data, pn, w0):
    res = {k: [] for k in ALL}
    surv_active = data.coverage.get("surveillance", {}).get("snapshot_days", 0) >= 30
    n_by_det = Counter()
    for sym, df in sorted(data.frames.items()):
        b = compute_features(sym, df)
        if b is None:
            continue
        lo = max(int(np.searchsorted(b.date, np.datetime64(w0, "D"))), WARMUP)
        cands, _ = detect_all(b, lo=lo)
        for c in cands:
            if c.detector not in WHITELIST:
                continue
            if BT.gate(b, c, data.surveillance.get(sym), surv_active) or pn.row_of.get(b.date[c.idx]) is None:
                continue
            n_by_det[c.detector] += 1
            ev = data.events.get(sym)
            for name, X in ALL.items():
                if name == "BASE":
                    tr = MG.simulate(b, c, ev)
                    tr.loss_stop = tr.reason == "stop" or (tr.reason == "stop_gap" and len(tr.exits) == 1)
                else:
                    tr = simulate_x(b, c, ev, X)
                res[name].append(tr)
    return res, n_by_det


def _half_ok(v: dict) -> bool:
    return bool(v.get("filled") and v["net_r"] is not None and v["net_r"] > 0 and (v["first_half"]["net_r"] or -1) > 0 and (v["second_half"]["net_r"] or -1) > 0)


def compute(data, out: dict) -> dict:
    pn = BT.build_panels(data)
    w0 = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS)]
    mid = pn.cal[max(0, len(pn.cal) - BT.WINDOW_BARS // 2)]
    mid64 = np.datetime64(mid, "D")
    weeks = len(pn.cal[pn.cal >= w0]) / 5.0
    out["window"] = {"start": str(w0.date()), "split": str(mid.date()), "end": str(pn.cal[-1].date()), "weeks": round(weeks, 1)}
    out["whitelist"] = list(WHITELIST)
    res, n_by_det = collect(data, pn, w0)
    out["candidates"] = dict(n_by_det)
    table = {}
    for det in (*WHITELIST, "BOTH"):
        table[det] = {}
        for name, trs in res.items():
            sel = [t for t in trs if det == "BOTH" or t.detector == det]
            n = sum(n_by_det.values()) if det == "BOTH" else n_by_det.get(det, 0)
            table[det][name] = block(sel, mid64, n, weeks)
    out["by_detector"] = table
    out["paired_vs_base_both"] = {name: paired_delta(res["BASE"], trs) for name, trs in res.items() if name != "BASE"}
    # deploy rules
    def passes(name):
        return all(_half_ok(table[d][name]) for d in WHITELIST)
    out["part1_rule"] = {"text": "positive net R on BOTH publishing detectors and in both halves of each (variants A-D)",
                         "passing": [n for n in PRIMARY if passes(n)],
                         "per_variant": {n: {d: {"full": table[d][n].get("net_r"), "h1": table[d][n]["first_half"]["net_r"] if table[d][n].get("filled") else None,
                                                 "h2": table[d][n]["second_half"]["net_r"] if table[d][n].get("filled") else None} for d in WHITELIST} for n in PRIMARY}}
    out["part2_rule"] = {"text": "same rule inside each mode",
                         "mode_5_session_passing": [n for n in H5 if passes(n)], "mode_20_session_passing": [n for n in H20 if passes(n)],
                         "note": "nothing is deployed until the owner picks the horizon"}
    out["note"] = ("The two detectors were selected for being positive in this same sample: every number here is optimistic by construction; "
                   "n is small per half (episodic_pivot about 50 per half).")
    return out


def run(conn=None, trigger: str = "manual") -> dict:
    from ingest import connect
    own = conn is None
    conn = conn or connect()
    t0 = time.time()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "trigger": trigger, "state": "running", "note": "Measurement only."}
    try:
        data = BT.load_from_db(conn)
        compute(data, out)
        out["state"] = "done"
    except Exception as e:                                                             # noqa: BLE001
        log.exception("whitelist exit study failed")
        out["state"], out["error"] = "failed", f"{type(e).__name__}: {e}"[:300]
    out["runtime_sec"] = round(time.time() - t0)
    try:
        with conn.cursor() as cur:
            cur.execute("insert into engine_settings (key, value, updated_at) values (%s, %s::jsonb, now()) "
                        "on conflict (key) do update set value = excluded.value, updated_at = now()", (KEY, json.dumps(out, default=str)))
        conn.commit()
    except Exception:                                                                  # noqa: BLE001
        log.exception("could not store the whitelist exit study")
    try:
        log_markdown(render_markdown(out))
    finally:
        if own:
            conn.close()
    return out


def render_markdown(d: dict) -> str:
    L = [f"# Exit study, publishing detectors only  [{d.get('state')}]  generated {d.get('generated_at')}  runtime {d.get('runtime_sec')}s"]
    if d.get("error"):
        L.append(f"ERROR: {d['error']}")
    L.append(f"whitelist {d.get('whitelist')} candidates {d.get('candidates')} window {d.get('window')}")
    for det, row in (d.get("by_detector") or {}).items():
        L.append(f"## {det}")
        for name, v in row.items():
            if not v.get("filled"):
                L.append(f"{name}: no trades")
                continue
            h1, h2 = v["first_half"], v["second_half"]
            L.append(f"{name}: n {v['filled']} netR {v['net_r']} CI{v['net_r_ci95']} gross {v['gross_r']} | halves {h1['net_r']} (n{h1['n']}) / {h2['net_r']} (n{h2['n']}) "
                     f"| win {v['win_pct']}% PF {v['pf']} | risk {v['risk_pct_mean']}% | stop-out {v['stop_out_pct']}% early {v['early_stop_out_pct']}% | bars {v['avg_bars']} "
                     f"| ret {v['avg_return_pct']}% | exits {v['exit_mix_pct']}")
    L.append("## paired vs BASE, both detectors")
    for n, v in (d.get("paired_vs_base_both") or {}).items():
        L.append(f"{n}: n {v.get('n')} delta R {v.get('mean_delta_r')} CI{v.get('ci95')}")
    L.append("## rules")
    L.append(json.dumps(d.get("part1_rule"), default=str))
    L.append(json.dumps(d.get("part2_rule"), default=str))
    L.append(str(d.get("note")))
    return "\n".join(L)


def log_markdown(md: str) -> int:
    n = 0
    for line in md.splitlines():
        for i in range(0, max(len(line), 1), 1400):
            log.info("%s%s", LOG_PREFIX, line[i:i + 1400])
            n += 1
    return n
