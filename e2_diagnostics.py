"""
e2_diagnostics.py — per-detector exhaustiveness.

  setups_detected        what the detector found (after the 3-bar cooldown)
  setups_expected        what the INDEPENDENT recognizer (e2_recognizers) says exists
  setups_matched         expected setups the detector also found (within +-MATCH_TOL bars)
  miss_rate              1 - matched / expected
  extra                  detections the recognizer does not agree with
  false_positive_rate    share of simulated trades stopped out within 5 bars of entry
  status                 'broken' when miss_rate > 50% or false_positive_rate > 60%;
                         'thin_sample' when fewer than MIN_SAMPLE setups exist to judge;
                         'ok' otherwise

This is a DIAGNOSTIC: it never switches a detector off. A 'broken' flag means the definition or
its implementation should be examined, and the missed/extra examples are listed to make that
possible.
"""
from __future__ import annotations

import numpy as np

from e2_features import Bars
from e2_recognizers import RECOGNIZERS, expected_bars

MATCH_TOL = 2
MIN_SAMPLE = 20
MISS_BROKEN, FP_BROKEN = 0.50, 0.60
FAIL_BARS = 5
STOP_REASONS = {"stop", "stop_gap", "breakeven_stop", "trail_stop"}


def compare_symbol(b: Bars, name: str, detected_idx: list[int], lo: int, hi: int) -> dict:
    exp = expected_bars(b, name, lo, hi)
    det = sorted(detected_idx)
    matched = [e for e in exp if any(abs(e - d) <= MATCH_TOL for d in det)]
    extra = [d for d in det if not any(abs(e - d) <= MATCH_TOL for e in exp)]
    missed = [e for e in exp if e not in matched]
    return {"expected": len(exp), "detected": len(det), "matched": len(matched),
            "missed": [str(b.date[x]) for x in missed[:5]], "extra": [str(b.date[x]) for x in extra[:5]],
            "n_extra": len(extra), "n_missed": len(missed)}


def aggregate(per_symbol: dict[str, dict[str, dict]], trades_by_det: dict[str, list]) -> dict:
    """per_symbol[symbol][detector] = compare_symbol(...); trades_by_det[detector] = [Trade]"""
    out = {}
    for name in RECOGNIZERS:
        exp = det = mat = n_extra = 0
        missed, extra = [], []
        for sym, d in per_symbol.items():
            r = d.get(name)
            if not r:
                continue
            exp += r["expected"]; det += r["detected"]; mat += r["matched"]; n_extra += r["n_extra"]
            missed += [(sym, x) for x in r["missed"]]
            extra += [(sym, x) for x in r["extra"]]
        trs = [t for t in trades_by_det.get(name, []) if not t.skipped]
        failed = sum(1 for t in trs if t.reason in STOP_REASONS and t.bars <= FAIL_BARS)
        miss = (1 - mat / exp) if exp else None
        fp = (failed / len(trs)) if trs else None
        if max(exp, det) < MIN_SAMPLE:
            status = "thin_sample"
        elif (miss is not None and miss > MISS_BROKEN) or (fp is not None and fp > FP_BROKEN):
            status = "broken"
        else:
            status = "ok"
        out[name] = {
            "setups_detected": det, "setups_expected": exp, "setups_matched": mat,
            "miss_rate": None if miss is None else round(miss, 3),
            "extra_detections": n_extra,
            "false_positive_rate": None if fp is None else round(fp, 3),
            "trades_simulated": len(trs), "status": status,
            "missed_examples": missed[:8], "extra_examples": extra[:8],
        }
    return out
