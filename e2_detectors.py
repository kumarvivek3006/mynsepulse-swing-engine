"""
e2_detectors.py — the registry. Runs EVERY detector on EVERY candidate bar; no detector
gates another and none is switched off by how it performed.

  detect_all(bars, lo, hi)   -> (candidates, errors)

`errors` is a list of (detector, message): a detector that raises is recorded and the
rest still run, so a bug is a visible line in the report, not a silent hole.

Dedupe: the same detector on the same symbol re-confirming within COOLDOWN bars is the
same setup seen again (e.g. a pocket pivot on two consecutive days); the first is kept.
That is bookkeeping, not a filter: the first confirmation is always kept, whatever it scores.
"""
from __future__ import annotations

import traceback

from e2_core import FAMILY, Candidate, ienv
from e2_det_bases import (asc_triangle, ascending_base, base_on_base, cup_handle, desc_triangle,
                          double_bottom, flag_pennant, flat_base, high_tight_flag, inverse_hs,
                          rounding_bottom, sym_triangle, vcp)
from e2_det_common import WARMUP
from e2_det_momentum import (breakout_retest, episodic_pivot, gap_and_go, momentum_burst,
                             pocket_pivot, pullback_ema, undercut_reclaim)
from e2_features import Bars

COOLDOWN = ienv("E2_COOLDOWN", 3)

DETECTORS = {
    "vcp": vcp, "flat_base": flat_base, "base_on_base": base_on_base, "cup_handle": cup_handle,
    "ascending_base": ascending_base, "flag_pennant": flag_pennant, "asc_triangle": asc_triangle,
    "sym_triangle": sym_triangle, "desc_triangle": desc_triangle, "high_tight_flag": high_tight_flag,
    "double_bottom": double_bottom, "rounding_bottom": rounding_bottom, "inverse_hs": inverse_hs,
    "undercut_reclaim": undercut_reclaim, "pullback_ema": pullback_ema, "breakout_retest": breakout_retest,
    "episodic_pivot": episodic_pivot, "momentum_burst": momentum_burst, "pocket_pivot": pocket_pivot,
    "gap_and_go": gap_and_go,
}
assert set(DETECTORS) == set(FAMILY), "every detector needs a family and vice versa"


def dedupe(cands: list[Candidate], cooldown: int = COOLDOWN) -> list[Candidate]:
    out: list[Candidate] = []
    last: dict[str, int] = {}
    for c in sorted(cands, key=lambda x: (x.detector, x.idx)):
        prev = last.get(c.detector)
        if prev is not None and c.idx - prev <= cooldown:
            continue
        last[c.detector] = c.idx
        out.append(c)
    return sorted(out, key=lambda x: (x.idx, x.detector))


def detect_all(b: Bars, lo: int = WARMUP, hi: int | None = None, ctx: dict | None = None,
               only: list[str] | None = None) -> tuple[list[Candidate], list[tuple[str, str]]]:
    """Candidates confirmed on bars [lo, hi). hi defaults to every bar."""
    hi = b.n if hi is None else min(hi, b.n)
    ctx = ctx or {}
    found: list[Candidate] = []
    errors: list[tuple[str, str]] = []
    for name, fn in DETECTORS.items():
        if only and name not in only:
            continue
        try:
            found.extend(fn(b, ctx, max(lo, WARMUP), hi))
        except Exception as e:                       # noqa: BLE001 — recorded, never swallowed silently
            errors.append((name, f"{type(e).__name__}: {e} @ {traceback.format_exc().splitlines()[-3].strip()}"))
    return dedupe(found), errors
