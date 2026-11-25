"""
e2_tier1.py — TIER 1 REPLACEMENT COMPONENTS (owner's order of 26 Nov 00:38 IST: replace, do not add).

Four components, each a 0-1 fraction in the same slot as the component it replaces, so the composite keeps its shape and the WEIGHTS decide what is real
(train-half refit, e2_t1study.py). Nothing here is read by the live scan: E2_TIER1 is not wired into e2_scorer.components until a refit has been reported.

  volume    the existing volume fraction with its linear relvol term (40%) replaced by tiers; OBV trend, prior dry-up and up/down volume (60%) are unchanged.
            relvol >= 2.5 -> 1.0 (+10 pts) | 1.5-2.5 -> 0.75 (+5) | 0.8-1.5 -> 0.5 (0) | < 0.8 -> 0.0 (-10)       [points mapped affinely: (pts + 10) / 20]
  sector    the existing 3-month sector RS rank as tiers: rank 1-3 -> 1.0 (+15) | middle -> 0.5 (0) | last 3 -> 0.0 (-15); unknown sector -> 0.5 (neutral)
  catalyst  the existing gap+volume proxy (or confirmed news) UNCHANGED, plus an earnings-date check: a results date within +/-5 calendar days of the signal date
            sets the fraction to 1.0 (+15). Where events_calendar holds no past results dates this equals the old component, and the study says so.
  delivery  NEW slot. Delivery % of the signal session: > 50 -> 1.0 (+10) | 35-50 -> 0.75 (+5) | 25-35 -> 0.5 (0) | < 25 -> 0.0 (-10); missing -> 0.5 (neutral).
            Timing caveat: the bhavcopy for a session is published after the close (typically evening), so a live postclose scan may not have the signal day's delivery;
            the study also reports the previous session's value (delivery_lag1), the live-feasible variant.
"""
from __future__ import annotations

import numpy as np

from e2_core import clamp
from e2_features import Bars
import e2_scorer as SC

EARNINGS_WINDOW_DAYS = 5
NEW_COMPONENTS = ["trend", "rs", "volume", "base", "catalyst", "sector", "regime", "delivery"]


def relvol_tier(rv: float | None) -> float:
    if rv is None or np.isnan(rv):
        return 0.5                       # unknown volume is neutral, not a penalty
    if rv >= 2.5:
        return 1.0
    if rv >= 1.5:
        return 0.75
    if rv >= 0.8:
        return 0.5
    return 0.0


def volume_fraction_t1(b: Bars, i: int) -> float:
    """SC.volume_fraction with the linear relvol term replaced by the tier fraction (the other 60% is identical code)."""
    f = 0.4 * relvol_tier(b.relvol[i])
    f += 0.2 * (b.obv_trend[i] > 0)
    f += 0.2 * bool(np.nan_to_num(b.vol_dry5[max(0, i - 10):i].astype(float)).max() > 0) if i > 0 else 0.0
    lo = max(1, i - 49)
    up = b.v[lo:i + 1][b.c[lo:i + 1] > b.c[lo - 1:i]].sum()
    dn = b.v[lo:i + 1][b.c[lo:i + 1] < b.c[lo - 1:i]].sum()
    if dn > 0:
        f += 0.2 * clamp((up / dn - 0.8) / 0.8, 0, 1)
    elif up > 0:
        f += 0.2
    return float(min(f, 1.0))


def sector_fraction_t1(rank: float | None, n: int) -> float:
    if rank is None or np.isnan(rank) or n < 6:
        return 0.5
    if rank <= 3:
        return 1.0
    if rank > n - 3:
        return 0.0
    return 0.5


def earnings_near(events: np.ndarray | None, date64: np.datetime64, days: int = EARNINGS_WINDOW_DAYS) -> bool:
    if events is None or len(events) == 0:
        return False
    d = np.datetime64(date64, "D")
    k = int(np.searchsorted(events, d - np.timedelta64(days, "D"), side="left"))
    return k < len(events) and events[k] <= d + np.timedelta64(days, "D")


def catalyst_fraction_t1(b: Bars, cand, events: np.ndarray | None) -> tuple[float, str]:
    """cand needs .detector, .meta, .idx (SC.catalyst_fraction's own reads). Earnings within the window lifts the fraction to 1.0; never lowers it."""
    base, src = SC.catalyst_fraction(b, cand)
    if earnings_near(events, b.date[cand.idx]):
        return 1.0, "earnings"
    return base, src


def delivery_fraction(pct: float | None) -> float:
    if pct is None or np.isnan(pct):
        return 0.5
    if pct > 50.0:
        return 1.0
    if pct >= 35.0:
        return 0.75
    if pct >= 25.0:
        return 0.5
    return 0.0


def t1_components(frac_old: dict, b: Bars, cand, rank: float | None, n_sec: int, delivery_pct: float | None,
                  events: np.ndarray | None) -> dict:
    """The old fractions with volume, sector and catalyst REPLACED and delivery added. frac_old is not modified."""
    f = {k: v for k, v in frac_old.items() if not k.startswith("_")}
    f["volume"] = volume_fraction_t1(b, cand.idx)
    f["sector"] = sector_fraction_t1(rank, n_sec)
    f["catalyst"], f["_catalyst_source"] = catalyst_fraction_t1(b, cand, events)
    f["delivery"] = delivery_fraction(delivery_pct)
    f["_delivery_missing"] = delivery_pct is None or bool(np.isnan(delivery_pct))
    return f
