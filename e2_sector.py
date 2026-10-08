"""
e2_sector.py — 11 sectors, equal-weighted composites, 3-month relative-strength rank.

Source of sector strength: NSE sector indices are NOT in the database (only NIFTY50,
NIFTY500 and INDIAVIX are ingested), so each sector is an equal-weighted composite of
its constituents in the scan universe: the average of their daily returns, chained.
That is recorded in every report as `sector_source = "equal_weight_composite"`.

The sector score is a RANK, not a gate: top 3 of 11 get a boost, bottom 3 a penalty,
through the 0-10 "sector alignment" component of the composite score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTORS = ["Financials", "IT", "Healthcare", "FMCG", "Auto", "Energy", "Metals & Mining",
           "Industrials", "Realty", "Chemicals & Materials", "Consumer & Telecom"]

# NSE industry names (Nifty 500 'Industry' column) -> one of the 11. Matching is by lower-case
# substring so small naming differences do not drop a stock into 'unknown'.
_RULES = [
    ("Financials", ("financial", "bank", "insurance")),
    ("IT", ("information technology", "software", "it -")),
    ("Healthcare", ("healthcare", "pharma", "hospital")),
    ("FMCG", ("fast moving", "fmcg")),
    ("Auto", ("automobile", "auto component", "auto ")),
    ("Energy", ("oil", "gas", "power", "coal", "energy", "utilities")),
    ("Metals & Mining", ("metal", "mining", "steel")),
    ("Realty", ("realty", "real estate")),
    ("Chemicals & Materials", ("chemical", "construction materials", "cement", "forest", "textile", "paper", "fertili")),
    ("Consumer & Telecom", ("consumer", "telecom", "media", "retail", "diversified", "durables", "hotel", "leisure")),
    ("Industrials", ("capital goods", "construction", "industrial", "services", "logistic", "defence", "engineering")),
]


def sector_of(industry: str | None, sector: str | None = None) -> str | None:
    """None when it cannot be mapped (such a stock gets a neutral sector score and no sector cap)."""
    for text in (industry, sector):
        if not text:
            continue
        t = text.lower()
        for name, keys in _RULES:
            if any(k in t for k in keys):
                return name
    return None


def composites(close: pd.DataFrame, sector_by_symbol: dict[str, str | None]) -> pd.DataFrame:
    """close: dates x symbols. Returns dates x sectors, each an equal-weighted index (start 100)."""
    ret = close.pct_change(fill_method=None)
    out = {}
    for s in SECTORS:
        cols = [c for c in close.columns if sector_by_symbol.get(c) == s]
        if len(cols) < 2:
            continue
        r = ret[cols].mean(axis=1, skipna=True).fillna(0.0)
        out[s] = (1 + r).cumprod() * 100.0
    return pd.DataFrame(out, index=close.index)


def rs_rank(comp: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
    """rank 1 = strongest 3-month return among the sectors that exist on that date."""
    r = comp / comp.shift(lookback) - 1.0
    return r.rank(axis=1, ascending=False, method="min")


def sector_points(rank: float | None, n: int) -> float:
    """0-10: rank 1 -> 10 ... last -> 0 (so the top 3 sit at >= 8 and the bottom 3 at <= 2 when n=11).
    Unknown sector or too few sectors -> 5 (neutral)."""
    if rank is None or np.isnan(rank) or n < 2:
        return 5.0
    return float(10.0 * (n - rank) / (n - 1))
