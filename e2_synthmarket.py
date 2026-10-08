"""
e2_synthmarket.py — a synthetic multi-stock market (market factor + sector factors + idiosyncratic
moves, volume that follows |return|, occasional gaps) used to exercise the WHOLE pipeline —
panels, detectors, gates, scoring, trade simulation, portfolio, report — without a database.
It makes no claim about real-market edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from e2_backtest import Data
from e2_sector import SECTORS

INDUSTRY_NAMES = {"Financials": "Financial Services", "IT": "Information Technology", "Healthcare": "Healthcare",
                  "FMCG": "Fast Moving Consumer Goods", "Auto": "Automobile and Auto Components",
                  "Energy": "Oil Gas & Consumable Fuels", "Metals & Mining": "Metals & Mining",
                  "Industrials": "Capital Goods", "Realty": "Realty", "Chemicals & Materials": "Chemicals",
                  "Consumer & Telecom": "Consumer Durables"}


def make_market(n_stocks: int = 80, n_days: int = 900, seed: int = 7, start: str = "2021-01-04") -> Data:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    # market factor with a drawdown phase in the middle so regimes actually change
    drift = np.where((np.arange(n_days) > n_days * 0.45) & (np.arange(n_days) < n_days * 0.6), -0.0015, 0.0006)
    vol = np.where((np.arange(n_days) > n_days * 0.45) & (np.arange(n_days) < n_days * 0.6), 0.014, 0.008)
    rm = drift + vol * rng.standard_normal(n_days)
    sec_f = {s: 0.005 * rng.standard_normal(n_days) + 0.0002 * rng.standard_normal() for s in SECTORS}
    frames, industry = {}, {}
    for k in range(n_stocks):
        sector = SECTORS[k % len(SECTORS)]
        beta = 0.6 + 0.8 * rng.random()
        idio = 0.011 + 0.01 * rng.random()
        r = beta * rm + sec_f[sector] + 0.0003 + idio * rng.standard_normal(n_days)
        jumps = rng.random(n_days) < 0.01
        r = np.where(jumps, r + rng.choice([-1, 1], n_days) * (0.04 + 0.05 * rng.random(n_days)), r)
        c = 100 * (1 + 15 * rng.random()) * np.cumprod(1 + r)
        gap = 0.004 * rng.standard_normal(n_days) + np.where(jumps, r * 0.7, 0)
        o = np.empty(n_days); o[0] = c[0]; o[1:] = c[:-1] * (1 + gap[1:])
        hi = np.maximum(o, c) * (1 + 0.006 * np.abs(rng.standard_normal(n_days)))
        lo = np.minimum(o, c) * (1 - 0.006 * np.abs(rng.standard_normal(n_days)))
        vol_base = 2e5 * (1 + 20 * rng.random())
        v = vol_base * np.exp(0.3 * rng.standard_normal(n_days)) * (1 + 40 * np.abs(r))
        sym = f"SYN{k:03d}"
        frames[sym] = pd.DataFrame({"trade_date": dates, "open": o, "high": hi, "low": lo, "close": c, "volume": v})
        industry[sym] = INDUSTRY_NAMES[sector] if k % 17 else None
    ix = pd.Series(1000 * np.cumprod(1 + rm), index=dates)
    vixv = pd.Series(15 + 400 * np.abs(pd.Series(rm).rolling(10).std().fillna(0.01).to_numpy()) , index=dates)
    cov = {"universe": {"scanned": n_stocks}, "surveillance": {"snapshot_days": 0},
           "results_calendar": {"rows": 0}}
    return Data(frames, ix, "SYNTH", vixv, industry, dict(industry), {}, {}, cov)
