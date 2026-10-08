"""
e2_synth.py — synthetic OHLCV with PLANTED setups, so a detector can be tested
against a known answer (recall) and against shapes that look similar but are
outside the definition (specificity).

No real market data is available where this is written; this is how the
detectors are exercised before the server-side validation run on the real
database. It proves a detector implements its definition. It does not prove
the definition makes money — that is the validation run's job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class Chart:
    def __init__(self, seed: int = 1, start_price: float = 100.0, vol_base: float = 1_000_000):
        self.rng = np.random.default_rng(seed)
        self.closes: list[float] = [start_price]
        self.vol: list[float] = [vol_base]
        self.vol_base = vol_base
        self.wick = 0.004
        self.gap_noise = 0.001
        self.overrides: dict[int, dict] = {}

    # ---- builders ----------------------------------------------------
    def drift(self, n: int, daily: float = 0.001, sigma: float = 0.012, vmult: float = 1.0):
        for _ in range(n):
            r = daily + sigma * self.rng.standard_normal()
            self.closes.append(self.closes[-1] * (1 + r))
            self.vol.append(self.vol_base * vmult * float(np.exp(0.25 * self.rng.standard_normal())))
        return self

    def path(self, points: list[tuple[int, float]], sigma: float = 0.004, vmult: float | list = 1.0):
        """piecewise-linear path through (bars, target_price) waypoints, with small noise."""
        for k, (n, target) in enumerate(points):
            start = self.closes[-1]
            for t in range(1, n + 1):
                base = start + (target - start) * t / n
                p = base * (1 + sigma * self.rng.standard_normal()) if t < n else target
                self.closes.append(p)
                vm = vmult[k] if isinstance(vmult, list) else vmult
                self.vol.append(self.vol_base * vm * float(np.exp(0.2 * self.rng.standard_normal())))
        return self

    def bar(self, close: float, vmult: float = 1.0, open_: float | None = None,
            high: float | None = None, low: float | None = None):
        self.closes.append(close)
        self.vol.append(self.vol_base * vmult)
        self.overrides[len(self.closes) - 1] = {k: v for k, v in
                                               (("open", open_), ("high", high), ("low", low)) if v is not None}
        return self

    @property
    def last(self) -> int:
        return len(self.closes) - 1

    # ---- output ------------------------------------------------------
    def frame(self, symbol: str = "SYN", start: str = "2023-01-02") -> pd.DataFrame:
        c = np.array(self.closes, float)
        n = len(c)
        o = np.empty(n)
        o[0] = c[0]
        o[1:] = c[:-1] * (1 + self.gap_noise * self.rng.standard_normal(n - 1))
        hi = np.maximum(o, c) * (1 + self.wick * np.abs(self.rng.standard_normal(n)))
        lo = np.minimum(o, c) * (1 - self.wick * np.abs(self.rng.standard_normal(n)))
        for i, ov in self.overrides.items():
            if "open" in ov:
                o[i] = ov["open"]
            hi[i] = max(ov.get("high", hi[i]), o[i], c[i])
            lo[i] = min(ov.get("low", lo[i]), o[i], c[i])
        dates = pd.bdate_range(start, periods=n)
        return pd.DataFrame({"trade_date": dates, "open": o, "high": hi, "low": lo, "close": c,
                             "volume": np.array(self.vol, float)})


def noise_chart(n: int = 600, seed: int = 0, sigma: float = 0.015, drift: float = 0.0005) -> pd.DataFrame:
    return Chart(seed).drift(n, drift, sigma).frame()
