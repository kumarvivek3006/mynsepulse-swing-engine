"""
e2_features.py — the ONE feature function used by the live shadow scan and by
the backtest. Same code, same inputs, same values (that is what makes the
validation mean something). Nothing here looks past bar i.

Input: a DataFrame with trade_date, open, high, low, close, volume (ADJUSTED).
Output: Bars, a thin holder of numpy arrays.

Feature groups (spec): trend (EMA 8/20, SMA 20/50/150/200), momentum (RSI, ROC),
volatility (ATR14, ATR%, 20-bar ATR contraction, Bollinger squeeze, historical
vol percentile), volume (relative volume, dry-up flag, OBV trend), structure
(swing pivots, 52-week extremes), gap and close position.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from e2_core import ema, rolling_max, rolling_mean, rolling_min, shift, swing_points

MIN_BARS = 210            # SMA200 needs 200; same floor the live engine uses


class Bars:
    """Arrays for one symbol. Attributes are numpy arrays of length n."""

    def __init__(self, symbol: str, df: pd.DataFrame):
        self.symbol = symbol
        self.n = len(df)
        d = df["trade_date"].values
        self.date = np.array(d, dtype="datetime64[D]")
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.v = df["volume"].to_numpy(float)
        self._sw: dict = {}
        self._compute()

    # -- swing pivots, cached by (kind, k) --------------------------------
    def swings(self, kind: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        key = (kind, k)
        if key not in self._sw:
            arr = self.l if kind == "low" else self.h
            self._sw[key] = swing_points(arr, k, kind)
        return self._sw[key]

    def pivots_before(self, kind: str, k: int, i: int, lo: int = 0) -> np.ndarray:
        """indices of confirmed swing points usable at bar i (confirm <= i) and >= lo."""
        j, conf = self.swings(kind, k)
        m = (conf <= i) & (j >= lo)
        return j[m]

    def _compute(self) -> None:
        o, h, l, c, v = self.o, self.h, self.l, self.c, self.v
        n = self.n
        s = pd.Series
        cs, vs = s(c), s(v)

        self.c_prev = shift(c, 1)
        self.gap_pct = np.where(self.c_prev > 0, (o / self.c_prev - 1.0) * 100.0, np.nan)
        rng = h - l
        self.rng = rng
        self.close_pos = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 0.5)

        # true range / ATR (Wilder)
        tr = np.maximum.reduce([h - l, np.abs(h - self.c_prev), np.abs(l - self.c_prev)])
        tr[0] = h[0] - l[0]
        self.tr = tr
        self.atr14 = s(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
        self.atr_pct = self.atr14 / c * 100.0
        self.atr_prev = shift(self.atr14, 1)
        self.atr_contr20 = self.atr14 / shift(self.atr14, 20)          # < 1 = contracting
        self.range_atr = rng / self.atr_prev

        # trend
        self.ema8 = ema(c, 8)
        self.ema20 = ema(c, 20)
        self.sma20 = rolling_mean(c, 20)
        self.sma50 = rolling_mean(c, 50)
        self.sma150 = rolling_mean(c, 150)
        self.sma200 = rolling_mean(c, 200)
        self.sma50_up = self.sma50 > shift(self.sma50, 10)
        self.sma200_up = self.sma200 > shift(self.sma200, 20)

        # volume (averages are of the PRIOR bars so today's surge is not in its own baseline)
        self.vavg50 = shift(rolling_mean(v, 50), 1)
        self.vavg20 = shift(rolling_mean(v, 20), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.relvol = v / self.vavg50
            self.relvol20 = v / self.vavg20
            self.vol_dry5 = (rolling_mean(v, 5) / self.vavg50) <= 0.75
        step = np.where(c > self.c_prev, v, np.where(c < self.c_prev, -v, 0.0))
        step[0] = 0.0
        self.obv = np.cumsum(step)
        self.obv_sma20 = rolling_mean(self.obv, 20)
        self.obv_trend = np.sign(self.obv - self.obv_sma20)            # +1 accumulating, -1 distributing

        # Bollinger squeeze: 20-bar band width in the lowest quintile of its last 126 bars
        sd20 = cs.rolling(20).std(ddof=0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            self.bb_width = 4.0 * sd20 / self.sma20
        bw = s(self.bb_width)
        q20 = bw.rolling(126, min_periods=60).quantile(0.2).to_numpy()
        self.bb_squeeze = self.bb_width <= q20

        # historical volatility percentile (20-day realised vol vs its own last 252 days)
        lr = np.log(c / np.where(self.c_prev > 0, self.c_prev, np.nan))
        hv = s(lr).rolling(20).std(ddof=0).to_numpy() * np.sqrt(252)
        self.hv20 = hv
        self.hv_pct = s(hv).rolling(252, min_periods=100).rank(pct=True).to_numpy() * 100.0

        # extremes and returns
        self.high52 = s(h).rolling(252, min_periods=100).max().to_numpy()
        self.low52 = s(l).rolling(252, min_periods=100).min().to_numpy()
        for k in (21, 63, 126, 252):
            with np.errstate(divide="ignore", invalid="ignore"):
                setattr(self, f"ret{k}", c / shift(c, k) - 1.0)

        # RSI14 (Wilder)
        d = np.diff(c, prepend=c[0])
        up = s(np.where(d > 0, d, 0.0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
        dn = s(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = up / dn
            self.rsi14 = 100 - 100 / (1 + rs)
        self.rsi14 = np.where(dn == 0, 100.0, self.rsi14)

        # liquidity (rupee turnover, crores; median of 20 bars) for the hard gate
        turn = c * v / 1e7
        self.turnover20_cr = s(turn).rolling(20).median().to_numpy()

    # convenient trend template (Minervini-style) used by the scorer
    def trend_ok(self, i: int) -> bool:
        return bool(i >= 200 and self.c[i] > self.sma50[i] > self.sma150[i] > self.sma200[i]
                    and self.sma200_up[i])


def compute_features(symbol: str, df: pd.DataFrame) -> Bars | None:
    """None when there is too little history — reported by the caller, never silent."""
    if df is None or len(df) < MIN_BARS:
        return None
    return Bars(symbol, df.reset_index(drop=True))
