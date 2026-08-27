"""
Upstox API v3 data client for the mynsepulse swing engine.

Deliberately narrow: this reads market data only. No order placement, no
portfolio access. The swing engine recommends; it does not trade.

Three Upstox-specific facts shape this module:

  1. Access tokens expire at 03:30 IST the following day regardless of when
     they were issued, and the standard OAuth flow returns no refresh token.
     So there is no "refresh" path — only a fresh authorisation each day.
     `TokenStore` therefore treats expiry as a hard wall and raises rather
     than silently proceeding with a dead token.

  2. Instruments are identified by ISIN-backed keys (`NSE_EQ|INE002A01018`),
     not tickers. This survives NSE symbol renames across a 3-year history,
     which ticker-keyed storage does not. We resolve tickers to instrument
     keys once, from the instrument master, and store the key.

  3. Published rate limits change and are not the same across endpoint
     families, so every limit here is configuration, not a constant. Set
     them from the current docs at deploy time rather than trusting values
     baked into code.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

API_BASE = os.environ.get("UPSTOX_API_BASE", "https://api.upstox.com")
INSTRUMENT_MASTER_URL = os.environ.get(
    "UPSTOX_INSTRUMENT_URL",
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
)

# Upstox publishes these and has revised them before. Treat as configuration.
REQ_PER_SEC = int(os.environ.get("UPSTOX_REQ_PER_SEC", "20"))
REQ_PER_MIN = int(os.environ.get("UPSTOX_REQ_PER_MIN", "250"))
REQ_PER_DAY = int(os.environ.get("UPSTOX_REQ_PER_DAY", "20000"))


class TokenExpired(RuntimeError):
    """No usable Upstox token. Callers must abort, not improvise."""


# ---------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------
class RateLimiter:
    """Sliding-window limiter across second, minute and day buckets."""

    def __init__(self, per_sec=REQ_PER_SEC, per_min=REQ_PER_MIN, per_day=REQ_PER_DAY):
        self.per_sec, self.per_min, self.per_day = per_sec, per_min, per_day
        self._calls: deque[float] = deque()
        self._day_count = 0
        self._day_stamp = datetime.now(IST).date()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                today = datetime.now(IST).date()
                if today != self._day_stamp:
                    self._day_stamp, self._day_count = today, 0

                if self._day_count >= self.per_day:
                    raise RuntimeError("Upstox daily request quota exhausted")

                while self._calls and now - self._calls[0] > 60:
                    self._calls.popleft()

                in_last_sec = sum(1 for t in self._calls if now - t < 1.0)
                if in_last_sec < self.per_sec and len(self._calls) < self.per_min:
                    self._calls.append(now)
                    self._day_count += 1
                    return

                wait = (1.05 - (now - self._calls[-1])) if in_last_sec >= self.per_sec \
                    else (60.5 - (now - self._calls[0]))
            time.sleep(max(wait, 0.05))

    @property
    def used_today(self) -> int:
        return self._day_count


# ---------------------------------------------------------------------
# Token handling
# ---------------------------------------------------------------------
def next_expiry_ist(now: datetime | None = None) -> datetime:
    """Upstox tokens die at 03:30 IST the following morning, always."""
    now = now or datetime.now(IST)
    wall = now.replace(hour=3, minute=30, second=0, microsecond=0)
    if now >= wall:
        wall += timedelta(days=1)
    return wall


@dataclass
class UpstoxCredentials:
    api_key: str
    api_secret: str
    redirect_uri: str


class TokenStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("TOKEN_STORE_PATH", "/data/upstox_token.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            log.warning("Token store corrupt; treating as empty")
            return {}

    def write(self, access_token: str, issued_at: datetime | None = None) -> None:
        issued = issued_at or datetime.now(IST)
        self.path.write_text(json.dumps({
            "access_token": access_token,
            "issued_at": issued.isoformat(),
            "expires_at": next_expiry_ist(issued).isoformat(),
        }))

    def valid_token(self, margin=timedelta(minutes=10)) -> str | None:
        row = self.read()
        token, expires = row.get("access_token"), row.get("expires_at")
        if not token or not expires:
            return None
        if datetime.fromisoformat(expires) - margin <= datetime.now(IST):
            return None
        return token


def exchange_auth_code(creds: UpstoxCredentials, code: str) -> str:
    """
    Documented OAuth exchange. `code` comes from the redirect after the user
    approves at the authorization dialog. Single use — it is consumed whether
    or not the exchange succeeds.
    """
    resp = requests.post(
        f"{API_BASE}/v2/login/authorization/token",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"},
        data={
            "code": code,
            "client_id": creds.api_key,
            "client_secret": creds.api_secret,
            "redirect_uri": creds.redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    payload = resp.json()
    if "access_token" not in payload:
        raise RuntimeError(f"Upstox token exchange failed: {payload}")
    return payload["access_token"]


def authorization_url(creds: UpstoxCredentials, state: str = "mynsepulse") -> str:
    return (
        f"{API_BASE}/v2/login/authorization/dialog"
        f"?client_id={creds.api_key}&redirect_uri={creds.redirect_uri}"
        f"&response_type=code&state={state}"
    )


# ---------------------------------------------------------------------
# Instrument master
# ---------------------------------------------------------------------
@dataclass
class Instrument:
    instrument_key: str
    trading_symbol: str
    name: str
    isin: str | None
    exchange: str
    segment: str
    instrument_type: str
    tick_size: float | None = None
    lot_size: int | None = None
    raw: dict = field(default_factory=dict)


class InstrumentMaster:
    """
    Downloads and caches the NSE instrument master.

    Cached to disk because the file is several MB and only changes once a
    day. Field names vary between the JSON and CSV builds of this file, so
    lookups are defensive rather than assuming one shape.
    """

    def __init__(self, cache_path: str | Path = "/data/upstox_instruments.json",
                 max_age: timedelta = timedelta(hours=20)):
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        self._by_symbol: dict[str, Instrument] = {}
        self._by_key: dict[str, Instrument] = {}

    def _download(self) -> list[dict]:
        # A plain requests default UA has been reported to get 403 here.
        resp = requests.get(
            INSTRUMENT_MASTER_URL,
            headers={"User-Agent": "mynsepulse/0.1", "Accept": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.content
        if INSTRUMENT_MASTER_URL.endswith(".gz"):
            body = gzip.decompress(body)
        return json.loads(body.decode("utf-8"))

    def _cache_fresh(self) -> bool:
        if not self.cache_path.exists():
            return False
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(
            self.cache_path.stat().st_mtime, tz=timezone.utc)
        return age < self.max_age

    def load(self, force: bool = False) -> None:
        if not force and self._cache_fresh():
            rows = json.loads(self.cache_path.read_text())
        else:
            rows = self._download()
            self.cache_path.write_text(json.dumps(rows))
            log.info("Instrument master refreshed: %d rows", len(rows))

        self._by_symbol.clear()
        self._by_key.clear()

        for row in rows:
            key = row.get("instrument_key") or row.get("instrumentKey")
            if not key:
                continue
            symbol = (row.get("trading_symbol") or row.get("tradingsymbol")
                      or row.get("tradingSymbol") or row.get("name") or "")
            inst = Instrument(
                instrument_key=key,
                trading_symbol=symbol.upper(),
                name=row.get("name", ""),
                isin=row.get("isin"),
                exchange=row.get("exchange", ""),
                segment=row.get("segment", ""),
                instrument_type=row.get("instrument_type") or row.get("instrumentType", ""),
                tick_size=row.get("tick_size"),
                lot_size=row.get("lot_size"),
                raw=row,
            )
            self._by_key[key] = inst
            if inst.trading_symbol:
                self._by_symbol.setdefault(inst.trading_symbol, inst)

    def equities(self) -> list[Instrument]:
        """Cash-segment NSE equities only — the swing universe."""
        return [
            i for i in self._by_key.values()
            if i.segment == "NSE_EQ" and i.instrument_type == "EQ"
        ]

    def resolve(self, ticker: str) -> Instrument | None:
        if not self._by_symbol:
            self.load()
        return self._by_symbol.get(ticker.upper())

    def resolve_many(self, tickers: list[str]) -> tuple[dict[str, str], list[str]]:
        """Returns (ticker -> instrument_key, unresolved tickers).

        Unresolved tickers are returned rather than skipped silently: a
        symbol missing from the universe is a data-quality event worth
        logging, not a shrug.
        """
        resolved, missing = {}, []
        for t in tickers:
            inst = self.resolve(t)
            if inst:
                resolved[t.upper()] = inst.instrument_key
            else:
                missing.append(t.upper())
        return resolved, missing


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------
class UpstoxClient:

    def __init__(self, creds: UpstoxCredentials | None = None,
                 store: TokenStore | None = None):
        self.creds = creds
        self.store = store or TokenStore()
        self.limiter = RateLimiter()
        self.session = requests.Session()

    def token(self) -> str:
        tok = self.store.valid_token()
        if not tok:
            raise TokenExpired(
                "No valid Upstox access token. Tokens expire at 03:30 IST daily "
                "and cannot be refreshed — a new authorisation is required. "
                "Scan aborted rather than run on stale data."
            )
        return tok

    def _get(self, path: str, params: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.token()}", "Accept": "application/json"}

        for attempt in range(4):
            self.limiter.acquire()
            resp = self.session.get(f"{API_BASE}{path}", params=params,
                                    headers=headers, timeout=45)

            if resp.status_code == 429:
                backoff = 2 ** attempt
                log.warning("Upstox rate limited on %s; sleeping %ss", path, backoff)
                time.sleep(backoff)
                continue

            if resp.status_code in (401, 403):
                raise TokenExpired(f"Upstox rejected the token on {path} ({resp.status_code})")

            try:
                return resp.json()
            except json.JSONDecodeError:
                log.error("Non-JSON response from %s: %s", path, resp.text[:200])
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Upstox request failed after retries: {path}")

    # -- history ------------------------------------------------------
    def historical_daily(self, instrument_key: str, start: date, end: date,
                         chunk_years: int = 1) -> list[list]:
        """
        Daily candles, oldest first: [timestamp, o, h, l, c, v, oi].

        Chunked by year even though v3 accepts wide ranges — bounded
        responses fail more predictably and make partial backfills
        resumable. Boundary duplicates are removed on stitch.
        """
        out: list[list] = []
        cursor = start

        while cursor <= end:
            chunk_end = min(date(cursor.year + chunk_years, cursor.month, cursor.day)
                            - timedelta(days=1), end)
            payload = self._get(
                f"/v3/historical-candle/{instrument_key}/days/1/"
                f"{chunk_end.isoformat()}/{cursor.isoformat()}"
            )
            if payload.get("status") == "success":
                out.extend(payload.get("data", {}).get("candles", []))
            else:
                log.error("history failed %s %s→%s: %s",
                          instrument_key, cursor, chunk_end, payload)
            cursor = chunk_end + timedelta(days=1)

        seen, deduped = set(), []
        for candle in sorted(out, key=lambda c: c[0]):
            if candle[0] not in seen:
                seen.add(candle[0])
                deduped.append(candle)
        return deduped

    def intraday_today(self, instrument_key: str) -> list[list]:
        payload = self._get(f"/v3/historical-candle/intraday/{instrument_key}/days/1")
        return payload.get("data", {}).get("candles", [])

    def quotes(self, instrument_keys: list[str]) -> dict:
        result: dict = {}
        for i in range(0, len(instrument_keys), 100):
            batch = instrument_keys[i:i + 100]
            payload = self._get("/v2/market-quote/quotes",
                                {"instrument_key": ",".join(batch)})
            result.update(payload.get("data", {}))
        return result
