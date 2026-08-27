"""
NSE data ingestion — the things a broker API does not provide.

Upstox gives price and volume. It does not tell you which stocks are in the
Nifty 500, which are under surveillance, or when a stock splits. Those come
from NSE directly, and NSE is the source of record for all three.

Access strategy, in order of preference:

  1. nsearchives.nseindia.com — static CSV files. No session, no cookies,
     no rate limiting in practice. Used wherever a file exists.
  2. www.nseindia.com/api/* — JSON endpoints. Undocumented, require a
     primed cookie jar and browser-like headers, and will reject a bare
     client. Used only where no archive file exists.

The archive path is preferred everywhere possible because the JSON API is
the fragile part of this system and will break without notice. When it
does, the failure must be loud: a silently empty surveillance list would
let the scanner recommend a stock in the ASM framework.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime

import requests

log = logging.getLogger(__name__)

ARCHIVES = "https://nsearchives.nseindia.com"
NSE_BASE = "https://www.nseindia.com"

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{NSE_BASE}/market-data/live-equity-market",
}

INDEX_CSV = {
    "nifty500": f"{ARCHIVES}/content/indices/ind_nifty500list.csv",
    "nifty50": f"{ARCHIVES}/content/indices/ind_nifty50list.csv",
    "niftynext50": f"{ARCHIVES}/content/indices/ind_niftynext50list.csv",
    "nifty100": f"{ARCHIVES}/content/indices/ind_nifty100list.csv",
    "niftymidcap150": f"{ARCHIVES}/content/indices/ind_niftymidcap150list.csv",
    "niftysmallcap250": f"{ARCHIVES}/content/indices/ind_niftysmallcap250list.csv",
}


class NSEUnavailable(RuntimeError):
    """NSE returned nothing usable. Callers must not treat this as 'empty'."""


# ---------------------------------------------------------------------
@dataclass
class Constituent:
    symbol: str
    company_name: str
    industry: str | None
    isin: str | None


@dataclass
class CorporateAction:
    symbol: str
    ex_date: date
    action_type: str          # split | bonus | dividend | rights | demerger | other
    ratio_from: float | None
    ratio_to: float | None
    purpose: str

    @property
    def adjustment_factor(self) -> float | None:
        """
        Multiplier applied to prices BEFORE the ex-date to make them
        comparable with prices after it.

        A 1:5 split (ratio_from=1, ratio_to=5) means one old share became
        five, so historical prices must be divided by 5 -> factor 0.2.
        Dividends are not adjusted: for swing trading on daily bars the
        distortion is smaller than the error introduced by guessing at
        gross-vs-net treatment.
        """
        if self.action_type not in ("split", "bonus"):
            return None
        if not self.ratio_from or not self.ratio_to:
            return None
        if self.action_type == "split":
            return self.ratio_from / self.ratio_to
        return self.ratio_from / (self.ratio_from + self.ratio_to)


class NSEClient:

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        self._primed = False

    # -- session ------------------------------------------------------
    def _prime(self) -> None:
        """NSE's JSON API rejects requests without cookies from a page visit."""
        if self._primed:
            return
        try:
            self.session.get(NSE_BASE, timeout=self.timeout)
            self.session.get(f"{NSE_BASE}/market-data/live-equity-market", timeout=self.timeout)
            self._primed = True
            time.sleep(0.5)
        except requests.RequestException as exc:
            raise NSEUnavailable(f"Could not prime NSE session: {exc}") from exc

    def _get_json(self, path: str, params: dict | None = None) -> dict | list:
        self._prime()
        for attempt in range(3):
            resp = self.session.get(f"{NSE_BASE}{path}", params=params, timeout=self.timeout)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    log.warning("NSE returned non-JSON for %s", path)
            elif resp.status_code in (401, 403):
                # Cookies went stale mid-run; re-prime once and retry.
                self._primed = False
                self._prime()
            time.sleep(2 ** attempt)
        raise NSEUnavailable(f"NSE JSON endpoint failed after retries: {path}")

    def _get_csv(self, url: str) -> list[dict]:
        resp = self.session.get(url, timeout=self.timeout)
        if resp.status_code != 200:
            raise NSEUnavailable(f"NSE archive returned {resp.status_code} for {url}")
        text = resp.content.decode("utf-8-sig", errors="replace")
        return list(csv.DictReader(io.StringIO(text)))

    # -- universe -----------------------------------------------------
    def index_constituents(self, index: str = "nifty500") -> list[Constituent]:
        url = INDEX_CSV.get(index.lower())
        if not url:
            raise ValueError(f"Unknown index: {index}")

        rows = self._get_csv(url)
        out = [
            Constituent(
                symbol=(r.get("Symbol") or "").strip().upper(),
                company_name=(r.get("Company Name") or "").strip(),
                industry=(r.get("Industry") or "").strip() or None,
                isin=(r.get("ISIN Code") or "").strip() or None,
            )
            for r in rows
            if (r.get("Symbol") or "").strip()
        ]

        # A truncated constituent file would quietly shrink the universe,
        # so sanity-check the count rather than trusting whatever arrived.
        expected = {"nifty500": 400, "nifty50": 45, "niftynext50": 45,
                    "nifty100": 90, "niftymidcap150": 130, "niftysmallcap250": 200}
        floor = expected.get(index.lower(), 1)
        if len(out) < floor:
            raise NSEUnavailable(
                f"{index} returned only {len(out)} constituents (expected >= {floor}). "
                "Refusing to shrink the universe on a partial file."
            )
        log.info("%s: %d constituents", index, len(out))
        return out

    # -- corporate actions --------------------------------------------
    RATIO_PATTERNS = [
        (re.compile(r"split.*?(\d+(?:\.\d+)?)\s*[/:-]\s*(\d+(?:\.\d+)?)", re.I), "split"),
        (re.compile(r"bonus.*?(\d+(?:\.\d+)?)\s*[/:-]\s*(\d+(?:\.\d+)?)", re.I), "bonus"),
        (re.compile(r"rights.*?(\d+(?:\.\d+)?)\s*[/:-]\s*(\d+(?:\.\d+)?)", re.I), "rights"),
    ]

    @classmethod
    def _parse_purpose(cls, purpose: str) -> tuple[str, float | None, float | None]:
        """
        NSE encodes corporate actions as free text, e.g.
        'Face Value Split From Rs.10/- To Re.1/-' or 'Bonus 1:2'.
        Ratios are extracted where recognisable; unrecognised strings are
        typed 'other' with no ratio so the adjustment step skips them
        rather than applying a wrong factor.
        """
        text = purpose or ""
        for pattern, kind in cls.RATIO_PATTERNS:
            m = pattern.search(text)
            if m:
                return kind, float(m.group(1)), float(m.group(2))

        low = text.lower()
        if "split" in low:
            # 'From Rs.10/- To Re.1/-' style
            nums = re.findall(r"(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)", low)
            if len(nums) >= 2:
                return "split", float(nums[1]), float(nums[0])
            return "split", None, None
        if "bonus" in low:
            return "bonus", None, None
        if "dividend" in low:
            return "dividend", None, None
        if "demerger" in low:
            return "demerger", None, None
        return "other", None, None

    def corporate_actions(self, from_date: date, to_date: date) -> list[CorporateAction]:
        raw = self._get_json("/api/corporates-corporateActions", {
            "index": "equities",
            "from_date": from_date.strftime("%d-%m-%Y"),
            "to_date": to_date.strftime("%d-%m-%Y"),
        })
        rows = raw if isinstance(raw, list) else raw.get("data", [])

        out: list[CorporateAction] = []
        for r in rows:
            ex_raw = r.get("exDate") or r.get("ex_date")
            symbol = (r.get("symbol") or "").strip().upper()
            if not ex_raw or not symbol:
                continue
            try:
                ex = datetime.strptime(ex_raw, "%d-%b-%Y").date()
            except ValueError:
                log.debug("Unparseable ex-date %r for %s", ex_raw, symbol)
                continue

            purpose = r.get("subject") or r.get("purpose") or ""
            kind, a, b = self._parse_purpose(purpose)
            out.append(CorporateAction(symbol, ex, kind, a, b, purpose))

        log.info("Corporate actions %s→%s: %d rows", from_date, to_date, len(out))
        return out

    # -- surveillance -------------------------------------------------
    def surveillance(self) -> dict[str, list[dict]]:
        """
        ASM and GSM lists. A stock in either is excluded by Gate 0.

        Raises rather than returning empty on failure. An empty
        surveillance list is indistinguishable from 'nothing is flagged',
        and that difference decides whether a restricted stock reaches
        the recommendation list.
        """
        result: dict[str, list[dict]] = {}
        for name, path in (("asm", "/api/reportASM"), ("gsm", "/api/reportGSM")):
            payload = self._get_json(path)
            rows: list[dict] = []
            if isinstance(payload, dict):
                for bucket in ("longterm", "shortterm", "data"):
                    section = payload.get(bucket)
                    if isinstance(section, dict):
                        rows.extend(section.get("data", []))
                    elif isinstance(section, list):
                        rows.extend(section)
            elif isinstance(payload, list):
                rows = payload
            result[name] = rows
            log.info("%s list: %d symbols", name.upper(), len(rows))
        return result

    def surveillance_symbols(self) -> set[str]:
        flat: set[str] = set()
        for rows in self.surveillance().values():
            for r in rows:
                sym = (r.get("symbol") or r.get("Symbol") or "").strip().upper()
                if sym:
                    flat.add(sym)
        return flat
