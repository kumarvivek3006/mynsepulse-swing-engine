"""
universe.py — the ONE definition of "what the scan looks at".

Step 1.1 / 1.2 of the rebuild.

Before: run_scan selected `where is_active`. sync_universe wrote
in_nifty500 correctly but nothing read it, so stocks that left the index on
27 Sep stayed active and kept being scanned beside the ones that joined (527,
not 500), and every future rebalance would have grown the set again. Breadth
is computed over this list, so the error fed the regime read too.

Now: active AND in the current Nifty 500 AND not on the exclusion list.

WHY AN EXPLICIT EXCLUSION LIST, NOT AN industry LIKE '%REIT%' RULE
  The three REITs (BIRET, EMBASSY, BAGMANE) are excluded because they are
  yield instruments with a different volatility and volume structure, not
  breakout candidates. A pattern-match on industry text would be a guess at
  data this code has never seen; an explicit list is exact, reviewable, and
  extended by one env var. BAGMANE is on it now although it has too little
  history to be scanned yet, so it cannot slip in when it reaches 210 bars.
  universe_report() lists any OTHER name whose company_name/industry looks
  like a trust, so a future listing is surfaced for a human decision rather
  than excluded or admitted silently.

Rows are NOT deactivated. is_active stays true so price history keeps
updating (open positions, later re-inclusion); exclusion is a scan-time
decision and is reported in the scan summary every run.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_EXCLUDED = ("BIRET", "EMBASSY", "BAGMANE")


def excluded_symbols() -> list[str]:
    raw = os.environ.get("SCAN_EXCLUDE_SYMBOLS")
    if raw is None:
        return list(DEFAULT_EXCLUDED)
    return sorted({x.strip().upper() for x in raw.split(",") if x.strip()})


@dataclass
class ScanUniverse:
    symbols: list[str]
    flagged: set[str]                      # under surveillance in the last 5 days
    detail: dict = field(default_factory=dict)


# One statement for the universe itself, one for the breakdown. The breakdown
# exists so the scan summary can say WHY the number is what it is.
UNIVERSE_SQL = """
    select s.symbol,
           coalesce(bool_or(v.symbol is not null), false) as flagged
    from symbols s
    left join surveillance v
           on v.symbol = s.symbol
          and v.as_of >= current_date - 5
    where s.is_active
      and s.in_nifty500
      and coalesce(s.series, '') <> 'INDEX'
      and not (s.symbol = any(%s))
    group by s.symbol
    order by s.symbol
"""

BREAKDOWN_SQL = """
    select count(*) filter (where is_active)                                    as active,
           count(*) filter (where is_active and in_nifty500)                    as in_index,
           count(*) filter (where is_active and not coalesce(in_nifty500, false)) as not_in_index,
           count(*) filter (where is_active and in_nifty500 and symbol = any(%s)) as excluded_listed
    from symbols
    where coalesce(series, '') <> 'INDEX'
"""


def load_scan_universe(conn) -> ScanUniverse:
    excluded = excluded_symbols()
    with conn.cursor() as cur:
        cur.execute(UNIVERSE_SQL, (excluded,))
        rows = cur.fetchall()
    with conn.cursor() as cur:
        cur.execute(BREAKDOWN_SQL, (excluded,))
        active, in_index, not_in_index, excluded_listed = cur.fetchone()

    symbols = [r[0] for r in rows]
    detail = {
        "active": int(active),
        "in_nifty500": int(in_index),
        "dropped_not_in_index": int(not_in_index),
        "dropped_excluded_symbols": int(excluded_listed),
        "exclusion_list": excluded,
        "scanned": len(symbols),
    }
    log.info("Scan universe: %d scanned (%d active, %d not in index dropped, "
             "%d excluded: %s)", len(symbols), active, not_in_index,
             excluded_listed, ",".join(excluded))
    return ScanUniverse(symbols=symbols, flagged={r[0] for r in rows if r[1]},
                        detail=detail)


# Names that LOOK like trusts but are not on the exclusion list. Reported for a
# human decision; never acted on automatically.
TRUST_LOOKALIKE_SQL = """
    select symbol, company_name, industry, in_nifty500, is_active
    from symbols
    where coalesce(series, '') <> 'INDEX'
      and not (symbol = any(%s))
      and (company_name ilike '%%reit%%' or company_name ilike '%%real estate investment%%'
           or company_name ilike '%%infrastructure investment trust%%'
           or company_name ilike '%%invit%%' or industry ilike '%%reit%%')
    order by symbol
"""


def trust_lookalikes(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(TRUST_LOOKALIKE_SQL, (excluded_symbols(),))
        return [dict(zip(("symbol", "company_name", "industry", "in_nifty500", "is_active"), r))
                for r in cur.fetchall()]
