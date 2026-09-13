"""
Fundamentals ingestion — Gate 2 inputs.

Scope is deliberately narrow, and the reason matters:

  * Promoter holding comes from NSE's shareholding-pattern endpoint. That
    endpoint returns promoter, public and employee-trust holdings — and
    NOTHING about pledged shares. Pledge is Column XIV of the detailed SHP
    filing document, which needs per-company document parsing. So the
    pledge vetoes in the spec are NOT satisfied here and must not be
    reported as if they were.

  * Quarterly revenue / net profit / EPS come from the results-comparison
    endpoint (NSE spells the path "results-comparision" — their typo, not
    a mistake here). Amounts arrive in Rupees LAKHS and are converted to
    Crores on write, because every threshold elsewhere is in Crores and a
    silent 100x unit error is exactly the kind of thing that would make a
    veto fire on the wrong companies.

The NSE endpoints are the fragile part of this system, so this uses the
maintained `nse` library rather than hand-rolled URLs — it tracks NSE's
cookie handling and path changes, which hand-rolled requests do not.

Run weekly, not daily: this data changes quarterly.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ingest import connect

log = logging.getLogger(__name__)

# Railway is a server environment; the library switches transport accordingly.
NSE_SERVER_MODE = os.environ.get("NSE_SERVER_MODE", "true").lower() == "true"
NSE_THROTTLE_SEC = float(os.environ.get("NSE_THROTTLE_SEC", "0.4"))
LAKHS_TO_CRORES = 100.0


# ---------------------------------------------------------------------
# Upstox fundamentals — parsers for the VERIFIED income-statement schema
# ---------------------------------------------------------------------
UPSTOX_PERIOD_FMTS = ("%b %Y", "%B %Y")


def parse_upstox_period(label: str) -> date | None:
    """
    "Mar 2025" -> 2025-03-31. Upstox labels a period by its END month, so
    the stored date is that month's last day — matching how the NSE path
    stored period_end, so both sources remain comparable.
    """
    if not label:
        return None
    for fmt in UPSTOX_PERIOD_FMTS:
        try:
            d = datetime.strptime(str(label).strip(), fmt).date()
            nxt = date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
            return nxt - timedelta(days=1)
        except ValueError:
            continue
    return None


def parse_income_statement(payload: dict) -> list[dict]:
    """
    Upstox income-statement -> one row per period.

    Schema confirmed from the API reference:
      data.income_statement[] = {category, history[{value, period, change}]}
      data.full_statement[]   = {particular, history[{period, value}]}

    Values are in CRORE (data.units_in). The NSE path returned LAKHS and
    divided by 100 — reusing that conversion here would be wrong by 100x,
    which is exactly the kind of silent unit error that makes a veto fire
    on the wrong companies.
    """
    data = (payload or {}).get("data") or {}
    by_period: dict[date, dict] = {}

    def touch(period_label):
        d = parse_upstox_period(period_label)
        if d is None:
            return None
        return by_period.setdefault(d, {"period_end": d})

    for block in data.get("income_statement") or []:
        category = (block.get("category") or "").lower()
        field = {"revenue": "revenue",
                 "operating_profit": "operating_profit",
                 "net_profit": "pat"}.get(category)
        if not field:
            continue
        for point in block.get("history") or []:
            row = touch(point.get("period"))
            if row is not None and point.get("value") is not None:
                row[field] = float(point["value"])

    # full_statement carries EPS and PBT, which the summary categories omit.
    wanted = {"eps - basic": "eps",
              "profit before tax": "pbt",
              "total revenue": "total_revenue"}
    for block in data.get("full_statement") or []:
        field = wanted.get((block.get("particular") or "").strip().lower())
        if not field:
            continue
        for point in block.get("history") or []:
            row = touch(point.get("period"))
            if row is not None and point.get("value") is not None:
                row[field] = float(point["value"])

    rows = []
    for d in sorted(by_period, reverse=True):
        row = by_period[d]
        rev, op = row.get("revenue"), row.get("operating_profit")
        # Operating margin computed from operating_profit directly — the
        # NSE path had to reconstruct it as PBT + interest - other income
        # because no operating line existed. Here it is reported.
        row["opm_pct"] = round(op / rev * 100, 2) if rev and op is not None and rev > 0 else None
        rows.append(row)
    return rows


def parse_share_holdings(payload: dict) -> list[dict]:
    """
    Promoter / FII / DII / public by quarter.

    Schema NOT yet verified against the reference — the published example
    was not retrievable. Parsed defensively across the plausible shapes and
    returns [] rather than guessing when none match, so a shape change
    surfaces as "no data" rather than as a table of nulls that reads like
    real data. Confirm with /jobs/fundamentals/upstox-probe first.
    """
    data = (payload or {}).get("data") or {}
    holdings = (data.get("share_holdings") or data.get("shareholding")
                or data.get("holdings") or [])
    if not isinstance(holdings, list):
        return []

    label_map = {
        "promoters": "promoter_pct", "promoter": "promoter_pct",
        "promoter and promoter group": "promoter_pct",
        "fii": "fii_pct", "fiis": "fii_pct", "foreign institutions": "fii_pct",
        "dii": "dii_pct", "diis": "dii_pct", "domestic institutions": "dii_pct",
        "public": "public_pct",
    }

    by_period: dict[date, dict] = {}
    for block in holdings:
        if not isinstance(block, dict):
            continue
        raw = (block.get("category") or block.get("holder_type")
               or block.get("particular") or block.get("type") or "")
        field = label_map.get(str(raw).strip().lower())
        if not field:
            continue
        for point in block.get("history") or []:
            d = parse_upstox_period(point.get("period"))
            if d is None or point.get("value") is None:
                continue
            by_period.setdefault(d, {"period_end": d})[field] = float(point["value"])

    return [by_period[d] for d in sorted(by_period, reverse=True)]


def _ratio_value(raw) -> float | None:
    """
    Key-ratio values arrive as STRINGS, and percentages carry a '%' suffix:
    "8.94%", "20.15". A bare float() throws on the former, which in the
    earlier draft was swallowed by a try/except — so every percentage ratio
    (ROA, ROE, ROCE) would have silently come back missing while P/E and
    P/B parsed fine. Exactly the kind of partial success that reads as
    working.
    """
    if raw is None:
        return None
    try:
        return float(str(raw).strip().rstrip("%").replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_key_ratios(payload: dict) -> dict:
    """
    ROE / ROCE / P/E / P/B / ROA / EV-EBITDA plus sector benchmarks.

    Schema VERIFIED against the API reference:
      data = [ {name, company_value, sector_value}, ... ]

    Note `data` is a BARE ARRAY, not an object with a "key_ratios" key —
    the earlier draft looked for data["key_ratios"] and would have found
    nothing on every call. ROE here supplies CANSLIM's return requirement,
    and sector_value gives relative context the NSE path never had.

    Debt/Equity is NOT in this response despite being a CANSLIM input; it
    has to come from the balance sheet.
    """
    data = (payload or {}).get("data")
    if not isinstance(data, list):
        return {}

    wanted = {"p/e": "pe", "p/b": "pb", "roa": "roa",
              "roe": "roe", "roce": "roce", "ev/ebitda": "ev_ebitda"}

    out: dict = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        field = wanted.get(str(item.get("name") or "").strip().lower())
        if not field:
            continue
        v = _ratio_value(item.get("company_value"))
        if v is not None:
            out[field] = v
        sv = _ratio_value(item.get("sector_value"))
        if sv is not None:
            out[f"{field}_sector"] = sv
    return out


class FundamentalsUnavailable(RuntimeError):
    """NSE returned nothing usable. Never treat as 'no data to report'."""


def _nse():
    from nse import NSE
    return NSE(download_folder="/data/nse", server=NSE_SERVER_MODE)


def _num(value) -> float | None:
    if value in (None, "", "-", "NA"):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _parse_date(value) -> date | None:
    if not value:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------
# Probe — run this before any bulk ingestion
# ---------------------------------------------------------------------
def probe(symbol: str = "RELIANCE") -> dict:
    """
    Fetch one symbol and report the actual response shape.

    Field names on these endpoints are undocumented and have changed
    before. Running 500 symbols against guessed key names would write
    500 rows of nulls that look like real data. This shows what the keys
    actually are first.
    """
    logging.basicConfig(level="INFO")
    out: dict = {"symbol": symbol}
    with _nse() as nse:
        try:
            sh = nse.shareholding(symbol)
            out["shareholding_rows"] = len(sh)
            out["shareholding_keys"] = sorted(sh[0].keys()) if sh else []
            out["shareholding_sample"] = sh[0] if sh else None
            dates = sorted(str(_parse_date(_first(r, DATE_KEYS))) for r in sh
                           if _parse_date(_first(r, DATE_KEYS)))
            out["shareholding_periods"] = dates
            out["shareholding_latest"] = dates[-1] if dates else None
            out["shareholding_xbrl_urls"] = sum(1 for r in sh if r.get("xbrl"))
        except Exception as exc:
            out["shareholding_error"] = str(exc)[:300]

        time.sleep(NSE_THROTTLE_SEC)

        try:
            rc = nse.results_comparison(symbol)
            rows = rc.get("resCmpData", []) if isinstance(rc, dict) else []
            out["results_rows"] = len(rows)
            out["results_keys"] = sorted(rows[0].keys()) if rows else []
            out["results_sample"] = rows[0] if rows else None
            periods = sorted(str(_parse_date(_first(r, TO_DT_KEYS))) for r in rows
                             if _parse_date(_first(r, TO_DT_KEYS)))
            out["results_periods"] = periods
            out["results_latest"] = periods[-1] if periods else None
        except Exception as exc:
            out["results_error"] = str(exc)[:300]

    return out


# ---------------------------------------------------------------------
# Shareholding
# ---------------------------------------------------------------------
PROMOTER_KEYS = ("pr_and_prgrp", "promoterAndPromoterGroup", "promoter")
PUBLIC_KEYS = ("public_val", "public", "publicShareholding")
TRUST_KEYS = ("employeeTrusts", "employee_trusts", "sharesHeldByEmployeeTrusts")
DATE_KEYS = ("date", "asOnDate", "as_on_date", "recordDate")


def _first(row: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def sync_shareholding(conn, symbols: list[str] | None = None) -> dict:
    """
    Promoter holding by quarter.

    Writes promoter_pct only. promoter_pledge_pct is left NULL because
    this source does not carry it — Gate 2's pledge vetoes stay disabled
    rather than silently evaluating against nulls.
    """
    with conn.cursor() as cur:
        if symbols is None:
            cur.execute("select symbol from symbols where is_active "
                        "and coalesce(series,'') <> 'INDEX' order by symbol")
            symbols = [r[0] for r in cur.fetchall()]

    with conn.cursor() as cur:
        cur.execute("""
            select count(*) from information_schema.columns
            where table_schema = 'swing' and table_name = 'shareholding'
              and column_name = 'xbrl_url'
        """)
        _has_xbrl_column = cur.fetchone()[0] > 0
    if not _has_xbrl_column:
        log.warning("shareholding.xbrl_url missing (migration 002 not applied); "
                    "promoter data will ingest, filing URLs will not")

    written = failed = empty = 0
    unknown_shape: list[str] = []

    with _nse() as nse:
        for sym in symbols:
            try:
                rows = nse.shareholding(sym)
            except Exception as exc:
                failed += 1
                log.debug("shareholding failed for %s: %s", sym, exc)
                time.sleep(NSE_THROTTLE_SEC)
                continue

            if not rows:
                empty += 1
                time.sleep(NSE_THROTTLE_SEC)
                continue

            for row in rows:
                period = _parse_date(_first(row, DATE_KEYS))
                promoter = _num(_first(row, PROMOTER_KEYS))
                if period is None or promoter is None:
                    if len(unknown_shape) < 5:
                        unknown_shape.append(f"{sym}:{sorted(row.keys())[:8]}")
                    continue

                # The SHP filing URL is the route to pledge data (Column XIV),
                # which this summary endpoint does not carry. Stored now so a
                # later parser has it without re-crawling.
                with conn.cursor() as cur:
                    if _has_xbrl_column:
                        cur.execute("""
                            insert into shareholding
                                (symbol, period_end, promoter_pct, fii_pct, dii_pct, xbrl_url)
                            values (%s, %s, %s, null, null, %s)
                            on conflict (symbol, period_end) do update set
                                promoter_pct = excluded.promoter_pct,
                                xbrl_url = coalesce(excluded.xbrl_url,
                                                    shareholding.xbrl_url)
                        """, (sym, period, promoter, row.get("xbrl")))
                    else:
                        # Migration 002 not applied. Promoter data is the part
                        # Gate 2 actually reads, so ingest it rather than
                        # failing the whole run over a column that only a
                        # future pledge parser needs.
                        cur.execute("""
                            insert into shareholding
                                (symbol, period_end, promoter_pct, fii_pct, dii_pct)
                            values (%s, %s, %s, null, null)
                            on conflict (symbol, period_end) do update set
                                promoter_pct = excluded.promoter_pct
                        """, (sym, period, promoter))
                    written += cur.rowcount
            conn.commit()
            time.sleep(NSE_THROTTLE_SEC)

    if written == 0:
        raise FundamentalsUnavailable(
            f"No shareholding rows written across {len(symbols)} symbols "
            f"({failed} errors, {empty} empty). Shape seen: {unknown_shape}"
        )

    log.info("Shareholding: %d rows, %d failed, %d empty", written, failed, empty)
    return {"written": written, "failed": failed, "empty": empty,
            "unknown_shape": unknown_shape}


# ---------------------------------------------------------------------
# Quarterly P&L
# ---------------------------------------------------------------------
# re_net_sale is revenue from operations. re_total_inc adds other income,
# which is lumpy — a one-off asset sale would show as revenue growth and
# quietly clear the declining-revenue veto. Operations first, always.
# re_int_earned is the bank/NBFC equivalent, which has no re_net_sale.
REV_KEYS = ("re_net_sale", "re_int_earned", "re_total_inc")
PAT_KEYS = ("re_net_profit", "re_proloss_ord_act", "re_con_pro_loss")
# re_basic_eps comes back null on live filings; the populated field is the
# continuing-operations one.
EPS_KEYS = ("re_basic_eps_for_cont_dic_opr", "re_basic_eps", "re_bsc_eps_bfr_exi")
TO_DT_KEYS = ("re_to_dt",)
PBT_KEYS = ("re_pro_loss_bef_tax",)
INTEREST_KEYS = ("re_int_new", "re_int_expd")
OTHER_INC_KEYS = ("re_oth_inc_new", "re_oth_inc")


def _operating_margin(row: dict, revenue_lakhs: float | None) -> float | None:
    """
    Operating margin, derived rather than reported.

    NSE publishes no operating-profit line, so it is rebuilt as
    PBT + interest - other income. Checked against RELIANCE Q3 FY25:
    this gives 8.4% where the filing's own note states 8.0%.

    An earlier version used PAT / revenue and called it OPM. That is the
    net profit margin — a different number, and one that moves with tax
    and one-off items rather than with the business.
    """
    if not revenue_lakhs or revenue_lakhs <= 0:
        return None
    pbt = _num(_first(row, PBT_KEYS))
    if pbt is None:
        return None
    interest = _num(_first(row, INTEREST_KEYS)) or 0.0
    other_income = _num(_first(row, OTHER_INC_KEYS)) or 0.0
    return (pbt + interest - other_income) / revenue_lakhs * 100


def sync_quarterly_results(conn, symbols: list[str] | None = None) -> dict:
    """
    Revenue, PAT and EPS per quarter.

    NSE returns amounts in Rupees Lakhs; they are stored in Crores. The
    conversion happens once, here, so no downstream threshold has to know
    about it.
    """
    with conn.cursor() as cur:
        if symbols is None:
            cur.execute("select symbol from symbols where is_active "
                        "and coalesce(series,'') <> 'INDEX' order by symbol")
            symbols = [r[0] for r in cur.fetchall()]

    written = failed = empty = 0
    unknown_shape: list[str] = []

    with _nse() as nse:
        for sym in symbols:
            try:
                payload = nse.results_comparison(sym)
            except Exception as exc:
                failed += 1
                log.debug("results_comparison failed for %s: %s", sym, exc)
                time.sleep(NSE_THROTTLE_SEC)
                continue

            rows = payload.get("resCmpData", []) if isinstance(payload, dict) else []
            if not rows:
                empty += 1
                time.sleep(NSE_THROTTLE_SEC)
                continue

            for row in rows:
                period = _parse_date(_first(row, TO_DT_KEYS))
                revenue_lakhs = _num(_first(row, REV_KEYS))
                pat_lakhs = _num(_first(row, PAT_KEYS))
                if period is None or (revenue_lakhs is None and pat_lakhs is None):
                    if len(unknown_shape) < 5:
                        unknown_shape.append(f"{sym}:{sorted(row.keys())[:8]}")
                    continue

                revenue = revenue_lakhs / LAKHS_TO_CRORES if revenue_lakhs is not None else None
                pat = pat_lakhs / LAKHS_TO_CRORES if pat_lakhs is not None else None
                opm = _operating_margin(row, revenue_lakhs)

                with conn.cursor() as cur:
                    cur.execute("""
                        insert into fundamentals_quarterly
                            (symbol, period_end, revenue, pat, eps, opm_pct, source)
                        values (%s, %s, %s, %s, %s, %s, 'nse_results_comparison')
                        on conflict (symbol, period_end) do update set
                            revenue = excluded.revenue,
                            pat     = excluded.pat,
                            eps     = excluded.eps,
                            opm_pct = excluded.opm_pct,
                            source  = excluded.source
                    """, (sym, period, revenue, pat,
                          _num(_first(row, EPS_KEYS)), opm))
                    written += cur.rowcount
            conn.commit()
            time.sleep(NSE_THROTTLE_SEC)

    if written == 0:
        raise FundamentalsUnavailable(
            f"No quarterly results written across {len(symbols)} symbols "
            f"({failed} errors, {empty} empty). Shape seen: {unknown_shape}"
        )

    log.info("Quarterly results: %d rows, %d failed, %d empty", written, failed, empty)
    return {"written": written, "failed": failed, "empty": empty,
            "unknown_shape": unknown_shape}


# ---------------------------------------------------------------------
# Gate 2 inputs, read back for the scan
# ---------------------------------------------------------------------
@dataclass
class FundamentalSnapshot:
    symbol: str
    promoter_pct: float | None
    promoter_pct_2q_ago: float | None
    revenue_trend: list[float]     # oldest -> newest, Crores
    pat_trend: list[float]
    has_data: bool


def load_snapshots(conn) -> dict[str, FundamentalSnapshot]:
    with conn.cursor() as cur:
        cur.execute("""
            select symbol, period_end, promoter_pct
            from shareholding
            where promoter_pct is not null
            order by symbol, period_end desc
        """)
        sh: dict[str, list] = {}
        for sym, period, pct in cur.fetchall():
            sh.setdefault(sym, []).append(float(pct))

        cur.execute("""
            select symbol, period_end, revenue, pat
            from fundamentals_quarterly
            order by symbol, period_end
        """)
        fin: dict[str, list] = {}
        for sym, period, rev, pat in cur.fetchall():
            fin.setdefault(sym, []).append(
                (float(rev) if rev is not None else None,
                 float(pat) if pat is not None else None))

    out: dict[str, FundamentalSnapshot] = {}
    for sym in set(sh) | set(fin):
        holdings = sh.get(sym, [])
        rows = fin.get(sym, [])
        out[sym] = FundamentalSnapshot(
            symbol=sym,
            promoter_pct=holdings[0] if holdings else None,
            promoter_pct_2q_ago=holdings[2] if len(holdings) > 2 else None,
            revenue_trend=[r for r, _ in rows if r is not None],
            pat_trend=[p for _, p in rows if p is not None],
            has_data=bool(holdings or rows),
        )
    return out


if __name__ == "__main__":
    import pprint
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        pprint.pprint(probe(sys.argv[2] if len(sys.argv) > 2 else "RELIANCE"))
    else:
        logging.basicConfig(level="INFO")
        conn = connect()
        try:
            pprint.pprint(sync_shareholding(conn))
            pprint.pprint(sync_quarterly_results(conn))
        finally:
            conn.close()
