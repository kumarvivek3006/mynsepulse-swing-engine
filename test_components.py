"""Component points in the forward log, GET /jobs/engine2-components, and the E2COMP| lines."""
from __future__ import annotations
import json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/eng")
os.environ.setdefault("INTERNAL_API_KEY", "k")
import ingest, e2_live, e2_service, e2_scan, e2_scorer as SC
PASS = FAIL = 0
def check(n, c, d=""):
    global PASS, FAIL
    PASS += bool(c); FAIL += (not c); print(("  ok    " if c else "  FAIL  ") + n, d if not c else "")

def E(s, d, dt, sc, sel, cp=None, rk=None, wn=None, cs="p"):
    e = {"s": s, "d": d, "dt": dt, "sc": sc, "q": 70.0, "pv": 100, "st": 95, "sec": "X", "sel": sel, "fam": "f"}
    if rk: e.update(rk=rk, wn=wn)
    if cp: e.update(cp=cp, cs=cs)
    return e
flog = [E("OLD", "episodic_pivot", "2026-11-05", 60.0, False),                                  # logged before components existed
        E("P1", "pocket_pivot", "2026-11-06", 80.0, True, [16, 15, 12, 18, 5, 8, 3], 1),
        E("P2", "pocket_pivot", "2026-11-06", 74.0, True, [14, 14, 10, 15, 6, 8, 3], 2),
        E("G1", "gap_and_go", "2026-11-06", 71.0, False, [14, 12, 9, 13, 5, 6, 3], 3, "max_positions"),
        E("E1", "episodic_pivot", "2026-11-06", 55.0, False, [10, 9, 8, 14, 4, 5, 2], 4, "max_positions", "c")]
conn = ingest.connect()
e2_live.put(conn, "engine2_shadow_log", flog)
rep = e2_service.components_report(conn, "gap_and_go,episodic_pivot", "2026-11-01", None)
r = {x["symbol"]: x for x in rep["rows"]}
check("only the filtered detectors are listed, old entry skipped", set(r) == {"G1", "E1"} and rep["entries_without_components"] == 1, str(rep["entries_without_components"]))
g = r["G1"]
check("points per component, total, rank, selected, why-not, cut-off gap",
      g["trend_score"] == 14 and g["rs_score"] == 12 and g["volume_score"] == 9 and g["base_score"] == 13 and g["catalyst_score"] == 5
      and g["sector_score"] == 6 and g["regime_score"] == 3 and g["score"] == 71.0 and g["rank"] == 3 and g["selected"] is False
      and g["why_not"] == "max_positions" and g["cutoff"] == 74.0 and g["points_below_cutoff"] == 3.0, str(g))
check("catalyst source decoded", r["E1"]["catalyst_source"] == "confirmed" and g["catalyst_source"] == "proxy")
s = rep["summary"]
check("summary covers every detector regardless of the filter", {"pocket_pivot", "gap_and_go", "episodic_pivot", "ALL_SELECTED", "ALL_NOT_SELECTED", "pocket_pivot:selected"} <= set(s), str(list(s)))
check("selected mean and the gap to it, per component",
      s["ALL_SELECTED"]["trend_mean"] == 15.0 and s["gap_and_go"]["minus_selected_mean"]["trend"] == -1.0 and s["episodic_pivot"]["minus_selected_mean"]["rs"] == -5.5,
      str(s["episodic_pivot"]["minus_selected_mean"]))
check("max points exposed", rep["max_points"]["trend"] == 20 and rep["components"] == SC.COMPONENTS)
check("rows can be omitted", e2_service.components_report(conn, None, None, None, False)["rows"] is None)

# E2COMP lines, incl. a stored row that has only the 0-1 fractions (scan stored before points existed)
out = {"as_of": "2026-11-05", "candidates": 3, "cutoff_score": 72.1, "scan_mode": "postclose", "all_candidates": [
    {"symbol": "RKFORGE", "detector": "gap_and_go", "rank": 15, "score": 71.1, "selected": False, "why_not": "max_positions",
     "components": {"trend": 0.7, "rs": 0.6, "volume": 0.5, "base": 0.711, "catalyst": 0.5, "sector": 0.6, "regime": 0.6}, "catalyst_source": "proxy"},
    {"symbol": "X", "detector": "pocket_pivot", "rank": 1, "score": 80.0, "selected": True,
     "points": {"trend": 16, "rs": 15, "volume": 12, "base": 18, "catalyst": 5, "sector": 8, "regime": 3}, "catalyst_source": "proxy"}]}
lines = e2_scan.component_lines("postclose", out)
j = json.loads(lines[0][len("E2COMP| "):])
row = dict(zip(j["cols"], j["rows"][0]))
check("one valid E2COMP line with column names", len(lines) == 1 and j["as_of"] == "2026-11-05" and j["cutoff"] == 72.1 and j["part"] == "1/1")
check("fractions x spec weights when points were not stored", row["trend"] == 14.0 and row["rs"] == 12.0 and row["volume"] == 7.5 and row["base"] == 14.2 and row["catalyst"] == 5.0 and row["sector"] == 6.0 and row["regime"] == 3.0, str(row))
check("chunking", len(e2_scan.component_lines("p", {"all_candidates": out["all_candidates"] * 25, "candidates": 50}, per_line=20)) == 3)

recs = []
class H(logging.Handler):
    def emit(self, r): recs.append(r.getMessage())
e2_service.log.addHandler(H()); e2_service.log.setLevel(logging.INFO)
e2_live.put(conn, "engine2_shadow_latest", out)
res = e2_service.log_components_latest()
check("boot replay writes the stored scan's components", res["lines"] == 1 and any(m.startswith("E2COMP| {") and "replay-postclose" in m for m in recs), str(recs)[:200])
print(f"\n{PASS} passed, {FAIL} failed"); sys.exit(1 if FAIL else 0)
