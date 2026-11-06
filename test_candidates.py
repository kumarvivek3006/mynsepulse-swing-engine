"""GET /jobs/engine2-candidates and the E2CAND| log: score, rank, cut-off, why-not and raw-vs-ranked, from the forward log."""
from __future__ import annotations
import json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/eng")
os.environ.setdefault("INTERNAL_API_KEY", "k")
import ingest, e2_live, e2_service
PASS = FAIL = 0
def check(n, c, d=""):
    global PASS, FAIL
    PASS += bool(c); FAIL += (not c); print(("  ok    " if c else "  FAIL  ") + n, d if not c else "")

E = lambda s, d, dt, sc, sel, rk=None, wn=None: {"s": s, "d": d, "dt": dt, "sc": sc, "q": 70.0, "pv": 100, "st": 95, "sec": "X", "sel": sel, "fam": "f",
                                                  **({"rk": rk, "wn": wn} if rk else {})}
flog = [E("A1", "pocket_pivot", "2026-10-22", 71.0, True), E("A2", "pocket_pivot", "2026-10-22", 69.0, True),
        E("A3", "pullback_ema", "2026-10-22", 66.0, True), E("A4", "pocket_pivot", "2026-10-22", 64.5, True),
        E("EP1", "episodic_pivot", "2026-10-22", 58.2, False),                       # old entry: no rank / why-not recorded
        E("A5", "pocket_pivot", "2026-10-22", 60.0, False),
        E("A1", "pocket_pivot", "2026-11-06", 80.0, True, 1), E("GG1", "gap_and_go", "2026-11-06", 70.0, False, 2, "max_positions"),
        E("A2", "pullback_ema", "2026-11-06", 75.0, True, 3)]
runs = [{"mode": "postclose", "status": "success", "as_of": "2026-10-22", "detector_candidates": {"episodic_pivot": 3, "pocket_pivot": 9, "gap_and_go": 0}}]
conn = ingest.connect()
with conn.cursor() as cur:
    cur.execute("delete from engine_settings where key in ('engine2_shadow_log','engine2_scan_runs')")
conn.commit()
e2_live.put(conn, "engine2_shadow_log", flog)
import e2_scan; e2_scan._put(conn, "engine2_scan_runs", runs)

rep = e2_service.candidates_report(conn, "episodic_pivot,gap_and_go", "2026-10-09", None)
d = {x["date"]: x for x in rep["days"]}
check("both days listed", set(d) == {"2026-10-22", "2026-11-06"}, str(list(d)))
x = d["2026-10-22"]
c = x["candidates"][0]
check("score, rank (derived when not recorded) and cut-off", c["score"] == 58.2 and c["rank"] == 6 and x["cutoff_score"] == 64.5 and c["points_below_cutoff"] == 6.3, str(c))
check("selected scores of the same day are shown", [s["score"] for s in x["selected"]] == [71.0, 69.0, 66.0, 64.5])
check("why_not says not_recorded for an old entry", c["why_not"] == "not_recorded")
check("raw 3 vs ranked 1 -> 2 removed before ranking (hard gate / no stop)", x["raw_vs_ranked"]["episodic_pivot"] == {"raw": 3, "ranked": 1, "removed_before_ranking": 2}, str(x.get("raw_vs_ranked")))
y = d["2026-11-06"]["candidates"][0]
check("recorded why_not is carried", y["why_not"] == "max_positions" and y["detector"] == "gap_and_go")

recs = []
class H(logging.Handler):
    def emit(self, r): recs.append(r.getMessage())
e2_service.log.addHandler(H()); e2_service.log.setLevel(logging.INFO)
out = e2_service.log_candidates()
lines = [m for m in recs if m.startswith("E2CAND| {")]
check("one E2CAND line per day, valid JSON", out["days"] == 2 and len(lines) == 2 and json.loads(lines[0][8:])["date"] == "2026-10-22", str(recs)[:300])
print(f"\n{PASS} passed, {FAIL} failed"); sys.exit(1 if FAIL else 0)
