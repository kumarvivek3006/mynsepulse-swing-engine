#!/usr/bin/env python3
"""
D1 decision rule — should BASE_STRATEGY switch to best_quality?

Takes the walk_forward block from two 3-year runs and applies the rule
exactly as specified. The walk-forward is the arbiter; aggregate expectancy
is reported but does NOT decide.

    python3 d1_decide.py first_valid.json best_quality.json

Each file is either the full backtest response or just the walk_forward
block — both are accepted, since pasting one or the other is equally
likely. Aggregate expectancy is picked up when present.

Run #29 (first_valid) is already known: 2/6 positive. Passing only the
best_quality file compares against that stored baseline.
"""

from __future__ import annotations

import json
import sys

RUN29_FIRST_VALID = {
    "label": "first_valid (Run #29, stored)",
    "windows": [
        {"window": 1, "expectancy_r": -0.048},
        {"window": 2, "expectancy_r": -1.065},
        {"window": 3, "expectancy_r": -0.030},
        {"window": 4, "expectancy_r": +0.447},
        {"window": 5, "expectancy_r": -0.658},
        {"window": 6, "expectancy_r": +1.390},
    ],
    "positive_windows": 2,
    "aggregate_expectancy_r": 0.199,
    "n": 175,
}


def extract(payload: dict) -> dict:
    """Accept a full run response, a metrics block, or a bare walk_forward."""
    node = payload
    for key in ("run", "metrics"):
        if isinstance(node, dict) and key in node and isinstance(node[key], dict):
            node = node[key]

    wf = node.get("walk_forward", node if "windows" in node else None)
    if not wf or "windows" not in wf:
        raise SystemExit("no walk_forward block found — paste the run JSON "
                         "or the walk_forward block itself")

    overall = (node.get("overall") or {}) if isinstance(node, dict) else {}
    return {
        "windows": wf["windows"],
        "positive_windows": wf.get(
            "positive_windows",
            sum(1 for w in wf["windows"]
                if (w.get("expectancy_r") or 0) > 0)),
        "underpowered": wf.get("underpowered_windows", []),
        "aggregate_expectancy_r": overall.get("expectancy_r"),
        "n": overall.get("filled"),
    }


def show(label: str, d: dict) -> None:
    print(f"\n{label}")
    print(f"  {'win':>3} {'n':>5} {'expectancy':>11}  sign")
    for w in d["windows"]:
        e = w.get("expectancy_r")
        sign = "positive" if (e or 0) > 0 else "negative"
        n = w.get("n", "-")
        print(f"  {w.get('window','?'):>3} {str(n):>5} {str(e):>11}  {sign}")
    print(f"  -> positive windows: {d['positive_windows']}/6")
    if d.get("underpowered"):
        print(f"  -> UNDERPOWERED windows (n<15): {d['underpowered']}")
        print("     a pass carried by these should be read as a fail")
    if d.get("aggregate_expectancy_r") is not None:
        print(f"  -> aggregate: {d['aggregate_expectancy_r']}R on n={d.get('n')}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)

    if len(args) == 1:
        base = dict(RUN29_FIRST_VALID)
        base["underpowered"] = []
        cand = extract(json.load(open(args[0])))
        base_label = RUN29_FIRST_VALID["label"]
    else:
        base = extract(json.load(open(args[0])))
        cand = extract(json.load(open(args[1])))
        base_label = "first_valid"

    show(base_label, base)
    show("best_quality (new run)", cand)

    bp, cp = base["positive_windows"], cand["positive_windows"]

    print("\n" + "=" * 62)
    print("DECISION RULE (walk-forward is the arbiter, not aggregate)")
    print("=" * 62)
    print(f"  first_valid positive windows : {bp}/6")
    print(f"  best_quality positive windows: {cp}/6")

    if cp >= 3:
        verdict = "ADOPT best_quality"
        action = ('setups.py: DEFAULT_BASE_STRATEGY = os.environ.get('
                  '"BASE_STRATEGY", "best_quality")')
        why = (f"{cp}/6 positive is at least one more than first_valid's {bp}/6 — "
               "the walk-forward confirms the aggregate win.")
    elif cp == 2:
        verdict = "DO NOT ADOPT — keep first_valid"
        action = "no code change; BASE_STRATEGY stays first_valid"
        why = ("2/6 matches first_valid. The aggregate win is therefore "
               "REGIME-DEPENDENT: best_quality earns more in the windows that "
               "already worked without winning any new ones.")
    else:
        verdict = "DO NOT ADOPT — and record the finding"
        action = "no code change; BASE_STRATEGY stays first_valid"
        why = (f"{cp}/6 is WORSE than first_valid's {bp}/6 despite a better "
               "aggregate. Better aggregate with worse walk-forward means the "
               "gain is concentrated in fewer periods — that is a finding "
               "worth recording, not just a rejected switch.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  REASON : {why}")
    print(f"  ACTION : {action}")

    agg_b, agg_c = base.get("aggregate_expectancy_r"), cand.get("aggregate_expectancy_r")
    if agg_b is not None and agg_c is not None and agg_c > agg_b and cp <= bp:
        print("\n  NOTE: aggregate favours best_quality while walk-forward does "
              "not. Per the brief, the walk-forward decides.")

    print("\n  ROLLBACK (either way): BASE_STRATEGY=first_valid in env.")
    print("  Both strategies remain available; neither is removed.")


if __name__ == "__main__":
    main()
