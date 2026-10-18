"""Apply the pre-registered analysis to a directory of runs.

    python -m experiments.observer.analyze /mnt/vault/projects/cultivated-learning/observer/runs

Writes analysis.json next to the runs and prints the tables. Every completed
run is included; incomplete runs are listed, never silently dropped.
"""
import json
import os
import statistics
import sys

from experiments.observer import metrics

ARMS = ("SELF", "SELF_PLUS", "PEER", "OTHER")
COMPONENTS = ("lists", "question", "length")
COMPARISONS = {
    "Q1": ("SELF_PLUS", "PEER", "primary: outside position, information equal"),
    "Q2": ("PEER", "OTHER", "exploratory: whose mind writes from outside"),
    "Q3": ("SELF", "SELF_PLUS", "exploratory: the observer's evidence given to the self"),
    "Q4": ("SELF", "PEER", "exploratory: shipped design vs outside author"),
}


def load_runs(root):
    runs, incomplete = {}, []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        summary_path = os.path.join(path, "summary.json")
        if not os.path.exists(summary_path):
            incomplete.append(name)
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get("completed") != 100 or not summary.get("endpoints"):
            incomplete.append(name)
            continue
        runs[(summary["arm"], summary["seed"])] = summary
    return runs, incomplete


def compare(runs, control, treatment):
    seeds = sorted(s for a, s in runs if a == control and (treatment, s) in runs)
    pairs = [(runs[(control, s)]["endpoints"]["M1"], runs[(treatment, s)]["endpoints"]["M1"]) for s in seeds]
    components = {
        c: [(runs[(control, s)]["endpoints"][c], runs[(treatment, s)]["endpoints"][c]) for s in seeds]
        for c in COMPONENTS
    }
    result = metrics.verdict(pairs, components)
    result["seeds"] = seeds
    result["pairs"] = pairs
    return result


def composition(summary):
    stats = summary["memory_stats"]
    total = stats.get("total", 0) or 1
    by_type = stats.get("by_type", {})
    return {t: by_type.get(t, 0) / total for t in ("episodic", "reflective", "semantic", "procedural")}


def arm_table(runs):
    table = {}
    for arm in ARMS:
        rows = [s for (a, _), s in runs.items() if a == arm]
        if not rows:
            continue

        def mean(fn, rows=rows):
            return statistics.fmean(fn(r) for r in rows)

        def spread(fn, rows=rows):
            return statistics.pstdev(fn(r) for r in rows) if len(rows) > 1 else 0.0

        table[arm] = {
            "n": len(rows),
            "M1": mean(lambda r: r["endpoints"]["M1"]),
            "M1_sd": spread(lambda r: r["endpoints"]["M1"]),
            **{c: mean(lambda r, c=c: r["endpoints"][c]) for c in COMPONENTS},
            "reflective_share": mean(lambda r: composition(r)["reflective"]),
            "reflective_share_range": [min(composition(r)["reflective"] for r in rows),
                                       max(composition(r)["reflective"] for r in rows)],
            "memories": mean(lambda r: r["memory_stats"].get("total", 0)),
            "final_directives": mean(lambda r: len(r["final_directives"])),
            "crucial_after_40": mean(lambda r: r["crucial_after_40"]),
            "important_to_note_after_42": mean(lambda r: r["important_to_note_after_42"]),
            "minutes": mean(lambda r: r["seconds"] / 60),
        }
    return table


def q5(table):
    shares = {arm: row["reflective_share"] for arm, row in table.items()}
    lo = min(row["reflective_share_range"][0] for row in table.values())
    hi = max(row["reflective_share_range"][1] for row in table.values())
    gap = max(shares.values()) - min(shares.values())
    holds = 0.40 <= lo and hi <= 0.55 and gap < 0.05
    return {"arm_means": shares, "min": lo, "max": hi, "max_gap": gap,
            "prediction": "every run in [0.40, 0.55] and arm means within 0.05",
            "holds": holds}


def analyze(root):
    runs, incomplete = load_runs(root)
    table = arm_table(runs)
    return {
        "runs": len(runs),
        "incomplete": incomplete,
        "arms": table,
        "comparisons": {q: {"control": c, "treatment": t, "label": label, **compare(runs, c, t)}
                        for q, (c, t, label) in COMPARISONS.items()},
        "Q5": q5(table) if table else None,
    }


def print_report(result):
    print(f"runs: {result['runs']}  incomplete: {result['incomplete'] or 'none'}\n")
    print(f"{'arm':10} {'n':>2} {'M1':>6} {'sd':>6} {'lists':>6} {'quest':>6} {'length':>6} {'refl%':>6} {'dirs':>5}")
    for arm, row in result["arms"].items():
        print(f"{arm:10} {row['n']:>2} {row['M1']:6.3f} {row['M1_sd']:6.3f} {row['lists']:6.3f} "
              f"{row['question']:6.3f} {row['length']:6.3f} {row['reflective_share'] * 100:6.1f} "
              f"{row['final_directives']:5.1f}")
    print()
    for q, comp in result["comparisons"].items():
        if not comp["n_pairs"]:
            continue
        comps = ", ".join(f"{k} {v:+.3f}" for k, v in comp["component_mean_diffs"].items())
        print(f"{q} {comp['control']} - {comp['treatment']}: mean {comp['mean_diff']:+.3f}, "
              f"treatment lower in {comp['wins']}/{comp['n_pairs']}, [{comps}] -> {comp['verdict']}"
              f"  ({comp['label']})")
    if result["Q5"]:
        q = result["Q5"]
        print(f"\nQ5 reflective share: {q['min'] * 100:.1f}-{q['max'] * 100:.1f}%, arm gap "
              f"{q['max_gap'] * 100:.1f} pp -> {'holds' if q['holds'] else 'does not hold'}")


def main(argv=None):
    argv = argv or sys.argv[1:]
    root = argv[0] if argv else "/mnt/vault/projects/cultivated-learning/observer/runs"
    result = analyze(root)
    with open(os.path.join(root, "analysis.json"), "w") as f:
        json.dump(result, f, indent=2)
    print_report(result)


if __name__ == "__main__":
    main()
