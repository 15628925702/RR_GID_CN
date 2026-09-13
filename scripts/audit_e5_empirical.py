"""Fail-closed audit for E5 empirical-kNN paper grids."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


METHODS = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize(rows: list[dict], metric: str) -> dict:
    out = {}
    for method in METHODS:
        values = [float(row[metric]) for row in rows if row.get("method") == method]
        if not values:
            out[method] = None
            continue
        mean = statistics.mean(values)
        se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        out[method] = {"n": len(values), "mean": mean, "se": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("r1", "r2"), required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = load_rows(args.rows)
    errors = []
    methods = sorted({row.get("method") for row in rows})
    budgets = sorted({int(row["budget"]) for row in rows})
    reps = sorted({int(row["replication"]) for row in rows})
    if methods != sorted(METHODS):
        errors.append("methods")
    if args.stage == "r1":
        expected_budgets = [400, 800, 1600, 3200]
        expected_rows = 4 * 4 * 20
        campaigns = ["gas_semisynthetic"]
        groups = collections.defaultdict(list)
        for row in rows:
            groups[(int(row["budget"]), int(row["replication"]))].append(row)
        if budgets != expected_budgets:
            errors.append("budgets")
        if any(row.get("experiment") != "gas_semisynthetic" for row in rows):
            errors.append("experiment")
    else:
        expected_budgets = [800, 1600, 3200]
        expected_rows = 3 * 3 * 4 * 20
        campaigns = sorted({row.get("campaign") for row in rows})
        if campaigns != ["batch10", "batch7", "batches8_9"]:
            errors.append("campaigns")
        groups = collections.defaultdict(list)
        for row in rows:
            groups[(row.get("campaign"), int(row["budget"]), int(row["replication"]))].append(row)
        if budgets != expected_budgets:
            errors.append("budgets")
        if any(row.get("experiment") != "gas_natural" for row in rows):
            errors.append("experiment")
    if len(rows) != expected_rows:
        errors.append(f"row_count:{len(rows)}!={expected_rows}")
    if reps != list(range(20)):
        errors.append("replications")
    for key, items in groups.items():
        if len(items) != 4:
            errors.append("group_size")
            break
        hashes = {item.get("target_draw_sha256") for item in items}
        if len(hashes) != 1 or not next(iter(hashes)):
            errors.append("paired_target")
            break
        if {item.get("method") for item in items} != set(METHODS):
            errors.append("group_methods")
            break
    if any(not math.isfinite(float(row.get("primary_loss", "nan"))) for row in rows):
        errors.append("finite_primary_loss")
    if any(not math.isfinite(float(row.get("design_ratio_main", "nan"))) for row in rows):
        errors.append("finite_design")
    metric = "primary_loss"
    by_budget = {}
    for budget in budgets:
        subset = [row for row in rows if int(row["budget"]) == budget]
        by_budget[str(budget)] = summarize(subset, metric)
    report = {
        "stage": f"e5_{args.stage}_empirical_knn",
        "passed": not errors,
        "failed_gates": errors,
        "row_count": len(rows),
        "expected_rows": expected_rows,
        "methods": methods,
        "budgets": budgets,
        "campaigns": campaigns,
        "replications": len(reps),
        "rows_sha256": hashlib.sha256(args.rows.read_bytes()).hexdigest(),
        "primary_loss_ci95": summarize(rows, metric),
        "primary_loss_by_budget": by_budget,
        "generator": "empirical_knn",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "failed_gates": errors, "row_count": len(rows), "out": str(args.out)}, indent=2))
    if errors:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
