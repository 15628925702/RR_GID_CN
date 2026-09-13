"""Fail-closed validation for the L1 B=50 four-method smoke run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    methods = tuple(str(value) for value in cfg["methods"])
    budgets = tuple(int(value) for value in cfg["budgets"])
    replications = range(int(cfg["replications"]))
    expected = {(method, budget, rep) for method in methods for budget in budgets for rep in replications}
    keys = [
        (str(row.get("method")), int(row.get("budget", -1)), int(row.get("replication", -1)))
        for row in rows
    ]
    failures: list[str] = []
    duplicates = sorted(key for key, count in Counter(keys).items() if count != 1)
    if set(keys) != expected or duplicates:
        failures.append(
            f"grid mismatch: missing={sorted(expected - set(keys))[:5]}, "
            f"extra={sorted(set(keys) - expected)[:5]}, duplicates={duplicates[:5]}"
        )

    paired: dict[tuple[int, int], set[str]] = defaultdict(set)
    method_risks: dict[str, list[float]] = defaultdict(list)
    devices = set()
    for row in rows:
        budget = int(row["budget"])
        rep = int(row["replication"])
        paired[(budget, rep)].add(str(row.get("target_draw_sha256")))
        if int(row.get("allocated_observations", -1)) != budget:
            failures.append(f"budget not exhausted exactly for {row.get('method')} B={budget} rep={rep}")
        for field in ("kl_raw", "risk_ratio", "design_ratio_main", "wall_seconds_total"):
            if not math.isfinite(float(row.get(field, math.nan))):
                failures.append(f"non-finite {field} for {row.get('method')} B={budget} rep={rep}")
        if str(row.get("score_backend")) != "cached_qmc":
            failures.append(f"unexpected score backend for {row.get('method')} B={budget} rep={rep}")
        if str(row.get("information_inner")) != "cached_qmc":
            failures.append(f"unexpected information backend for {row.get('method')} B={budget} rep={rep}")
        if row.get("gold_converged") is not None:
            failures.append(f"cached-only row incorrectly claims a gold convergence status for {row.get('method')} B={budget} rep={rep}")
        if int((row.get("workload") or {}).get("conditional_rejection_calls", -1)) != 0:
            failures.append(f"conditional rejection used for {row.get('method')} B={budget} rep={rep}")
        if str(row.get("evidence_level")) != "L1" or str(row.get("experiment_mode")) != "smoke":
            failures.append(f"incorrect evidence label for {row.get('method')} B={budget} rep={rep}")
        device = str((row.get("runtime") or {}).get("device", ""))
        devices.add(device)
        if not device.startswith("cuda:0:"):
            failures.append(f"CUDA device not recorded for {row.get('method')} B={budget} rep={rep}")
        method_risks[str(row["method"])].append(float(row["risk_ratio"]))

    broken_pairing = sorted(key for key, hashes in paired.items() if len(hashes) != 1)
    if broken_pairing:
        failures.append(f"paired target hashes differ: {broken_pairing[:5]}")

    report = {
        "passed": not failures,
        "evidence_level": "L1",
        "publication_eligible": False,
        "failed_gates": failures,
        "warnings": [
            "B=50 and pilot=10 are insufficient for a scientific risk-ranking claim in the 12-dimensional model.",
            "L2 query, aggregate-information, and end-to-end calibration remain separate prerequisites for bulk runs.",
        ],
        "rows_path": str(args.rows),
        "config_path": str(args.config),
        "row_count": len(rows),
        "devices": sorted(devices),
        "risk_ratio_by_method": dict(sorted(method_risks.items())),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
