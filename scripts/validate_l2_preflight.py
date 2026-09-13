"""Combine query and aggregate-information probes into an L2 preflight report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--information", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    query_report = load(args.query)
    information_report = load(args.information)
    query = query_report.get("query") or {}
    information = information_report.get("information") or {}
    failures = []
    if not query.get("query_ok", False):
        failures.append("query_calibration_failed")
    order_reports = list(query.get("orders") or [])
    if not order_reports or any(float(item.get("gold_converged_frac", 0.0)) != 1.0 for item in order_reports):
        failures.append("query_gold_not_fully_converged")
    if not information.get("information_ok", False):
        failures.append("aggregate_information_calibration_failed")
    if not information.get("gold_converged", False):
        failures.append("information_gold_not_fully_converged")
    if information.get("comparison_object") != "allocation-weighted aggregate M over selected panels":
        failures.append("information_compared_wrong_object")

    preflight_passed = not failures
    outstanding = []
    if int(query.get("n_queries", 0)) < 200:
        outstanding.append("full_200_query_grid_not_run")
    if (
        int(information.get("n_panels", 0)) < 20
        or int(information.get("n_betas", 0)) < 5
        or float(information.get("allocation_coverage", 0.0)) < 0.95
    ):
        outstanding.append("full_information_grid_not_run")
    outstanding.append("end_to_end_calibration_not_run")
    report = {
        "passed": False,
        "preflight_passed": preflight_passed,
        "evidence_level": "L2-preflight",
        "publication_eligible": False,
        "failed_gates": failures + outstanding,
        "warnings": [
            "The 20-query and 2-panel information probes are preflights, not the full registered L2 grid.",
            "No bulk experiment is authorized until the full query, information, and paired end-to-end gates pass.",
        ],
        "query_path": str(args.query),
        "information_path": str(args.information),
        "query": query,
        "information": information,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
