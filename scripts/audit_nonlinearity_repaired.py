"""Fail-closed audit for a repaired synthetic nonlinearity sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    yaml = __import__("yaml")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    alphas = [float(value) for value in cfg["alphas"]]
    budgets = [int(value) for value in cfg["budgets"]]
    reps = int(cfg["replications"])
    expected = {(method, alpha, budget, rep) for method in METHODS for alpha in alphas for budget in budgets for rep in range(reps)}
    observed = {
        (str(row.get("method")), float(row.get("alpha", math.nan)), int(row.get("budget", -1)), int(row.get("replication", -1)))
        for row in rows
    }
    failures: list[str] = []
    if observed != expected:
        failures.append(f"grid mismatch: missing={sorted(expected - observed)[:5]} extra={sorted(observed - expected)[:5]}")
    if len(rows) != len(expected):
        failures.append(f"row_count={len(rows)} expected={len(expected)}")
    if not args.certificate.exists():
        failures.append("missing fixed_qmc equivalence certificate")
    else:
        certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
        if not certificate.get("passed") or int(certificate.get("order", -1)) != int(cfg["score_qmc_order"]):
            failures.append("fixed_qmc equivalence certificate failed or order mismatch")
    expected_information = "direct_qmc" if str(cfg.get("information_backend")) == "direct_qmc" else "cached_qmc"
    expected_scoring_steps = int(cfg["scoring_steps"])
    paired: dict[tuple[float, int, int], set[str]] = defaultdict(set)
    for row in rows:
        label = f"{row.get('method')} alpha={row.get('alpha')} B={row.get('budget')} rep={row.get('replication')}"
        if str(row.get("score_backend")) not in {"fixed_qmc", "cached_qmc"}:
            failures.append(f"{label}: score_backend")
        if str(row.get("information_inner")) != expected_information:
            failures.append(f"{label}: information_inner")
        if int(row.get("scoring_steps", -1)) != expected_scoring_steps:
            failures.append(f"{label}: scoring_steps")
        for key, expected_value in (
            ("score_qmc_order", int(cfg["score_qmc_order"])),
            ("score_mu_qmc_order", int(cfg.get("score_mu_qmc_order", cfg["score_qmc_order"]))),
            ("information_qmc_order", int(cfg["information_qmc_order"])),
            ("budget", int(row.get("budget", -1))),
        ):
            if key == "budget":
                continue
            if int(row.get(key, -1)) != expected_value:
                failures.append(f"{label}: {key}")
        if int(row.get("allocated_observations", -1)) != int(row.get("budget", -2)):
            failures.append(f"{label}: budget not exhausted")
        for key in ("risk_ratio", "risk_ratio_raw", "kl_raw", "design_ratio_main", "wall_seconds_total"):
            try:
                if not math.isfinite(float(row[key])):
                    failures.append(f"{label}: non-finite {key}")
            except Exception:
                failures.append(f"{label}: missing {key}")
        paired[(float(row["alpha"]), int(row["budget"]), int(row["replication"]))].add(str(row.get("target_draw_sha256")))
    for key, hashes in paired.items():
        if len(hashes) != 1:
            failures.append(f"paired target hash mismatch: {key}")

    report = {
        "schema_version": "synthetic-nonlinearity-repaired-audit-v1",
        "passed": not failures,
        "publication_eligible": False,
        "evidence_level": str(cfg.get("evidence_level", "L4-repaired")),
        "failed_gates": failures,
        "rows_path": str(args.rows),
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "rows_sha256": sha256(args.rows),
        "certificate_path": str(args.certificate),
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "alphas": alphas,
        "budgets": budgets,
        "replications": reps,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
