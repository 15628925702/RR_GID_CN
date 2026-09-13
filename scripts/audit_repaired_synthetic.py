"""Fail-closed audit for the calibrated streaming synthetic bulk rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


METHODS = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--certificate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    expected_score_order = int(cfg["score_qmc_order"])
    expected_mu_order = int(cfg.get("score_mu_qmc_order", expected_score_order))
    expected_information_order = int(cfg["information_qmc_order"])
    expected_scoring_steps = int(cfg["scoring_steps"])
    expected_information_inner = (
        "direct_qmc"
        if str(cfg.get("information_backend", "cached_qmc_cross")) == "direct_qmc"
        else "cached_qmc"
    )
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    budgets = [int(x) for x in cfg["budgets"]]
    reps = range(int(cfg["replications"]))
    expected = {(m, b, r) for m in METHODS for b in budgets for r in reps}
    observed = {(str(row.get("method")), int(row.get("budget", -1)), int(row.get("replication", -1))) for row in rows}
    failures: list[str] = []
    if observed != expected:
        failures.append(f"grid mismatch: missing={sorted(expected-observed)[:5]} extra={sorted(observed-expected)[:5]}")
    if len(rows) != len(expected):
        failures.append(f"row_count={len(rows)} expected={len(expected)}")
    if not args.certificate.exists():
        failures.append("missing fixed_qmc equivalence certificate")
    else:
        cert = json.loads(args.certificate.read_text(encoding="utf-8"))
        if not cert.get("passed") or int(cert.get("order", -1)) != int(cfg["score_qmc_order"]):
            failures.append("fixed_qmc equivalence certificate failed or order mismatch")

    paired: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in rows:
        label = f"{row.get('method')} B={row.get('budget')} rep={row.get('replication')}"
        if str(row.get("score_backend")) not in {"fixed_qmc", "cached_qmc"}:
            failures.append(f"{label}: score_backend")
        if str(row.get("information_inner")) != expected_information_inner:
            failures.append(f"{label}: information_inner")
        if int(row.get("score_qmc_order", -1)) != expected_score_order or int(row.get("score_mu_qmc_order", -1)) != expected_mu_order:
            failures.append(f"{label}: score order")
        if int(row.get("information_qmc_order", -1)) != expected_information_order:
            failures.append(f"{label}: information order")
        if int(row.get("scoring_steps", -1)) != expected_scoring_steps:
            failures.append(f"{label}: scoring_steps")
        if int(row.get("allocated_observations", -1)) != int(row.get("budget", -2)):
            failures.append(f"{label}: budget not exhausted")
        for key in ("risk_ratio", "risk_ratio_raw", "kl_raw", "design_ratio_main", "wall_seconds_total"):
            try:
                if not __import__("math").isfinite(float(row[key])):
                    failures.append(f"{label}: non-finite {key}")
            except Exception:
                failures.append(f"{label}: missing {key}")
        paired[(int(row["budget"]), int(row["replication"]))].add(str(row.get("target_draw_sha256")))
    bad_pairs = [key for key, hashes in paired.items() if len(hashes) != 1]
    if bad_pairs:
        failures.append(f"paired target hash mismatch: {bad_pairs[:5]}")

    report = {
        "schema_version": "synthetic-order14-repaired-audit-v1",
        "passed": not failures,
        "publication_eligible": False,
        "evidence_level": "L4-repaired",
        "failed_gates": failures,
        "rows_path": str(args.rows),
        "config_path": str(args.config),
        "config_sha256": sha256(args.config),
        "rows_sha256": sha256(args.rows),
        "certificate_path": str(args.certificate),
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
