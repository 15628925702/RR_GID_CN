"""Fail-closed audit for the order-14 reuse sweep with the PDF checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ("RR-GID", "Discriminative Score OED")


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
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    yaml = __import__("yaml")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_sequences = int(cfg["sequences"])
    n_campaigns = int(cfg["campaigns"][-1])
    expected = {
        (method, sequence, campaign)
        for method in METHODS
        for sequence in range(n_sequences)
        for campaign in range(n_campaigns)
    }
    observed = {
        (str(row.get("method")), int(row.get("sequence", -1)), int(row.get("campaign_index", -1)))
        for row in rows
    }
    failures: list[str] = []
    if observed != expected:
        failures.append(f"grid mismatch: missing={sorted(expected - observed)[:5]} extra={sorted(observed - expected)[:5]}")
    if len(rows) != len(expected) or len({(r["sequence"], r["campaign_index"], r["method"]) for r in rows}) != len(expected):
        failures.append(f"row_count={len(rows)} expected={len(expected)}")

    paired: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in rows:
        label = f"{row.get('method')} seq={row.get('sequence')} camp={row.get('campaign_index')}"
        if row.get("method") not in METHODS:
            failures.append(f"{label}: method")
        if int(row.get("budget", -1)) != int(cfg["budgets"][0]):
            failures.append(f"{label}: budget")
        if row.get("score_backend") != "cached_qmc" or row.get("information_inner") != "cached_qmc":
            failures.append(f"{label}: backend")
        if int(row.get("score_qmc_order", -1)) != int(cfg["score_qmc_order"]):
            failures.append(f"{label}: score_qmc_order")
        if int(row.get("score_mu_qmc_order", -1)) != int(cfg["score_mu_qmc_order"]):
            failures.append(f"{label}: score_mu_qmc_order")
        if int(row.get("information_qmc_order", -1)) != int(cfg["information_qmc_order"]):
            failures.append(f"{label}: information_qmc_order")
        for key in ("risk_ratio", "risk_ratio_raw", "kl_raw", "c_star", "wall_seconds_total"):
            try:
                if not math.isfinite(float(row[key])):
                    failures.append(f"{label}: non-finite {key}")
            except Exception:
                failures.append(f"{label}: missing {key}")
        try:
            if abs(float(row["risk_ratio_raw"]) - int(row["budget"]) * float(row["kl_raw"]) / float(row["c_star"])) > 1e-9:
                failures.append(f"{label}: risk_ratio_raw")
        except Exception:
            failures.append(f"{label}: risk_ratio_raw unavailable")
        paired[(int(row["sequence"]), int(row["campaign_index"]))].add(str(row.get("target_draw_sha256")))
        if "cuda:0" not in str((row.get("runtime") or {}).get("device", "")):
            failures.append(f"{label}: non-CUDA runtime")
    for key, hashes in paired.items():
        if len(hashes) != 1:
            failures.append(f"paired target hash mismatch: {key}")

    checkpoints = [int(t) for t in cfg["campaigns"]]
    summaries = {}
    for t in checkpoints:
        camp = t - 1
        for method in METHODS:
            vals = [float(r["risk_ratio_raw"]) for r in rows if r["method"] == method and int(r["campaign_index"]) == camp]
            if len(vals) != n_sequences:
                failures.append(f"checkpoint T={t} {method}: expected {n_sequences} rows, got {len(vals)}")
            else:
                mean = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                summaries[f"T={t}/{method}"] = {
                    "mean": mean,
                    "ci95": [mean - 1.96 * sd / math.sqrt(len(vals)), mean + 1.96 * sd / math.sqrt(len(vals))],
                }

    report = {
        "schema_version": "reuse-order14-t50-audit-v1",
        "passed": not failures,
        "publication_eligible": False,
        "evidence_level": str(cfg.get("evidence_level", "L4-repaired")),
        "failed_gates": failures,
        "rows_path": str(args.rows),
        "config_path": str(args.config),
        "rows_sha256": sha256(args.rows),
        "config_sha256": sha256(args.config),
        "row_count": len(rows),
        "expected_row_count": len(expected),
        "sequences": n_sequences,
        "campaigns": n_campaigns,
        "report_checkpoints": checkpoints,
        "risk_ratio_ci95": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
