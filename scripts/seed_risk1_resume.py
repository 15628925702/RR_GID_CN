"""Seed the 10-rep RISK-1 runner from completed isolated checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _calibration_module():
    path = ROOT / "scripts" / "calibrate_fast_backend.py"
    spec = importlib.util.spec_from_file_location("calibrate_fast_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", nargs="+", type=Path, required=True)
    parser.add_argument("--gold", nargs="+", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--information", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.fast) != len(args.gold):
        raise ValueError("fast/gold checkpoint counts differ")

    module = _calibration_module()
    rows = []
    for fast_path, gold_path in zip(args.fast, args.gold):
        fast = _load(fast_path)["result"]
        gold = _load(gold_path)["result"]
        rep = int(fast["replication"])
        if rep != int(gold["replication"]):
            raise ValueError("replication mismatch")
        if fast["target_draw_sha256"] != gold["target_draw_sha256"]:
            raise ValueError("target draw hash mismatch")
        _write(args.run_dir / f"e2e_rep{rep}_fast.json", fast)
        _write(args.run_dir / f"e2e_rep{rep}_gold.json", gold)
        row = {
            "replication": rep,
            "fast": float(fast["risk_ratio"]),
            "gold": float(gold["risk_ratio"]),
            "abs_delta": float(abs(fast["risk_ratio"] - gold["risk_ratio"])),
            "signed_delta": float(fast["risk_ratio"] - gold["risk_ratio"]),
            "target_draw_sha256": fast["target_draw_sha256"],
            "gold_converged": bool(gold.get("gold_converged", False)),
            "fast_seconds": float(fast.get("wall_seconds_total", 0.0)),
            "gold_seconds": float(gold.get("wall_seconds_total", 0.0)),
            "score_qmc_order": 13,
            "information_qmc_order": 12,
            "information_basis_dtype": "float32",
            "budget": 8000,
            "evidence_level": "L2-e2e",
            "fast_backend": fast.get("score_backend"),
            "gold_backend": gold.get("score_backend"),
            "fast_runtime": fast.get("runtime"),
            "gold_runtime": gold.get("runtime"),
            "fast_workload": fast.get("workload"),
            "gold_workload": gold.get("workload"),
            "fast_kl_raw": fast.get("kl_raw"),
            "gold_kl_raw": gold.get("kl_raw"),
            "fast_beta_hat": fast.get("beta_hat"),
            "gold_beta_hat": gold.get("beta_hat"),
            "fast_update_diagnostics": fast.get("update_diagnostics"),
            "gold_update_diagnostics": gold.get("update_diagnostics"),
            "gold_information_diagnostics": gold.get("gold_information_diagnostics"),
            "gold_score_diagnostics": gold.get("gold_score_diagnostics"),
            "imported_checkpoint": True,
        }
        rows.append(row)
    rows.sort(key=lambda row: row["replication"])
    expected = list(range(len(rows)))
    if [row["replication"] for row in rows] != expected:
        raise ValueError(f"checkpoints must start contiguously at zero: expected {expected}")

    jsonl = "".join(json.dumps(row) + "\n" for row in rows)
    _write(args.run_dir / "e2e.jsonl", jsonl)
    e2e = module.evaluate_e2e_gate(rows, n_e2e=10, cal={"end_to_end_mean_abs_delta": 0.15})
    e2e.update({
        "n": len(rows),
        "rows": rows,
        "score_qmc_order": 13,
        "information_qmc_order": 12,
        "adaptive_max_order": 18,
        "budget": 8000,
        "top_panels": 20,
        "allocation_coverage": 0.977230176218399,
        "evidence_level": "L2-e2e-in-progress",
        "publication_eligible": False,
    })
    query = _load(args.query)["query"]
    information = _load(args.information)["information"]
    report = {
        "schema_version": "risk1-order13-10rep-v1",
        "component_provenance": {
            "query_source": str(args.query),
            "information_source": str(args.information),
            "imported_fast_sources": [str(path) for path in args.fast],
            "imported_gold_sources": [str(path) for path in args.gold],
        },
        "query": query,
        "information": information,
        "end_to_end": e2e,
        "query_ok": bool(query.get("query_ok")),
        "information_ok": bool(information.get("information_ok")),
        "passed": False,
    }
    _write(args.run_dir / "report.json", report)
    print(json.dumps({
        "run_dir": str(args.run_dir),
        "seeded_replications": expected,
        "next_replication": len(rows),
        "failed_gates_while_incomplete": e2e["failed_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
