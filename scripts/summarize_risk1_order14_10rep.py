"""Assemble the frozen order-14 RISK-1 10-replication certificate.

Combines the isolated fixed_qmc fast checkpoints, the isolated adaptive-gold
checkpoints, and the already-passing query / information certificates into a
single fail-closed report.  Refuses to emit ``passed: true`` unless the query,
information and paired end-to-end gates all pass.
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--information", type=Path, required=True)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--score-order", type=int, default=14)
    parser.add_argument("--information-order", type=int, default=12)
    parser.add_argument("--adaptive-max-order", type=int, default=18)
    parser.add_argument("--mean-abs-tolerance", type=float, default=0.15)
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    module = _calibration_module()

    rows = []
    sources = []
    for rep in range(int(args.replications)):
        fast_path = run_dir / f"fast_order{args.score_order}_rep{rep}.json"
        gold_path = run_dir / f"gold_order{args.adaptive_max_order}_rep{rep}.json"
        for path in (fast_path, gold_path):
            if not path.exists():
                raise FileNotFoundError(f"missing checkpoint: {path}")
        fast_payload = _load(fast_path)
        gold_payload = _load(gold_path)
        fast = fast_payload["result"]
        gold = gold_payload["result"]
        if int(fast["replication"]) != rep or int(gold["replication"]) != rep:
            raise ValueError(f"replication mismatch at rep {rep}")
        if fast["target_draw_sha256"] != gold["target_draw_sha256"]:
            raise ValueError(f"target draw hash mismatch at rep {rep}")
        _write(run_dir / f"e2e_rep{rep}_fast.json", fast)
        _write(run_dir / f"e2e_rep{rep}_gold.json", gold)
        rows.append({
            "replication": rep,
            "fast": float(fast["risk_ratio"]),
            "gold": float(gold["risk_ratio"]),
            "abs_delta": float(abs(fast["risk_ratio"] - gold["risk_ratio"])),
            "signed_delta": float(fast["risk_ratio"] - gold["risk_ratio"]),
            "target_draw_sha256": fast["target_draw_sha256"],
            "gold_converged": bool(gold.get("gold_converged", False)),
            "fast_seconds": float(fast.get("wall_seconds_total", 0.0)),
            "gold_seconds": float(gold.get("wall_seconds_total", 0.0)),
            "score_qmc_order": int(args.score_order),
            "information_qmc_order": int(args.information_order),
            "information_basis_dtype": "float32",
            "budget": int(args.budget),
            "evidence_level": "L2-e2e",
            "fast_backend": str(fast_payload.get("score_backend")),
            "fast_score_scrambles": int(fast_payload.get("score_scrambles", 1) or 1),
            "gold_backend": str(gold.get("score_backend")),
            "gold_information_inner": gold_payload.get("information_inner"),
            "gold_adaptive": gold_payload.get("adaptive"),
            "fast_runtime": fast.get("runtime"),
            "gold_runtime": gold.get("runtime"),
            "fast_workload": fast.get("workload"),
            "gold_workload": gold.get("workload"),
            "fast_kl_raw": fast.get("kl_raw"),
            "gold_kl_raw": gold.get("kl_raw"),
            "fast_beta_hat": fast.get("beta_hat"),
            "gold_beta_hat": gold.get("beta_hat"),
            "fast_update_diagnostics": fast.get("update_diagnostics"),
            "gold_information_diagnostics": gold.get("gold_information_diagnostics"),
            "gold_score_diagnostics": gold.get("gold_score_diagnostics"),
            "imported_checkpoint": True,
        })
        sources.append({
            "replication": rep,
            "fast_source": str(fast_path),
            "gold_source": str(gold_path),
            "fast_sha256": hashlib.sha256(fast_path.read_bytes()).hexdigest(),
            "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        })

    jsonl = "".join(json.dumps(row) + "\n" for row in rows)
    _write(run_dir / "e2e.jsonl", jsonl)

    e2e = module.evaluate_e2e_gate(
        rows, n_e2e=int(args.replications),
        cal={"end_to_end_mean_abs_delta": float(args.mean_abs_tolerance)},
    )
    e2e.update({
        "n": len(rows),
        "rows": rows,
        "score_qmc_order": int(args.score_order),
        "information_qmc_order": int(args.information_order),
        "adaptive_max_order": int(args.adaptive_max_order),
        "budget": int(args.budget),
        "top_panels": 20,
        "allocation_coverage": 0.977230176218399,
        "evidence_level": "L2-e2e",
        "publication_eligible": False,
    })

    query = _load(args.query)["query"]
    information = _load(args.information)["information"]
    query_ok = bool(query.get("query_ok"))
    information_ok = bool(information.get("information_ok"))
    passed = bool(query_ok and information_ok and e2e.get("ok"))

    report = {
        "schema_version": "risk1-order14-10rep-v1",
        "component_provenance": {
            "query_source": str(args.query),
            "information_source": str(args.information),
            "checkpoints": sources,
            "equivalence_certificate": str(run_dir / "equivalence_certificate.json"),
        },
        "query": query,
        "information": information,
        "end_to_end": e2e,
        "query_ok": query_ok,
        "information_ok": information_ok,
        "end_to_end_ok": bool(e2e.get("ok")),
        "passed": passed,
    }
    _write(run_dir / "report.json", report)

    print(json.dumps({
        "out": str(run_dir / "report.json"),
        "n": len(rows),
        "query_ok": query_ok,
        "information_ok": information_ok,
        "end_to_end_ok": bool(e2e.get("ok")),
        "passed": passed,
        "mean_abs_delta": e2e["mean_abs_delta"],
        "signed_mean_delta": e2e["signed_mean_delta"],
        "signed_paired_se": e2e["signed_paired_se"],
        "gold_converged_frac": e2e["gold_converged_frac"],
        "failed_gates": e2e["failed_gates"],
    }, indent=2))
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
