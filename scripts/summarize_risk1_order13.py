"""Assemble the frozen order-13 RISK-1 preflight from isolated checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts" / "calibrate_fast_backend.py"
    spec = importlib.util.spec_from_file_location("calibrate_fast_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", nargs=2, type=Path, required=True)
    parser.add_argument("--gold", nargs=2, type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--information", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fast = [_load(path)["result"] for path in args.fast]
    gold = [_load(path)["result"] for path in args.gold]
    rows = []
    for fast_row, gold_row in zip(fast, gold):
        if fast_row["replication"] != gold_row["replication"]:
            raise ValueError("replication mismatch")
        if fast_row["target_draw_sha256"] != gold_row["target_draw_sha256"]:
            raise ValueError("target draw hash mismatch")
        rows.append({
            "replication": int(fast_row["replication"]),
            "fast": float(fast_row["risk_ratio"]),
            "gold": float(gold_row["risk_ratio"]),
            "signed_delta": float(fast_row["risk_ratio"] - gold_row["risk_ratio"]),
            "abs_delta": float(abs(fast_row["risk_ratio"] - gold_row["risk_ratio"])),
            "target_draw_sha256": fast_row["target_draw_sha256"],
            "gold_converged": bool(gold_row["gold_converged"]),
            "fast_seconds": float(fast_row["wall_seconds_total"]),
            "gold_seconds": float(gold_row["wall_seconds_total"]),
            "fast_beta_hat": fast_row["beta_hat"],
            "gold_beta_hat": gold_row["beta_hat"],
        })
    cal = {"end_to_end_mean_abs_delta": 0.15}
    e2e = _module().evaluate_e2e_gate(rows, n_e2e=2, cal=cal)
    e2e.update({
        "n": 2,
        "rows": rows,
        "score_qmc_order": 13,
        "information_qmc_order": 12,
        "adaptive_max_order": 18,
        "adaptive_atol": 2e-5,
        "adaptive_rtol": 2e-4,
        "adaptive_scrambles": 4,
        "budget": 8000,
        "top_panels": 20,
        "allocation_coverage": 0.977230176218399,
        "evidence_level": "L2-e2e-preflight",
        "publication_eligible": False,
    })
    query = _load(args.query)["query"]
    information = _load(args.information)["information"]
    report = {
        "schema_version": "risk1-order13-preflight-v1",
        "query": query,
        "information": information,
        "end_to_end": e2e,
        "query_ok": bool(query.get("query_ok")),
        "information_ok": bool(information.get("information_ok")),
        "end_to_end_ok": bool(e2e.get("ok")),
    }
    report["passed"] = bool(report["query_ok"] and report["information_ok"] and report["end_to_end_ok"])
    report["next_action"] = "expand e2e to 10 paired replications" if report["passed"] else "stop and diagnose"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "query_ok": report["query_ok"],
        "information_ok": report["information_ok"],
        "end_to_end_ok": report["end_to_end_ok"],
        "passed": report["passed"],
        "mean_abs_delta": e2e["mean_abs_delta"],
        "signed_paired_se": e2e["signed_paired_se"],
        "failed_gates": e2e["failed_gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
