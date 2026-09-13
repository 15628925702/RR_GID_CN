"""Run an isolated adaptive-gold calibration replication.

The normal e2e runner evaluates cached and gold routes in one long process.
This helper is deliberately gold-only so a completed adaptive run is persisted
even when a later paired comparison or verbose diagnostics is interrupted.
It is an L2 diagnostic; it never marks a gate as passed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rr_gid_cn.conditional_backend import build_panel_information_basis  # noqa: E402
from rr_gid_cn.paper_run import run_paper_replication  # noqa: E402
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--top-panels", type=int, default=20)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--score-order", type=int, default=13)
    parser.add_argument("--info-order", type=int, default=None)
    parser.add_argument("--max-order", type=int, default=18)
    parser.add_argument("--start-order", type=int, default=8)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--scrambles", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--scoring-steps", type=int, default=1)
    parser.add_argument(
        "--information-inner",
        choices=("cached_qmc", "exact_adaptive"),
        default="exact_adaptive",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--basis-root", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prepared_full = pickle.loads(args.prepared.read_bytes())
    allocation = np.asarray(prepared_full["designs"]["oracle RR-GID"], dtype=float)
    indices = np.argsort(-allocation, kind="stable")[: int(args.top_panels)]
    panels_all = all_pairs()
    panels = tuple(panels_all[int(i)] for i in indices)
    prepared = dict(prepared_full)
    prepared["information"] = np.asarray(prepared_full["information"])[indices]
    prepared["designs"] = {
        key: np.asarray(values, dtype=float)[indices]
        for key, values in prepared_full["designs"].items()
    }
    for key, values in prepared["designs"].items():
        prepared["designs"][key] = values / np.sum(values)

    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    rep = int(args.replication)
    seed = 202609000 + 17 * (rep + 1)
    entry = {
        "replication": rep,
        "replication_seed": seed,
        "target_draw_seed": seed + 3,
        "pilot_or_design_seed": seed + 11,
        "score_seed_root": seed + 4,
        "information_seed_root": seed + 7000,
    }
    info_order = int(args.info_order or cfg.get("information_qmc_order", 12))
    storage = args.basis_root / f"rep{rep}"
    info_basis = build_panel_information_basis(
        mixture,
        prepared["reference"],
        panels,
        scale=scale,
        outer_rows=int(cfg.get("information_outer_rows", 128)),
        order=info_order,
        seed=entry["information_seed_root"],
        dtype=str(cfg.get("dtype", "float32")),
        scrambles=int(cfg.get("information_scrambles", 2)),
        storage_dir=storage,
    )
    result = run_paper_replication(
        mixture,
        scale,
        panels,
        int(args.budget),
        seed,
        prepared,
        seed_manifest_entry=entry,
        scoring_steps=int(args.scoring_steps),
        scoring_max_step_norm=float(cfg.get("scoring_max_step_norm", 2.0)),
        score_qmc_order=int(args.score_order),
        score_backend="exact_adaptive",
        information_inner=str(args.information_inner),
        information_qmc_order=info_order,
        information_outer_rows=int(cfg.get("information_outer_rows", 128)),
        information_scrambles=int(cfg.get("information_scrambles", 2)),
        hot_dtype=str(cfg.get("dtype", "float32")),
        policies=("oracle RR-GID",),
        pilot_schedule=cfg.get("pilot_schedule"),
        adaptive_start_order=int(args.start_order),
        adaptive_max_order=int(args.max_order),
        adaptive_atol=float(args.atol),
        adaptive_rtol=float(args.rtol),
        adaptive_scrambles=int(args.scrambles),
        adaptive_chunk_rows=int(args.chunk_rows),
        information_storage_dir=storage,
        info_basis=info_basis,
    )[0]
    payload = {
        "evidence_level": "L2-gold-only-diagnostic",
        "gold_run": True,
        "replication": rep,
        "budget": int(args.budget),
        "score_order": int(args.score_order),
        "information_order": info_order,
        "information_inner": str(args.information_inner),
        "adaptive": {
            "start_order": int(args.start_order),
            "max_order": int(args.max_order),
            "atol": float(args.atol),
            "rtol": float(args.rtol),
            "scrambles": int(args.scrambles),
            "chunk_rows": int(args.chunk_rows),
        },
        "panel_restriction": {
            "selected_panel_indices": [int(i) for i in indices],
            "selected_panels": [list(panel) for panel in panels],
            "allocation_coverage": float(np.sum(allocation[indices])),
            "total_panels": int(len(panels_all)),
        },
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "risk_ratio": result["risk_ratio"],
        "kl_raw": result["kl_raw"],
        "gold_converged": result["gold_converged"],
        "wall_seconds_total": result["wall_seconds_total"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
