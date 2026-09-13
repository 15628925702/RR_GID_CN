"""Run one cached-QMC calibration replication without the adaptive gold route.

This is a diagnostic companion to ``calibrate_fast_backend.py``.  It uses the
same seed manifest convention and top-allocation panel restriction as the L2
probe, but writes a standalone result so an unfinished gold run cannot be
mistaken for a gate decision.
"""

from __future__ import annotations

import argparse
import importlib.util
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


def _load_calibration_module():
    path = ROOT / "scripts" / "calibrate_fast_backend.py"
    spec = importlib.util.spec_from_file_location("calibrate_fast_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _top_panels(module, panels, prepared, count: int):
    return module._restrict_e2e_panels(panels, prepared, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--replication", type=int, default=1)
    parser.add_argument("--top-panels", type=int, default=20)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--score-order", type=int, default=13)
    parser.add_argument(
        "--score-backend",
        choices=("cached_qmc", "fixed_qmc"),
        default="cached_qmc",
        help="Fast diagnostic score path; fixed_qmc streams the same nodes without a full basis",
    )
    parser.add_argument("--score-scrambles", type=int, default=1)
    parser.add_argument("--info-order", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--basis-root", type=Path, required=True)
    parser.add_argument(
        "--reuse-information-basis",
        action="store_true",
        help="Open a complete existing information memmap set read-only after size validation",
    )
    parser.add_argument(
        "--score-storage-root",
        type=Path,
        default=None,
        help="Optional disk-backed score-basis root for orders that exceed host RAM",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prepared = pickle.loads(args.prepared.read_bytes())
    module = _load_calibration_module()
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    panels, prepared, restriction = _top_panels(module, all_pairs(), prepared, args.top_panels)
    info_order = int(args.info_order or cfg.get("information_qmc_order", 12))
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
    storage = args.basis_root / f"rep{rep}"
    score_storage = args.score_storage_root / f"rep{rep}" if args.score_storage_root else None
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
        reuse_existing=bool(args.reuse_information_basis),
    )
    cal = cfg.get("calibration", {})
    rows = run_paper_replication(
        mixture,
        scale,
        panels,
        int(args.budget),
        seed,
        prepared,
        seed_manifest_entry=entry,
        scoring_steps=int(cfg.get("scoring_steps", 2)),
        scoring_max_step_norm=float(cfg.get("scoring_max_step_norm", 2.0)),
        score_qmc_order=int(args.score_order),
        score_backend=str(args.score_backend),
        information_qmc_order=info_order,
        information_outer_rows=int(cfg.get("information_outer_rows", 128)),
        information_scrambles=int(cfg.get("information_scrambles", 2)),
        hot_dtype=str(cfg.get("dtype", "float32")),
        policies=("oracle RR-GID",),
        pilot_schedule=cfg.get("pilot_schedule"),
        information_storage_dir=storage,
        adaptive_start_order=int(cal.get("adaptive_start_order", 8)),
        adaptive_max_order=int(cal.get("adaptive_max_order", 16)),
        adaptive_atol=float(cal.get("adaptive_atol", 2e-6)),
        adaptive_rtol=float(cal.get("adaptive_rtol", 2e-5)),
        adaptive_scrambles=int(cal.get("adaptive_scrambles", 4)),
        adaptive_chunk_rows=int(cal.get("adaptive_chunk_rows", 32)),
        info_basis=info_basis,
        score_storage_dir=score_storage,
        score_scrambles=int(args.score_scrambles),
    )
    result = {
        "evidence_level": "L2-fast-only-diagnostic",
        "gold_run": False,
        "score_backend": str(args.score_backend),
        "score_scrambles": int(args.score_scrambles),
        "score_order": int(args.score_order),
        "information_order": info_order,
        "replication": rep,
        "budget": int(args.budget),
        "panel_restriction": restriction,
        "score_storage_dir": str(score_storage) if score_storage is not None else None,
        "result": rows[0],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "risk_ratio": rows[0]["risk_ratio"],
        "kl_raw": rows[0]["kl_raw"],
        "score_order": args.score_order,
        "wall_seconds_total": rows[0]["wall_seconds_total"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
