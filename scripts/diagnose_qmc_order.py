"""Small fixed-query diagnostic for cached versus adaptive conditional means."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rr_gid_cn.conditional_backend import (  # noqa: E402
    AdaptiveConditionalMean,
    build_conditional_feature_basis,
    evaluate_conditional_basis,
)
from rr_gid_cn.synthetic_oracle import (  # noqa: E402
    make_frozen_mixture,
    reference_scale,
    sample_full,
    tilted_conditional_mean_qmc_nested,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", nargs=2, type=int, default=(5, 11))
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=202609020)
    parser.add_argument("--beta-seed", type=int, default=20260903)
    parser.add_argument("--score-seed", type=int, default=202609004)
    parser.add_argument("--orders", nargs="+", type=int, default=(12, 14, 16))
    parser.add_argument("--max-order", type=int, default=16)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--scrambles", type=int, default=4)
    args = parser.parse_args()

    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    full = sample_full(mixture, max(args.rows, 20), seed=args.seed)
    panel = tuple(args.panel)
    rows = full[: args.rows][:, list(panel)]
    rng = np.random.default_rng(args.beta_seed)
    beta = rng.normal(0.0, 0.35, 12)
    beta *= min(1.0, 2.0 / max(float(np.linalg.norm(beta)), 1e-12))

    gold_basis = AdaptiveConditionalMean(
        mixture,
        rows,
        panel,
        seed=args.score_seed,
        scale=scale,
        start_order=8,
        max_order=args.max_order,
        atol=args.atol,
        rtol=args.rtol,
        scrambles=args.scrambles,
        chunk_rows=args.rows,
    )
    gold = gold_basis.conditional_mean(beta)
    report = {
        "panel": list(panel),
        "rows": int(args.rows),
        "gold_converged": bool(gold_basis.last_converged),
        "gold_diagnostics": gold_basis.last_diagnostics,
        "orders": [],
    }
    for order in args.orders:
        basis = build_conditional_feature_basis(
            mixture,
            rows,
            panel,
            order=order,
            seed=args.score_seed,
            scale=scale,
            dtype="float64",
        )
        estimate = evaluate_conditional_basis(beta, basis)
        rel = np.linalg.norm(estimate - gold, axis=1) / np.maximum(np.linalg.norm(gold, axis=1), 1e-8)
        report["orders"].append(
            {
                "order": int(order),
                "relative_error_by_row": rel.tolist(),
                "max_relative_error": float(rel.max()),
                "max_absolute_coordinate_error": float(np.max(np.abs(estimate - gold))),
            }
        )
        direct = tilted_conditional_mean_qmc_nested(
            mixture,
            beta,
            rows,
            panel,
            order,
            seed=args.score_seed,
            scale=scale,
            return_orders=(order,),
        )[order]
        report["orders"][-1].update(
            {
                "basis_vs_direct_max_abs": float(np.max(np.abs(estimate - direct))),
                "basis_vs_direct_rel": float(
                    np.max(
                        np.linalg.norm(estimate - direct, axis=1)
                        / np.maximum(np.linalg.norm(direct, axis=1), 1e-8)
                    )
                ),
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
