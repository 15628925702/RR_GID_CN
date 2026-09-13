"""Verify that the fixed_qmc score path equals the cached_qmc basis path.

Both paths are supposed to evaluate the same nested Sobol nodes at the same
seed; ``fixed_qmc`` only avoids retaining the full ``(rows, components, nodes,
rank)`` tensor.  This script checks that claim numerically at a low order so
the order-14 fixed_qmc RISK-1 evidence is comparable to the order-13
cached_qmc baseline.

Writes a JSON certificate; exits non-zero if the tolerance is exceeded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rr_gid_cn.conditional_backend import (  # noqa: E402
    build_conditional_feature_basis,
    uncached_qmc_mean,
)
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--panels", type=int, default=3)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=202609999)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=2000, seed=2026)
    rng = np.random.default_rng(args.seed)
    panels = all_pairs()[: int(args.panels)]

    betas = [np.zeros(12)]
    betas.append(rng.normal(scale=0.35, size=12))
    betas.append(rng.normal(scale=1.0, size=12))

    per_panel = []
    worst = 0.0
    for panel in panels:
        rows = rng.uniform(-1.2, 1.2, size=(int(args.rows), len(panel)))
        basis = build_conditional_feature_basis(
            mixture, rows, panel, order=int(args.order), seed=int(args.seed),
            scale=scale, dtype="float32", feature_fn=None,
        )
        panel_worst = 0.0
        for beta in betas:
            cached_mean = np.asarray(basis.conditional_mean(beta), dtype=np.float64)
            fixed_mean = np.asarray(
                uncached_qmc_mean(
                    mixture, beta, rows, panel, int(args.order),
                    int(args.seed), scale, feature_fn=None,
                ),
                dtype=np.float64,
            )
            delta = float(np.max(np.abs(cached_mean - fixed_mean)))
            panel_worst = max(panel_worst, delta)
        worst = max(worst, panel_worst)
        per_panel.append({
            "panel": [int(i) for i in panel],
            "max_abs_difference": panel_worst,
            "n_betas": len(betas),
            "cached_dtype": "float32",
        })
        print(f"panel={panel} max_abs_diff={panel_worst:.3e}", flush=True)

    payload = {
        "schema_version": "fixed-vs-cached-equivalence-v1",
        "order": int(args.order),
        "scrambles": 1,
        "seed": int(args.seed),
        "n_rows_per_panel": int(args.rows),
        "n_betas": len(betas),
        "tolerance": float(args.tolerance),
        "max_abs_difference": worst,
        "per_panel": per_panel,
        "passed": bool(worst <= float(args.tolerance)),
        "note": (
            "cached basis stores float32 nodes; the tolerance reflects that "
            "storage dtype, not an algorithmic difference."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "max_abs_difference": worst,
                      "passed": payload["passed"]}, indent=2))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
