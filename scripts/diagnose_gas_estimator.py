"""Gas estimator identifiability diagnostic (not a paper grid).

Answers three yes/no questions before E5-v2 bulk:
1. Is ||beta*|| > 2, and does D(proj_2 beta*, beta*) recover the 0.43 floor?
2. Does unconstrained full-observation MLE drive D down with B?
3. How large is the R2 split / A-mismatch floor?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run_gas_semisynthetic as r1
from rr_gid_cn.empirical_generator import EmpiricalKNNGenerator
from rr_gid_cn.gas_preprocess import transform_features


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--output", type=Path, default=Path("results/e5_v2_diag/diagnose_gas_estimator.json"))
    ap.add_argument("--budgets", type=int, nargs="+", default=[400, 800, 1600, 3200])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    data = np.load(args.data)
    ref_val = np.asarray(data["ref_val"], dtype=float)
    fn = lambda x: transform_features(x, data["mean"], data["std"], data["pcs"])
    phi_val = fn(ref_val)
    t0 = perf_counter()
    beta_star = r1.calibrate_beta(phi_val, seed=2026)
    beta_star_norm = float(np.linalg.norm(beta_star))
    beta_proj2 = r1._project(beta_star, bound=4.0, norm_cap=2.0)
    report = {
        "schema_version": "gas-estimator-diag-v1",
        "data": str(args.data),
        "n_ref_val": int(len(ref_val)),
        "n_ref_train": int(len(data["ref_train"])),
        "beta_star": beta_star.tolist(),
        "beta_star_norm": beta_star_norm,
        "D_zero": r1.bregman_projection_loss(phi_val, np.zeros(16), beta_star),
        "D_proj2": r1.bregman_projection_loss(phi_val, beta_proj2, beta_star),
        "cap2_binds": bool(beta_star_norm > 2.0 + 1e-8),
        "full_obs": [],
        "r2_floors": {},
        "seconds": 0.0,
    }
    print(
        f"beta_star_norm={beta_star_norm:.4f} D(0)={report['D_zero']:.4f} "
        f"D(proj2)={report['D_proj2']:.4f} cap2_binds={report['cap2_binds']}",
        flush=True,
    )
    rng_logits = phi_val @ beta_star
    w = np.exp(rng_logits - rng_logits.max())
    w /= w.sum()
    for budget in args.budgets:
        for rep in range(int(args.reps)):
            rng = np.random.default_rng(202609130 + int(budget) * 100 + rep)
            target = ref_val[rng.choice(len(ref_val), size=int(budget), replace=True, p=w)]
            mu = fn(target).mean(0)
            unc = r1.solve_pilot_beta(mu, phi_val, theta_bound=8.0, norm_cap=None)
            cap = r1.solve_pilot_beta(mu, phi_val, theta_bound=4.0, norm_cap=2.0)
            row = {
                "budget": int(budget),
                "replication": int(rep),
                "D_unconstrained": r1.bregman_projection_loss(phi_val, unc, beta_star),
                "D_cap2": r1.bregman_projection_loss(phi_val, cap, beta_star),
                "beta_unconstrained_norm": float(np.linalg.norm(unc)),
                "beta_cap2_norm": float(np.linalg.norm(cap)),
            }
            report["full_obs"].append(row)
            print(
                f"full-obs B={budget} rep={rep} "
                f"D_unc={row['D_unconstrained']:.4f} ||b||={row['beta_unconstrained_norm']:.3f} "
                f"D_cap2={row['D_cap2']:.4f}",
                flush=True,
            )
    gen = EmpiricalKNNGenerator(np.asarray(data["ref_train"], dtype=float), feature_fn=fn, neighbors=60)
    ref_pool_phi = fn(gen.sample_full(8000, seed=42))
    for camp, x_key, phi_key in (
        ("batch7", "x_batch7", "phi_batch7"),
        ("batches8_9", "x_batches89", "phi_batches89"),
        ("batch10", "x_batch10", "phi_batch10"),
    ):
        x = np.asarray(data[x_key], dtype=float)
        phi = np.asarray(data[phi_key], dtype=float)
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(x))
        half = len(x) // 2
        pool_phi = fn(x[perm[:half]])
        test_phi = phi[perm[half:]]
        beta_full = r1.solve_pilot_beta(phi.mean(0), ref_pool_phi, theta_bound=4.0, norm_cap=None)
        beta_test = r1.solve_pilot_beta(test_phi.mean(0), ref_pool_phi, theta_bound=4.0, norm_cap=None)
        beta_pool = r1.solve_pilot_beta(pool_phi.mean(0), ref_pool_phi, theta_bound=4.0, norm_cap=None)
        report["r2_floors"][camp] = {
            "n": int(len(x)),
            "beta_full_norm": float(np.linalg.norm(beta_full)),
            "D_pool_vs_test": r1.bregman_projection_loss(ref_pool_phi, beta_pool, beta_test),
            "D_full_vs_test": r1.bregman_projection_loss(ref_pool_phi, beta_full, beta_test),
            "D_zero_vs_full": r1.bregman_projection_loss(ref_pool_phi, np.zeros(16), beta_full),
        }
        print(f"R2 {camp} n={len(x)} {report['r2_floors'][camp]}", flush=True)
    report["seconds"] = float(perf_counter() - t0)
    by_b = {}
    for row in report["full_obs"]:
        by_b.setdefault(row["budget"], []).append(row)
    report["full_obs_mean"] = {
        str(b): {
            "D_unconstrained": float(np.mean([r["D_unconstrained"] for r in rows])),
            "D_cap2": float(np.mean([r["D_cap2"] for r in rows])),
        }
        for b, rows in by_b.items()
    }
    unc3200 = report["full_obs_mean"].get("3200", {}).get("D_unconstrained")
    report["pass_unconstrained_recovers"] = bool(unc3200 is not None and unc3200 < 0.05)
    report["pass_cap2_explains_floor"] = bool(report["D_proj2"] > 0.2)
    _write(args.output, report)
    print(json.dumps({k: report[k] for k in ("beta_star_norm", "D_proj2", "full_obs_mean", "r2_floors", "pass_unconstrained_recovers", "pass_cap2_explains_floor")}, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
