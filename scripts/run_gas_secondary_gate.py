"""RISK-3: fail-closed Gas secondary-metric gate.

Rebuilds the max-budget, replication-0 Gas natural-drift cells from the frozen
seed manifest, verifies that the primary family-projection loss reproduces the
audited source rows, and then evaluates the secondary metrics the PDF reports
for real data: C2ST AUC, held-out moment RMSE, importance ESS, unconditional
and conditional acceptance rates, ``lambda_min(M_c(p))`` and the RR-GID
Frank-Wolfe gap.

Exits non-zero when any threshold in ``configs/validation/gas_secondary_v1.yaml``
fails.  Never writes ``passed: true`` on an incomplete grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run_gas_natural as r2  # noqa: E402
import run_gas_semisynthetic as r1  # noqa: E402
from p10_r2_formal import c2st_auc, fit_pc2, heldout_moment_rmse  # noqa: E402
from rr_gid_cn.empirical_generator import load_gas_generator  # noqa: E402
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.p4_integrity import compute_pilot_budget  # noqa: E402
from rr_gid_cn.policies import frank_wolfe, objective, uniform_probabilities  # noqa: E402
from rr_gid_cn.s1_gate import discriminative_design, largest_remainder_counts  # noqa: E402
from rr_gid_cn.vaeac import learned_information  # noqa: E402

METHODS = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_rows(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[(row["campaign"], int(row["budget"]), int(row["replication"]),
                 row.get("method") or row.get("policy"))] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/validation/gas_secondary_v1.yaml")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    started = perf_counter()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = Path(args.out or cfg["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = cfg["thresholds"]
    rep = int(cfg.get("replication", 0))

    source_cfg_path = Path(cfg["source_config"])
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8"))
    require_reproduction = bool(cfg.get("require_source_reproduction", True))
    source_rows_path = Path(cfg["source_rows"]) if cfg.get("source_rows") else None
    source_rows = _load_rows(source_rows_path) if require_reproduction and source_rows_path and source_rows_path.exists() else {}
    budget = max(int(b) for b in source_cfg["budgets"])
    campaigns = list(source_cfg["campaigns"])

    data_path = Path(cfg["data"])
    data = np.load(data_path)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    pcs2 = fit_pc2(data["ref_train"], mean, std)
    fn = lambda x: transform_features(x, mean, std, pcs)  # noqa: E731
    sensor_pairs = panel_library()
    coord_panels = tuple(
        tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs
    )

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(source_cfg.get("generator_checkpoint") or source_cfg.get("checkpoint") or "")
    if not ckpt_path.exists():
        raise SystemExit(f"missing Gas generator artifact: {ckpt_path}")
    gen = load_gas_generator(source_cfg, data, fn, device=device)

    full_pool_x = gen.sample_full(8000, seed=42)
    full_pool_phi = fn(full_pool_x)
    phi_ref = fn(data["ref_val"])
    fisher_ref = np.cov(phi_ref, rowvar=False)
    _, oracle_infos, p_star, _ = r1.build_oracle(data["ref_val"], phi_ref, fn, coord_panels)
    a_info = r1.gas_a_optimal_information(phi_ref, sensor_pairs)
    costs = np.ones(len(coord_panels))
    n_samples = int(cfg.get("generated_samples", 2000))
    n_cond_queries = int(cfg.get("conditional_queries", 8))
    n_cond_draws = int(cfg.get("conditional_draws", 8))

    rows: list[dict] = []
    for camp in campaigns:
        x_key, phi_key = r2.CAMPAIGN_KEYS[camp]
        campaign_x, campaign_phi = data[x_key], data[phi_key]
        entry = r2.seed_entry(camp, budget, rep)
        rng = np.random.default_rng(int(entry["replication_seed"]))
        perm = rng.permutation(len(campaign_x))
        half = len(campaign_x) // 2
        pool_x = campaign_x[perm[:half]]
        test_x = campaign_x[perm[half:]]
        test_phi = campaign_phi[perm[half:]]
        beta_dag = r2.fit_beta_dag(full_pool_phi, test_phi.mean(0))
        rng_t = np.random.default_rng(int(entry["target_draw_seed"]))
        target = pool_x[rng_t.choice(len(pool_x), size=int(budget), replace=True)]
        target_hash = hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()
        pilot_budget = compute_pilot_budget(source_cfg["pilot_schedule"], int(budget))
        remaining = int(budget) - int(pilot_budget)
        pilot_counts = r1.gas_balanced_pilot_counts(coord_panels, pilot_budget)
        pilot_obs = []
        cursor = 0
        for panel, count in zip(coord_panels, pilot_counts):
            for row in target[cursor: cursor + int(count)]:
                pilot_obs.append((panel, row[list(panel)]))
            cursor += int(count)
        mu_pil = r1.gas_pilot_ht_moment(fn, pilot_obs, pilot_counts, coord_panels)
        pool_phi = fn(pool_x)
        beta_tilde = r1.solve_pilot_beta(mu_pil, pool_phi)
        if getattr(gen, "kind", "") == "empirical_knn":
            n_outer = min(len(data["ref_val"]), int(source_cfg.get("information_outer_rows", 512)))
            rng_info = np.random.default_rng(int(entry["information_seed_root"]))
            outer_idx = rng_info.choice(len(data["ref_val"]), size=n_outer, replace=False)
            _, learned_stack, _ = gen.panel_information(beta_tilde, coord_panels, data["ref_val"][outer_idx])
        else:
            _, learned_stack = learned_information(
                gen, beta_tilde, coord_panels,
                n_tilted=int(source_cfg.get("gen_info_tilted", 64)),
                n_conditional=int(source_cfg.get("gen_info_cond", 16)),
                seed=int(entry["information_seed_root"]), feature_fn=fn,
            )
        learned = {panel: learned_stack[i] for i, panel in enumerate(coord_panels)}
        rr_p, rr_gap, _ = frank_wolfe(
            fisher_ref, learned_stack, costs, uniform_probabilities(len(coord_panels)),
            tolerance=1e-4, max_iter=300,
        )
        a_p, _, _ = frank_wolfe(
            np.eye(16), a_info, costs, uniform_probabilities(len(coord_panels)),
            tolerance=1e-4, max_iter=300,
        )
        disc_p = discriminative_design(
            data["ref_train"], data["ref_val"], beta_tilde, coord_panels, np.ones(128),
            seed=int(entry["pilot_or_design_seed"]),
            hidden=int(source_cfg.get("mlp_hidden", 64)),
            steps=int(source_cfg.get("mlp_steps", 100)), feature_fn=fn,
        )
        policy_p = {
            "Uniform SQD": uniform_probabilities(len(coord_panels)),
            "A-OSQD": a_p,
            "RR-GID": rr_p,
            "Discriminative Score OED": disc_p,
        }
        m_c = np.einsum("s,sij->ij", rr_p, learned_stack)
        lambda_min_rr = float(np.linalg.eigvalsh((m_c + m_c.T) / 2.0).min())
        phi_star = objective(fisher_ref, oracle_infos, p_star, costs)
        print(f"[{camp}] B={budget} rep={rep} designs ready", flush=True)

        for name in METHODS:
            probabilities = np.asarray(policy_p[name], dtype=float)
            counts = largest_remainder_counts(probabilities, remaining)
            main_obs = []
            cursor = int(pilot_budget)
            for panel, count in zip(coord_panels, counts):
                for row in target[cursor: cursor + int(count)]:
                    main_obs.append((panel, row[list(panel)]))
                cursor += int(count)
            beta_hat = r1.score_beta(
                gen, beta_tilde, pilot_obs + main_obs, coord_panels, pool_phi, learned, fn,
                n_cond=int(source_cfg.get("gen_info_cond", 16)),
                seed=int(entry["score_seed_root"]),
                steps=int(source_cfg["scoring_steps"]),
                max_step_norm=float(source_cfg["scoring_max_step_norm"]),
            )
            loss = r1.bregman_projection_loss(full_pool_phi, beta_hat, beta_dag)
            samples, acc, ess = gen.tilted_full_diagnostics(
                beta_hat, n_samples, int(entry["score_seed_root"]) + 33)
            mean_rmse, std_rmse = heldout_moment_rmse(samples, data["ref_val"], mean, std, pcs2)
            auc = c2st_auc(samples, data["ref_val"], rng_seed=int(entry["replication_seed"]) % 10000)
            auc_vs_target = c2st_auc(samples, test_x, rng_seed=int(entry["replication_seed"]) % 10000)

            cond_acc = []
            cond_rng = np.random.default_rng(int(entry["score_seed_root"]) + 71)
            for panel in list(coord_panels)[:n_cond_queries]:
                observed = target[int(cond_rng.integers(0, len(target)))][list(panel)]
                _, cacc, _ = gen.tilted_conditional_diagnostics(
                    beta_hat, observed, panel, n_cond_draws,
                    int(cond_rng.integers(2**31 - 1)),
                )
                cond_acc.append(float(cacc))
            conditional_acceptance = float(np.min(cond_acc)) if cond_acc else 0.0

            source_key = (camp, budget, rep, name)
            source_row = source_rows.get(source_key)
            require_reproduction = bool(cfg.get("require_source_reproduction", True))
            if require_reproduction and source_row is None:
                raise KeyError(f"missing audited source row for {source_key}")
            reproduction_delta = (
                abs(float(source_row["primary_loss"]) - float(loss)) if source_row is not None else None
            )
            target_hash_match = (
                source_row["target_draw_sha256"] == target_hash if source_row is not None else True
            )

            failed = []
            if require_reproduction and not target_hash_match:
                failed.append("target_draw_hash_mismatch")
            if require_reproduction and reproduction_delta is not None and reproduction_delta > float(thresholds["primary_reproduction_atol"]):
                failed.append("primary_reproduction")
            if auc > float(thresholds["c2st_auc_max"]):
                failed.append("c2st_auc_max")
            if ess < float(thresholds["importance_ess_min"]):
                failed.append("importance_ess_min")
            if acc < float(thresholds["unconditional_acceptance_min"]):
                failed.append("unconditional_acceptance_min")
            if conditional_acceptance < float(thresholds["conditional_acceptance_min"]):
                failed.append("conditional_acceptance_min")
            if name == "RR-GID":
                if lambda_min_rr <= float(thresholds["lambda_min_min"]):
                    failed.append("lambda_min_min")
                if float(rr_gap) > float(thresholds["rr_fw_gap_max"]):
                    failed.append("rr_fw_gap_max")

            row = {
                "campaign": camp, "method": name, "budget": int(budget), "replication": rep,
                "primary_loss": float(loss),
                "source_primary_loss": None if source_row is None else float(source_row["primary_loss"]),
                "primary_reproduction_delta": reproduction_delta,
                "target_draw_sha256": target_hash,
                "target_draw_hash_match": target_hash_match,
                "heldout_mean_rmse": float(mean_rmse),
                "heldout_std_rmse": float(std_rmse),
                "c2st_auc": float(auc),
                "c2st_auc_vs_target": float(auc_vs_target),
                "generator_ess": float(ess),
                "unconditional_acceptance": float(acc),
                "conditional_acceptance_min": conditional_acceptance,
                "lambda_min_Mc": lambda_min_rr if name == "RR-GID" else None,
                "rr_fw_gap": float(rr_gap) if name == "RR-GID" else None,
                "failed_gates": failed,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in
                              ("campaign", "method", "c2st_auc", "generator_ess",
                               "unconditional_acceptance", "failed_gates")}), flush=True)

    failed_any = [f"{r['campaign']}:{r['method']}:{g}" for r in rows for g in r["failed_gates"]]
    complete = len(rows) == len(campaigns) * len(METHODS)
    report = {
        "schema_version": cfg["schema_version"],
        "stage": "risk3_gas_secondary_v1",
        "passed": bool(complete and not failed_any),
        "failed_gates": failed_any,
        "budget": int(budget),
        "replication": rep,
        "campaigns": campaigns,
        "methods": list(METHODS),
        "expected_rows": len(campaigns) * len(METHODS),
        "row_count": len(rows),
        "thresholds": thresholds,
        "rows": rows,
        "provenance": {
            "code_commit": _git_commit(),
            "config": str(args.config),
            "config_sha256": _sha256(args.config),
            "source_config": str(source_cfg_path),
            "source_config_sha256": _sha256(source_cfg_path),
            "expected_source_config_sha256": None,
            "source_rows": None if source_rows_path is None else str(source_rows_path),
            "source_rows_sha256": None if not (source_rows_path and source_rows_path.exists()) else _sha256(source_rows_path),
            "data": str(data_path),
            "data_sha256": _sha256(data_path),
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": _sha256(ckpt_path),
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda if device == "cuda" else None,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": perf_counter() - started,
    }
    _write_json(out_dir / "report.json", report)
    print(json.dumps({"passed": report["passed"], "failed_gates": failed_any,
                      "rows": report["row_count"], "out": str(out_dir / "report.json")}, indent=2))
    if not report["passed"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
