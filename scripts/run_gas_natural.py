"""Paper E5-R2: Gas natural-drift family-projection curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import run_gas_semisynthetic as r1
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.p4_integrity import compute_pilot_budget, sha256_file, validate_experiment_mode
from rr_gid_cn.policies import frank_wolfe, objective, uniform_probabilities
from rr_gid_cn.empirical_generator import load_gas_generator
from rr_gid_cn.s1_gate import discriminative_design, largest_remainder_counts
from rr_gid_cn.vaeac import learned_information

CAMPAIGN_KEYS = {
    "batch7": ("x_batch7", "phi_batch7"),
    "batches8_9": ("x_batches89", "phi_batches89"),
    "batch10": ("x_batch10", "phi_batch10"),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def seed_entry(campaign: str, budget: int, replication: int) -> dict:
    tag = {"batch7": 1, "batches8_9": 2, "batch10": 3}[campaign]
    seed = 202609200 + tag * 100000 + int(budget) * 1000 + int(replication)
    return {
        "campaign": campaign,
        "budget": int(budget),
        "replication": int(replication),
        "replication_seed": seed,
        "target_draw_seed": seed + 3,
        "pilot_or_design_seed": seed + 11,
        "score_seed_root": seed + 4,
        "information_seed_root": seed + 7000,
    }


def fit_beta_dag(phi_pool: np.ndarray, mu_t: np.ndarray) -> np.ndarray:
    beta = np.zeros(16)
    for _ in range(30):
        logits = phi_pool @ beta
        w = np.exp(logits - logits.max())
        w /= w.sum()
        mu = w @ phi_pool
        diff = np.asarray(mu_t) - mu
        if np.linalg.norm(diff) < 1e-6:
            break
        fisher = np.cov(phi_pool, rowvar=False, aweights=w) + 1e-3 * np.eye(16)
        step = np.linalg.solve(fisher, diff)
        nrm = float(np.linalg.norm(step))
        if nrm > 2.0:
            step *= 2.0 / nrm
        beta = r1._project(beta + step)
    return beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/gas_natural.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--campaign", type=str, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument("--output-path", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_experiment_mode(cfg, paper=True, formal=True)
    data = np.load(args.data)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    sensor_pairs = panel_library()
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs)
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    ckpt_path = Path(cfg["generator_checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"missing Gas generator artifact: {ckpt_path}")
    gen = load_gas_generator(cfg, data, fn, device=device)
    if args.profile:
        print(f"sampling A-hat pool device={device}", flush=True)
    full_pool_x = gen.sample_full(8000, seed=42)
    full_pool_phi = fn(full_pool_x)
    phi_ref = fn(data["ref_val"])
    fisher_ref = np.cov(phi_ref, rowvar=False)
    _, oracle_infos, p_star, _ = r1.build_oracle(data["ref_val"], phi_ref, fn, coord_panels)
    a_info = r1.gas_a_optimal_information(phi_ref, sensor_pairs)
    campaigns = [args.campaign] if args.campaign else list(cfg["campaigns"])
    budgets = [int(args.budget)] if args.budget is not None else [int(b) for b in cfg["budgets"]]
    replications = int(cfg["replications"])
    start, end = (0, replications) if args.rep_range is None else (int(args.rep_range[0]), int(args.rep_range[1]))
    methods = tuple(cfg["methods"])
    manifest_path = Path(cfg["seed_manifest"])
    if not manifest_path.exists():
        rows = [seed_entry(c, b, r) for c in cfg["campaigns"] for b in cfg["budgets"] for r in range(replications)]
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    by_key = {(row["campaign"], int(row["budget"]), int(row["replication"])): row
              for row in json.loads(manifest_path.read_text(encoding="utf-8")).get("rows", [])}
    out = Path(args.output_path or cfg["output_path"]) / "rows.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["campaign"], row.get("method") or row.get("policy"), int(row["budget"]), int(row.get("replication", 0))))
    commit = _git_commit()
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    ckpt_sha = sha256_file(ckpt_path)
    data_sha = sha256_file(args.data)
    seed_manifest_sha = sha256_file(manifest_path)
    runtime = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "pytorch": torch.__version__ if "torch" in locals() else None,
        "cuda_runtime": torch.version.cuda if device == "cuda" else None,
        "score_backend": cfg.get("score_backend"),
        "information_backend": cfg.get("information_backend"),
    }
    manifest_payload = {
        "schema_version": "gas-natural-v1-provenance",
        "stage": "P10",
        "experiment": "gas_natural_v1",
        "status": "running",
        "config": str(args.config),
        "config_sha256": config_sha,
        "data": str(args.data),
        "data_sha256": data_sha,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "seed_manifest": str(manifest_path),
        "seed_manifest_sha256": seed_manifest_sha,
        "rows": str(out),
        "methods": list(methods),
        "campaigns": campaigns,
        "budgets": budgets,
        "replications": replications,
        "resume_supported": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
    }
    provenance_path = out.parent / "provenance.json"
    if args.resume and provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        if old.get("config_sha256") != config_sha or old.get("checkpoint_sha256") != ckpt_sha or old.get("data_sha256") != data_sha:
            raise RuntimeError("--resume refused: config/checkpoint/data hash mismatch")
    _write_json(provenance_path, manifest_payload)
    t_wall = perf_counter()
    for camp in campaigns:
        x_key, phi_key = CAMPAIGN_KEYS[camp]
        campaign_x, campaign_phi = data[x_key], data[phi_key]
        for budget in budgets:
            for rep in range(start, end):
                if args.max_hours is not None and (perf_counter() - t_wall) / 3600.0 >= float(args.max_hours):
                    print(f"max-hours reached at {camp} B={budget} rep={rep}", flush=True)
                    print(f"wrote {out}")
                    return
                pending = [m for m in methods if (camp, m, int(budget), int(rep)) not in done]
                if not pending:
                    continue
                entry = dict(by_key.get((camp, int(budget), int(rep))) or seed_entry(camp, int(budget), int(rep)))
                entry["replication"] = int(rep)
                rng = np.random.default_rng(int(entry["replication_seed"]))
                perm = rng.permutation(len(campaign_x))
                half = len(campaign_x) // 2
                pool_x = campaign_x[perm[:half]]
                test_phi = campaign_phi[perm[half:]]
                beta_dag = fit_beta_dag(full_pool_phi, test_phi.mean(0))
                rng_t = np.random.default_rng(int(entry["target_draw_seed"]))
                target = pool_x[rng_t.choice(len(pool_x), size=int(budget), replace=True)]
                target_hash = hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()
                pilot_budget = compute_pilot_budget(cfg["pilot_schedule"], int(budget))
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
                    n_outer = min(len(data["ref_val"]), int(cfg.get("information_outer_rows", 512)))
                    rng_info = np.random.default_rng(int(entry["information_seed_root"]))
                    outer_idx = rng_info.choice(len(data["ref_val"]), size=n_outer, replace=False)
                    _, learned_stack, _ = gen.panel_information(beta_tilde, coord_panels, data["ref_val"][outer_idx])
                else:
                    _, learned_stack = learned_information(
                        gen, beta_tilde, coord_panels,
                        n_tilted=int(cfg.get("gen_info_tilted", 64)),
                        n_conditional=int(cfg.get("gen_info_cond", 16)),
                        seed=int(entry["information_seed_root"]), feature_fn=fn,
                    )
                learned = {panel: learned_stack[i] for i, panel in enumerate(coord_panels)}
                rr_p, rr_gap, _ = frank_wolfe(
                    fisher_ref, learned_stack, np.ones(len(coord_panels)),
                    uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300,
                )
                a_p, _, _ = frank_wolfe(
                    np.eye(16), a_info, np.ones(len(coord_panels)),
                    uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300,
                )
                disc_p = None
                if "Discriminative Score OED" in pending:
                    disc_p = discriminative_design(
                        data["ref_train"], data["ref_val"], beta_tilde, coord_panels, np.ones(128),
                        seed=int(entry["pilot_or_design_seed"]),
                        hidden=int(cfg.get("mlp_hidden", 64)),
                        steps=int(cfg.get("mlp_steps", 100)), feature_fn=fn,
                    )
                policy_p = {
                    "Uniform SQD": uniform_probabilities(len(coord_panels)),
                    "A-OSQD": a_p,
                    "RR-GID": rr_p,
                    "Discriminative Score OED": disc_p,
                }
                costs = np.ones(len(coord_panels))
                phi_star = objective(fisher_ref, oracle_infos, p_star, costs)
                rows = []
                for name in pending:
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
                        n_cond=int(cfg.get("gen_info_cond", 16)),
                        seed=int(entry["score_seed_root"]),
                        steps=int(cfg["scoring_steps"]),
                        max_step_norm=float(cfg["scoring_max_step_norm"]),
                    )
                    loss = r1.bregman_projection_loss(full_pool_phi, beta_hat, beta_dag)
                    rows.append({
                        "schema_version": "paper-result-v1",
                        "experiment": "gas_natural",
                        "environment": "gas",
                        "campaign": camp,
                        "method": name,
                        "policy": name,
                        "budget": int(budget),
                        "replication": int(rep),
                        "primary_metric": "family_projection_bregman",
                        "primary_loss": float(loss),
                        "projection_loss": float(loss),
                        "design_ratio_main": float(objective(fisher_ref, oracle_infos, probabilities, costs) / max(phi_star, 1e-12)),
                        "target_draw_sha256": target_hash,
                        "pilot_budget": int(pilot_counts.sum()),
                        "fw_gap": float(rr_gap) if name == "RR-GID" else None,
                        "code_commit": commit,
                        "config_sha256": config_sha,
                        "artifact_sha256": ckpt_sha,
                        "data_sha256": data_sha,
                        "data_path": str(args.data),
                        "config_path": str(args.config),
                        "checkpoint_path": str(ckpt_path),
                        "seed_manifest_sha256": seed_manifest_sha,
                        "experiment_mode": cfg.get("experiment_mode"),
                        "provenance_schema": "gas-natural-v1-provenance",
                        "replication_seed": int(entry["replication_seed"]),
                        "target_draw_seed": int(entry["target_draw_seed"]),
                        "pilot_or_design_seed": int(entry["pilot_or_design_seed"]),
                        "score_seed_root": int(entry["score_seed_root"]),
                        "information_seed_root": int(entry["information_seed_root"]),
                        "device": device,
                        "gpu": runtime["gpu"],
                        "pytorch": runtime["pytorch"],
                        "cuda_runtime": runtime["cuda_runtime"],
                        "score_backend": cfg.get("score_backend"),
                        "information_backend": cfg.get("information_backend"),
                        "score_qmc_order": int(cfg.get("score_qmc_order", 0)),
                        "information_qmc_order": int(cfg.get("information_qmc_order", 0)),
                        "allocated_observations": int(budget),
                    })
                r1._append_jsonl(out, rows)
                if args.profile:
                    for row in rows:
                        print(
                            f"{camp} {row['method']} B={budget} rep={rep} loss={row['projection_loss']:.4f}",
                            flush=True,
                        )
                done.update((camp, row["method"], int(budget), int(rep)) for row in rows)
    manifest_payload["status"] = "completed"
    manifest_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_payload["row_count"] = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip()) if out.exists() else 0
    _write_json(provenance_path, manifest_payload)
    _write_json(out.parent / "summary.json", {
        "stage": "P10", "experiment": "gas_natural_v1", "status": "completed",
        "row_count": manifest_payload["row_count"], "expected_rows": len(cfg["campaigns"]) * len(cfg["budgets"]) * replications * len(methods),
        "rows_path": str(out), "provenance_path": str(provenance_path), "config_sha256": config_sha,
    })
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
