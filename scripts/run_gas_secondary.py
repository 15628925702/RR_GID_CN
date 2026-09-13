"""Once-per-campaign secondary checks at max Gas budget (RMSE + C2ST)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run_gas_natural as r2
import run_gas_semisynthetic as r1
from p10_r2_formal import c2st_auc, fit_pc2, heldout_moment_rmse
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.p4_integrity import compute_pilot_budget
from rr_gid_cn.policies import frank_wolfe, uniform_probabilities
from rr_gid_cn.s1_gate import discriminative_design, largest_remainder_counts
from rr_gid_cn.vaeac import VAEACGenerator, learned_information, load_vaeac_checkpoint


def main() -> None:
    cfg = yaml.safe_load(Path("configs/paper/gas_natural.yaml").read_text(encoding="utf-8"))
    data = np.load("data/gas/processed/gas_processed.npz")
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    pcs2 = fit_pc2(data["ref_train"], mean, std)
    fn = lambda x: transform_features(x, mean, std, pcs)
    sensor_pairs = panel_library()
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_vaeac_checkpoint(cfg["generator_checkpoint"], device=device, expected_dim=128)
    gen = VAEACGenerator(model, ckpt.get("scale", np.ones(128)), alpha=0.0, device=device, feature_fn=fn)
    full_pool_x = gen.sample_full(8000, seed=42)
    full_pool_phi = fn(full_pool_x)
    phi_ref = fn(data["ref_val"])
    fisher_ref = np.cov(phi_ref, rowvar=False)
    a_info = r1.gas_a_optimal_information(phi_ref, sensor_pairs)
    budget = max(int(b) for b in cfg["budgets"])
    out_rows = []
    for camp in cfg["campaigns"]:
        x_key, phi_key = r2.CAMPAIGN_KEYS[camp]
        campaign_x, campaign_phi = data[x_key], data[phi_key]
        entry = r2.seed_entry(camp, budget, 0)
        rng = np.random.default_rng(int(entry["replication_seed"]))
        perm = rng.permutation(len(campaign_x))
        half = len(campaign_x) // 2
        pool_x = campaign_x[perm[:half]]
        test_x = campaign_x[perm[half:]]
        test_phi = campaign_phi[perm[half:]]
        beta_dag = r2.fit_beta_dag(full_pool_phi, test_phi.mean(0))
        rng_t = np.random.default_rng(int(entry["target_draw_seed"]))
        target = pool_x[rng_t.choice(len(pool_x), size=int(budget), replace=True)]
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
        _, learned_stack = learned_information(
            gen, beta_tilde, coord_panels,
            n_tilted=int(cfg.get("gen_info_tilted", 64)),
            n_conditional=int(cfg.get("gen_info_cond", 16)),
            seed=int(entry["information_seed_root"]), feature_fn=fn,
        )
        learned = {panel: learned_stack[i] for i, panel in enumerate(coord_panels)}
        rr_p, _, _ = frank_wolfe(
            fisher_ref, learned_stack, np.ones(len(coord_panels)),
            uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300,
        )
        a_p, _, _ = frank_wolfe(
            np.eye(16), a_info, np.ones(len(coord_panels)),
            uniform_probabilities(len(coord_panels)), tolerance=1e-4, max_iter=300,
        )
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
        print(f"{camp} B={budget} designs ready", flush=True)
        for name, probabilities in policy_p.items():
            counts = largest_remainder_counts(np.asarray(probabilities, dtype=float), remaining)
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
            gen_samples, acc, ess = gen.tilted_full_diagnostics(beta_hat, 2000, int(entry["score_seed_root"]) + 33)
            mean_rmse, std_rmse = heldout_moment_rmse(gen_samples, test_x, mean, std, pcs2)
            auc = c2st_auc(gen_samples, test_x, rng_seed=int(entry["replication_seed"]) % 10000)
            loss = r1.bregman_projection_loss(full_pool_phi, beta_hat, beta_dag)
            row = {
                "campaign": camp, "method": name, "budget": int(budget),
                "projection_loss": float(loss),
                "heldout_mean_rmse": float(mean_rmse),
                "heldout_std_rmse": float(std_rmse),
                "c2st_auc": float(auc),
                "generator_ess": float(ess),
                "unconditional_acceptance": float(acc),
            }
            out_rows.append(row)
            print(json.dumps(row), flush=True)
    out = Path("results/paper/gas_natural/secondary_max_budget.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"budget": budget, "rows": out_rows}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
