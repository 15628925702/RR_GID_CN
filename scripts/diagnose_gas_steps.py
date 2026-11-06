from pathlib import Path
import os, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import run_gas_semisynthetic as r1
from rr_gid_cn.empirical_generator import EmpiricalKNNGenerator
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.policies import uniform_probabilities
from rr_gid_cn.s1_gate import largest_remainder_counts


def run_budget(budget, steps_list):
    data = np.load(ROOT / "data/gas/processed/gas_processed.npz")
    ref_train = np.asarray(data["ref_train"], float)
    ref_val = np.asarray(data["ref_val"], float)
    fn = lambda x: transform_features(x, data["mean"], data["std"], data["pcs"])
    phi_val = fn(ref_val)
    beta_star = r1.calibrate_beta(phi_val, seed=2026)
    rng = np.random.default_rng(203009103 + budget)
    logits = phi_val @ beta_star
    w = np.exp(logits - logits.max()); w /= w.sum()
    target = ref_val[rng.choice(len(ref_val), size=budget, replace=True, p=w)]
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in panel_library())
    counts = largest_remainder_counts(uniform_probabilities(len(coord_panels)), budget)
    obs = []
    cursor = 0
    for panel, count in zip(coord_panels, counts):
        for row in target[cursor: cursor + int(count)]:
            obs.append((panel, row[list(panel)]))
        cursor += int(count)
    gen = EmpiricalKNNGenerator(ref_train, feature_fn=fn, neighbors=60)
    learned = {panel: np.eye(16) for panel in coord_panels}
    mu_full = fn(target).mean(0)
    full = r1.solve_pilot_beta(mu_full, phi_val, theta_bound=8.0, norm_cap=None)
    print("B", budget, "full-obs", float(np.linalg.norm(full)), r1.bregman_projection_loss(phi_val, full, beta_star), flush=True)
    for steps in steps_list:
        for cap in (None, 3.0):
            hat = r1.score_beta(
                gen, np.zeros(16), obs, coord_panels, phi_val, learned, fn,
                n_cond=32, seed=1, steps=steps, max_step_norm=0.5,
                theta_bound=4.0, norm_cap=cap,
            )
            print(
                "B", budget, "J", steps, "cap", cap,
                "bnorm", round(float(np.linalg.norm(hat)), 3),
                "D", round(r1.bregman_projection_loss(phi_val, hat, beta_star), 4),
                flush=True,
            )


if __name__ == "__main__":
    run_budget(400, [4, 8, 12])
    run_budget(3200, [4, 8, 12])
