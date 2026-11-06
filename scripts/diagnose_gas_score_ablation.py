"""One-replication scoring ablation for Gas R1. Not a paper grid."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def main() -> None:
    data = np.load(ROOT / "data/gas/processed/gas_processed.npz")
    ref_train = np.asarray(data["ref_train"], float)
    ref_val = np.asarray(data["ref_val"], float)
    fn = lambda x: transform_features(x, data["mean"], data["std"], data["pcs"])
    phi_val = fn(ref_val)
    beta_star = r1.calibrate_beta(phi_val, seed=2026)
    budget = 400
    rng = np.random.default_rng(203009103)
    logits = phi_val @ beta_star
    w = np.exp(logits - logits.max()); w /= w.sum()
    target = ref_val[rng.choice(len(ref_val), size=budget, replace=True, p=w)]
    sensor_pairs = panel_library()
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs)
    counts = largest_remainder_counts(uniform_probabilities(len(coord_panels)), budget)
    obs = []
    cursor = 0
    for panel, count in zip(coord_panels, counts):
        for row in target[cursor: cursor + int(count)]:
            obs.append((panel, row[list(panel)]))
        cursor += int(count)

    mu_full = fn(target).mean(0)
    full_mle = r1.solve_pilot_beta(mu_full, phi_val, theta_bound=8.0, norm_cap=None)
    print("full-obs MLE", float(np.linalg.norm(full_mle)), r1.bregman_projection_loss(phi_val, full_mle, beta_star))

    learned = {panel: np.eye(16) for panel in coord_panels}
    donors = {
        "ref_train": EmpiricalKNNGenerator(ref_train, feature_fn=fn, neighbors=60),
        "ref_val": EmpiricalKNNGenerator(ref_val, feature_fn=fn, neighbors=60),
    }
    for name, gen in donors.items():
        hat = r1.score_beta(
            gen, np.zeros(16), obs, coord_panels, phi_val, learned, fn,
            n_cond=32, seed=1, steps=12, max_step_norm=0.5,
            theta_bound=4.0, norm_cap=None,
        )
        print(name, "uncap", float(np.linalg.norm(hat)), r1.bregman_projection_loss(phi_val, hat, beta_star))
        hat2 = r1.score_beta(
            gen, np.zeros(16), obs, coord_panels, phi_val, learned, fn,
            n_cond=32, seed=1, steps=12, max_step_norm=0.5,
            theta_bound=4.0, norm_cap=2.0,
        )
        print(name, "cap2", float(np.linalg.norm(hat2)), r1.bregman_projection_loss(phi_val, hat2, beta_star))

    # Oracle completions: plug in the true full-row features as "conditional means".
    class OracleGen:
        kind = "oracle"
        neighbors = 60
        reference = ref_val

    grouped = {}
    truth = {}
    cursor = 0
    for panel, count in zip(coord_panels, counts):
        rows = target[cursor: cursor + int(count)]
        grouped[panel] = rows
        truth[panel] = fn(rows)
        cursor += int(count)

    beta = np.zeros(16)
    for _ in range(12):
        logits = phi_val @ beta
        ww = np.exp(logits - logits.max()); ww /= ww.sum()
        mu = ww @ phi_val
        u = np.zeros(16)
        h = 1e-2 * np.eye(16)
        n_obs = 0
        for panel, rows in grouped.items():
            means = truth[panel]
            u += (means - mu).sum(0)
            h += len(rows) * np.eye(16)
            n_obs += len(rows)
        raw = np.linalg.solve(0.5 * (h + h.T), u)
        nrm = float(np.linalg.norm(raw))
        if nrm > 0.5:
            raw *= 0.5 / nrm
        bar_m = mu + u / max(n_obs, 1)
        current = r1.log_partition(phi_val, beta) - float(beta @ bar_m)
        scale = 1.0
        acc = beta
        while scale > 1e-4:
            cand = r1._project(beta + scale * raw, bound=4.0, norm_cap=None)
            trial = r1.log_partition(phi_val, cand) - float(cand @ bar_m)
            if trial <= current + 1e-9:
                acc = cand
                break
            scale *= 0.5
        beta = acc
    print("oracle-full-phi", float(np.linalg.norm(beta)), r1.bregman_projection_loss(phi_val, beta, beta_star))


if __name__ == "__main__":
    main()
