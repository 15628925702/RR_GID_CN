"""Why v6mix R2 fails on batches 8-9: I_S(0) vs tilted gate, vs kNN designs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rr_gid_cn.empirical_generator import (  # noqa: E402
    empirical_information_basis,
    load_gas_generator,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import information_fidelity_report, panel_sensitivities  # noqa: E402
from rr_gid_cn.policies import frank_wolfe, uniform_probabilities  # noqa: E402
from run_gas_semisynthetic import calibrate_beta  # noqa: E402


def topk(scores, k=10):
    return np.argsort(-np.asarray(scores))[:k]


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def main() -> None:
    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train = np.asarray(data["ref_train"], float)
    ref_val = np.asarray(data["ref_val"], float)
    camp = np.asarray(data["x_batches89"], float)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    phi_train, phi_val = fn(ref_train), fn(ref_val)
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in panel_library())
    beta_t = calibrate_beta(phi_val, seed=2026, target_ess=0.5)
    official_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    outer = ref_val[official_idx]
    outer_phi = fn(outer)
    print("kNN I_S tilted and zero", flush=True)
    fisher_t, truth_t, _ = empirical_information_basis(ref_train, phi_train, outer, outer_phi, panels, beta_t, neighbors=60)
    fisher_0, truth_0, _ = empirical_information_basis(
        ref_train, phi_train, outer, outer_phi, panels, np.zeros(16), neighbors=60,
    )
    knn = load_gas_generator(
        {"generator_kind": "empirical_knn", "generator_neighbors": 60,
         "checkpoint": "paper_results/checkpoints/empirical_knn_gas_v1.json"},
        data, fn, "cpu",
    )
    mix = load_gas_generator(
        {"generator_kind": "vaeac",
         "checkpoint": r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix.pt"},
        data, fn, "cuda",
    )
    print("mix I_S", flush=True)
    _, mix_t, _ = mix.panel_information(beta_t, panels, outer, estimator="phi_net")
    _, mix_0, _ = mix.panel_information(np.zeros(16), panels, outer, estimator="phi_net")
    for name, fisher, truth, cand in (
        ("tilt", fisher_t, truth_t, mix_t),
        ("zero", fisher_0, truth_0, mix_0),
    ):
        rep = information_fidelity_report(fisher, truth, cand)
        print(
            name,
            f"S={rep['panel_ranking_spearman']:.3f}",
            f"top={rep['top_k_overlap_count']}/10",
            f"design={rep['validation_design_ratio']:.3f}",
            "failed", rep["failed_gates"],
            flush=True,
        )
    s_knn0 = panel_sensitivities(fisher_0, truth_0)
    s_mix0 = panel_sensitivities(fisher_0, mix_0)
    s_knn_t = panel_sensitivities(fisher_t, truth_t)
    print("top10 knn0", topk(s_knn0).tolist())
    print("top10 mix0", topk(s_mix0).tolist())
    print("top10 knnt", topk(s_knn_t).tolist())
    print("overlap knn0 vs knnt", len(set(topk(s_knn0)) & set(topk(s_knn_t))))
    print("overlap mix0 vs knn0", len(set(topk(s_mix0)) & set(topk(s_knn0))))

    uni = uniform_probabilities(len(panels))
    p_knn, _, _ = frank_wolfe(fisher_0, truth_0, np.ones(len(panels)), uni, tolerance=1e-4, max_iter=300)
    p_mix, _, _ = frank_wolfe(fisher_0, mix_0, np.ones(len(panels)), uni, tolerance=1e-4, max_iter=300)
    p_mix_fmix, _, _ = frank_wolfe(
        np.cov(outer_phi, rowvar=False) * 0 + mix.panel_information(np.zeros(16), panels, outer, estimator="phi_net")[0],
        mix_0, np.ones(len(panels)), uni, tolerance=1e-4, max_iter=300,
    )
    print("TV knn-design vs mix-design at 0, F_ref", tv(p_knn, p_mix))
    print("mass top10 knn", float(p_knn[topk(s_knn0)].sum()), "mix", float(p_mix[topk(s_knn0)].sum()))

    rng = np.random.default_rng(0)
    knn_s = knn.sample_full(2048, seed=0)
    mix_s = mix.sample_full(2048, seed=0)
    camp_s = camp[rng.choice(len(camp), size=2048, replace=True)]
    from p10_r2_formal import c2st_auc

    print("C2ST knn vs camp89", c2st_auc(knn_s, camp_s, rng_seed=0))
    print("C2ST mix vs camp89", c2st_auc(mix_s, camp_s, rng_seed=0))
    print("C2ST knn vs ref_val", c2st_auc(knn_s, ref_val[:2048] if len(ref_val) >= 2048 else ref_val, rng_seed=0))
    print("C2ST mix vs ref_val", c2st_auc(mix_s, ref_val, rng_seed=0))
    Path("results/opt/diag_r2_v6mix.json").write_text(
        json.dumps(
            {
                "tilt": information_fidelity_report(fisher_t, truth_t, mix_t),
                "zero": information_fidelity_report(fisher_0, truth_0, mix_0),
                "tv_design_zero": tv(p_knn, p_mix),
            },
            default=float,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
