"""Diagnose RISK-2 top-k failure and calibrate I_S without leaking the official outer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rr_gid_cn.empirical_generator import (
    empirical_information_basis,
    load_gas_generator,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.generator_validation import information_fidelity_report, panel_sensitivities
from rr_gid_cn.policies import uniform_probabilities
from run_gas_semisynthetic import calibrate_beta


def coord_panels():
    pairs = panel_library()
    return tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in pairs), pairs


def report_head(name, fisher, truth, cand, k=10):
    rep = information_fidelity_report(fisher, truth, cand)
    scores_t = panel_sensitivities(fisher, truth, uniform_probabilities(len(truth)))
    scores_c = panel_sensitivities(fisher, cand, uniform_probabilities(len(cand)))
    top_t = np.argsort(-scores_t, kind="stable")[:k]
    top_c = np.argsort(-scores_c, kind="stable")[:k]
    print(f"\n==== {name} ====")
    print(
        f"spearman={rep['panel_ranking_spearman']:.4f} "
        f"design={rep['validation_design_ratio']:.4f} "
        f"top{k}={rep['top_k_overlap_count']}/{k} "
        f"failed={rep['failed_gates']}"
    )
    print("truth top:", [int(i) for i in top_t])
    print("cand  top:", [int(i) for i in top_c])
    print("overlap:", sorted(set(map(int, top_t)).intersection(map(int, top_c))))
    print("i  pair     teacher  student  stud_rank  gap")
    ranks_c = np.argsort(np.argsort(-scores_c, kind="stable"), kind="stable")
    for i in top_t:
        a, b = panel_library()[int(i)]
        print(
            f"{int(i):3d} ({a:2d},{b:2d})  {scores_t[i]:8.4f} {scores_c[i]:8.4f}  "
            f"{int(ranks_c[i])+1:3d}  {scores_c[i]-scores_t[i]:+.4f}"
        )
    return rep, scores_t, scores_c


def scale_infos(infos, scales):
    return np.asarray(infos, dtype=float) * np.asarray(scales, dtype=float)[:, None, None]


def blend_infos(neural, knn, mix):
    mix = float(mix)
    return (1.0 - mix) * np.asarray(neural) + mix * np.asarray(knn)


def fit_scales(fisher, neural_infos, knn_infos, n_iter=80, lr=0.25):
    """Per-panel positive scales so sensitivities track the kNN teacher."""
    scales = np.ones(len(neural_infos), dtype=float)
    target = panel_sensitivities(fisher, knn_infos, uniform_probabilities(len(knn_infos)))
    target = target / max(float(np.std(target)), 1e-8)
    for _ in range(n_iter):
        cand = scale_infos(neural_infos, scales)
        scores = panel_sensitivities(fisher, cand, uniform_probabilities(len(cand)))
        scores = scores / max(float(np.std(scores)), 1e-8)
        # relative residual; keep scales positive
        scales *= np.exp(np.clip(lr * (target - scores), -0.3, 0.3))
        scales = np.clip(scales, 0.25, 4.0)
        scales /= np.median(scales)
    return scales


def main() -> None:
    data = np.load(ROOT / "data/gas/processed/gas_processed.npz")
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    donor_x = np.asarray(data["ref_train"], dtype=float)
    donor_phi = fn(donor_x)
    val_x = np.asarray(data["ref_val"], dtype=float)
    val_phi = fn(val_x)
    beta = calibrate_beta(val_phi, seed=2026, target_ess=0.5)
    panels, pairs = coord_panels()
    rng = np.random.default_rng(202609700)
    n_outer = 512
    official_idx = rng.choice(len(val_x), size=n_outer, replace=False)
    official_mask = np.zeros(len(val_x), dtype=bool)
    official_mask[official_idx] = True
    cal_pool = np.arange(len(val_x))[~official_mask]
    spec = {
        "generator_kind": "vaeac",
        "checkpoint": r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\vaeac_gas_opt_v4phi.pt",
    }
    gen = load_gas_generator(spec, data, fn, device="cuda")
    assert gen.phi_net is not None

    def knn_basis(outer):
        return empirical_information_basis(
            donor_x, donor_phi, outer, fn(outer), panels, beta, neighbors=60,
        )

    def neural_basis(outer):
        return gen.panel_information(beta, panels, outer, estimator="phi_net")

    fisher, truth, _ = knn_basis(val_x[official_idx])
    _, neural, _ = neural_basis(val_x[official_idx])
    report_head("official 512  raw phi-net", fisher, truth, neural)

    # kNN top-k stability on two disjoint val halves
    rng2 = np.random.default_rng(0)
    a = rng2.choice(len(val_x), size=512, replace=False)
    b = rng2.choice(np.setdiff1d(np.arange(len(val_x)), a), size=min(512, len(val_x) - 512), replace=False)
    _, ia, _ = knn_basis(val_x[a])
    _, ib, _ = knn_basis(val_x[b])
    fa = np.cov(val_phi[a], rowvar=False)
    sa = panel_sensitivities(fa, ia)
    sb = panel_sensitivities(np.cov(val_phi[b], rowvar=False), ib)
    topa = set(map(int, np.argsort(-sa)[:10]))
    topb = set(map(int, np.argsort(-sb)[:10]))
    print("\n==== kNN top-10 stability across two val 512s ====")
    print("overlap", len(topa & topb), "sets", sorted(topa), sorted(topb))

    # Fit scales on calibration val (disjoint from official 512)
    cal_n = min(512, len(cal_pool))
    cal_idx = np.random.default_rng(7).choice(cal_pool, size=cal_n, replace=False)
    fisher_c, knn_c, _ = knn_basis(val_x[cal_idx])
    _, neu_c, _ = neural_basis(val_x[cal_idx])
    scales = fit_scales(fisher_c, neu_c, knn_c)
    scaled = scale_infos(neural, scales)
    report_head("official 512  scaled phi-net (cal on held-out val)", fisher, truth, scaled)

    print("\n==== blend sweep on held-out cal, then official ====")
    rows = []
    for mix in (0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0):
        blended_c = blend_infos(neu_c, knn_c, mix)
        cal_rep = information_fidelity_report(fisher_c, knn_c, blended_c)
        blended_o = blend_infos(neural, truth, mix)
        off_rep = information_fidelity_report(fisher, truth, blended_o)
        row = {
            "mix": mix,
            "cal_spearman": cal_rep["panel_ranking_spearman"],
            "cal_topk": cal_rep["top_k_overlap_count"],
            "cal_design": cal_rep["validation_design_ratio"],
            "cal_failed": cal_rep["failed_gates"],
            "off_spearman": off_rep["panel_ranking_spearman"],
            "off_topk": off_rep["top_k_overlap_count"],
            "off_design": off_rep["validation_design_ratio"],
            "off_failed": off_rep["failed_gates"],
        }
        rows.append(row)
        print(
            f"mix={mix:.2f}  cal S={row['cal_spearman']:.3f} top={row['cal_topk']} {row['cal_failed']}  "
            f"| off S={row['off_spearman']:.3f} top={row['off_topk']} {row['off_failed']}"
        )

    passing = [r for r in rows if not r["cal_failed"]]
    chosen = min(passing, key=lambda r: r["mix"]) if passing else rows[-1]
    print("\nchosen mix from cal-pass (else kernel):", chosen)

    scaled_blend_rows = []
    for mix in (0.0, 0.15, 0.25, 0.35, 0.5):
        cand = blend_infos(scale_infos(neural, scales), truth, mix)
        cand_c = blend_infos(scale_infos(neu_c, scales), knn_c, mix)
        cal_rep = information_fidelity_report(fisher_c, knn_c, cand_c)
        off_rep = information_fidelity_report(fisher, truth, cand)
        scaled_blend_rows.append(
            {
                "mix": mix,
                "cal_failed": cal_rep["failed_gates"],
                "off_failed": off_rep["failed_gates"],
                "off_spearman": off_rep["panel_ranking_spearman"],
                "off_topk": off_rep["top_k_overlap_count"],
                "off_design": off_rep["validation_design_ratio"],
            }
        )
        print(
            f"scale+mix={mix:.2f} cal {cal_rep['failed_gates']} | "
            f"off S={off_rep['panel_ranking_spearman']:.3f} top={off_rep['top_k_overlap_count']} {off_rep['failed_gates']}"
        )

    out = {
        "official_raw_failed": information_fidelity_report(fisher, truth, neural)["failed_gates"],
        "blend_rows": rows,
        "scaled_blend_rows": scaled_blend_rows,
        "chosen": chosen,
        "scales_median": float(np.median(scales)),
        "scales_max": float(np.max(scales)),
        "scales_min": float(np.min(scales)),
        "knn_topk_stability": int(len(topa & topb)),
    }
    path = Path(r"H:\RR_GID_CN_data\active\opt_20260914\risk_gates\diag_topk_cal.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
