"""Fast Gas ranking probe: kNN I_S vs VAEAC prior-decode I_S on the same outer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rr_gid_cn.empirical_generator import empirical_information_basis
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.generator_validation import information_fidelity_report
from rr_gid_cn.vaeac import load_vaeac_checkpoint, VAEACGenerator
from run_gas_semisynthetic import calibrate_beta


def prior_decode_phi(gen, outer_x, panel, fn):
    center = np.asarray(gen.model.data_mean, dtype=np.float64)
    spread = np.asarray(gen.model.data_std, dtype=np.float64)
    cols = list(panel)
    obs = torch.zeros(len(outer_x), gen.dimension, device=gen.device)
    mask = torch.zeros_like(obs)
    scaled = (np.asarray(outer_x[:, cols], dtype=np.float64) - center[cols]) / np.maximum(spread[cols], 1e-12)
    obs[:, cols] = torch.as_tensor(scaled, dtype=torch.float32, device=gen.device)
    mask[:, cols] = 1.0
    with torch.no_grad():
        p_mu, _ = gen.model.prior(obs, mask)
        recon, _ = gen.model.decode_params(p_mu, mask, obs)
        mixed = (mask * obs + (1.0 - mask) * recon).cpu().numpy()
    raw = mixed * spread + center
    raw[:, cols] = outer_x[:, cols]
    return fn(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--n-outer", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    data = np.load(args.data)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    donor_x = np.asarray(data["ref_train"], dtype=float)
    donor_phi = fn(donor_x)
    val_x = np.asarray(data["ref_val"], dtype=float)
    val_phi = fn(val_x)
    beta = calibrate_beta(val_phi, seed=2026, target_ess=0.5)
    rng = np.random.default_rng(202609700)
    n_outer = min(len(val_x), int(args.n_outer))
    idx = rng.choice(len(val_x), size=n_outer, replace=False)
    outer_x = val_x[idx]
    outer_phi = val_phi[idx]
    pairs = panel_library()
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in pairs)
    fisher, truth, _ = empirical_information_basis(
        donor_x, donor_phi, outer_x, outer_phi, panels, beta, neighbors=60,
    )
    model, _ = load_vaeac_checkpoint(str(args.checkpoint), device=args.device, expected_dim=128)
    gen = VAEACGenerator(model, np.ones(128), alpha=0.0, device=args.device, feature_fn=fn)
    from rr_gid_cn.empirical_generator import (
        _project_psd,
        _stable_tilt_weights,
        empirical_conditional_means,
    )
    from rr_gid_cn.generator_validation import weighted_covariance

    outer_w = _stable_tilt_weights(outer_phi, beta)
    donor_w = _stable_tilt_weights(donor_phi, beta)
    infos = []
    rmses = []
    for i, panel in enumerate(panels):
        cond = prior_decode_phi(gen, outer_x, panel, fn)
        infos.append(_project_psd(weighted_covariance(cond, outer_w)))
        if i < 8:
            teacher = empirical_conditional_means(
                donor_x, donor_phi, outer_x[:, list(panel)], panel,
                donor_tilt_weights=donor_w, neighbors=60,
            )
            rmses.append(float(np.sqrt(np.mean((cond - teacher) ** 2))))
        if (i + 1) % 30 == 0:
            print(f"prior-decode panels {i+1}/{len(panels)}", flush=True)
    candidate = np.asarray(infos)
    report = information_fidelity_report(fisher, truth, candidate)
    fake = gen.sample_full(min(1187, len(val_x)), 0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([val_x[: len(fake)], fake]).astype(np.float32)
    y = np.concatenate([np.ones(len(fake)), np.zeros(len(fake))])
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    aucs = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in splitter.split(x, y):
        clf = LogisticRegression(max_iter=5000)
        clf.fit(x[tr], y[tr])
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(x[te])[:, 1])))
    payload = {
        "checkpoint": str(args.checkpoint),
        "n_outer": n_outer,
        "panel_ranking_spearman": report["panel_ranking_spearman"],
        "top_k_overlap_count": report["top_k_overlap_count"],
        "validation_design_ratio": report["validation_design_ratio"],
        "prior_decode_knn_rmse_first8": float(np.mean(rmses)) if rmses else None,
        "c2st_auc_ref_val": float(np.mean(aucs)),
        "beta_norm": float(np.linalg.norm(beta)),
        "failed_gates": report["failed_gates"],
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
