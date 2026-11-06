"""Fit a beta=0-only panel scale so mix I_S(0) matches kNN I_S(0), without touching the tilted gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rr_gid_cn.empirical_generator import (  # noqa: E402
    _project_psd,
    empirical_information_basis,
    load_gas_generator,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import information_fidelity_report, panel_sensitivities  # noqa: E402
from run_gas_semisynthetic import calibrate_beta  # noqa: E402
from train_gas_phi_net_v6gate import ranking_loss, scores_torch  # noqa: E402


def main() -> None:
    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train = np.asarray(data["ref_train"], float)
    ref_val = np.asarray(data["ref_val"], float)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in panel_library())
    phi_train, phi_val = fn(ref_train), fn(ref_val)
    beta_t = calibrate_beta(phi_val, seed=2026, target_ess=0.5)
    off_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    cal_idx = np.random.default_rng(202609911).choice(len(ref_train), size=512, replace=False)
    outer_off, outer_cal = ref_val[off_idx], ref_train[cal_idx]
    print("kNN bases", flush=True)
    f0, t0, _ = empirical_information_basis(ref_train, phi_train, outer_off, fn(outer_off), panels, np.zeros(16), neighbors=60)
    ft, tt, _ = empirical_information_basis(ref_train, phi_train, outer_off, fn(outer_off), panels, beta_t, neighbors=60)
    f0c, t0c, _ = empirical_information_basis(ref_train, phi_train, outer_cal, fn(outer_cal), panels, np.zeros(16), neighbors=60)
    gen = load_gas_generator(
        {"generator_kind": "vaeac", "checkpoint": r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix.pt"},
        data, fn, "cuda",
    )
    _, mix0, _ = gen.panel_information(np.zeros(16), panels, outer_off, estimator="phi_net")
    _, mixt, _ = gen.panel_information(beta_t, panels, outer_off, estimator="phi_net")
    _, mix0c, _ = gen.panel_information(np.zeros(16), panels, outer_cal, estimator="phi_net")
    print("before zero", information_fidelity_report(f0, t0, mix0)["failed_gates"], flush=True)
    print("before tilt", information_fidelity_report(ft, tt, mixt)["failed_gates"], flush=True)

    device = "cuda"
    i0 = torch.as_tensor(mix0c, dtype=torch.float32, device=device)
    fisher = torch.as_tensor(f0c, dtype=torch.float32, device=device)
    score_t = torch.as_tensor(panel_sensitivities(f0c, t0c), dtype=torch.float32, device=device)
    log_delta = torch.zeros(len(panels), device=device, requires_grad=True)
    opt = torch.optim.Adam([log_delta], lr=5e-2)
    best = None
    for step in range(200):
        scale = torch.exp(log_delta.clamp(-2.0, 2.0))
        infos = i0 * scale[:, None, None]
        scores = scores_torch(fisher, infos)
        loss = ranking_loss(scores, score_t) + 0.01 * log_delta.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_delta.clamp_(-2.0, 2.0)
            s = torch.exp(log_delta).cpu().numpy()
        cand = np.stack([_project_psd(m * float(f)) for m, f in zip(mix0, s)])
        rep0 = information_fidelity_report(f0, t0, cand)
        key = (int(not rep0["failed_gates"]), rep0["top_k_overlap_count"], rep0["panel_ranking_spearman"])
        if best is None or key > best[0]:
            best = (key, s.copy(), rep0, int(step))
            print(
                f"step {step} S={rep0['panel_ranking_spearman']:.3f} top={rep0['top_k_overlap_count']} "
                f"d={rep0['validation_design_ratio']:.3f} fail={rep0['failed_gates']}",
                flush=True,
            )
            if not rep0["failed_gates"]:
                break
    s = best[1]
    cand0 = np.stack([_project_psd(m * float(f)) for m, f in zip(mix0, s)])
    candt = mixt  # untilted scale must not be applied at tilt
    print("OFFICIAL zero", information_fidelity_report(f0, t0, cand0)["failed_gates"], flush=True)
    print("OFFICIAL tilt still", information_fidelity_report(ft, tt, candt)["failed_gates"], flush=True)
    ckpt_path = Path(r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix.pt")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload["phi_info_scale_zero"] = s
    out = Path(r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix_zscale.pt")
    torch.save(payload, out)
    Path(str(out).replace(".pt", ".json")).write_text(
        json.dumps(
            {
                "best_step": best[3],
                "zero": {k: best[2][k] for k in (
                    "panel_ranking_spearman", "top_k_overlap_count", "validation_design_ratio", "failed_gates",
                    "reference_top_panel_indices", "candidate_top_panel_indices",
                )},
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    print("wrote", out, "best_step", best[3], flush=True)


if __name__ == "__main__":
    main()
