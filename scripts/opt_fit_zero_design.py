"""Fit beta=0 mix scales on several TRAIN outers so I_S(0) copies kNN designs, then check official gate."""
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
from rr_gid_cn.policies import frank_wolfe, uniform_probabilities  # noqa: E402
from train_gas_phi_net_v6gate import ranking_loss, scores_torch  # noqa: E402


def main() -> None:
    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train = np.asarray(data["ref_train"], float)
    ref_val = np.asarray(data["ref_val"], float)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in panel_library())
    phi_train = fn(ref_train)
    gen = load_gas_generator(
        {"generator_kind": "vaeac", "checkpoint": r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix.pt"},
        data, fn, "cuda",
    )
    batches = []
    for seed in (11, 22, 33):
        idx = np.random.default_rng(202609800 + seed).choice(len(ref_train), size=256, replace=False)
        outer = ref_train[idx]
        print("knn train outer", seed, flush=True)
        fisher, truth, _ = empirical_information_basis(
            ref_train, phi_train, outer, fn(outer), panels, np.zeros(16), neighbors=60,
        )
        _, mix, _ = gen.panel_information(np.zeros(16), panels, outer, estimator="phi_net")
        batches.append((fisher, truth, mix, panel_sensitivities(fisher, truth)))
    off_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    outer_o = ref_val[off_idx]
    print("knn official zero", flush=True)
    f0, t0, _ = empirical_information_basis(ref_train, phi_train, outer_o, fn(outer_o), panels, np.zeros(16), neighbors=60)
    _, mix0, _ = gen.panel_information(np.zeros(16), panels, outer_o, estimator="phi_net")

    device = "cuda"
    log_delta = torch.zeros(len(panels), device=device, requires_grad=True)
    opt = torch.optim.Adam([log_delta], lr=4e-2)
    packed = [
        (
            torch.as_tensor(f, dtype=torch.float32, device=device),
            torch.as_tensor(m, dtype=torch.float32, device=device),
            torch.as_tensor(s, dtype=torch.float32, device=device),
        )
        for f, _, m, s in batches
    ]
    best = None
    for step in range(250):
        opt.zero_grad()
        scale = torch.exp(log_delta.clamp(-2.0, 2.0))
        loss = 0.01 * log_delta.pow(2).mean()
        for fisher, mix, score_t in packed:
            infos = mix * scale[:, None, None]
            scores = scores_torch(fisher, infos)
            loss = loss + ranking_loss(scores, score_t)
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_delta.clamp_(-2.0, 2.0)
        if step % 25 == 0 or step == 249:
            s = torch.exp(log_delta.detach()).cpu().numpy()
            cand = np.stack([_project_psd(m * float(f)) for m, f in zip(mix0, s)])
            rep = information_fidelity_report(f0, t0, cand)
            uni = uniform_probabilities(len(panels))
            p_k, _, _ = frank_wolfe(f0, t0, np.ones(len(panels)), uni, tolerance=1e-4, max_iter=300)
            p_s, _, _ = frank_wolfe(f0, cand, np.ones(len(panels)), uni, tolerance=1e-4, max_iter=300)
            tv = 0.5 * float(np.abs(p_k - p_s).sum())
            print(
                f"step {step} {rep['panel_ranking_spearman']:.3f} top={rep['top_k_overlap_count']} "
                f"d={rep['validation_design_ratio']:.3f} tv={tv:.3f} fail={rep['failed_gates']}",
                flush=True,
            )
            key = (int(not rep["failed_gates"]), -tv, rep["top_k_overlap_count"], rep["panel_ranking_spearman"])
            if best is None or key > best[0]:
                best = (key, s.copy(), rep, tv, step)
                if not rep["failed_gates"] and tv < 0.25:
                    break
    s = best[1]
    payload = torch.load(
        r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix.pt",
        map_location="cpu", weights_only=False,
    )
    payload["phi_info_scale_zero"] = s
    out = Path(r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix_zdes.pt")
    torch.save(payload, out)
    Path(str(out).replace(".pt", ".json")).write_text(
        json.dumps({"step": best[4], "tv": best[3], "zero": {
            k: best[2][k] for k in ("panel_ranking_spearman", "top_k_overlap_count", "validation_design_ratio", "failed_gates")
        }}, indent=2, default=float),
        encoding="utf-8",
    )
    print("wrote", out, "tv", best[3], "fail", best[2]["failed_gates"], flush=True)


if __name__ == "__main__":
    main()
