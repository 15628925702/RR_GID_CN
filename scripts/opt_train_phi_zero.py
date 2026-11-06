"""Train a beta=0-only phi-net to copy untilted kNN E[phi|x_S] and I_S ranking."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rr_gid_cn.empirical_generator import (  # noqa: E402
    _stable_tilt_weights,
    empirical_conditional_means,
    empirical_information_basis,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import information_fidelity_report  # noqa: E402
from rr_gid_cn.vaeac import PanelPhiNet  # noqa: E402
from train_gas_phi_net_v6gate import (  # noqa: E402
    fmt,
    forward_all,
    infos_from_means,
    infos_torch,
    numpy_means,
    ranking_loss,
    scores_torch,
)


def main() -> None:
    t0 = time.time()
    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train = np.asarray(data["ref_train"], float)
    ref_val = np.asarray(data["ref_val"], float)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    phi_train = fn(ref_train)
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in panel_library())
    cache = np.load(r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\gas_phi_teacher_train.npz")
    q_idx = cache["q_idx"]
    queries = ref_train[q_idx]
    w0 = _stable_tilt_weights(phi_train, np.zeros(16))
    teacher_path = Path(r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\gas_phi_teacher_zero.npz")
    if teacher_path.exists():
        teacher = np.load(teacher_path)["teacher"]
        print("loaded zero teacher", teacher.shape, flush=True)
    else:
        print("building untilted kNN teacher", flush=True)
        rows = []
        for i, panel in enumerate(panels):
            rows.append(
                empirical_conditional_means(
                    ref_train, phi_train, queries[:, list(panel)], panel,
                    donor_tilt_weights=w0, neighbors=60,
                )
            )
            if (i + 1) % 20 == 0:
                print("  panel", i + 1, flush=True)
        teacher = np.stack(rows)
        np.savez(teacher_path, teacher=teacher, q_idx=q_idx)
        print("wrote", teacher_path, flush=True)

    off_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    outer_o = ref_val[off_idx]
    print("official kNN zero", flush=True)
    f0, t0, _ = empirical_information_basis(
        ref_train, phi_train, outer_o, fn(outer_o), panels, np.zeros(16), neighbors=60,
    )
    w_o = _stable_tilt_weights(fn(outer_o), np.zeros(16))

    ckpt = torch.load(
        r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix.pt",
        map_location="cpu", weights_only=False,
    )
    center = np.asarray(ckpt["data_mean"], float)
    spread = np.maximum(np.asarray(ckpt["data_std"], float), 1e-12)
    hidden = int(ckpt.get("phi_net_hidden", 128))
    device = "cuda"
    net = PanelPhiNet(n_panels=len(panels), hidden=hidden).to(device)
    net.load_state_dict(ckpt["phi_net"])
    obs = np.zeros((len(panels), len(queries), 16), np.float32)
    obs_o = np.zeros((len(panels), len(outer_o), 16), np.float32)
    for i, panel in enumerate(panels):
        cols = list(panel)
        obs[i] = (queries[:, cols] - center[cols]) / spread[cols]
        obs_o[i] = (outer_o[:, cols] - center[cols]) / spread[cols]
    obs_t = torch.as_tensor(obs, device=device)
    obs_o_t = torch.as_tensor(obs_o, device=device)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
    w_q = torch.as_tensor(_stable_tilt_weights(fn(queries), np.zeros(16)), dtype=torch.float32, device=device)
    fisher_np = f0 * 0 + 0  # placeholder
    from rr_gid_cn.empirical_generator import _project_psd
    from rr_gid_cn.generator_validation import weighted_covariance, panel_sensitivities
    fisher_q = _project_psd(weighted_covariance(fn(queries), _stable_tilt_weights(fn(queries), np.zeros(16))))
    t_info = infos_from_means(teacher, _stable_tilt_weights(fn(queries), np.zeros(16)))
    score_t = torch.as_tensor(panel_sensitivities(fisher_q, t_info), dtype=torch.float32, device=device)
    fisher_t = torch.as_tensor(fisher_q, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    best = None
    rng = np.random.default_rng(202609920)
    for epoch in range(80):
        net.train()
        sel = rng.choice(len(queries), size=256, replace=False)
        sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
        pred = forward_all(net, obs_t.index_select(1, sel_t))
        mean_term = torch.mean((pred - teacher_t.index_select(1, sel_t)).pow(2))
        infos = infos_torch(pred, w_q.index_select(0, sel_t))
        # ranking on the subsample is noisy; keep mean-term dominant
        loss = mean_term
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == 79:
            means = numpy_means(net, obs_o_t)
            cand = infos_from_means(means, w_o)
            rep = information_fidelity_report(f0, t0, cand)
            print(f"epoch {epoch} mse={float(mean_term):.4f} {fmt(rep)}", flush=True)
            key = (int(not rep["failed_gates"]), rep["top_k_overlap_count"], rep["panel_ranking_spearman"])
            if best is None or key > best[0]:
                best = (key, {k: v.detach().cpu() for k, v in net.state_dict().items()}, rep, epoch)
                if not rep["failed_gates"]:
                    break
    payload = dict(ckpt)
    payload["phi_net_zero"] = best[1]
    out = Path(r"H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6mix_zero.pt")
    torch.save(payload, out)
    Path(str(out).replace(".pt", ".json")).write_text(
        json.dumps({"epoch": best[3], "zero": {
            k: best[2][k] for k in ("panel_ranking_spearman", "top_k_overlap_count", "validation_design_ratio", "failed_gates")
        }, "seconds": time.time() - t0}, indent=2, default=float),
        encoding="utf-8",
    )
    print("wrote", out, "best epoch", best[3], fmt(best[2]), flush=True)


if __name__ == "__main__":
    main()
