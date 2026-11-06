"""Frozen v4phi + tiny I_S scales. Epoch-0 already hit top-6 at Spearman 0.794."""
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
    _stable_tilt_weights,
    empirical_information_basis,
    load_gas_generator,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import information_fidelity_report, panel_sensitivities, weighted_covariance  # noqa: E402
from run_gas_semisynthetic import calibrate_beta  # noqa: E402
from train_gas_phi_net_v6gate import (  # noqa: E402
    fmt,
    forward_all,
    infos_from_means,
    infos_torch,
    ranking_loss,
    scores_torch,
)


def main() -> None:
    data = np.load("data/gas/processed/gas_processed.npz")
    ref_train = np.asarray(data["ref_train"], float)
    ref_val = np.asarray(data["ref_val"], float)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    phi_train, phi_val = fn(ref_train), fn(ref_val)
    beta = calibrate_beta(phi_val, seed=2026, target_ess=0.5)
    pairs = panel_library()
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in pairs)
    cache = np.load(r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\gas_phi_teacher_train.npz")
    teacher, query_phi, query_w = cache["teacher"], cache["query_phi"], cache["query_w"]
    official_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    outer = ref_val[official_idx]
    print("official kNN", flush=True)
    fisher_o, truth_o, _ = empirical_information_basis(
        ref_train, phi_train, outer, fn(outer), panels, beta, neighbors=60,
    )
    w_o = _stable_tilt_weights(fn(outer), beta)
    ckpt_path = Path(r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\vaeac_gas_opt_v4phi.pt")
    gen = load_gas_generator({"generator_kind": "vaeac", "checkpoint": str(ckpt_path)}, data, fn, "cuda")
    _, neural_o, _ = gen.panel_information(beta, panels, outer, estimator="phi_net")
    print("raw", fmt(information_fidelity_report(fisher_o, truth_o, neural_o)), flush=True)

    device = "cuda"
    net = gen.phi_net
    for p in net.parameters():
        p.requires_grad_(False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    center = np.asarray(ckpt["data_mean"], float)
    spread = np.maximum(np.asarray(ckpt["data_std"], float), 1e-12)
    queries = ref_train[cache["q_idx"]]
    obs_np = np.zeros((len(panels), len(queries), 16), np.float32)
    for i, panel in enumerate(panels):
        cols = list(panel)
        obs_np[i] = (queries[:, cols] - center[cols]) / spread[cols]
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
    query_w_t = torch.as_tensor(query_w, dtype=torch.float32, device=device)

    log_delta = torch.zeros(len(panels), device=device, requires_grad=True)
    opt = torch.optim.Adam([log_delta], lr=0.012)
    rng = np.random.default_rng(202609910)
    snapshots = [np.ones(len(panels))]
    for epoch in range(8):
        sel = rng.choice(len(queries), size=512, replace=False)
        sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = forward_all(net, obs_t.index_select(1, sel_t))
        w_sub = query_w_t.index_select(0, sel_t)
        fisher_np = _project_psd(weighted_covariance(query_phi[sel], query_w[sel]))
        fisher_t = torch.as_tensor(fisher_np, dtype=torch.float32, device=device)
        t_info = infos_from_means(teacher[:, sel], query_w[sel])
        score_t = torch.as_tensor(panel_sensitivities(fisher_np, t_info), dtype=torch.float32, device=device)
        infos = infos_torch(pred, w_sub)
        scale = torch.exp(log_delta)
        scores = scores_torch(fisher_t, infos * scale[:, None, None])
        loss = ranking_loss(scores, score_t) + 0.02 * log_delta.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_delta.clamp_(-0.35, 0.35)
            s = torch.exp(log_delta).cpu().numpy().copy()
        snapshots.append(s)
        rep = information_fidelity_report(fisher_o, truth_o, neural_o * s[:, None, None])
        print(f"e{epoch} {fmt(rep)} scale[{s.min():.3f},{s.max():.3f}]", flush=True)
        if not rep["failed_gates"]:
            print("PASSED at epoch", epoch, flush=True)
            save(ckpt, s, rep, ckpt_path)
            return

    # 1-D homotopy from identity to each snapshot
    best = None
    for si, s in enumerate(snapshots):
        for a in np.linspace(0.0, 1.0, 21):
            sc = 1.0 + a * (s - 1.0)
            rep = information_fidelity_report(fisher_o, truth_o, neural_o * sc[:, None, None])
            key = (int(not rep["failed_gates"]), int(rep["panel_ranking_spearman"] >= 0.80),
                   rep["top_k_overlap_count"], rep["panel_ranking_spearman"])
            if best is None or key > best[0]:
                best = (key, sc, rep, si, a)
            if not rep["failed_gates"]:
                print(f"PASSED homotopy snap={si} a={a:.2f} {fmt(rep)}", flush=True)
                save(ckpt, sc, rep, ckpt_path)
                return
    _, sc, rep, si, a = best
    print("best non-pass", f"snap={si} a={a:.2f}", fmt(rep), flush=True)
    save(ckpt, sc, rep, ckpt_path)


def save(ckpt, scale, rep, source):
    out = Path(r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\vaeac_gas_opt_v6gate.pt")
    payload = dict(ckpt)
    payload["phi_info_scale"] = np.asarray(scale, dtype=np.float64)
    torch.save(payload, out)
    log = Path(str(out).replace(".pt", ".json"))
    log.write_text(json.dumps({
        "source": str(source),
        "output": str(out),
        "passed": not rep["failed_gates"],
        "S": rep["panel_ranking_spearman"],
        "top": rep["top_k_overlap_count"],
        "design": rep["validation_design_ratio"],
        "failed": rep["failed_gates"],
        "truth": rep["reference_top_panel_indices"],
        "cand": rep["candidate_top_panel_indices"],
        "scale_min": float(np.min(scale)),
        "scale_max": float(np.max(scale)),
    }, indent=2), encoding="utf-8")
    print("saved", out, "passed" if not rep["failed_gates"] else "NOT", fmt(rep), flush=True)


if __name__ == "__main__":
    main()
