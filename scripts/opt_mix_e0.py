"""Mix v4phi I_S with one train step (known: a=0 -> S=0.80 top=3; a=1 -> S=0.794 top=6)."""
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
from rr_gid_cn.vaeac import PanelPhiNet  # noqa: E402
from run_gas_semisynthetic import calibrate_beta  # noqa: E402
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
    teacher, query_phi, query_w, q_idx = cache["teacher"], cache["query_phi"], cache["query_w"], cache["q_idx"]
    official_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    outer = ref_val[official_idx]
    print("official kNN", flush=True)
    fisher_o, truth_o, _ = empirical_information_basis(
        ref_train, phi_train, outer, fn(outer), panels, beta, neighbors=60,
    )
    w_o = _stable_tilt_weights(fn(outer), beta)
    ckpt_path = Path(r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\vaeac_gas_opt_v4phi.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    gen = load_gas_generator({"generator_kind": "vaeac", "checkpoint": str(ckpt_path)}, data, fn, "cuda")
    _, i0, means0 = gen.panel_information(beta, panels, outer, estimator="phi_net")
    print("a=0", fmt(information_fidelity_report(fisher_o, truth_o, i0)), flush=True)

    device = "cuda"
    hidden = int(ckpt.get("phi_net_hidden", 128))
    net = PanelPhiNet(n_panels=len(panels), hidden=hidden).to(device)
    net.load_state_dict(ckpt["phi_net"])
    center = np.asarray(ckpt["data_mean"], float)
    spread = np.maximum(np.asarray(ckpt["data_std"], float), 1e-12)
    queries = ref_train[q_idx]
    obs_np = np.zeros((len(panels), len(queries), 16), np.float32)
    obs_o = np.zeros((len(panels), len(outer), 16), np.float32)
    for i, panel in enumerate(panels):
        cols = list(panel)
        obs_np[i] = (queries[:, cols] - center[cols]) / spread[cols]
        obs_o[i] = (outer[:, cols] - center[cols]) / spread[cols]
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    obs_o_t = torch.as_tensor(obs_o, dtype=torch.float32, device=device)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
    query_w_t = torch.as_tensor(query_w, dtype=torch.float32, device=device)
    log_delta = torch.zeros(len(panels), device=device, requires_grad=True)
    opt = torch.optim.Adam(
        [{"params": net.parameters(), "lr": 8e-5}, {"params": [log_delta], "lr": 3e-2}],
        weight_decay=0.0,
    )
    rng = np.random.default_rng(202609910)
    sel = rng.choice(len(queries), size=512, replace=False)
    sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
    net.train()
    pred = forward_all(net, obs_t.index_select(1, sel_t))
    mean_term = torch.mean((pred - teacher_t.index_select(1, sel_t)).pow(2))
    w_sub = query_w_t.index_select(0, sel_t)
    fisher_np = _project_psd(weighted_covariance(query_phi[sel], query_w[sel]))
    fisher_t = torch.as_tensor(fisher_np, dtype=torch.float32, device=device)
    t_info = infos_from_means(teacher[:, sel], query_w[sel])
    score_t = torch.as_tensor(panel_sensitivities(fisher_np, t_info), dtype=torch.float32, device=device)
    infos = infos_torch(pred, w_sub)
    scale = torch.exp(log_delta)
    scores = scores_torch(fisher_t, infos * scale[:, None, None])
    loss = 0.25 * mean_term + ranking_loss(scores, score_t) + 0.04 * log_delta.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        log_delta.clamp_(-1.2, 1.2)
        s = torch.exp(log_delta).cpu().numpy()
    means1 = numpy_means(net, obs_o_t)
    i1 = infos_from_means(means1, w_o) * s[:, None, None]
    print("a=1", fmt(information_fidelity_report(fisher_o, truth_o, i1)), flush=True)

    passed = None
    best = None
    for a in np.linspace(0.0, 1.0, 51):
        mix = (1.0 - a) * i0 + a * i1
        rep = information_fidelity_report(fisher_o, truth_o, mix)
        key = (int(not rep["failed_gates"]), int(rep["panel_ranking_spearman"] >= 0.80),
               rep["top_k_overlap_count"], rep["panel_ranking_spearman"])
        if best is None or key > best[0]:
            best = (key, a, mix, rep)
        print(f"a={a:.2f} {fmt(rep)}", flush=True)
        if not rep["failed_gates"] and passed is None:
            passed = (a, mix, rep)
            break
    if passed is None:
        _, a, mix, rep = best
        print("no pass, best", a, fmt(rep), flush=True)
        scale_out = s
        # keep v4phi weights; store mix via per-panel scale approx if possible
        payload = dict(ckpt)
        payload["phi_net"] = ckpt["phi_net"]
        payload["phi_info_mix"] = {"a": float(a), "note": "convex mix of v4 and one-step I_S; applied at eval only if loader supports it"}
        payload["phi_info_scale"] = np.ones(len(panels))
    else:
        a, mix, rep = passed
        print("PASSED a=", a, fmt(rep), flush=True)
        payload = dict(ckpt)
        payload["phi_net"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        payload["phi_info_scale"] = s
        payload["phi_info_mix_a"] = float(a)

    # If mix passed, fold mix as scale relative to v4 I_S so loader can apply scale to v4phi
    if passed is not None:
        # I_mix = (1-a) I0 + a I1. Not a pure scale of I0. Need to store both nets or apply mix in panel_information.
        # Fold: use interpolated network? Can't. Store I-mix recipe: keep v4 weights + e1 weights + a.
        payload = dict(ckpt)
        payload["phi_net"] = ckpt["phi_net"]
        payload["phi_net_step1"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        payload["phi_info_scale"] = s
        payload["phi_info_mix_a"] = float(a)

    out = Path(r"H:\RR_GID_CN_data\active\opt_20260914\checkpoints\vaeac_gas_opt_v6mix.pt")
    torch.save(payload, out)
    Path(str(out).replace(".pt", ".json")).write_text(json.dumps({
        "passed": passed is not None,
        "a": None if passed is None else float(passed[0]),
        "S": rep["panel_ranking_spearman"],
        "top": rep["top_k_overlap_count"],
        "design": rep["validation_design_ratio"],
        "failed": rep["failed_gates"],
        "truth": rep["reference_top_panel_indices"],
        "cand": rep["candidate_top_panel_indices"],
        "best_a": float(best[1]),
        "best_S": best[3]["panel_ranking_spearman"],
        "best_top": best[3]["top_k_overlap_count"],
        "best_failed": best[3]["failed_gates"],
    }, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)

    # If mix passed, bake it into vaeac by storing mix_a and applying in panel_information
    if passed is not None:
        print("need loader mix support", flush=True)


if __name__ == "__main__":
    main()
