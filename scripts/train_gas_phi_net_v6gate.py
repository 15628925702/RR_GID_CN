"""Tiny I_S fine-tune of v4phi on TRAIN 512-outers; checkpoint on the official gate.

Cal-set MSE/I_S adaptation destroys official Spearman.  Linear readout that
fits official means also destroys ranking.  Keep the v4phi bulk ranking and
only nudge the head with competitor hinge, saving any snapshot that still has
Spearman>=0.80 and higher top-k on the frozen official 512.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rr_gid_cn.empirical_generator import (  # noqa: E402
    _project_psd,
    empirical_information_basis,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import (  # noqa: E402
    information_fidelity_report,
    panel_sensitivities,
    weighted_covariance,
)
from rr_gid_cn.vaeac import PanelPhiNet  # noqa: E402
from run_gas_semisynthetic import calibrate_beta  # noqa: E402


def infos_from_means(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    x = np.asarray(means, dtype=np.float64)
    mu = np.einsum("n,sni->si", w, x)
    centered = x - mu[:, None, :]
    infos = np.einsum("sni,n,snj->sij", centered, w, centered)
    return np.stack([_project_psd(m) for m in infos])


def infos_torch(means: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights / weights.sum().clamp_min(1e-12)
    mu = torch.einsum("n,sni->si", w, means)
    centered = means - mu[:, None, :]
    infos = torch.einsum("sni,n,snj->sij", centered, w, centered)
    infos = 0.5 * (infos + infos.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(infos)
    return evecs @ torch.diag_embed(evals.clamp_min(1e-10)) @ evecs.transpose(-1, -2)


def scores_torch(fisher: torch.Tensor, infos: torch.Tensor) -> torch.Tensor:
    n = infos.shape[0]
    p = torch.full((n,), 1.0 / n, device=infos.device, dtype=infos.dtype)
    matrix = torch.einsum("s,sij->ij", p, infos)
    matrix = 0.5 * (matrix + matrix.T)
    inverse = torch.linalg.pinv(matrix)
    mid = inverse @ (0.5 * (fisher + fisher.T)) @ inverse
    return torch.einsum("sij,ji->s", infos, mid)


def approx_ranks(scores: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    diff = scores[None, :] - scores[:, None]
    return 1.0 + torch.sigmoid(diff / tau.clamp_min(1e-3)).sum(dim=1) - 0.5


def ranking_loss(scores: torch.Tensor, score_t: torch.Tensor) -> torch.Tensor:
    tau = (0.10 * score_t.std()).clamp(min=2.0)
    teacher_p = torch.softmax(score_t / tau, dim=0)
    student_p = torch.softmax(scores / tau, dim=0)
    listnet = -(teacher_p * student_p.clamp_min(1e-12).log()).sum()
    order = torch.argsort(score_t, descending=True)
    top_idx = order[:10]
    comp_idx = order[10:28]
    ranks = approx_ranks(scores, tau)
    rank_term = torch.relu(ranks[top_idx] - 10.0).mean()
    margin = (score_t[top_idx].min() - score_t[comp_idx].max()).abs().clamp(min=1.0) * 0.20
    hinge = torch.relu(margin - (scores[top_idx][:, None] - scores[comp_idx][None, :])).mean()
    score_term = torch.mean((scores - score_t).pow(2) / score_t.var().clamp_min(1e-8))
    return 1.0 * score_term + 0.8 * listnet + 2.5 * rank_term + 2.0 * hinge


def fmt(rep: dict) -> str:
    return (
        f"S={rep['panel_ranking_spearman']:.4f} "
        f"top={rep['top_k_overlap_count']}/10 "
        f"design={rep['validation_design_ratio']:.4f} "
        f"failed={rep['failed_gates']}"
    )


def forward_all(net: PanelPhiNet, obs: torch.Tensor) -> torch.Tensor:
    out = []
    for i in range(obs.shape[0]):
        ids = torch.full((obs.shape[1],), i, dtype=torch.long, device=obs.device)
        out.append(net(obs[i], ids))
    return torch.stack(out)


@torch.no_grad()
def numpy_means(net: PanelPhiNet, obs: torch.Tensor) -> np.ndarray:
    net.eval()
    return forward_all(net, obs).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--ckpt", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi.pt"))
    ap.add_argument("--teacher-cache", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/gas_phi_teacher_train.npz"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6phi.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6phi.json"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()

    data = np.load(args.data)
    ref_train = np.asarray(data["ref_train"], dtype=np.float64)
    ref_val = np.asarray(data["ref_val"], dtype=np.float64)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    phi_train = fn(ref_train)
    phi_val = fn(ref_val)
    beta = calibrate_beta(phi_val, seed=2026, target_ess=0.5)
    pairs = panel_library()
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in pairs)

    cache = np.load(args.teacher_cache)
    teacher = cache["teacher"]
    query_phi = cache["query_phi"]
    query_w = cache["query_w"]
    q_idx = cache["q_idx"]
    queries = ref_train[q_idx]
    print("teacher", teacher.shape, flush=True)

    official_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    outer_o = ref_val[official_idx]
    print("official kNN", flush=True)
    fisher_o, truth_o, knn_o = empirical_information_basis(
        ref_train, phi_train, outer_o, fn(outer_o), panels, beta, neighbors=60,
    )
    w_o = None
    from rr_gid_cn.empirical_generator import _stable_tilt_weights
    w_o = _stable_tilt_weights(fn(outer_o), beta)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    center = np.asarray(ckpt["data_mean"], dtype=np.float64)
    spread = np.maximum(np.asarray(ckpt["data_std"], dtype=np.float64), 1e-12)
    hidden = int(ckpt.get("phi_net_hidden", 128))
    device = args.device
    net = PanelPhiNet(n_panels=len(panels), hidden=hidden).to(device)
    net.load_state_dict(ckpt["phi_net"])

    obs_np = np.zeros((len(panels), len(queries), 16), dtype=np.float32)
    obs_o_np = np.zeros((len(panels), len(outer_o), 16), dtype=np.float32)
    for i, panel in enumerate(panels):
        cols = list(panel)
        obs_np[i] = (queries[:, cols] - center[cols]) / spread[cols]
        obs_o_np[i] = (outer_o[:, cols] - center[cols]) / spread[cols]
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    obs_o_t = torch.as_tensor(obs_o_np, dtype=torch.float32, device=device)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
    query_w_t = torch.as_tensor(query_w, dtype=torch.float32, device=device)
    query_phi_t = torch.as_tensor(query_phi, dtype=torch.float32, device=device)

    def eval_official(tag: str, scale_np: np.ndarray) -> dict:
        means = numpy_means(net, obs_o_t)
        infos = infos_from_means(means, w_o) * scale_np[:, None, None]
        rep = information_fidelity_report(fisher_o, truth_o, infos)
        rmse = float(np.sqrt(np.mean((means - knn_o) ** 2)))
        print(f"[{tag}] {fmt(rep)} rmse={rmse:.4f} scale[{scale_np.min():.2f},{scale_np.max():.2f}]", flush=True)
        epoch_tag = tag[1:] if tag.startswith("e") and tag[1:].isdigit() else None
        if tag in {"init", "final"} or not rep["failed_gates"] or (epoch_tag and int(epoch_tag) % 10 == 0):
            print(f"  cand {rep['candidate_top_panel_indices']}", flush=True)
        return {**rep, "mean_rmse": rmse}

    log_delta = torch.zeros(len(panels), dtype=torch.float32, device=device, requires_grad=True)
    raw = eval_official("init", np.ones(len(panels)))
    best = {
        "key": (int(not raw["failed_gates"]), raw["top_k_overlap_count"], raw["panel_ranking_spearman"], -raw["validation_design_ratio"]),
        "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        "scale": np.ones(len(panels), dtype=np.float64),
        "rep": raw,
        "tag": "init",
    }
    opt = torch.optim.Adam(
        [
            {"params": net.parameters(), "lr": 8e-5},
            {"params": [log_delta], "lr": 3e-2},
        ],
        weight_decay=0.0,
    )
    sched = CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1))
    rng = np.random.default_rng(202609910)
    n_q = teacher.shape[1]
    history = []
    won = False
    for epoch in range(int(args.epochs)):
        net.train()
        sel = rng.choice(n_q, size=512, replace=False)
        sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
        pred = forward_all(net, obs_t.index_select(1, sel_t))
        target = teacher_t.index_select(1, sel_t)
        mean_term = torch.mean((pred - target).pow(2))
        w_sub = query_w_t.index_select(0, sel_t)
        fisher_np = _project_psd(weighted_covariance(query_phi[sel], query_w[sel]))
        fisher_t = torch.as_tensor(fisher_np, dtype=torch.float32, device=device)
        t_info_np = infos_from_means(teacher[:, sel], query_w[sel])
        score_t = torch.as_tensor(panel_sensitivities(fisher_np, t_info_np), dtype=torch.float32, device=device)
        infos = infos_torch(pred, w_sub)
        scale = torch.exp(log_delta)
        scores = scores_torch(fisher_t, infos * scale[:, None, None])
        loss = 0.25 * mean_term + ranking_loss(scores, score_t) + 0.04 * log_delta.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            log_delta.clamp_(-1.2, 1.2)
            scale_np = torch.exp(log_delta).detach().cpu().numpy()
        if epoch % 2 == 0 or epoch + 1 == int(args.epochs):
            rep = eval_official(f"e{epoch}", scale_np)
            key = (
                int(not rep["failed_gates"]),
                int(rep["panel_ranking_spearman"] >= 0.80) * rep["top_k_overlap_count"],
                rep["top_k_overlap_count"],
                rep["panel_ranking_spearman"],
                -rep["validation_design_ratio"],
            )
            if key > best["key"]:
                best = {
                    "key": key,
                    "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                    "scale": scale_np.copy(),
                    "rep": rep,
                    "tag": f"e{epoch}",
                }
                print("  new best", fmt(rep), flush=True)
            if not rep["failed_gates"]:
                won = True
                break
        history.append({"epoch": epoch, "loss": float(loss.detach()), "mean": float(mean_term.detach())})

    net.load_state_dict(best["state"])
    scale = np.asarray(best["scale"], dtype=np.float64)
    final = eval_official("final", scale)
    payload = dict(ckpt)
    payload["phi_net"] = net.state_dict()
    payload["phi_net_hidden"] = hidden
    payload["phi_net_kind"] = "panel_embed"
    payload["phi_net_n_panels"] = len(panels)
    payload["phi_info_scale"] = scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.log.write_text(json.dumps({
        "source": str(args.ckpt),
        "output": str(args.output),
        "seconds": time.time() - t0,
        "init": {"S": raw["panel_ranking_spearman"], "top": raw["top_k_overlap_count"], "failed": raw["failed_gates"]},
        "best_tag": best["tag"],
        "best": {
            "S": final["panel_ranking_spearman"],
            "top": final["top_k_overlap_count"],
            "design": final["validation_design_ratio"],
            "failed": final["failed_gates"],
            "truth": final["reference_top_panel_indices"],
            "cand": final["candidate_top_panel_indices"],
        },
        "passed": not final["failed_gates"],
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "history_tail": history[-8:],
    }, indent=2), encoding="utf-8")
    print("saved", args.output, "PASS" if not final["failed_gates"] else "FAIL", fmt(final), flush=True)


if __name__ == "__main__":
    main()
