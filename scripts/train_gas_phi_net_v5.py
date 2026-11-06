"""RISK-2 top-k fix: distill kNN E[phi|x_S], then match I_S on 512-outers.

v4 failed top-10 because (1) mean MSE stayed ~0.62 while ranking losses dominated,
(2) the hinge compared teacher top-10 with the bottom 110 instead of near-competitors,
(3) I_S was recomputed on a fresh 256-subset every step, so the head target jittered.
This trainer follows the RISK-2 protocol: same outer table, tilt-weighted covariance
of E[phi|x_S], then Spearman / top-10 / design-ratio on panel_sensitivities.
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
    _stable_tilt_weights,
    empirical_conditional_means,
    empirical_information_basis,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import (  # noqa: E402
    information_fidelity_report,
    panel_sensitivities,
    weighted_covariance as weighted_covariance_np,
)
from rr_gid_cn.vaeac import PanelPhiNet  # noqa: E402
from run_gas_semisynthetic import calibrate_beta  # noqa: E402


def _infos_from_means(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    x = np.asarray(means, dtype=np.float64)
    mu = np.einsum("n,sni->si", w, x)
    centered = x - mu[:, None, :]
    infos = np.einsum("sni,n,snj->sij", centered, w, centered)
    return np.stack([_project_psd(m) for m in infos])


def _infos_from_means_torch(means: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    w = weights / weights.sum().clamp_min(1e-12)
    mu = torch.einsum("n,sni->si", w, means)
    centered = means - mu[:, None, :]
    infos = torch.einsum("sni,n,snj->sij", centered, w, centered)
    infos = 0.5 * (infos + infos.transpose(-1, -2))
    evals, evecs = torch.linalg.eigh(infos)
    return evecs @ torch.diag_embed(evals.clamp_min(1e-10)) @ evecs.transpose(-1, -2)


def _scores_torch(fisher: torch.Tensor, infos: torch.Tensor) -> torch.Tensor:
    n = infos.shape[0]
    p = torch.full((n,), 1.0 / n, device=infos.device, dtype=infos.dtype)
    matrix = torch.einsum("s,sij->ij", p, infos)
    matrix = 0.5 * (matrix + matrix.T)
    inverse = torch.linalg.pinv(matrix)
    mid = inverse @ (0.5 * (fisher + fisher.T)) @ inverse
    return torch.einsum("sij,ji->s", infos, mid)


def _approx_ranks(scores: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    diff = scores[None, :] - scores[:, None]
    return 1.0 + torch.sigmoid(diff / tau.clamp_min(1e-3)).sum(dim=1) - 0.5


def _forward_all(net: PanelPhiNet, obs: torch.Tensor) -> torch.Tensor:
    preds = []
    for i in range(obs.shape[0]):
        ids = torch.full((obs.shape[1],), i, dtype=torch.long, device=obs.device)
        preds.append(net(obs[i], ids))
    return torch.stack(preds)


@torch.no_grad()
def _numpy_means(net: PanelPhiNet, obs: torch.Tensor) -> np.ndarray:
    net.eval()
    return _forward_all(net, obs).cpu().numpy()


def _report(fisher, truth, cand) -> dict:
    return information_fidelity_report(fisher, truth, cand)


def _fmt(rep: dict) -> str:
    return (
        f"S={rep['panel_ranking_spearman']:.4f} "
        f"top={rep['top_k_overlap_count']}/10 "
        f"design={rep['validation_design_ratio']:.4f} "
        f"failed={rep['failed_gates']}"
    )


def _sqrtm(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values, floor)
    return (vectors * np.sqrt(values)) @ vectors.T


def _invsqrtm(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    values = np.maximum(values, floor)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


def _fit_cov_maps(student: np.ndarray, teacher: np.ndarray, weights: np.ndarray):
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    maps = []
    for i in range(len(student)):
        s = np.asarray(student[i], dtype=np.float64)
        t = np.asarray(teacher[i], dtype=np.float64)
        cs = _project_psd(weighted_covariance_np(s, w))
        ct = _project_psd(weighted_covariance_np(t, w))
        a = _invsqrtm(cs) @ _sqrtm(ct)
        mu_s = w @ s
        mu_t = w @ t
        maps.append((a, mu_s, mu_t))
    return maps


def _apply_cov_maps(means: np.ndarray, maps) -> np.ndarray:
    out = np.empty_like(means, dtype=np.float64)
    for i, (a, mu_s, mu_t) in enumerate(maps):
        out[i] = (np.asarray(means[i], dtype=np.float64) - mu_s) @ a + mu_t
    return out


def _fold_cov_maps(net: PanelPhiNet, maps) -> None:
    for i, (a, mu_s, mu_t) in enumerate(maps):
        linear = net.nets[i][-1]
        w = linear.weight.detach().cpu().numpy().astype(np.float64)
        b = linear.bias.detach().cpu().numpy().astype(np.float64)
        a = np.asarray(a, dtype=np.float64)
        w_new = a.T @ w
        b_new = b @ a + (mu_t - mu_s @ a)
        linear.weight.data.copy_(torch.as_tensor(w_new, dtype=linear.weight.dtype))
        linear.bias.data.copy_(torch.as_tensor(b_new, dtype=linear.bias.dtype))


def _passed(rep: dict) -> bool:
    return not rep["failed_gates"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--vaeac", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v3.pt"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v5phi.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v5phi.json"))
    ap.add_argument("--mean-epochs", type=int, default=120)
    ap.add_argument("--rank-epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-queries", type=int, default=0, help="0 = all ref_train")
    ap.add_argument("--outer-size", type=int, default=512)
    ap.add_argument("--lr-mean", type=float, default=2e-3)
    ap.add_argument("--lr-rank", type=float, default=4e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-rank-if-passed", action="store_true", default=True)
    args = ap.parse_args()

    data = np.load(args.data)
    ref_train = np.asarray(data["ref_train"], dtype=np.float64)
    ref_val = np.asarray(data["ref_val"], dtype=np.float64)
    mean = np.asarray(data["mean"], dtype=np.float64)
    std = np.asarray(data["std"], dtype=np.float64)
    pcs = np.asarray(data["pcs"], dtype=np.float64)
    fn = lambda x: transform_features(x, mean, std, pcs)
    phi_train = fn(ref_train)
    phi_val = fn(ref_val)
    beta = calibrate_beta(phi_val, seed=2026, target_ess=0.5)
    tilt = _stable_tilt_weights(phi_train, beta)
    pairs = panel_library()
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in pairs)
    device = args.device
    t0 = time.time()

    n_q = len(ref_train) if int(args.n_queries) <= 0 else min(int(args.n_queries), len(ref_train))
    rng = np.random.default_rng(202609802)
    q_idx = rng.choice(len(ref_train), size=n_q, replace=False)
    queries = ref_train[q_idx]
    query_phi = phi_train[q_idx]
    query_w = _stable_tilt_weights(query_phi, beta)
    print(f"precomputing kNN teacher {n_q} x {len(panels)}  n_train={len(ref_train)}", flush=True)
    teacher = np.zeros((len(panels), n_q, 16), dtype=np.float32)
    observed = np.zeros((len(panels), n_q, 16), dtype=np.float32)
    for i, panel in enumerate(panels):
        observed[i] = queries[:, list(panel)]
        teacher[i] = empirical_conditional_means(
            ref_train, phi_train, observed[i], panel,
            donor_tilt_weights=tilt, neighbors=60,
        )
        if (i + 1) % 30 == 0:
            print(f"  teacher {i + 1}/{len(panels)}", flush=True)

    ckpt = torch.load(args.vaeac, map_location="cpu", weights_only=False)
    center = np.asarray(ckpt["data_mean"], dtype=np.float64)
    spread = np.maximum(np.asarray(ckpt["data_std"], dtype=np.float64), 1e-12)
    obs_np = np.zeros_like(observed)
    for i, panel in enumerate(panels):
        cols = list(panel)
        obs_np[i] = (observed[i] - center[cols]) / spread[cols]
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
    query_w_t = torch.as_tensor(query_w, dtype=torch.float32, device=device)
    query_phi_t = torch.as_tensor(query_phi, dtype=torch.float32, device=device)

    official_rng = np.random.default_rng(202609700)
    official_idx = official_rng.choice(len(ref_val), size=min(512, len(ref_val)), replace=False)
    outer_o = ref_val[official_idx]
    print("computing official 512 kNN basis", flush=True)
    fisher_o, truth_o, knn_means_o = empirical_information_basis(
        ref_train, phi_train, outer_o, fn(outer_o), panels, beta, neighbors=60,
    )
    obs_o = np.zeros((len(panels), len(outer_o), 16), dtype=np.float32)
    for i, panel in enumerate(panels):
        cols = list(panel)
        obs_o[i] = (outer_o[:, cols] - center[cols]) / spread[cols]
    obs_o_t = torch.as_tensor(obs_o, dtype=torch.float32, device=device)
    outer_w_o = _stable_tilt_weights(fn(outer_o), beta)

    monitor_n = min(int(args.outer_size), max(64, n_q // 5))
    monitor_idx = np.arange(n_q - monitor_n, n_q)
    train_idx = np.arange(0, n_q - monitor_n)
    fisher_m = _project_psd(weighted_covariance_np(query_phi[monitor_idx], query_w[monitor_idx]))
    truth_m = _infos_from_means(teacher[:, monitor_idx], query_w[monitor_idx])

    net = PanelPhiNet(n_panels=len(panels), hidden=int(args.hidden)).to(device)
    history = []

    def eval_official(tag: str) -> dict:
        means = _numpy_means(net, obs_o_t)
        cand = _infos_from_means(means, outer_w_o)
        rep = _report(fisher_o, truth_o, cand)
        mean_rmse = float(np.sqrt(np.mean((means - knn_means_o) ** 2)))
        print(f"[{tag}] official512 {_fmt(rep)}  mean_rmse={mean_rmse:.4f}", flush=True)
        print(f"[{tag}] truth top {rep['reference_top_panel_indices']}", flush=True)
        print(f"[{tag}] cand  top {rep['candidate_top_panel_indices']}", flush=True)
        return {**rep, "mean_rmse": mean_rmse, "tag": tag}

    def eval_monitor(tag: str) -> dict:
        means = _numpy_means(net, obs_t[:, monitor_idx])
        cand = _infos_from_means(means, query_w[monitor_idx])
        rep = _report(fisher_m, truth_m, cand)
        print(f"[{tag}] monitor {_fmt(rep)}", flush=True)
        return rep

    def train_mean(epochs: int, lr: float, tag: str) -> None:
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))
        batch = min(1024, len(train_idx))
        rng_ep = np.random.default_rng(11)
        for epoch in range(epochs):
            net.train()
            sel = rng_ep.choice(train_idx, size=batch, replace=False)
            sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
            pred = _forward_all(net, obs_t.index_select(1, sel_t))
            target = teacher_t.index_select(1, sel_t)
            loss = torch.mean((pred - target).pow(2))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            row = {"phase": tag, "epoch": epoch, "mean": float(loss.detach())}
            history.append(row)
            if epoch % 20 == 0 or epoch + 1 == epochs:
                print(f"{tag} epoch {epoch} mean {row['mean']:.5f}", flush=True)

    def train_rank(epochs: int, lr: float) -> None:
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))
        outer = min(int(args.outer_size), len(train_idx))
        batch_mean = min(1024, len(train_idx))
        rng_ep = np.random.default_rng(23)
        for epoch in range(epochs):
            net.train()
            sel_m = rng_ep.choice(train_idx, size=batch_mean, replace=False)
            sel = rng_ep.choice(train_idx, size=outer, replace=False)
            sel_m_t = torch.as_tensor(sel_m, dtype=torch.long, device=device)
            sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
            pred_m = _forward_all(net, obs_t.index_select(1, sel_m_t))
            mean_term = torch.mean((pred_m - teacher_t.index_select(1, sel_m_t)).pow(2))
            pred = _forward_all(net, obs_t.index_select(1, sel_t))
            w_sub = query_w_t.index_select(0, sel_t)
            w_np = query_w[sel]
            fisher_np = _project_psd(weighted_covariance_np(query_phi[sel], w_np))
            fisher_t = torch.as_tensor(fisher_np, dtype=torch.float32, device=device)
            teacher_infos_np = _infos_from_means(teacher[:, sel], w_np)
            teacher_infos = torch.as_tensor(teacher_infos_np, dtype=torch.float32, device=device)
            score_t = torch.as_tensor(
                panel_sensitivities(fisher_np, teacher_infos_np),
                dtype=torch.float32, device=device,
            )
            infos = _infos_from_means_torch(pred, w_sub)
            scores = _scores_torch(fisher_t, infos)
            info_term = torch.mean((infos - teacher_infos).pow(2)) / teacher_infos.pow(2).mean().clamp_min(1e-8)
            score_term = torch.mean((scores - score_t).pow(2) / score_t.var().clamp_min(1e-8))
            tau = (0.08 * score_t.std()).clamp(min=1.0)
            teacher_p = torch.softmax(score_t / tau, dim=0)
            student_p = torch.softmax(scores / tau, dim=0)
            listnet = -(teacher_p * student_p.clamp_min(1e-12).log()).sum()
            order = torch.argsort(score_t, descending=True)
            top_idx = order[:10]
            comp_idx = order[10:30]
            ranks = _approx_ranks(scores, tau)
            rank_term = torch.relu(ranks[top_idx] - 10.0).mean()
            margin = (score_t[top_idx].min() - score_t[comp_idx].max()).abs().clamp(min=1.0) * 0.25
            hinge = torch.relu(margin - (scores[top_idx][:, None] - scores[comp_idx][None, :])).mean()
            # keep the current student impostors from sitting in the head
            stud_order = torch.argsort(scores, descending=True)
            impostors = stud_order[:10]
            push = torch.relu(scores[impostors] - scores[top_idx].min()).mean()
            loss = (
                1.0 * mean_term
                + 4.0 * info_term
                + 2.0 * score_term
                + 0.5 * listnet
                + 3.0 * rank_term
                + 2.0 * hinge
                + 1.0 * push
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            row = {
                "phase": "rank",
                "epoch": epoch,
                "mean": float(mean_term.detach()),
                "info": float(info_term.detach()),
                "score": float(score_term.detach()),
                "rank": float(rank_term.detach()),
                "hinge": float(hinge.detach()),
                "loss": float(loss.detach()),
            }
            history.append(row)
            if epoch % 10 == 0 or epoch + 1 == epochs:
                print(
                    f"rank epoch {epoch} loss {row['loss']:.4f} mean {row['mean']:.4f} "
                    f"info {row['info']:.4f} rank {row['rank']:.4f} hinge {row['hinge']:.4f}",
                    flush=True,
                )
                eval_monitor(f"rank{epoch}")

    print("phase 1: mean distillation", flush=True)
    train_mean(int(args.mean_epochs), float(args.lr_mean), "mean")
    eval_monitor("after_mean")
    official_mean = eval_official("after_mean")

    print("phase 1b: fold train covariance maps into last layers", flush=True)
    student_train = _numpy_means(
        net, obs_t.index_select(1, torch.as_tensor(train_idx, dtype=torch.long, device=device))
    )
    maps = _fit_cov_maps(student_train, teacher[:, train_idx], query_w[train_idx])
    raw_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    monitor_before = eval_monitor("cal_before")
    _fold_cov_maps(net, maps)
    monitor_after = eval_monitor("cal_after")
    official_cal = eval_official("after_covcal")
    cal_better = (
        monitor_after["top_k_overlap_count"],
        monitor_after["panel_ranking_spearman"],
        -monitor_after["validation_design_ratio"],
    ) >= (
        monitor_before["top_k_overlap_count"],
        monitor_before["panel_ranking_spearman"],
        -monitor_before["validation_design_ratio"],
    )
    if not cal_better:
        print("covariance fold hurt monitor; reverting last-layer maps", flush=True)
        net.load_state_dict(raw_state)
        official_cal = eval_official("reverted_covcal")

    current_official = official_cal if cal_better else official_mean
    do_rank = not (_passed(current_official) and bool(args.skip_rank_if_passed))
    official_rank = None
    if do_rank:
        print("phase 2: I_S / top-k rank matching on random train 512 outers", flush=True)
        state_before_rank = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        monitor_before_rank = eval_monitor("before_rank")
        train_rank(int(args.rank_epochs), float(args.lr_rank))
        monitor_after_rank = eval_monitor("after_rank")
        official_rank = eval_official("after_rank")
        rank_better = (
            monitor_after_rank["top_k_overlap_count"],
            monitor_after_rank["panel_ranking_spearman"],
            -monitor_after_rank["validation_design_ratio"],
        ) >= (
            monitor_before_rank["top_k_overlap_count"],
            monitor_before_rank["panel_ranking_spearman"],
            -monitor_before_rank["validation_design_ratio"],
        )
        if not rank_better:
            print("rank phase hurt monitor; restoring pre-rank weights", flush=True)
            net.load_state_dict(state_before_rank)
            current_official = eval_official("restored_prerank")
        else:
            current_official = official_rank
    else:
        print("official 512 already passed after mean/calibration; skip rank phase", flush=True)
    best_official = current_official

    payload = dict(ckpt)
    payload["phi_net"] = net.state_dict()
    payload["phi_net_hidden"] = int(args.hidden)
    payload["phi_net_kind"] = "panel_embed"
    payload["phi_net_n_panels"] = len(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    log = {
        "output": str(args.output),
        "vaeac": str(args.vaeac),
        "n_queries": n_q,
        "hidden": int(args.hidden),
        "mean_epochs": int(args.mean_epochs),
        "rank_epochs": 0 if not do_rank else int(args.rank_epochs),
        "train_seconds": time.time() - t0,
        "official_after_mean": {k: official_mean[k] for k in (
            "panel_ranking_spearman", "top_k_overlap_count", "validation_design_ratio",
            "failed_gates", "mean_rmse", "reference_top_panel_indices", "candidate_top_panel_indices",
        )},
        "official_after_covcal": {k: official_cal[k] for k in (
            "panel_ranking_spearman", "top_k_overlap_count", "validation_design_ratio",
            "failed_gates", "mean_rmse", "reference_top_panel_indices", "candidate_top_panel_indices",
        )},
        "official_after_rank": None if official_rank is None else {k: official_rank[k] for k in (
            "panel_ranking_spearman", "top_k_overlap_count", "validation_design_ratio",
            "failed_gates", "mean_rmse", "reference_top_panel_indices", "candidate_top_panel_indices",
        )},
        "official_best": {k: best_official[k] for k in (
            "panel_ranking_spearman", "top_k_overlap_count", "validation_design_ratio",
            "failed_gates", "mean_rmse", "reference_top_panel_indices", "candidate_top_panel_indices",
        )},
        "history_tail": history[-20:],
    }
    args.log.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("saved", args.output, flush=True)
    print("best official", _fmt(best_official), flush=True)


if __name__ == "__main__":
    main()
