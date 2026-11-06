"""Parametric copy of the Gas kNN operator: 16-d panel observation -> E[phi | x_S]."""
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

from rr_gid_cn.empirical_generator import _project_psd, _stable_tilt_weights, empirical_conditional_means
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.generator_validation import panel_sensitivities, weighted_covariance as weighted_covariance_np
from rr_gid_cn.vaeac import PanelPhiNet
from run_gas_semisynthetic import calibrate_beta


def weighted_cov(features, weights):
    w = weights / weights.sum().clamp_min(1e-12)
    mu = w @ features
    centered = features - mu
    return (centered * w[:, None]).T @ centered


def panel_scores_torch(fisher, infos):
    n = infos.shape[0]
    p = torch.full((n,), 1.0 / n, device=infos.device, dtype=infos.dtype)
    matrix = torch.einsum("s,sij->ij", p, infos)
    matrix = 0.5 * (matrix + matrix.T)
    inverse = torch.linalg.pinv(matrix)
    mid = inverse @ (0.5 * (fisher + fisher.T)) @ inverse
    return torch.einsum("sij,ji->s", infos, mid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--vaeac", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v3.pt"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi.json"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-queries", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--batch-queries", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = np.load(args.data)
    ref_train = np.asarray(data["ref_train"], dtype=np.float64)
    mean = np.asarray(data["mean"], dtype=np.float64)
    std = np.asarray(data["std"], dtype=np.float64)
    pcs = np.asarray(data["pcs"], dtype=np.float64)
    phi_train = transform_features(ref_train, mean, std, pcs)
    beta = calibrate_beta(phi_train, seed=2026, target_ess=0.5)
    tilt = _stable_tilt_weights(phi_train, beta)
    pairs = panel_library()
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in pairs)
    rng = np.random.default_rng(202609802)
    n_q = min(int(args.n_queries), len(ref_train))
    q_idx = rng.choice(len(ref_train), size=n_q, replace=False)
    queries = ref_train[q_idx]
    query_phi = phi_train[q_idx]
    query_w = _stable_tilt_weights(query_phi, beta)
    print(f"precomputing kNN teacher {n_q} x {len(panels)}", flush=True)
    teacher = np.zeros((len(panels), n_q, 16), dtype=np.float32)
    observed = np.zeros((len(panels), n_q, 16), dtype=np.float32)
    for i, panel in enumerate(panels):
        observed[i] = queries[:, list(panel)]
        teacher[i] = empirical_conditional_means(
            ref_train, phi_train, observed[i], panel,
            donor_tilt_weights=tilt, neighbors=60,
        )
        if (i + 1) % 30 == 0:
            print(f"  teacher {i+1}/{len(panels)}", flush=True)

    ckpt = torch.load(args.vaeac, map_location="cpu", weights_only=False)
    center = np.asarray(ckpt["data_mean"], dtype=np.float64)
    spread = np.maximum(np.asarray(ckpt["data_std"], dtype=np.float64), 1e-12)
    device = args.device
    net = PanelPhiNet(n_panels=len(panels), hidden=int(args.hidden)).to(device)
    if ckpt.get("phi_net") is not None:
        try:
            net.load_state_dict(ckpt["phi_net"])
            print("resumed phi_net weights", flush=True)
        except Exception as exc:
            print("could not resume phi_net:", exc, flush=True)
    obs_np = np.zeros((len(panels), n_q, 16), dtype=np.float32)
    for i, panel in enumerate(panels):
        cols = list(panel)
        obs_np[i] = (observed[i] - center[cols]) / spread[cols]
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=float(args.lr), weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1))
    history = []
    t0 = time.time()
    rng_ep = np.random.default_rng(1)
    batch_q = min(int(args.batch_queries), n_q)
    for epoch in range(int(args.epochs)):
        net.train()
        opt.zero_grad()
        sel = rng_ep.choice(n_q, size=batch_q, replace=False)
        sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
        phi_sub = query_phi[sel]
        w_np = _stable_tilt_weights(phi_sub, beta)
        w_sub = torch.as_tensor(w_np, dtype=torch.float32, device=device)
        fisher_np = _project_psd(weighted_covariance_np(phi_sub, w_np))
        teacher_infos = np.stack([
            _project_psd(weighted_covariance_np(teacher[i, sel], w_np)) for i in range(len(panels))
        ])
        score_np = panel_sensitivities(fisher_np, teacher_infos)
        fisher_t = torch.as_tensor(fisher_np, dtype=torch.float32, device=device)
        score_t = torch.as_tensor(score_np, dtype=torch.float32, device=device)
        preds = []
        for i in range(len(panels)):
            ids = torch.full((batch_q,), i, dtype=torch.long, device=device)
            preds.append(net(obs_t[i].index_select(0, sel_t), ids))
        stacked = torch.stack(preds)
        target = teacher_t.index_select(1, sel_t)
        mean_term = torch.mean((stacked - target).pow(2))
        infos = torch.stack([weighted_cov(stacked[i], w_sub) for i in range(len(panels))])
        scores = panel_scores_torch(fisher_t, infos)
        score_term = torch.mean((scores - score_t).pow(2) / (score_t.var().clamp_min(1e-8)))
        teacher_p = torch.softmax(score_t / 0.2, dim=0)
        student_p = torch.softmax(scores / 0.2, dim=0)
        listnet = -(teacher_p * student_p.clamp_min(1e-12).log()).sum()
        top_idx = torch.topk(score_t, 10).indices
        rest_idx = torch.topk(-score_t, max(len(score_t) - 10, 1)).indices
        hinge = torch.relu(-(scores[top_idx][:, None] - scores[rest_idx][None, :])).mean()
        loss = 0.2 * mean_term + 8.0 * score_term + 2.0 * listnet + 20.0 * hinge
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        row = {
            "epoch": epoch,
            "mean": float(mean_term.detach()),
            "score": float(score_term.detach()),
            "loss": float(loss.detach()),
        }
        history.append(row)
        if epoch % 10 == 0 or epoch + 1 == int(args.epochs):
            print(
                f"epoch {epoch} loss {row['loss']:.5f} mean {row['mean']:.5f} score {row['score']:.5f}",
                flush=True,
            )

    payload = dict(ckpt)
    payload["phi_net"] = net.state_dict()
    payload["phi_net_hidden"] = int(args.hidden)
    payload["phi_net_kind"] = "panel_embed"
    payload["phi_net_n_panels"] = len(panels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.log.write_text(json.dumps({
        "output": str(args.output),
        "vaeac": str(args.vaeac),
        "epochs": int(args.epochs),
        "n_queries": n_q,
        "train_seconds": time.time() - t0,
        "mean_first": history[0]["mean"],
        "mean_last": history[-1]["mean"],
        "score_last": history[-1]["score"],
        "history": history[-15:],
    }, indent=2), encoding="utf-8")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
