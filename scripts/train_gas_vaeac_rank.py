"""Fine-tune Gas VAEAC so I_S ranking matches tilt-weighted kNN on a frozen query table."""
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

from rr_gid_cn.empirical_generator import _stable_tilt_weights, empirical_conditional_means
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.vaeac import VAEAC, load_vaeac_checkpoint
from run_gas_semisynthetic import calibrate_beta
from train_gas_vaeac_opt import transform_features_torch


def prior_decode_student(model, query_raw, panel, center_t, spread_t, mean_t, std_t, pcs_t):
    cols = list(panel)
    n = query_raw.shape[0]
    dim = model.dim
    device = center_t.device
    obs = torch.zeros(n, dim, device=device)
    mask = torch.zeros(n, dim, device=device)
    obs[:, cols] = (query_raw[:, cols] - center_t[cols]) / spread_t[cols].clamp_min(1e-12)
    mask[:, cols] = 1.0
    p_mu, _ = model.prior(obs, mask)
    recon, _ = model.decode_params(p_mu, mask, obs)
    mixed = mask * obs + (1.0 - mask) * recon
    raw = mixed * spread_t + center_t
    raw = raw.clone()
    raw[:, cols] = query_raw[:, cols]
    return transform_features_torch(raw, mean_t, std_t, pcs_t)


def weighted_cov_torch(features, weights):
    w = weights / weights.sum().clamp_min(1e-12)
    mu = w @ features
    centered = features - mu
    return (centered * w[:, None]).T @ centered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--init", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v3.pt"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4.json"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-queries", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cov-weight", type=float, default=2.0)
    ap.add_argument("--elbo-weight", type=float, default=0.15)
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
    rng = np.random.default_rng(202609801)
    n_q = min(int(args.n_queries), len(ref_train))
    q_idx = rng.choice(len(ref_train), size=n_q, replace=False)
    queries = ref_train[q_idx]
    query_phi = phi_train[q_idx]
    query_w = _stable_tilt_weights(query_phi, beta)
    print(f"precomputing kNN teacher on {n_q} queries x {len(panels)} panels", flush=True)
    teacher = np.zeros((len(panels), n_q, 16), dtype=np.float64)
    for i, panel in enumerate(panels):
        teacher[i] = empirical_conditional_means(
            ref_train, phi_train, queries[:, list(panel)], panel,
            donor_tilt_weights=tilt, neighbors=60,
        )
        if (i + 1) % 30 == 0:
            print(f"  teacher {i+1}/{len(panels)}", flush=True)

    model, payload = load_vaeac_checkpoint(str(args.init), device=args.device, expected_dim=128)
    model.train()
    device = args.device
    center_t = torch.as_tensor(model.data_mean, dtype=torch.float32, device=device)
    spread_t = torch.as_tensor(model.data_std, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    pcs_t = torch.as_tensor(pcs, dtype=torch.float32, device=device)
    query_t = torch.as_tensor(queries, dtype=torch.float32, device=device)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
    weight_t = torch.as_tensor(query_w, dtype=torch.float32, device=device)
    raw = torch.as_tensor((ref_train - np.asarray(model.data_mean)) / np.asarray(model.data_std), dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    sched = CosineAnnealingLR(opt, T_max=max(int(args.epochs), 1))
    history = []
    t0 = time.time()
    rng_np = np.random.default_rng(0)
    for epoch in range(int(args.epochs)):
        model.train()
        mean_acc = 0.0
        cov_acc = 0.0
        opt.zero_grad()
        for i, panel in enumerate(panels):
            student = prior_decode_student(model, query_t, panel, center_t, spread_t, mean_t, std_t, pcs_t)
            target = teacher_t[i]
            mean_term = torch.mean((student - target).pow(2))
            cov_term = torch.mean(
                (weighted_cov_torch(student, weight_t) - weighted_cov_torch(target, weight_t)).pow(2)
            )
            ((mean_term + float(args.cov_weight) * cov_term) / len(panels)).backward()
            mean_acc += float(mean_term.detach())
            cov_acc += float(cov_term.detach())
        mean_acc /= len(panels)
        cov_acc /= len(panels)
        elbo_val = 0.0
        if float(args.elbo_weight) > 0.0:
            idx = torch.as_tensor(rng_np.choice(len(raw), size=256, replace=False), device=device)
            xb = raw[idx]
            mask = torch.zeros_like(xb)
            for j in range(len(xb)):
                u = rng_np.random()
                if u < 0.40:
                    pass
                elif u < 0.50:
                    mask[j] = 1.0
                else:
                    mask[j, list(panels[int(rng_np.integers(len(panels)))])] = 1.0
            recon_mu, recon_logvar, q_mu, q_logvar, p_mu, p_logvar = model(xb, mask)
            missing = 1.0 - mask
            nll = 0.5 * (recon_logvar + (xb - recon_mu).pow(2) * torch.exp(-recon_logvar))
            n_miss = missing.sum(-1).clamp_min(1.0)
            recon = (nll * missing).sum(-1) / n_miss
            kl = model.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            elbo = recon.mean() + 0.3 * kl.mean()
            (float(args.elbo_weight) * elbo).backward()
            elbo_val = float(elbo.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        row = {
            "epoch": epoch,
            "loss": mean_acc + float(args.cov_weight) * cov_acc + float(args.elbo_weight) * elbo_val,
            "mean": mean_acc,
            "cov": cov_acc,
            "elbo": elbo_val,
        }
        history.append(row)
        if epoch % 5 == 0 or epoch + 1 == int(args.epochs):
            print(
                f"epoch {epoch} loss {row['loss']:.5f} mean {row['mean']:.5f} "
                f"cov {row['cov']:.5f} elbo {row['elbo']:.4f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "data_mean": model.data_mean,
            "data_std": model.data_std,
            "alpha": 0.0,
            "reference_n": int(len(ref_train)),
            "opt_track": True,
            "rank_finetune": True,
            "init": str(args.init),
        },
        args.output,
    )
    record = {
        "output": str(args.output),
        "init": str(args.init),
        "epochs": int(args.epochs),
        "n_queries": n_q,
        "train_seconds": time.time() - t0,
        "loss_first": history[0]["loss"],
        "loss_last": history[-1]["loss"],
        "mean_last": history[-1]["mean"],
        "cov_last": history[-1]["cov"],
        "history": history[-20:],
    }
    args.log.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
