"""Retrain Gas VAEAC with enough empty-mask mass for the unconditional law.

Do not overwrite H:/RR_GID_CN_data/reusable/checkpoints/vaeac_gas_128_v1.pt.
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

from rr_gid_cn.empirical_generator import _stable_tilt_weights, empirical_conditional_means
from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.vaeac import VAEAC, VAEACGenerator


def transform_features_torch(x, mean, std, pcs):
    z = (x - mean) / std.clamp_min(1e-12)
    scores = torch.stack(
        [torch.einsum("...i,i->...", z[..., i * 8:(i + 1) * 8], pcs[i]) for i in range(16)],
        dim=-1,
    )
    return torch.cat([torch.tanh(scores[..., :8]), torch.tanh(scores[..., :8] * scores[..., 8:])], dim=-1)


def _knn_teacher_phi(donor_x, donor_phi, x_raw, panel_ids, panels, tilt, neighbors):
    """panel_ids: -1 empty, -2 full, >=0 panel index."""
    out = np.repeat(donor_phi.mean(0, keepdims=True), len(x_raw), axis=0)
    groups = {}
    for i, pid in enumerate(panel_ids):
        if int(pid) < 0:
            continue
        groups.setdefault(int(pid), []).append(i)
    for pid, rows in groups.items():
        panel = panels[pid]
        observed = x_raw[rows][:, list(panel)]
        out[rows] = empirical_conditional_means(
            donor_x, donor_phi, observed, panel,
            donor_tilt_weights=tilt, neighbors=neighbors,
        )
    return out


def train_opt(model, reference, panels, *, epochs, batch, lr, device, seed,
              empty_frac=0.40, full_frac=0.10, beta_final=0.3,
              feature_mean=None, feature_std=None, feature_pcs=None,
              distill_weight=1.0, neighbors=60, tilt_beta=None):
    model = model.to(device)
    raw = np.asarray(reference, dtype=np.float64)
    center = raw.mean(0)
    spread = raw.std(0, ddof=1)
    spread = np.where(spread < 1e-6, 1.0, spread)
    model.data_mean = center.astype(np.float64)
    model.data_std = spread.astype(np.float64)
    x = torch.as_tensor((raw - center) / spread, dtype=torch.float32, device=device)
    center_t = torch.as_tensor(center, dtype=torch.float32, device=device)
    spread_t = torch.as_tensor(spread, dtype=torch.float32, device=device)
    mean_t = torch.as_tensor(feature_mean, dtype=torch.float32, device=device) if feature_mean is not None else None
    std_t = torch.as_tensor(feature_std, dtype=torch.float32, device=device) if feature_std is not None else None
    pcs_t = torch.as_tensor(feature_pcs, dtype=torch.float32, device=device) if feature_pcs is not None else None
    phi_donor = transform_features(raw, feature_mean, feature_std, feature_pcs)
    if tilt_beta is None:
        tilt = np.full(len(raw), 1.0 / len(raw))
    else:
        tilt = _stable_tilt_weights(phi_donor, np.asarray(tilt_beta, dtype=float))
    uncond_phi = tilt @ phi_donor
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))
    rng = np.random.default_rng(seed)
    panels = list(panels)
    history = []
    for epoch in range(epochs):
        beta_kl = beta_final * min(1.0, (epoch + 1) / max(10, epochs // 5))
        perm = torch.randperm(len(x), device=device)
        total, distill_total, steps = 0.0, 0.0, 0
        for start in range(0, len(x), batch):
            idx = perm[start:start + batch]
            xb = x[idx]
            mask = torch.zeros(len(xb), model.dim, device=device)
            panel_ids = np.full(len(xb), -1, dtype=np.int32)
            for j in range(len(xb)):
                u = rng.random()
                if u < empty_frac:
                    panel_ids[j] = -1
                elif u < empty_frac + full_frac:
                    mask[j] = 1.0
                    panel_ids[j] = -2
                else:
                    pid = int(rng.integers(len(panels)))
                    mask[j, list(panels[pid])] = 1.0
                    panel_ids[j] = pid
            recon_mu, recon_logvar, q_mu, q_logvar, p_mu, p_logvar = model(xb, mask)
            missing = 1.0 - mask
            nll = 0.5 * (recon_logvar + (xb - recon_mu).pow(2) * torch.exp(-recon_logvar))
            n_miss = missing.sum(-1).clamp_min(1.0)
            recon_loss = (nll * missing).sum(-1) / n_miss
            observed = mask
            n_obs = observed.sum(-1).clamp_min(1.0)
            recon_obs = (nll * observed).sum(-1) / n_obs
            recon = torch.where(missing.sum(-1) > 0, recon_loss, recon_obs)
            kl = model.kl_divergence(q_mu, q_logvar, p_mu, p_logvar)
            loss = recon.mean() + beta_kl * kl.mean()
            distill = torch.zeros((), device=device)
            if distill_weight > 0.0 and mean_t is not None:
                # Eval samples z ~ p(z | x_S), not q(z | x). Distill the prior decode.
                x_masked = xb * mask
                prior_recon, _ = model.decode_params(p_mu, mask, x_masked)
                x_std = mask * xb + missing * prior_recon
                student = transform_features_torch(x_std * spread_t + center_t, mean_t, std_t, pcs_t)
                x_raw_true = (xb * spread_t + center_t).detach().cpu().numpy()
                teacher = _knn_teacher_phi(
                    raw, phi_donor, x_raw_true, panel_ids, panels, tilt, neighbors,
                )
                teacher_t = torch.as_tensor(teacher, dtype=torch.float32, device=device)
                uncond_t = torch.as_tensor(uncond_phi, dtype=torch.float32, device=device)
                panel_rows = torch.as_tensor(panel_ids >= 0, device=device)
                empty_rows = torch.as_tensor(panel_ids == -1, device=device)
                terms = []
                if bool(panel_rows.any()):
                    terms.append(torch.mean((student[panel_rows] - teacher_t[panel_rows]).pow(2)))
                if bool(empty_rows.any()):
                    terms.append(0.25 * torch.mean((student[empty_rows] - uncond_t).pow(2)))
                if terms:
                    distill = terms[0] if len(terms) == 1 else terms[0] + terms[1]
                    loss = loss + float(distill_weight) * distill
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            total += float(loss.detach())
            distill_total += float(distill.detach()) if torch.is_tensor(distill) else 0.0
            steps += 1
        sched.step()
        history.append({
            "epoch": epoch,
            "loss": total / max(steps, 1),
            "distill": distill_total / max(steps, 1),
            "beta_kl": beta_kl,
        })
        if epoch % 10 == 0 or epoch + 1 == epochs:
            print(
                f"epoch {epoch} loss {history[-1]['loss']:.4f} "
                f"distill {history[-1]['distill']:.4f} beta_kl {beta_kl:.3f}",
                flush=True,
            )
    return model, history


def c2st_auc(real: np.ndarray, fake: np.ndarray, seed: int = 0) -> float:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score
    except ImportError:
        mu = real.mean(0)
        delta = np.abs(fake.mean(0) - mu) / np.maximum(real.std(0), 1e-6)
        return float(min(1.0, 0.5 + 0.5 * min(1.0, delta.mean())))

    x = np.vstack([real, fake]).astype(np.float32)
    y = np.concatenate([np.ones(len(real)), np.zeros(len(fake))])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    aucs = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in splitter.split(x, y):
        clf = LogisticRegression(max_iter=200)
        clf.fit(x[tr], y[tr])
        proba = clf.predict_proba(x[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], proba)))
    return float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v3.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v3.json"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--distill-weight", type=float, default=3.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None)
    args = ap.parse_args()
    data = np.load(args.data)
    ref_train, ref_val = data["ref_train"], data["ref_val"]
    sensor_pairs = panel_library()
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.eval_only:
        from rr_gid_cn.vaeac import load_vaeac_checkpoint
        ckpt = args.checkpoint or Path("H:/RR_GID_CN_data/reusable/checkpoints/vaeac_gas_128_v1.pt")
        model, _ = load_vaeac_checkpoint(str(ckpt), device=args.device, expected_dim=128)
        gen = VAEACGenerator(model, np.ones(128), alpha=0.0, device=args.device)
        fake = gen.sample_full(min(4000, len(ref_val)), 0)
        auc = c2st_auc(ref_val[: len(fake)], fake, seed=0)
        print("C2ST", auc, "ckpt", ckpt)
        return
    from run_gas_semisynthetic import calibrate_beta

    phi_train = transform_features(
        ref_train,
        np.asarray(data["mean"], dtype=np.float64),
        np.asarray(data["std"], dtype=np.float64),
        np.asarray(data["pcs"], dtype=np.float64),
    )
    tilt_beta = calibrate_beta(phi_train, seed=2026, target_ess=0.5)
    model = VAEAC(dim=128, latent=args.latent, hidden=args.hidden, seed=0)
    model, hist = train_opt(
        model, ref_train, coord_panels,
        epochs=args.epochs, batch=args.batch, lr=args.lr, device=args.device, seed=0,
        feature_mean=np.asarray(data["mean"], dtype=np.float64),
        feature_std=np.asarray(data["std"], dtype=np.float64),
        feature_pcs=np.asarray(data["pcs"], dtype=np.float64),
        distill_weight=float(args.distill_weight),
        tilt_beta=tilt_beta,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "data_mean": model.data_mean,
            "data_std": model.data_std,
            "alpha": 0.0,
            "reference_n": int(len(ref_train)),
            "validation_n": int(len(ref_val)),
            "opt_track": True,
        },
        args.output,
    )
    gen = VAEACGenerator(model, np.ones(128), alpha=0.0, device=args.device)
    fake = gen.sample_full(min(4000, len(ref_val)), 0)
    auc = c2st_auc(ref_val[: len(fake)], fake, seed=0)
    record = {
        "output": str(args.output),
        "epochs": args.epochs,
        "hidden": args.hidden,
        "latent": args.latent,
        "train_seconds": time.time() - t0,
        "loss_first": hist[0]["loss"],
        "loss_last": hist[-1]["loss"],
        "distill_weight": float(args.distill_weight),
        "c2st_auc_ref_val": auc,
        "history": hist[-20:],
    }
    args.log.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("saved", args.output, "C2ST", auc, flush=True)


if __name__ == "__main__":
    main()
