"""Fit phi-net I_S scales on held-out val (not the official 512).

Train kNN top-10 and val kNN top-10 only overlap 3/10, so train-fitted scales
cannot recover the RISK-2 reference. Official 512 is a val draw; calibrate on
the complementary val rows.
"""
from __future__ import annotations

import argparse
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
    _project_psd,
    empirical_information_basis,
    load_gas_generator,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import (  # noqa: E402
    information_fidelity_report,
    panel_sensitivities,
    spearman_correlation,
    weighted_covariance,
)
from run_gas_semisynthetic import calibrate_beta  # noqa: E402


def infos_from_means(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    x = np.asarray(means, dtype=np.float64)
    mu = np.einsum("n,sni->si", w, x)
    centered = x - mu[:, None, :]
    infos = np.einsum("sni,n,snj->sij", centered, w, centered)
    return np.stack([_project_psd(m) for m in infos])


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
    tau = (0.08 * score_t.std()).clamp(min=1.0)
    teacher_p = torch.softmax(score_t / tau, dim=0)
    student_p = torch.softmax(scores / tau, dim=0)
    listnet = -(teacher_p * student_p.clamp_min(1e-12).log()).sum()
    order = torch.argsort(score_t, descending=True)
    top_idx = order[:10]
    comp_idx = order[10:35]
    ranks = approx_ranks(scores, tau)
    rank_term = torch.relu(ranks[top_idx] - 10.0).mean()
    margin = (score_t[top_idx].min() - score_t[comp_idx].max()).abs().clamp(min=1.0) * 0.35
    hinge = torch.relu(margin - (scores[top_idx][:, None] - scores[comp_idx][None, :])).mean()
    impostors = torch.argsort(scores, descending=True)[:10]
    push = torch.relu(scores[impostors] - scores[top_idx].min()).mean()
    score_mse = torch.mean((scores - score_t).pow(2) / score_t.var().clamp_min(1e-8))
    return 1.0 * listnet + 6.0 * rank_term + 4.0 * hinge + 2.0 * push + 0.25 * score_mse


def overlap_count(fisher, truth, cand, k=10) -> tuple[int, float]:
    t = panel_sensitivities(fisher, truth)
    c = panel_sensitivities(fisher, cand)
    top_t = set(map(int, np.argsort(-t, kind="stable")[:k]))
    top_c = set(map(int, np.argsort(-c, kind="stable")[:k]))
    return len(top_t & top_c), spearman_correlation(t, c)


def fmt(rep: dict) -> str:
    return (
        f"S={rep['panel_ranking_spearman']:.4f} "
        f"top={rep['top_k_overlap_count']}/10 "
        f"design={rep['validation_design_ratio']:.4f} "
        f"failed={rep['failed_gates']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--ckpt", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi.pt"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi_valscale.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi_valscale.json"))
    ap.add_argument("--n-subsets", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=180)
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

    official_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    official_mask = np.zeros(len(ref_val), dtype=bool)
    official_mask[official_idx] = True
    cal_idx = np.flatnonzero(~official_mask)
    print(f"cal n={len(cal_idx)} official n={official_mask.sum()}", flush=True)

    print("kNN basis official / cal / full-val", flush=True)
    fisher_o, truth_o, _ = empirical_information_basis(
        ref_train, phi_train, ref_val[official_idx], phi_val[official_idx], panels, beta, neighbors=60,
    )
    fisher_c, truth_c, knn_means_c = empirical_information_basis(
        ref_train, phi_train, ref_val[cal_idx], phi_val[cal_idx], panels, beta, neighbors=60,
    )
    spec = {"generator_kind": "vaeac", "checkpoint": str(args.ckpt)}
    gen = load_gas_generator(spec, data, fn, device=args.device)
    _, neural_o, _ = gen.panel_information(beta, panels, ref_val[official_idx], estimator="phi_net")
    _, neural_c, student_means_c = gen.panel_information(beta, panels, ref_val[cal_idx], estimator="phi_net")
    raw_o = information_fidelity_report(fisher_o, truth_o, neural_o)
    raw_c = information_fidelity_report(fisher_c, truth_c, neural_c)
    print("raw official", fmt(raw_o), flush=True)
    print("raw cal675", fmt(raw_c), flush=True)
    off_top = set(raw_o["reference_top_panel_indices"])
    cal_top = set(raw_c["reference_top_panel_indices"])
    print("kNN top-10 overlap official vs cal675", len(off_top & cal_top), sorted(off_top & cal_top), flush=True)

    from rr_gid_cn.empirical_generator import _stable_tilt_weights
    cal_w = _stable_tilt_weights(phi_val[cal_idx], beta)
    rng = np.random.default_rng(202609901)
    pack = []
    outer = min(512, len(cal_idx))
    for _ in range(int(args.n_subsets)):
        sel = rng.choice(len(cal_idx), size=outer, replace=False)
        fisher = _project_psd(weighted_covariance(phi_val[cal_idx][sel], cal_w[sel]))
        t_info = infos_from_means(knn_means_c[:, sel], cal_w[sel])
        s_info = infos_from_means(student_means_c[:, sel], cal_w[sel])
        t_sc = panel_sensitivities(fisher, t_info)
        pack.append((
            torch.as_tensor(fisher, dtype=torch.float32),
            torch.as_tensor(t_info, dtype=torch.float32),
            torch.as_tensor(s_info, dtype=torch.float32),
            torch.as_tensor(t_sc, dtype=torch.float32),
        ))

    # hold out 8 subsets for monitor (not used in the loss)
    monitor = pack[:8]
    train_pack = pack[8:]
    delta = torch.zeros(len(panels), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.06)
    best, best_key = None, None
    history = []
    fisher_c_t = torch.as_tensor(fisher_c, dtype=torch.float32)
    neural_c_t = torch.as_tensor(neural_c, dtype=torch.float32)
    score_c_t = torch.as_tensor(panel_sensitivities(fisher_c, truth_c), dtype=torch.float32)

    for epoch in range(int(args.epochs)):
        opt.zero_grad()
        scale = torch.exp(delta)
        loss = 0.05 * delta.pow(2).mean()
        loss = loss + 0.8 * ranking_loss(scores_torch(fisher_c_t, scale[:, None, None] * neural_c_t), score_c_t)
        pick = np.random.default_rng(epoch + 9).choice(len(train_pack), size=min(8, len(train_pack)), replace=False)
        for j in pick:
            fisher_t, _ti, s_info, score_t = train_pack[int(j)]
            loss = loss + ranking_loss(scores_torch(fisher_t, scale[:, None, None] * s_info), score_t) / len(pick)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-1.8, 1.8)
            scale_np = torch.exp(delta).cpu().numpy()
        tops, spears = [], []
        for fisher_t, t_info, s_info, _st in monitor:
            top, spear = overlap_count(fisher_t.numpy(), t_info.numpy(), scale_np[:, None, None] * s_info.numpy())
            tops.append(top)
            spears.append(spear)
        cal_rep = information_fidelity_report(fisher_c, truth_c, scale_np[:, None, None] * neural_c)
        key = (
            int(np.mean(tops) >= 5 and np.mean(spears) >= 0.80),
            float(np.mean(tops)),
            float(np.mean(spears)),
            int(not cal_rep["failed_gates"]),
            cal_rep["top_k_overlap_count"],
            cal_rep["panel_ranking_spearman"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best = scale_np.copy()
        if epoch % 20 == 0 or epoch + 1 == int(args.epochs):
            print(
                f"epoch {epoch} loss {float(loss.detach()):.3f} "
                f"mon top={np.mean(tops):.2f} S={np.mean(spears):.3f} "
                f"cal675 {fmt(cal_rep)}",
                flush=True,
            )
        history.append({"epoch": epoch, "mon_top": float(np.mean(tops)), "mon_S": float(np.mean(spears))})

    scale = np.asarray(best, dtype=np.float64)
    official = information_fidelity_report(fisher_o, truth_o, neural_o * scale[:, None, None])
    cal_best = information_fidelity_report(fisher_c, truth_c, neural_c * scale[:, None, None])
    print("best cal675", fmt(cal_best), flush=True)
    print("best official", fmt(official), flush=True)
    print("truth", official["reference_top_panel_indices"], flush=True)
    print("cand ", official["candidate_top_panel_indices"], flush=True)

    payload = dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    payload["phi_info_scale"] = scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.log.write_text(json.dumps({
        "source": str(args.ckpt),
        "output": str(args.output),
        "seconds": time.time() - t0,
        "raw_official": {"S": raw_o["panel_ranking_spearman"], "top": raw_o["top_k_overlap_count"], "failed": raw_o["failed_gates"]},
        "raw_cal": {"S": raw_c["panel_ranking_spearman"], "top": raw_c["top_k_overlap_count"], "failed": raw_c["failed_gates"]},
        "best_cal": {"S": cal_best["panel_ranking_spearman"], "top": cal_best["top_k_overlap_count"], "failed": cal_best["failed_gates"]},
        "best_official": {
            "S": official["panel_ranking_spearman"],
            "top": official["top_k_overlap_count"],
            "design": official["validation_design_ratio"],
            "failed": official["failed_gates"],
            "truth": official["reference_top_panel_indices"],
            "cand": official["candidate_top_panel_indices"],
        },
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "best_key": list(best_key),
        "history_tail": history[-6:],
    }, indent=2), encoding="utf-8")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
