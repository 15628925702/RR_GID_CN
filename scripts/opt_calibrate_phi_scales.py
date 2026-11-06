"""Calibrate per-panel I_S scales on frozen phi-net, no official-512 labels.

v4phi already has official Spearman 0.80; top-10 fails because the head is
compressed. A 120-vector of positive scales is linear in panel_sensitivities
and can swap near-ties without retraining E[phi|x_S]. Scales are fit on many
random TRAIN 512-outers plus the full-train ranking, then evaluated on the
official val 512.
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
    _stable_tilt_weights,
    empirical_conditional_means,
    empirical_information_basis,
    load_gas_generator,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import (  # noqa: E402
    information_fidelity_report,
    panel_sensitivities,
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
    comp_idx = order[10:30]
    ranks = approx_ranks(scores, tau)
    rank_term = torch.relu(ranks[top_idx] - 10.0).mean()
    margin = (score_t[top_idx].min() - score_t[comp_idx].max()).abs().clamp(min=1.0) * 0.25
    hinge = torch.relu(margin - (scores[top_idx][:, None] - scores[comp_idx][None, :])).mean()
    impostors = torch.argsort(scores, descending=True)[:10]
    push = torch.relu(scores[impostors] - scores[top_idx].min()).mean()
    score_term = torch.mean((scores - score_t).pow(2) / score_t.var().clamp_min(1e-8))
    return 1.5 * listnet + 4.0 * rank_term + 3.0 * hinge + 1.5 * push + 0.5 * score_term


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
    ap.add_argument("--teacher-cache", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/gas_phi_teacher_train.npz"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi_scaled.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi_scaled.json"))
    ap.add_argument("--n-subsets", type=int, default=48)
    ap.add_argument("--outer-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=250)
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
    tilt = _stable_tilt_weights(phi_train, beta)

    if args.teacher_cache.exists():
        cache = np.load(args.teacher_cache)
        q_idx = cache["q_idx"]
        teacher = cache["teacher"]
        query_phi = cache["query_phi"]
        query_w = cache["query_w"]
        print("loaded teacher cache", args.teacher_cache, teacher.shape, flush=True)
    else:
        q_idx = np.arange(len(ref_train))
        query_phi = phi_train
        query_w = _stable_tilt_weights(query_phi, beta)
        teacher = np.zeros((len(panels), len(ref_train), 16), dtype=np.float32)
        print(f"computing train teacher {len(ref_train)} x {len(panels)}", flush=True)
        for i, panel in enumerate(panels):
            teacher[i] = empirical_conditional_means(
                ref_train, phi_train, ref_train[:, list(panel)], panel,
                donor_tilt_weights=tilt, neighbors=60,
            )
            if (i + 1) % 30 == 0:
                print(f"  teacher {i + 1}/{len(panels)}", flush=True)
        args.teacher_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.teacher_cache,
            q_idx=q_idx, teacher=teacher, query_phi=query_phi, query_w=query_w, beta=beta,
        )
        print("wrote", args.teacher_cache, flush=True)

    spec = {"generator_kind": "vaeac", "checkpoint": str(args.ckpt)}
    gen = load_gas_generator(spec, data, fn, device=args.device)
    queries = ref_train[q_idx]
    print("computing frozen phi-net means on train", flush=True)
    student = np.zeros_like(teacher)
    for i, panel in enumerate(panels):
        student[i] = gen.conditional_mean_phi_net(queries[:, list(panel)], panel, i)
        if (i + 1) % 30 == 0:
            print(f"  student {i + 1}/{len(panels)}", flush=True)

    n_q = teacher.shape[1]
    rng = np.random.default_rng(202609880)
    subsets = [rng.choice(n_q, size=min(int(args.outer_size), n_q), replace=False) for _ in range(int(args.n_subsets))]
    pack = []
    teacher_top_freq = np.zeros(len(panels), dtype=np.float64)
    student_top_freq = np.zeros(len(panels), dtype=np.float64)
    for sel in subsets:
        fisher = _project_psd(weighted_covariance(query_phi[sel], query_w[sel]))
        t_info = infos_from_means(teacher[:, sel], query_w[sel])
        s_info = infos_from_means(student[:, sel], query_w[sel])
        t_sc = panel_sensitivities(fisher, t_info)
        s_sc = panel_sensitivities(fisher, s_info)
        teacher_top_freq[np.argsort(-t_sc)[:10]] += 1
        student_top_freq[np.argsort(-s_sc)[:10]] += 1
        pack.append((
            torch.as_tensor(fisher, dtype=torch.float32),
            torch.as_tensor(t_info, dtype=torch.float32),
            torch.as_tensor(s_info, dtype=torch.float32),
            torch.as_tensor(t_sc, dtype=torch.float32),
        ))
    teacher_top_freq /= len(subsets)
    student_top_freq /= len(subsets)
    stable_head = np.argsort(-teacher_top_freq)[:15]
    print("stable teacher head freq", {int(i): round(float(teacher_top_freq[i]), 3) for i in stable_head}, flush=True)
    print("student freq on those", {int(i): round(float(student_top_freq[i]), 3) for i in stable_head}, flush=True)

    fisher_full = _project_psd(weighted_covariance(query_phi, query_w))
    t_full = infos_from_means(teacher, query_w)
    s_full = infos_from_means(student, query_w)
    t_full_sc = panel_sensitivities(fisher_full, t_full)
    print("full-train raw", fmt(information_fidelity_report(fisher_full, t_full, s_full)), flush=True)

    official_idx = np.random.default_rng(202609700).choice(len(ref_val), size=min(512, len(ref_val)), replace=False)
    outer_o = ref_val[official_idx]
    print("official 512 kNN basis", flush=True)
    fisher_o, truth_o, _ = empirical_information_basis(
        ref_train, phi_train, outer_o, fn(outer_o), panels, beta, neighbors=60,
    )
    _, neural_o, _ = gen.panel_information(beta, panels, outer_o, estimator="phi_net")
    raw_official = information_fidelity_report(fisher_o, truth_o, neural_o)
    print("official raw", fmt(raw_official), flush=True)

    delta = torch.zeros(len(panels), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.08)
    head_idx = torch.as_tensor(stable_head, dtype=torch.long)
    best = None
    best_key = None
    history = []
    for epoch in range(int(args.epochs)):
        opt.zero_grad()
        scale = torch.exp(delta)
        loss = 0.02 * delta.pow(2).mean()
        loss = loss - 0.15 * delta.index_select(0, head_idx).mean()
        fisher_f = torch.as_tensor(fisher_full, dtype=torch.float32)
        infos_f = scale[:, None, None] * torch.as_tensor(s_full, dtype=torch.float32)
        score_f = scores_torch(fisher_f, infos_f)
        score_ft = torch.as_tensor(t_full_sc, dtype=torch.float32)
        loss = loss + 0.75 * ranking_loss(score_f, score_ft)
        pick = np.random.default_rng(epoch + 3).choice(len(pack), size=min(12, len(pack)), replace=False)
        for j in pick:
            fisher_t, _t_info, s_info, score_t = pack[int(j)]
            scores = scores_torch(fisher_t, scale[:, None, None] * s_info)
            loss = loss + ranking_loss(scores, score_t) / len(pick)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-2.2, 2.2)
            scale_np = torch.exp(delta).cpu().numpy()
        # monitor: mean top-k / spearman over all train subsets
        spears, tops = [], []
        for fisher_t, t_info, s_info, _score_t in pack:
            cand = (scale_np[:, None, None] * s_info.numpy())
            rep = information_fidelity_report(fisher_t.numpy(), t_info.numpy(), cand)
            spears.append(rep["panel_ranking_spearman"])
            tops.append(rep["top_k_overlap_count"])
        full_rep = information_fidelity_report(fisher_full, t_full, scale_np[:, None, None] * s_full)
        key = (
            int(np.mean(tops) >= 5.0 and np.mean(spears) >= 0.80 and not full_rep["failed_gates"]),
            float(np.mean(tops)),
            float(np.mean(spears)),
            int(not full_rep["failed_gates"]),
            full_rep["top_k_overlap_count"],
            full_rep["panel_ranking_spearman"],
        )
        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "subset_top": float(np.mean(tops)),
            "subset_S": float(np.mean(spears)),
            "full_top": full_rep["top_k_overlap_count"],
            "full_S": full_rep["panel_ranking_spearman"],
        }
        history.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best = scale_np.copy()
        if epoch % 25 == 0 or epoch + 1 == int(args.epochs):
            print(
                f"epoch {epoch} loss {row['loss']:.3f} subset top={row['subset_top']:.2f} "
                f"S={row['subset_S']:.3f} full {fmt(full_rep)}",
                flush=True,
            )

    scale = np.asarray(best, dtype=np.float64)
    scaled_o = neural_o * scale[:, None, None]
    official = information_fidelity_report(fisher_o, truth_o, scaled_o)
    print("official scaled", fmt(official), flush=True)
    print("official truth", official["reference_top_panel_indices"], flush=True)
    print("official cand ", official["candidate_top_panel_indices"], flush=True)
    print("scale min/median/max", float(scale.min()), float(np.median(scale)), float(scale.max()), flush=True)

    payload = dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    payload["phi_info_scale"] = scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    log = {
        "source": str(args.ckpt),
        "output": str(args.output),
        "seconds": time.time() - t0,
        "raw_official": {
            "panel_ranking_spearman": raw_official["panel_ranking_spearman"],
            "top_k_overlap_count": raw_official["top_k_overlap_count"],
            "validation_design_ratio": raw_official["validation_design_ratio"],
            "failed_gates": raw_official["failed_gates"],
        },
        "scaled_official": {
            "panel_ranking_spearman": official["panel_ranking_spearman"],
            "top_k_overlap_count": official["top_k_overlap_count"],
            "validation_design_ratio": official["validation_design_ratio"],
            "failed_gates": official["failed_gates"],
            "reference_top_panel_indices": official["reference_top_panel_indices"],
            "candidate_top_panel_indices": official["candidate_top_panel_indices"],
        },
        "stable_teacher_head": [int(i) for i in stable_head],
        "teacher_top_freq": teacher_top_freq.tolist(),
        "student_top_freq": student_top_freq.tolist(),
        "scale": scale.tolist(),
        "history_tail": history[-8:],
        "best_key": list(best_key),
    }
    args.log.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
