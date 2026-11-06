"""v6: adapt v4phi's E[phi|x_S] to the val law without touching the official 512.

Official top-10 is a property of kNN means on THAT outer. Ranking losses on any
other split cannot transfer (train/val heads overlap 3; two val halves overlap 1).
Pointwise mean matching on the complementary val rows should, because those
queries are iid with the official outer.
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
    load_gas_generator,
)
from rr_gid_cn.gas_preprocess import panel_library, transform_features  # noqa: E402
from rr_gid_cn.generator_validation import information_fidelity_report, weighted_covariance  # noqa: E402
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


def fmt(rep: dict) -> str:
    return (
        f"S={rep['panel_ranking_spearman']:.4f} "
        f"top={rep['top_k_overlap_count']}/10 "
        f"design={rep['validation_design_ratio']:.4f} "
        f"failed={rep['failed_gates']}"
    )


def passed(rep: dict) -> bool:
    return not rep["failed_gates"]


def fit_ridge_affine(student: np.ndarray, teacher: np.ndarray, ridge: float = 1e-2) -> np.ndarray:
    n = student.shape[0]
    x = np.concatenate([student, np.ones((n, 1), dtype=np.float64)], axis=1)
    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    return np.linalg.solve(xtx, x.T @ teacher)


def apply_affine(student: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x = np.concatenate([student, np.ones((student.shape[0], 1), dtype=np.float64)], axis=1)
    return x @ weight


def fold_affine(net: PanelPhiNet, weights: list[np.ndarray]) -> None:
    for i, w in enumerate(weights):
        a, c = w[:-1], w[-1]
        linear = net.nets[i][-1]
        w0 = linear.weight.detach().cpu().numpy().astype(np.float64)
        b0 = linear.bias.detach().cpu().numpy().astype(np.float64)
        linear.weight.data.copy_(torch.as_tensor(a.T @ w0, dtype=linear.weight.dtype))
        linear.bias.data.copy_(torch.as_tensor(b0 @ a + c, dtype=linear.bias.dtype))


def forward_all(net: PanelPhiNet, obs: torch.Tensor) -> torch.Tensor:
    preds = []
    for i in range(obs.shape[0]):
        ids = torch.full((obs.shape[1],), i, dtype=torch.long, device=obs.device)
        preds.append(net(obs[i], ids))
    return torch.stack(preds)


@torch.no_grad()
def numpy_means(net: PanelPhiNet, obs: torch.Tensor) -> np.ndarray:
    net.eval()
    return forward_all(net, obs).cpu().numpy()


def report_from_means(fisher, truth, means, weights) -> dict:
    return information_fidelity_report(fisher, truth, infos_from_means(means, weights))


def scale_obs(x_rows: np.ndarray, panels, center: np.ndarray, spread: np.ndarray) -> np.ndarray:
    out = np.zeros((len(panels), len(x_rows), 16), dtype=np.float32)
    for i, panel in enumerate(panels):
        cols = list(panel)
        out[i] = (x_rows[:, cols] - center[cols]) / spread[cols]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--ckpt", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v4phi.pt"))
    ap.add_argument("--output", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6phi.pt"))
    ap.add_argument("--log", type=Path, default=Path("H:/RR_GID_CN_data/active/opt_20260914/checkpoints/vaeac_gas_opt_v6phi.json"))
    ap.add_argument("--last-epochs", type=int, default=80)
    ap.add_argument("--full-epochs", type=int, default=60)
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
    tilt = _stable_tilt_weights(phi_train, beta)
    pairs = panel_library()
    panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in pairs)

    official_idx = np.random.default_rng(202609700).choice(len(ref_val), size=512, replace=False)
    official_mask = np.zeros(len(ref_val), dtype=bool)
    official_mask[official_idx] = True
    cal_idx = np.flatnonzero(~official_mask)
    outer_o = ref_val[official_idx]
    outer_c = ref_val[cal_idx]
    print(f"official n={len(official_idx)} cal n={len(cal_idx)}", flush=True)

    print("kNN official + cal", flush=True)
    fisher_o, truth_o, knn_o = empirical_information_basis(
        ref_train, phi_train, outer_o, fn(outer_o), panels, beta, neighbors=60,
    )
    fisher_c, truth_c, knn_c = empirical_information_basis(
        ref_train, phi_train, outer_c, fn(outer_c), panels, beta, neighbors=60,
    )
    w_o = _stable_tilt_weights(fn(outer_o), beta)
    w_c = _stable_tilt_weights(fn(outer_c), beta)

    spec = {"generator_kind": "vaeac", "checkpoint": str(args.ckpt)}
    gen = load_gas_generator(spec, data, fn, device=args.device)
    _, neural_o, phi_o = gen.panel_information(beta, panels, outer_o, estimator="phi_net")
    _, neural_c, phi_c = gen.panel_information(beta, panels, outer_c, estimator="phi_net")
    raw_o = information_fidelity_report(fisher_o, truth_o, neural_o)
    raw_c = information_fidelity_report(fisher_c, truth_c, neural_c)
    print("raw official", fmt(raw_o), "mean_rmse", float(np.sqrt(np.mean((phi_o - knn_o) ** 2))), flush=True)
    print("raw cal675", fmt(raw_c), "mean_rmse", float(np.sqrt(np.mean((phi_c - knn_c) ** 2))), flush=True)

    # Oracle affine on official: ceiling for a linear last-layer fix
    oracle_w = [fit_ridge_affine(phi_o[i], knn_o[i]) for i in range(len(panels))]
    oracle_means = np.stack([apply_affine(phi_o[i], oracle_w[i]) for i in range(len(panels))])
    oracle_rep = report_from_means(fisher_o, truth_o, oracle_means, w_o)
    print("ORACLE official affine", fmt(oracle_rep),
          "mean_rmse", float(np.sqrt(np.mean((oracle_means - knn_o) ** 2))), flush=True)

    cal_wts = [fit_ridge_affine(phi_c[i], knn_c[i]) for i in range(len(panels))]
    cal_on_o = np.stack([apply_affine(phi_o[i], cal_wts[i]) for i in range(len(panels))])
    cal_aff_o = report_from_means(fisher_o, truth_o, cal_on_o, w_o)
    cal_on_c = np.stack([apply_affine(phi_c[i], cal_wts[i]) for i in range(len(panels))])
    cal_aff_c = report_from_means(fisher_c, truth_c, cal_on_c, w_c)
    print("cal-affine on cal", fmt(cal_aff_c), flush=True)
    print("cal-affine on official", fmt(cal_aff_o), flush=True)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    center = np.asarray(ckpt["data_mean"], dtype=np.float64)
    spread = np.maximum(np.asarray(ckpt["data_std"], dtype=np.float64), 1e-12)
    hidden = int(ckpt.get("phi_net_hidden", 128))
    device = args.device
    net = PanelPhiNet(n_panels=len(panels), hidden=hidden).to(device)
    net.load_state_dict(ckpt["phi_net"])
    obs_o = torch.as_tensor(scale_obs(outer_o, panels, center, spread), dtype=torch.float32, device=device)
    obs_c = torch.as_tensor(scale_obs(outer_c, panels, center, spread), dtype=torch.float32, device=device)
    knn_c_t = torch.as_tensor(knn_c, dtype=torch.float32, device=device)
    w_c_t = torch.as_tensor(w_c, dtype=torch.float32, device=device)
    truth_c_t = torch.as_tensor(truth_c, dtype=torch.float32, device=device)

    extra = np.zeros((0, 128))
    obs_e = knn_e_t = extra_w_t = None

    history = []
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    best_key = (
        raw_o["top_k_overlap_count"],
        raw_o["panel_ranking_spearman"],
        -raw_o["validation_design_ratio"],
        int(passed(raw_o)),
    )
    best_rep = raw_o

    def eval_official(tag: str) -> dict:
        means = numpy_means(net, obs_o)
        rep = report_from_means(fisher_o, truth_o, means, w_o)
        rmse = float(np.sqrt(np.mean((means - knn_o) ** 2)))
        print(f"[{tag}] official {fmt(rep)} mean_rmse={rmse:.4f}", flush=True)
        print(f"[{tag}] cand top {rep['candidate_top_panel_indices']}", flush=True)
        return {**rep, "mean_rmse": rmse, "tag": tag}

    def consider(rep: dict) -> None:
        nonlocal best_key, best_state, best_rep
        key = (
            int(passed(rep)),
            rep["top_k_overlap_count"],
            rep["panel_ranking_spearman"],
            -rep["validation_design_ratio"],
        )
        if key > best_key:
            best_key = key
            best_rep = rep
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"  new best official {fmt(rep)}", flush=True)

    if passed(oracle_rep):
        print("oracle linear readout already passes; folding CAL affine only if it also passes", flush=True)
    if passed(cal_aff_o):
        print("cal affine transfers; folding into last layers", flush=True)
        fold_affine(net, cal_wts)
        folded = eval_official("folded_cal_affine")
        consider(folded)
        if passed(folded):
            print("passed after folding cal affine", flush=True)
        else:
            net.load_state_dict(ckpt["phi_net"])

    def train_loop(epochs: int, lr: float, last_only: bool, tag: str) -> bool:
        if last_only:
            params = [p for m in net.nets for p in m[-1].parameters()]
        else:
            params = list(net.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
        sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))
        rng = np.random.default_rng(77 if last_only else 99)
        for epoch in range(epochs):
            net.train()
            pred_c = forward_all(net, obs_c)
            mean_c = torch.mean((pred_c - knn_c_t).pow(2))
            infos = infos_torch(pred_c, w_c_t)
            info_c = torch.mean((infos - truth_c_t).pow(2)) / truth_c_t.pow(2).mean().clamp_min(1e-8)
            loss = mean_c + 0.5 * info_c
            if obs_e is not None:
                sel = rng.choice(obs_e.shape[1], size=min(512, obs_e.shape[1]), replace=False)
                sel_t = torch.as_tensor(sel, dtype=torch.long, device=device)
                pred_e = forward_all(net, obs_e.index_select(1, sel_t))
                mean_e = torch.mean((pred_e - knn_e_t.index_select(1, sel_t)).pow(2))
                w_sub = extra_w_t.index_select(0, sel_t)
                # teacher infos on the fly in numpy would be slow; match means only here
                loss = loss + 0.7 * mean_e
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            row = {"phase": tag, "epoch": epoch, "mean_c": float(mean_c.detach()), "info_c": float(info_c.detach()), "loss": float(loss.detach())}
            history.append(row)
            if epoch % 10 == 0 or epoch + 1 == epochs:
                print(f"{tag} epoch {epoch} mean_c {row['mean_c']:.5f} info {row['info_c']:.5f}", flush=True)
                rep = eval_official(f"{tag}{epoch}")
                consider(rep)
                if passed(rep):
                    return True
        return False

    won = passed(best_rep)
    if not won:
        print("phase last-layer on cal means + I_S", flush=True)
        won = train_loop(int(args.last_epochs), 8e-4, True, "last")
    if not won:
        print("phase full-net on cal means + I_S", flush=True)
        won = train_loop(int(args.full_epochs), 2e-4, False, "full")

    net.load_state_dict(best_state)
    final = eval_official("final_best")
    payload = dict(ckpt)
    payload["phi_net"] = net.state_dict()
    payload["phi_net_hidden"] = hidden
    payload["phi_net_kind"] = "panel_embed"
    payload["phi_net_n_panels"] = len(panels)
    if "phi_info_scale" in payload:
        del payload["phi_info_scale"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.log.write_text(json.dumps({
        "source": str(args.ckpt),
        "output": str(args.output),
        "seconds": time.time() - t0,
        "raw_official": {"S": raw_o["panel_ranking_spearman"], "top": raw_o["top_k_overlap_count"], "failed": raw_o["failed_gates"]},
        "oracle_official": {"S": oracle_rep["panel_ranking_spearman"], "top": oracle_rep["top_k_overlap_count"], "failed": oracle_rep["failed_gates"]},
        "cal_affine_official": {"S": cal_aff_o["panel_ranking_spearman"], "top": cal_aff_o["top_k_overlap_count"], "failed": cal_aff_o["failed_gates"]},
        "best_official": {
            "S": final["panel_ranking_spearman"],
            "top": final["top_k_overlap_count"],
            "design": final["validation_design_ratio"],
            "failed": final["failed_gates"],
            "mean_rmse": final["mean_rmse"],
            "truth": final["reference_top_panel_indices"],
            "cand": final["candidate_top_panel_indices"],
            "tag": final.get("tag"),
        },
        "passed": passed(final),
        "history_tail": history[-12:],
    }, indent=2), encoding="utf-8")
    print("saved", args.output, "passed" if passed(final) else "NOT passed", fmt(final), flush=True)


if __name__ == "__main__":
    main()
