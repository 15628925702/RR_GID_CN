"""Paper E5-R1: Gas well-specified semi-synthetic budget curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from rr_gid_cn.gas_preprocess import panel_library, transform_features
from rr_gid_cn.p4_integrity import compute_pilot_budget, sha256_file, validate_experiment_mode
from rr_gid_cn.policies import frank_wolfe, objective, uniform_probabilities
from rr_gid_cn.s1_gate import discriminative_design, largest_remainder_counts
from rr_gid_cn.empirical_generator import load_gas_generator
from rr_gid_cn.vaeac import imp_conditional_mean_from_generator, learned_information


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_keys(path: Path) -> set[tuple[str, int, int]]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((str(row.get("method") or row.get("policy")), int(row["budget"]), int(row.get("replication", 0))))
    return keys


def seed_entry(budget: int, replication: int) -> dict:
    seed = 202609100 + int(budget) * 1000 + int(replication)
    return {
        "budget": int(budget),
        "replication": int(replication),
        "replication_seed": seed,
        "target_draw_seed": seed + 3,
        "pilot_or_design_seed": seed + 11,
        "score_seed_root": seed + 4,
        "information_seed_root": seed + 7000,
    }


def write_manifest(path: Path, budgets, replications: int) -> None:
    rows = [seed_entry(budget, rep) for budget in budgets for rep in range(int(replications))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def log_partition(phi: np.ndarray, beta: np.ndarray) -> float:
    logits = np.asarray(phi) @ np.asarray(beta)
    return float(np.logaddexp.reduce(logits) - np.log(len(phi)))


def bregman_projection_loss(phi_pool: np.ndarray, beta_hat: np.ndarray, beta_dag: np.ndarray) -> float:
    phi_pool = np.asarray(phi_pool, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    beta_dag = np.asarray(beta_dag, dtype=float)
    logits = phi_pool @ beta_dag
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    grad = weights @ phi_pool
    raw = log_partition(phi_pool, beta_hat) - log_partition(phi_pool, beta_dag) - float(grad @ (beta_hat - beta_dag))
    if raw < -1e-8:
        raise FloatingPointError(f"negative Bregman divergence {raw:.6g}")
    return float(max(raw, 0.0))


def calibrate_beta(phi: np.ndarray, seed: int = 2026, target_ess: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=phi.shape[1])
    direction /= np.linalg.norm(direction)
    target = target_ess * len(phi)
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mag = (lo + hi) / 2
        logits = phi @ (mag * direction)
        weights = np.exp(logits - logits.max())
        ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
        if ess > target:
            lo = mag
        else:
            hi = mag
    return ((lo + hi) / 2) * direction


def empirical_conditional_mean(phi: np.ndarray, ref_pool: np.ndarray, observed: np.ndarray, panel, k: int = 60) -> np.ndarray:
    idx = list(panel)
    pool = ref_pool[:, idx]
    xp = np.atleast_2d(np.asarray(observed))
    d2 = np.sum(xp ** 2, axis=1)[:, None] + np.sum(pool ** 2, axis=1)[None, :] - 2.0 * (xp @ pool.T)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    d2_nn = np.take_along_axis(d2, nn, axis=1)
    band = np.median(d2_nn, axis=1)
    wk = np.exp(-d2_nn / (2 * band[:, None] ** 2 + 1e-8))
    wk /= wk.sum(axis=1, keepdims=True)
    return np.sum(wk[:, :, None] * phi[nn], axis=1)


def gas_a_optimal_information(phi_ref: np.ndarray, sensor_pairs) -> np.ndarray:
    x = np.asarray(phi_ref, dtype=float)
    dim = x.shape[1]
    inv_full = np.linalg.inv(np.cov(x, rowvar=False) + 1e-6 * np.eye(dim))
    infos = []
    for a, b in sensor_pairs:
        idx = sorted(set([a, b] + ([a + 8] if a + 8 < dim else []) + ([b + 8] if b + 8 < dim else [])))
        info = np.zeros((dim, dim))
        info[np.ix_(idx, idx)] = inv_full[np.ix_(idx, idx)]
        infos.append(info)
    return np.asarray(infos)


def gas_balanced_pilot_counts(coord_panels, budget: int) -> np.ndarray:
    counts = np.zeros(len(coord_panels), dtype=int)
    indices = []
    for sensor in range(8):
        panel = tuple(j for s in (sensor, sensor + 8) for j in range(s * 8, (s + 1) * 8))
        if panel in coord_panels:
            indices.append(coord_panels.index(panel))
    if not indices:
        return np.floor(budget * uniform_probabilities(len(coord_panels))).astype(int)
    base, rem = divmod(int(budget), len(indices))
    counts[indices] = base
    counts[indices[:rem]] += 1
    return counts


def gas_pilot_ht_moment(fn, pilot_obs, pilot_counts, coord_panels) -> np.ndarray:
    n0 = int(pilot_counts.sum())
    values = np.zeros(16)
    if n0 <= 0:
        return values
    sensor_of = {j: j // 8 for j in range(128)}
    panel_sensors = [set(sensor_of[j] for j in panel) for panel in coord_panels]
    rho = np.zeros(16)
    for a in range(8):
        rho[a] = sum(c for c, ps in zip(pilot_counts, panel_sensors) if a in ps) / n0
        rho[8 + a] = sum(c for c, ps in zip(pilot_counts, panel_sensors) if {a, a + 8}.issubset(ps)) / n0
    collected = {a: [] for a in range(16)}
    for panel, observed in pilot_obs:
        ps = set(sensor_of[j] for j in panel)
        full = np.zeros(128)
        full[list(panel)] = observed
        phi_row = fn(full[None, :])[0]
        for a in range(8):
            if a in ps:
                collected[a].append(phi_row[a])
            if {a, a + 8}.issubset(ps):
                collected[8 + a].append(phi_row[8 + a])
    for a in range(16):
        if rho[a] > 0 and collected[a]:
            values[a] = float(np.sum(collected[a])) / (n0 * rho[a])
    return values


def solve_pilot_beta(mu_pil, phi_val, steps=30, theta_bound=4.0, norm_cap=2.0) -> np.ndarray:
    beta = np.zeros(16)
    target = np.asarray(mu_pil, dtype=float)
    features = np.asarray(phi_val, dtype=float)
    for _ in range(max(steps, 100)):
        z = features @ beta
        w = np.exp(z - z.max())
        w /= w.sum()
        mu = w @ features
        diff = target - mu
        if np.linalg.norm(diff) < 1e-8:
            break
        fisher = np.cov(features, rowvar=False, aweights=w) + 1e-3 * np.eye(16)
        direction = np.linalg.solve(fisher, diff)
        step = 1.0
        current = log_partition(features, beta) - float(beta @ target)
        candidate = beta
        while step > 1e-5:
            candidate = np.clip(beta + step * direction, -theta_bound, theta_bound)
            nrm = float(np.linalg.norm(candidate))
            if nrm > norm_cap:
                candidate *= norm_cap / nrm
            if log_partition(features, candidate) - float(candidate @ target) <= current + 1e-9:
                break
            step *= 0.5
        if step <= 1e-5 or np.linalg.norm(candidate - beta) < 1e-8:
            break
        beta = candidate
    return beta


def build_oracle(ref_val, phi_val, fn, coord_panels, n_oracle=600, seed=1234):
    rng = np.random.default_rng(seed)
    n = min(len(ref_val), int(n_oracle))
    idx = rng.choice(len(ref_val), size=n, replace=False)
    oracle_x = ref_val[idx]
    oracle_phi = phi_val[idx]
    mu = oracle_phi.mean(0)
    infos = []
    for panel in coord_panels:
        projected = empirical_conditional_mean(oracle_phi, oracle_x, oracle_x[:, list(panel)], panel) - mu
        cov = np.cov(projected, rowvar=False)
        vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
        infos.append((vecs * np.maximum(vals, 1e-10)) @ vecs.T)
    infos = np.asarray(infos)
    fisher = np.cov(phi_val, rowvar=False)
    p_star, gap, _ = frank_wolfe(
        fisher, infos, np.ones(len(coord_panels)), uniform_probabilities(len(coord_panels)),
        tolerance=1e-4, max_iter=300,
    )
    return fisher, infos, p_star, float(gap)


def _project(beta, bound=4.0, norm_cap=2.0) -> np.ndarray:
    x = np.clip(np.asarray(beta, dtype=float), -bound, bound)
    nrm = float(np.linalg.norm(x))
    if nrm > norm_cap:
        x *= norm_cap / nrm
    return x


def score_beta(gen, beta, observations, coord_panels, phi_val, info_by_panel, fn, n_cond, seed, steps, max_step_norm):
    beta_hat = np.asarray(beta, dtype=float).copy()
    grouped = {}
    for panel, row in observations:
        grouped.setdefault(tuple(panel), []).append(np.asarray(row))
    panel_id = {tuple(panel): i for i, panel in enumerate(coord_panels)}
    for update in range(int(steps)):
        logits = phi_val @ beta_hat
        w = np.exp(logits - logits.max())
        w /= w.sum()
        mu = w @ phi_val
        u = np.zeros(16)
        h = 1e-2 * np.eye(16)
        for panel, rows in grouped.items():
            batch = np.asarray(rows)
            means = imp_conditional_mean_from_generator(
                gen, beta_hat, batch, panel, n=int(n_cond),
                seed=int(seed) + update * 1000 + panel_id[panel],
                feature_fn=fn,
            )
            u += (means - mu).sum(0)
            h += len(batch) * np.asarray(info_by_panel[panel], dtype=float)
        step = np.linalg.solve(0.5 * (h + h.T), u)
        nrm = float(np.linalg.norm(step))
        if nrm > float(max_step_norm) > 0:
            step *= float(max_step_norm) / nrm
        beta_hat = _project(beta_hat + step)
    return beta_hat


def run_replication(
    *,
    ref_train, ref_val, phi_val, fn, gen, coord_panels, sensor_pairs,
    fisher, oracle_infos, p_star, budget, entry, cfg, methods,
):
    beta_true = calibrate_beta(phi_val, seed=2026)
    rng = np.random.default_rng(int(entry["target_draw_seed"]))
    logits = phi_val @ beta_true
    w = np.exp(logits - logits.max())
    w /= w.sum()
    target = ref_val[rng.choice(len(ref_val), size=int(budget), replace=True, p=w)]
    target_hash = hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()
    pilot_budget = compute_pilot_budget(cfg["pilot_schedule"], int(budget))
    remaining = int(budget) - int(pilot_budget)
    pilot_counts = gas_balanced_pilot_counts(coord_panels, pilot_budget)
    pilot_obs = []
    cursor = 0
    for panel, count in zip(coord_panels, pilot_counts):
        for row in target[cursor: cursor + int(count)]:
            pilot_obs.append((panel, row[list(panel)]))
        cursor += int(count)
    mu_pil = gas_pilot_ht_moment(fn, pilot_obs, pilot_counts, coord_panels)
    beta_tilde = solve_pilot_beta(mu_pil, phi_val)

    t_info = perf_counter()
    if getattr(gen, "kind", "") == "empirical_knn":
        n_outer = min(len(ref_val), int(cfg.get("information_outer_rows", 512)))
        rng_info = np.random.default_rng(int(entry["information_seed_root"]))
        outer_idx = rng_info.choice(len(ref_val), size=n_outer, replace=False)
        _, learned_stack, _ = gen.panel_information(beta_tilde, coord_panels, ref_val[outer_idx])
    else:
        _, learned_stack = learned_information(
            gen, beta_tilde, coord_panels,
            n_tilted=int(cfg.get("gen_info_tilted", 64)),
            n_conditional=int(cfg.get("gen_info_cond", 16)),
            seed=int(entry["information_seed_root"]),
            feature_fn=fn,
        )
    time_info = perf_counter() - t_info
    learned = {panel: learned_stack[i] for i, panel in enumerate(coord_panels)}
    rr_p, rr_gap, _ = frank_wolfe(
        fisher, learned_stack, np.ones(len(coord_panels)), uniform_probabilities(len(coord_panels)),
        tolerance=1e-4, max_iter=300,
    )
    t_a = perf_counter()
    a_info = gas_a_optimal_information(phi_val, sensor_pairs)
    a_p, _, _ = frank_wolfe(
        np.eye(16), a_info, np.ones(len(coord_panels)), uniform_probabilities(len(coord_panels)),
        tolerance=1e-4, max_iter=300,
    )
    time_a = perf_counter() - t_a
    t_disc = perf_counter()
    disc_p = None
    if "Discriminative Score OED" in methods:
        disc_p = discriminative_design(
            ref_train, ref_val, beta_tilde, coord_panels, np.ones(128),
            seed=int(entry["pilot_or_design_seed"]),
            hidden=int(cfg.get("mlp_hidden", 64)),
            steps=int(cfg.get("mlp_steps", 100)),
            feature_fn=fn,
        )
    time_disc = perf_counter() - t_disc
    policy_p = {
        "Uniform SQD": uniform_probabilities(len(coord_panels)),
        "A-OSQD": a_p,
        "RR-GID": rr_p,
        "Discriminative Score OED": disc_p,
    }
    costs = np.ones(len(coord_panels))
    phi_star = objective(fisher, oracle_infos, p_star, costs)
    rows = []
    for name in methods:
        probabilities = np.asarray(policy_p[name], dtype=float)
        counts = largest_remainder_counts(probabilities, remaining)
        main_obs = []
        cursor = int(pilot_budget)
        for panel, count in zip(coord_panels, counts):
            for row in target[cursor: cursor + int(count)]:
                main_obs.append((panel, row[list(panel)]))
            cursor += int(count)
        t_est = perf_counter()
        beta_hat = score_beta(
            gen, beta_tilde, pilot_obs + main_obs, coord_panels, phi_val, learned, fn,
            n_cond=int(cfg.get("gen_info_cond", 16)),
            seed=int(entry["score_seed_root"]),
            steps=int(cfg["scoring_steps"]),
            max_step_norm=float(cfg["scoring_max_step_norm"]),
        )
        time_est = perf_counter() - t_est
        loss = bregman_projection_loss(phi_val, beta_hat, beta_true)
        design_ratio = float(objective(fisher, oracle_infos, probabilities, costs) / max(phi_star, 1e-12))
        rows.append({
            "schema_version": "paper-result-v1",
            "experiment": "gas_semisynthetic",
            "environment": "gas",
            "method": name,
            "policy": name,
            "budget": int(budget),
            "replication": int(entry["replication"]),
            "primary_metric": "family_projection_bregman",
            "primary_loss": float(loss),
            "projection_loss": float(loss),
            "risk_ratio": float(loss),
            "design_ratio_main": design_ratio,
            "beta_hat": beta_hat.tolist(),
            "pilot_budget": int(pilot_counts.sum()),
            "scoring_steps": int(cfg["scoring_steps"]),
            "allocated_observations": int(pilot_counts.sum() + counts.sum()),
            "target_draw_sha256": target_hash,
            "fw_gap": float(rr_gap) if name == "RR-GID" else None,
            "wall_seconds_total": float(time_info + time_a + time_disc + time_est),
            "wall_seconds_design": float(time_info + time_a + time_disc),
            "wall_seconds_estimation": float(time_est),
            "generator_training_seconds": 0.0,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/gas_semisynthetic.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/gas/processed/gas_processed.npz"))
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument("--output-path", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_experiment_mode(cfg, paper=True, formal=True)
    if not args.data.exists():
        raise SystemExit(f"missing processed gas data: {args.data}")
    ckpt_path = Path(cfg["generator_checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"missing Gas generator artifact: {ckpt_path}")
    data = np.load(args.data)
    ref_train, ref_val = data["ref_train"], data["ref_val"]
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    phi_val = fn(ref_val)
    sensor_pairs = panel_library()
    coord_panels = tuple(tuple(j for s in pair for j in range(s * 8, (s + 1) * 8)) for pair in sensor_pairs)
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    gen = load_gas_generator(cfg, data, fn, device=device)
    if args.profile:
        print(f"building empirical oracle on {len(coord_panels)} panels device={device}", flush=True)
    fisher, oracle_infos, p_star, fw_gap = build_oracle(ref_val, phi_val, fn, coord_panels)
    if args.profile:
        print(f"oracle fw_gap={fw_gap:.3f}", flush=True)
    budgets = [int(args.budget)] if args.budget is not None else [int(b) for b in cfg["budgets"]]
    replications = int(cfg["replications"])
    start, end = (0, replications) if args.rep_range is None else (int(args.rep_range[0]), int(args.rep_range[1]))
    methods = tuple(cfg["methods"])
    manifest_path = Path(cfg["seed_manifest"])
    if not manifest_path.exists():
        write_manifest(manifest_path, cfg["budgets"], replications)
    by_key = {}
    for row in json.loads(manifest_path.read_text(encoding="utf-8")).get("rows", []):
        by_key[(int(row["budget"]), int(row["replication"]))] = row
    out = Path(args.output_path or cfg["output_path"]) / "rows.jsonl"
    done = _done_keys(out) if args.resume else set()
    commit = _git_commit()
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    ckpt_sha = sha256_file(ckpt_path)
    data_sha = sha256_file(args.data)
    t_wall = perf_counter()
    for budget in budgets:
        for rep in range(start, end):
            if args.max_hours is not None and (perf_counter() - t_wall) / 3600.0 >= float(args.max_hours):
                print(f"max-hours {args.max_hours} reached at B={budget} rep={rep}", flush=True)
                print(f"wrote {out}")
                return
            pending = [m for m in methods if (m, int(budget), int(rep)) not in done]
            if not pending:
                continue
            entry = dict(by_key.get((int(budget), int(rep))) or seed_entry(int(budget), int(rep)))
            entry["replication"] = int(rep)
            if args.profile:
                print(f"B={budget} rep={rep} start", flush=True)
            rows = run_replication(
                ref_train=ref_train, ref_val=ref_val, phi_val=phi_val, fn=fn, gen=gen,
                coord_panels=coord_panels, sensor_pairs=sensor_pairs, fisher=fisher,
                oracle_infos=oracle_infos, p_star=p_star, budget=int(budget), entry=entry,
                cfg=cfg, methods=tuple(pending),
            )
            for row in rows:
                row.update({
                    "code_commit": commit,
                    "config_sha256": config_sha,
                    "artifact_sha256": ckpt_sha,
                    "data_sha256": data_sha,
                    "replication_seed": int(entry["replication_seed"]),
                    "target_draw_seed": int(entry["target_draw_seed"]),
                    "pilot_or_design_seed": int(entry["pilot_or_design_seed"]),
                    "score_seed_root": int(entry["score_seed_root"]),
                    "information_seed_root": int(entry["information_seed_root"]),
                    "device": device,
                    "score_backend": cfg.get("score_backend"),
                    "information_backend": cfg.get("information_backend"),
                    "score_qmc_order": cfg.get("score_qmc_order"),
                    "information_qmc_order": cfg.get("information_qmc_order"),
                })
            _append_jsonl(out, rows)
            if args.profile:
                for row in rows:
                    print(
                        f"{row['method']} B={budget} rep={rep} "
                        f"loss={row['projection_loss']:.4f} design={row['design_ratio_main']:.3f}",
                        flush=True,
                    )
            done.update((row["method"], int(budget), int(rep)) for row in rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
