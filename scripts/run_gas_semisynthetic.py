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
from rr_gid_cn.empirical_generator import (
    empirical_conditional_means,
    load_gas_generator,
    _stable_tilt_weights,
)


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


def scale_beta_to_ess(phi: np.ndarray, direction: np.ndarray, target_ess: float = 0.5) -> np.ndarray:
    """Keep the HT direction, put the radius on the same ESS as beta-star."""
    features = np.asarray(phi, dtype=float)
    vec = np.asarray(direction, dtype=float)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 1e-12:
        return np.zeros(features.shape[1], dtype=float)
    vec = vec / nrm
    target = float(target_ess) * len(features)
    lo, hi = 0.0, 8.0
    for _ in range(60):
        mag = 0.5 * (lo + hi)
        logits = features @ (mag * vec)
        weights = np.exp(logits - float(np.max(logits)))
        ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
        if ess > target:
            lo = mag
        else:
            hi = mag
    return ((lo + hi) / 2.0) * vec


def calibrate_beta(phi: np.ndarray, seed: int = 2026, target_ess: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=phi.shape[1])
    direction /= np.linalg.norm(direction)
    return scale_beta_to_ess(phi, direction, target_ess=target_ess)


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


def gas_pilot_knn_moment(gen, fn, pilot_obs, beta=None) -> np.ndarray:
    """Complete-data moment via frozen-generator conditionals on the pilot.

    HT on observed-only features systematically overshoots mu(beta*) and the
    inverse map lands nearly orthogonal to the tilt. Averaging kNN completions
    at beta=0 is the same residual the RR scorer uses.
    """
    if not pilot_obs:
        return np.zeros(16)
    donor_phi = fn(gen.reference)
    tilt = np.zeros(16) if beta is None else np.asarray(beta, dtype=float)
    donor_w = _stable_tilt_weights(donor_phi, tilt)
    acc = []
    for panel, observed in pilot_obs:
        means = empirical_conditional_means(
            gen.reference,
            donor_phi,
            np.atleast_2d(np.asarray(observed)),
            panel,
            donor_tilt_weights=donor_w,
            neighbors=int(getattr(gen, "neighbors", 60)),
        )
        acc.append(means[0])
    return np.mean(np.asarray(acc), axis=0)


def pilot_moment(cfg, *, gen, fn, pilot_obs, pilot_counts, coord_panels) -> np.ndarray:
    kind = str((cfg or {}).get("pilot_moment") or "ht").strip().lower()
    if kind in {"knn", "generator", "complete"}:
        return gas_pilot_knn_moment(gen, fn, pilot_obs)
    return gas_pilot_ht_moment(fn, pilot_obs, pilot_counts, coord_panels)


def cfg_norm_cap(cfg: dict | None, default: float | None = 2.0):
    """Legacy Gas configs omit the key and keep the VAEAC-era radius-2 cap.

    Explicit ``theta_norm_cap: null`` disables the Euclidean ball, matching E1.
    """
    if not cfg or "theta_norm_cap" not in cfg:
        return default
    value = cfg.get("theta_norm_cap")
    if value is None or value == "null":
        return None
    return float(value)


def cfg_pilot_norm_cap(cfg: dict | None, default: float | None = 2.0):
    if not cfg or "pilot_norm_cap" not in cfg:
        return default
    value = cfg.get("pilot_norm_cap")
    if value is None or value == "null":
        return None
    return float(value)


def cfg_theta_bound(cfg: dict | None, default: float = 4.0) -> float:
    if not cfg or cfg.get("theta_bound") is None:
        return float(default)
    return float(cfg.get("theta_bound"))


def scoring_init_beta(beta_tilde: np.ndarray, cfg: dict | None) -> np.ndarray:
    """kNN score from 0 unless explicitly asked to start at the capped pilot.

    Starting on the pilot ball walks the misspecified EE onto ``theta_norm_cap``
    as soon as B is large; the Gas diagnosis that stayed near D≈0.09 started at 0.
    """
    kind = str((cfg or {}).get("scoring_init") or "zero").strip().lower()
    if kind in {"pilot", "beta_tilde", "tilde"}:
        return np.asarray(beta_tilde, dtype=float).copy()
    return np.zeros_like(np.asarray(beta_tilde, dtype=float))


def information_beta(beta_tilde: np.ndarray, cfg: dict | None, beta_true: np.ndarray | None = None) -> np.ndarray:
    """Where RR-GID evaluates frozen-generator I_S.

    ``pilot`` (default) uses the HT pilot. ``zero`` evaluates the same kNN
    operator at the reference, matching A-OSQD's information location.
    ``truth`` / ``oracle`` is a diagnostic ceiling that uses beta-star; not a
    paper method. ``information_scale`` shrinks the pilot before evaluation.
    """
    kind = str((cfg or {}).get("information_at") or "pilot").strip().lower()
    if kind in {"zero", "0", "reference", "q0"}:
        return np.zeros_like(np.asarray(beta_tilde, dtype=float))
    if kind in {"truth", "true", "oracle", "beta_star", "betastar"}:
        if beta_true is None:
            raise ValueError("information_at=truth requires beta_true")
        return np.asarray(beta_true, dtype=float).copy()
    beta = np.asarray(beta_tilde, dtype=float).copy()
    scale = (cfg or {}).get("information_scale")
    if scale is not None and scale != "null":
        beta = beta * float(scale)
    return beta


def rr_design_fisher(
    cfg: dict | None,
    ref_fisher: np.ndarray,
    gen_fisher: np.ndarray | None,
    pool_fisher: np.ndarray | None = None,
) -> np.ndarray:
    """Frank-Wolfe Φ for RR-GID.

    ``generator`` uses the frozen-generator Fisher at the same β as I_S.
    ``reference`` uses empirical Fisher on Q0 (R1-consistent; R2 scoring does
    not live on Q0). ``ahat`` uses Fisher of the A-hat pool that scoring and
    D_A actually use, with the same 1e-3 ridge as Newton.
    """
    kind = str((cfg or {}).get("rr_fisher") or "generator").strip().lower()
    if kind in {"reference", "ref", "oracle", "q0"}:
        return np.asarray(ref_fisher, dtype=float)
    if kind in {"ahat", "pool", "d_a", "da"}:
        if pool_fisher is None:
            raise ValueError("rr_fisher=ahat requires pool_fisher")
        return np.asarray(pool_fisher, dtype=float)
    if gen_fisher is None:
        return np.asarray(ref_fisher, dtype=float)
    return np.asarray(gen_fisher, dtype=float)


def design_panel_information(gen, info_beta, coord_panels, ref_val, cfg, info_seed, *, sensor_pairs, phi_val, feature_fn=None):
    """I_S for Frank-Wolfe. Gated VAEAC uses phi-net mix, not Monte Carlo learned_information."""
    n_outer = min(len(ref_val), int(cfg.get("information_outer_rows", 512)))
    rng_info = np.random.default_rng(int(info_seed))
    outer_idx = rng_info.choice(len(ref_val), size=n_outer, replace=False)
    outer_x = ref_val[outer_idx]
    info_src = str((cfg or {}).get("information_source") or "generator").strip().lower()
    if info_src in {"empirical_knn", "knn", "empirical"}:
        from rr_gid_cn.empirical_generator import EmpiricalKNNGenerator
        knn = EmpiricalKNNGenerator(
            np.asarray(getattr(gen, "reference"), dtype=float),
            feature_fn=feature_fn or getattr(gen, "feature_fn", None),
            neighbors=int(getattr(gen, "neighbors", 60)),
        )
        gen_fisher, learned_stack, _ = knn.panel_information(info_beta, coord_panels, outer_x)
        _, a_stack, _ = knn.panel_information(np.zeros(16), coord_panels, outer_x)
        ridge = cfg.get("information_ridge")
        if ridge is not None and float(ridge) > 0.0:
            learned_stack = np.asarray(learned_stack, dtype=float) + float(ridge) * np.eye(16)[None, :, :]
        return gen_fisher, learned_stack, a_stack
    use_matched = getattr(gen, "phi_net", None) is not None or getattr(gen, "kind", "") == "empirical_knn"
    if use_matched:
        kwargs = {}
        if getattr(gen, "phi_net", None) is not None:
            kwargs["estimator"] = "phi_net"
        gen_fisher, learned_stack, _ = gen.panel_information(info_beta, coord_panels, outer_x, **kwargs)
        ridge = cfg.get("information_ridge")
        if ridge is not None and float(ridge) > 0.0:
            learned_stack = np.asarray(learned_stack, dtype=float) + float(ridge) * np.eye(16)[None, :, :]
        _, a_stack, _ = gen.panel_information(np.zeros(16), coord_panels, outer_x, **kwargs)
        return gen_fisher, learned_stack, a_stack
    from rr_gid_cn.vaeac import learned_information
    _, learned_stack = learned_information(
        gen, info_beta, coord_panels,
        n_tilted=int(cfg.get("gen_info_tilted", 64)),
        n_conditional=int(cfg.get("gen_info_cond", 16)),
        seed=int(info_seed),
        feature_fn=feature_fn,
    )
    return None, learned_stack, gas_a_optimal_information(phi_val, sensor_pairs)


def mix_uniform(probabilities: np.ndarray, mix: float) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    p = np.maximum(p, 0.0)
    p = p / max(float(p.sum()), 1e-12)
    mix = float(max(0.0, min(1.0, mix)))
    if mix <= 0.0:
        return p
    uniform = uniform_probabilities(len(p))
    out = (1.0 - mix) * p + mix * uniform
    return out / out.sum()


def cap_panel_share(probabilities: np.ndarray, max_share: float | None) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    p = np.maximum(p, 0.0)
    p = p / max(float(p.sum()), 1e-12)
    if max_share is None:
        return p
    cap = float(max_share)
    if cap <= 0.0 or cap >= 1.0:
        return p
    for _ in range(32):
        over = p > cap + 1e-12
        if not np.any(over):
            break
        excess = float(p[over].sum() - cap * int(over.sum()))
        p[over] = cap
        free = ~over
        if not np.any(free) or excess <= 0.0:
            break
        mass = float(p[free].sum())
        p[free] += excess * (p[free] / mass if mass > 1e-12 else 1.0 / int(free.sum()))
    p = np.maximum(p, 0.0)
    return p / max(float(p.sum()), 1e-12)


def stabilize_design(probabilities: np.ndarray, cfg: dict | None) -> np.ndarray:
    mix = float((cfg or {}).get("design_uniform_mix", 0.0) or 0.0)
    max_share = (cfg or {}).get("design_max_share")
    p = mix_uniform(probabilities, mix)
    return cap_panel_share(p, None if max_share is None else float(max_share))


def allocation_stats(probabilities: np.ndarray) -> dict:
    p = np.asarray(probabilities, dtype=float)
    p = np.maximum(p, 0.0)
    p = p / max(float(p.sum()), 1e-12)
    entropy = float(-(p * np.log(np.maximum(p, 1e-300))).sum())
    return {
        "design_entropy": entropy,
        "n_effective_panels": float(np.exp(entropy)),
        "n_mass_panels": int(np.sum(p > 1.0 / max(len(p), 1) / 5.0)),
        "max_panel_share": float(p.max()),
    }


def solve_pilot_beta(
    mu_pil,
    phi_val,
    steps=100,
    theta_bound=4.0,
    norm_cap=2.0,
    ridge=1e-3,
    max_step_norm=None,
    ess_target=None,
) -> np.ndarray:
    beta = np.zeros(16)
    target = np.asarray(mu_pil, dtype=float)
    features = np.asarray(phi_val, dtype=float)
    n_iter = max(int(steps), 1)
    ridge = float(ridge)
    for _ in range(n_iter):
        z = features @ beta
        w = np.exp(z - z.max())
        w /= w.sum()
        mu = w @ features
        diff = target - mu
        if np.linalg.norm(diff) < 1e-8:
            break
        fisher = np.cov(features, rowvar=False, aweights=w) + ridge * np.eye(16)
        direction = np.linalg.solve(fisher, diff)
        if max_step_norm is not None:
            cap = float(max_step_norm)
            nrm = float(np.linalg.norm(direction))
            if cap > 0.0 and nrm > cap:
                direction = direction * (cap / nrm)
        step = 1.0
        current = log_partition(features, beta) - float(beta @ target)
        candidate = beta
        while step > 1e-5:
            candidate = _project(beta + step * direction, bound=theta_bound, norm_cap=norm_cap)
            if log_partition(features, candidate) - float(candidate @ target) <= current + 1e-9:
                break
            step *= 0.5
        if step <= 1e-5 or np.linalg.norm(candidate - beta) < 1e-8:
            break
        beta = candidate
    if ess_target is not None and float(np.linalg.norm(beta)) > 1e-12:
        beta = scale_beta_to_ess(features, beta, target_ess=float(ess_target))
        beta = _project(beta, bound=theta_bound, norm_cap=norm_cap)
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
    if norm_cap is None:
        return x
    nrm = float(np.linalg.norm(x))
    cap = float(norm_cap)
    if cap > 0.0 and nrm > cap:
        x *= cap / nrm
    return x


def score_beta(
    gen, beta, observations, coord_panels, phi_val, info_by_panel, fn, n_cond, seed, steps, max_step_norm,
    theta_bound=4.0, norm_cap=None, stop_on_cap=False,
):
    beta_hat = np.asarray(beta, dtype=float).copy()
    grouped = {}
    for panel, row in observations:
        grouped.setdefault(tuple(panel), []).append(np.asarray(row))
    panel_id = {tuple(panel): i for i, panel in enumerate(coord_panels)}
    use_exact = getattr(gen, "kind", "") == "empirical_knn"
    donor_phi = fn(gen.reference) if use_exact else None
    for update in range(int(steps)):
        logits = phi_val @ beta_hat
        w = np.exp(logits - logits.max())
        w /= w.sum()
        mu = w @ phi_val
        fisher = np.cov(phi_val, rowvar=False, aweights=w) + 1e-3 * np.eye(16)
        u = np.zeros(16)
        n_obs = 0
        donor_w = _stable_tilt_weights(donor_phi, beta_hat) if use_exact else None
        for panel, rows in grouped.items():
            batch = np.asarray(rows)
            if use_exact:
                means = empirical_conditional_means(
                    gen.reference,
                    donor_phi,
                    batch,
                    panel,
                    donor_tilt_weights=donor_w,
                    neighbors=int(getattr(gen, "neighbors", 60)),
                )
            else:
                from rr_gid_cn.vaeac import imp_conditional_mean_from_generator
                means = imp_conditional_mean_from_generator(
                    gen, beta_hat, batch, panel, n=int(n_cond),
                    seed=int(seed) + update * 1000 + panel_id[panel],
                    feature_fn=fn,
                )
            u += (means - mu).sum(0)
            n_obs += len(batch)
        h = max(n_obs, 1) * fisher
        raw = np.linalg.solve(0.5 * (h + h.T), u)
        nrm = float(np.linalg.norm(raw))
        if nrm > float(max_step_norm) > 0:
            raw = raw * (float(max_step_norm) / nrm)
        bar_m = mu + u / max(n_obs, 1)
        current = log_partition(phi_val, beta_hat) - float(beta_hat @ bar_m)
        step_scale = 1.0
        accepted = beta_hat
        while step_scale > 1e-4:
            candidate = _project(beta_hat + step_scale * raw, bound=theta_bound, norm_cap=norm_cap)
            trial = log_partition(phi_val, candidate) - float(candidate @ bar_m)
            if trial <= current + 1e-9:
                accepted = candidate
                break
            step_scale *= 0.5
        if float(np.linalg.norm(accepted - beta_hat)) < 1e-10:
            break
        beta_hat = accepted
        if stop_on_cap and norm_cap is not None and float(np.linalg.norm(beta_hat)) >= float(norm_cap) - 1e-8:
            break
    return beta_hat


def apply_score_radius(beta_hat: np.ndarray, beta_tilde: np.ndarray, cfg: dict | None) -> np.ndarray:
    """Sit the scorer on the pilot sphere instead of slamming theta_norm_cap.

    Hats currently hit radius 3 while ``||beta*||=1.22`` and the HT pilot lives
    at radius 2. Rescaling to ``||beta_tilde||`` uses only the pilot.
    """
    kind = str((cfg or {}).get("score_radius") or "cap").strip().lower()
    x = np.asarray(beta_hat, dtype=float)
    if kind not in {"pilot", "tilde"}:
        return x
    radius = float(np.linalg.norm(beta_tilde))
    nrm = float(np.linalg.norm(x))
    if nrm <= 1e-12 or radius <= 0.0:
        return x
    return x * (radius / nrm)


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
    mu_pil = pilot_moment(cfg, gen=gen, fn=fn, pilot_obs=pilot_obs, pilot_counts=pilot_counts, coord_panels=coord_panels)
    theta_bound = cfg_theta_bound(cfg)
    pilot_cap = cfg_pilot_norm_cap(cfg)
    norm_cap = cfg_norm_cap(cfg, default=3.0)
    pilot_steps = cfg.get("pilot_steps")
    pilot_ridge = cfg.get("pilot_ridge")
    pilot_max_step = cfg.get("pilot_max_step_norm")
    pilot_ess = cfg.get("pilot_ess_target")
    beta_tilde = solve_pilot_beta(
        mu_pil, phi_val,
        steps=100 if pilot_steps is None else int(pilot_steps),
        theta_bound=theta_bound,
        norm_cap=pilot_cap,
        ridge=1e-3 if pilot_ridge is None else float(pilot_ridge),
        max_step_norm=None if pilot_max_step in (None, "null") else float(pilot_max_step),
        ess_target=None if pilot_ess in (None, "null") else float(pilot_ess),
    )
    score_init = scoring_init_beta(beta_tilde, cfg)
    info_kind = str(cfg.get("information_at") or "pilot").strip().lower()
    if info_kind in {"score_pilot", "scored", "newton_pilot"}:
        # Two-stage: Newton on the pilot only, then I_S at that hat.
        # I_S(beta*) showed the information location is the remaining gap.
        info_beta = score_beta(
            gen, np.zeros(16), pilot_obs, coord_panels, phi_val, {}, fn,
            n_cond=int(cfg.get("gen_info_cond", 16)),
            seed=int(entry["score_seed_root"]) + 17,
            steps=int(cfg.get("info_score_steps", cfg["scoring_steps"])),
            max_step_norm=float(cfg["scoring_max_step_norm"]),
            theta_bound=theta_bound,
            norm_cap=norm_cap,
        )
    else:
        info_beta = information_beta(beta_tilde, cfg, beta_true=beta_true)

    t_info = perf_counter()
    gen_fisher, learned_stack, a_stack = design_panel_information(
        gen, info_beta, coord_panels, ref_val, cfg, entry["information_seed_root"],
        sensor_pairs=sensor_pairs, phi_val=phi_val, feature_fn=fn,
    )
    time_info = perf_counter() - t_info
    learned = {panel: learned_stack[i] for i, panel in enumerate(coord_panels)}
    rr_fisher = rr_design_fisher(cfg, fisher, gen_fisher, pool_fisher=fisher)
    rr_p, rr_gap, _ = frank_wolfe(
        rr_fisher, learned_stack, np.ones(len(coord_panels)), uniform_probabilities(len(coord_panels)),
        tolerance=1e-4, max_iter=300,
    )
    rr_p = stabilize_design(rr_p, cfg)
    t_a = perf_counter()
    a_p, _, _ = frank_wolfe(
        np.eye(16), a_stack, np.ones(len(coord_panels)), uniform_probabilities(len(coord_panels)),
        tolerance=1e-4, max_iter=300,
    )
    a_p = stabilize_design(a_p, cfg)
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
        disc_p = stabilize_design(disc_p, cfg)
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
        include_pilot = cfg.get("score_include_pilot", True)
        if include_pilot is False or str(include_pilot).strip().lower() in {"false", "0", "no"}:
            score_obs = main_obs
        else:
            score_obs = pilot_obs + main_obs
        stop_on_cap = str(cfg.get("score_stop_on_cap") or "").strip().lower() in {"1", "true", "yes", "on"}
        beta_hat = score_beta(
            gen, score_init, score_obs, coord_panels, phi_val, learned, fn,
            n_cond=int(cfg.get("gen_info_cond", 16)),
            seed=int(entry["score_seed_root"]),
            steps=int(cfg["scoring_steps"]),
            max_step_norm=float(cfg["scoring_max_step_norm"]),
            theta_bound=theta_bound,
            norm_cap=norm_cap,
            stop_on_cap=stop_on_cap,
        )
        beta_hat = apply_score_radius(beta_hat, beta_tilde, cfg)
        time_est = perf_counter() - t_est
        loss = bregman_projection_loss(phi_val, beta_hat, beta_true)
        design_ratio = float(objective(fisher, oracle_infos, probabilities, costs) / max(phi_star, 1e-12))
        stats = allocation_stats(probabilities)
        beta_norm = float(np.linalg.norm(beta_hat))
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
            "beta_hat_norm": beta_norm,
            "beta_star_norm": float(np.linalg.norm(beta_true)),
            "beta_tilde_norm": float(np.linalg.norm(beta_tilde)),
            "score_init_norm": float(np.linalg.norm(score_init)),
            "pilot_ess_target": cfg.get("pilot_ess_target"),
            "pilot_steps": cfg.get("pilot_steps"),
            "information_at": str(cfg.get("information_at") or "pilot"),
            "info_beta_norm": float(np.linalg.norm(info_beta)),
            "rr_fisher": str(cfg.get("rr_fisher") or "generator"),
            "pilot_moment": str(cfg.get("pilot_moment") or "ht"),
            "score_radius": str(cfg.get("score_radius") or "cap"),
            "design_uniform_mix": float(cfg.get("design_uniform_mix") or 0.0),
            "design_max_share": cfg.get("design_max_share"),
            "hit_norm_cap": bool(norm_cap is not None and beta_norm >= float(norm_cap) - 1e-8),
            "theta_norm_cap": None if norm_cap is None else float(norm_cap),
            "theta_bound": float(theta_bound),
            "pilot_budget": int(pilot_counts.sum()),
            "scoring_steps": int(cfg["scoring_steps"]),
            "allocated_observations": int(pilot_counts.sum() + counts.sum()),
            "target_draw_sha256": target_hash,
            "fw_gap": float(rr_gap) if name == "RR-GID" else None,
            "wall_seconds_total": float(time_info + time_a + time_disc + time_est),
            "wall_seconds_design": float(time_info + time_a + time_disc),
            "wall_seconds_estimation": float(time_est),
            "generator_training_seconds": 0.0,
            **stats,
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
