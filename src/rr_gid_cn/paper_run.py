"""Paper bulk path: cached QMC scoring with a shared pilot and one estimator."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .conditional_backend import (
    AdaptiveConditionalMean,
    FixedConditionalMean,
    GroupedPanelData,
    IndexedAdaptiveConditionalMean,
    PanelInformationBasis,
    ReferenceMomentCache,
    build_conditional_feature_basis,
    build_direct_panel_information_basis,
    build_full_law_moment_cache,
    build_panel_information_basis,
    build_shared_score_bases,
    build_shared_adaptive_score_bases,
    cache_counters,
    cholesky_solve,
    DirectConditionalMean,
    gold_information_from_basis,
    reset_cache_counters,
)
from .p4_integrity import compute_pilot_budget
from .policies import frank_wolfe, objective, uniform_probabilities
from .s1_gate import (
    _peak_memory,
    _runtime_device,
    _sync_cuda,
    balanced_pilot_counts,
    design_metrics,
    discriminative_design,
    largest_remainder_counts,
    pilot_ht_moment,
    solve_pilot_beta,
)
from .synthetic_oracle import (
    feature_map,
    reset_workload_counters,
    tilted_full_sample,
    workload_counters,
)


PAPER_METHODS = (
    "Uniform SQD",
    "A-OSQD",
    "Discriminative Score OED",
    "RR-GID",
)


def _project(beta, theta_bound=4.0, norm_cap=None, l1_cap=None):
    from .s1_gate import _project_theta
    return _project_theta(beta, theta_bound, norm_cap, l1_cap)


def final_rr_step(
    beta_start: np.ndarray,
    grouped: GroupedPanelData,
    score_bases: dict,
    info_at_beta: dict,
    reference_cache: ReferenceMomentCache,
    *,
    step_size: float = 1.0,
    max_step_norm: float | None = 2.0,
    theta_bound: float = 4.0,
    norm_cap=None,
    l1_cap=None,
    mu: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    rank = int(np.asarray(beta_start).shape[0])
    H = np.zeros((rank, rank), dtype=np.float64)
    pieces = []
    t_score = perf_counter()
    if mu is None:
        mu, _ = reference_cache.moments(beta_start)
    else:
        mu = np.asarray(mu, dtype=np.float64)
    for panel, count in zip(grouped.panel_ids, grouped.counts):
        means = score_bases[panel].conditional_mean(beta_start)
        pieces.append(means - mu)
        H += int(count) * np.asarray(info_at_beta[panel], dtype=np.float64)
    time_score = perf_counter() - t_score
    U = np.concatenate(pieces, axis=0).sum(axis=0) if pieces else np.zeros(rank)
    H = 0.5 * (H + H.T)
    t_solve = perf_counter()
    step, solve_diag = cholesky_solve(H, U)
    time_solve = perf_counter() - t_solve
    scaled = float(step_size) * step
    if max_step_norm is not None:
        nrm = float(np.linalg.norm(scaled))
        if nrm > float(max_step_norm) > 0.0:
            scaled *= float(max_step_norm) / nrm
    updated = _project(np.asarray(beta_start) + scaled, theta_bound, norm_cap, l1_cap)
    return updated, {
        **solve_diag,
        "score_norm": float(np.linalg.norm(U)),
        "raw_step_norm": float(np.linalg.norm(step)),
        "applied_step_norm": float(np.linalg.norm(updated - beta_start)),
        "time_score": float(time_score),
        "time_H": 0.0,
        "time_mu": 0.0,
        "time_linear_solve": float(time_solve),
    }


def _build_score_bases(
    mixture,
    grouped: GroupedPanelData,
    scale,
    order,
    seed,
    dtype,
    feature_fn=None,
    storage_dir=None,
):
    bases = {}
    t0 = perf_counter()
    spill_root = Path(storage_dir) if storage_dir is not None else None
    if spill_root is not None:
        spill_root.mkdir(parents=True, exist_ok=True)
    for panel in grouped.panel_ids:
        token = "_".join(str(int(i)) for i in panel)
        bases[panel] = build_conditional_feature_basis(
            mixture, grouped.rows(panel), panel,
            order=int(order), seed=int(seed), scale=scale, dtype=dtype,
            feature_fn=feature_fn,
            storage_path=(spill_root / f"score_{token}.dat") if spill_root is not None else None,
        )
    return bases, perf_counter() - t0


def run_paper_replication(
    mixture,
    scale,
    panels,
    budget: int,
    seed: int,
    prepared: dict,
    *,
    scoring_steps: int = 4,
    scoring_max_step_norm: float = 2.0,
    score_qmc_order: int = 8,
    information_qmc_order: int = 6,
    information_outer_rows: int = 128,
    information_scrambles: int = 2,
    hot_dtype: str = "float32",
    policies: tuple[str, ...] = PAPER_METHODS,
    pilot_schedule: dict | None = None,
    seed_manifest_entry: dict | None = None,
    mlp_hidden: int = 64,
    mlp_steps: int = 100,
    validation_size: int = 4000,
    theta_norm_cap: float | None = None,
    theta_l1_cap: float | None = None,
    pilot_norm_cap: float = 2.0,
    reproducibility: dict | None = None,
    info_basis: PanelInformationBasis | None = None,
    design_info_basis=None,
    feature_fn=None,
    generator_training_seconds: float = 0.0,
    score_backend: str = "cached_qmc",
    information_inner: str = "cached_qmc",
    adaptive_start_order: int = 8,
    adaptive_max_order: int = 16,
    adaptive_atol: float = 2e-6,
    adaptive_rtol: float = 2e-5,
    adaptive_scrambles: int = 4,
    adaptive_chunk_rows: int = 32,
    information_storage_dir=None,
    information_reuse_existing: bool = False,
    score_storage_dir=None,
    score_scrambles: int = 1,
    score_mu_order: int = 12,
    score_mu_scrambles: int = 4,
    share_score_basis: bool = True,
    score_build_on_cuda: bool = False,
    score_direct_cuda: bool = False,
) -> list[dict[str, Any]]:
    reset_workload_counters()
    reset_cache_counters()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    reference = prepared["reference"]
    beta_true = prepared["beta_true"]
    fisher = prepared["fisher"]
    oracle_information = prepared["information"]
    designs = prepared["designs"]
    c_star = float(prepared["oracle_constant"]["half_phi_oracle"])
    # Preserve ``None`` for the built-in feature map.  The exact conditional
    # QMC path uses this sentinel to select its CUDA implementation; wrapping
    # the default in a lambda silently forced gold scoring onto NumPy.
    fn = feature_fn
    seeds = dict(seed_manifest_entry or {})
    replication_seed = int(seeds.get("replication_seed", seed))
    target_draw_seed = int(seeds.get("target_draw_seed", seed + 3))
    design_seed = int(seeds.get("pilot_or_design_seed", seed + 11))
    score_seed = int(seeds.get("score_seed_root", seed + 4))
    info_seed = int(seeds.get("information_seed_root", seed + 7000))

    if pilot_schedule is None:
        pilot_schedule = {
            "kind": "anchored_power",
            "anchor_budget": 8000,
            "anchor_pilot": 400,
            "exponent": 0.5,
            "max_fraction": 0.2,
        }
    pilot_budget = compute_pilot_budget(pilot_schedule, int(budget))
    remaining = int(budget) - int(pilot_budget)

    t_target = perf_counter()
    _sync_cuda()
    target_full = tilted_full_sample(mixture, beta_true, int(budget), target_draw_seed, scale, feature_fn=fn)
    _sync_cuda()
    time_target = perf_counter() - t_target
    target_hash = hashlib.sha256(np.ascontiguousarray(target_full).tobytes()).hexdigest()

    t_ref = perf_counter()
    ref_cache = ReferenceMomentCache.build(reference, scale, beta_true=beta_true, feature_fn=fn)
    score_mu_cache = build_full_law_moment_cache(
        mixture, scale, order=int(score_mu_order), scrambles=int(score_mu_scrambles),
        seed=score_seed + 19, feature_fn=fn,
    )
    time_ref = perf_counter() - t_ref

    t_pilot = perf_counter()
    pilot_counts = balanced_pilot_counts(panels, pilot_budget)
    pilot_observations = []
    cursor = 0
    for panel, count in zip(panels, pilot_counts):
        for row in target_full[cursor: cursor + int(count)]:
            pilot_observations.append((panel, row[list(panel)]))
        cursor += int(count)
    pilot_mu, pilot_rho = pilot_ht_moment(pilot_observations, pilot_counts, panels, scale, reference)
    beta_tilde = solve_pilot_beta(
        pilot_mu, reference, scale,
        norm_cap=theta_norm_cap if theta_norm_cap is not None else pilot_norm_cap,
        l1_cap=theta_l1_cap,
        features=ref_cache.phi_reference,
    )
    time_pilot = perf_counter() - t_pilot

    t_info = perf_counter()
    if info_basis is None:
        if str(information_inner) == "direct_qmc":
            info_basis = build_direct_panel_information_basis(
                mixture, reference, panels, scale=scale,
                outer_rows=int(information_outer_rows), order=int(information_qmc_order),
                seed=info_seed, feature_fn=fn,
                row_chunk=max(1, int(adaptive_chunk_rows)),
            )
        else:
            info_basis = build_panel_information_basis(
                mixture, reference, panels, scale=scale,
                outer_rows=int(information_outer_rows), order=int(information_qmc_order),
                seed=info_seed, dtype=hot_dtype, scrambles=int(information_scrambles),
                feature_fn=fn,
                storage_dir=information_storage_dir,
                reuse_existing=bool(information_reuse_existing),
            )
    design_basis = design_info_basis if design_info_basis is not None else info_basis
    info_at_pilot = design_basis.information(beta_tilde, panels)
    time_info_build = perf_counter() - t_info
    if design_info_basis is not None:
        est_info_at_pilot = info_basis.information(beta_tilde, panels)
    else:
        est_info_at_pilot = info_at_pilot

    plugin = np.stack([info_at_pilot[panel] for panel in panels], axis=0)
    t_fw = perf_counter()
    rr_p, rr_gap, rr_iters = frank_wolfe(
        fisher, plugin, np.ones(len(panels)), uniform_probabilities(len(panels)),
        tolerance=1e-6, max_iter=300,
    )
    time_fw = perf_counter() - t_fw

    t_disc = perf_counter()
    disc_p = None
    if "Discriminative Score OED" in policies:
        from .synthetic_oracle import sample_full
        validation = sample_full(mixture, int(validation_size), design_seed + 544)
        disc_p = discriminative_design(
            reference, validation, beta_tilde, panels, scale, design_seed,
            hidden=mlp_hidden, steps=mlp_steps, feature_fn=fn,
        )
    time_disc = perf_counter() - t_disc

    policy_probs = {
        "Uniform SQD": designs["Uniform SQD"],
        "A-OSQD": designs["A-OSQD"],
        "RR-GID": rr_p,
        "oracle RR-GID": designs["oracle RR-GID"],
    }
    if disc_p is not None:
        policy_probs["Discriminative Score OED"] = disc_p

    # Materialise all policy allocations first.  For fixed/cached QMC we can
    # then construct a deduplicated per-panel basis once and expose indexed
    # views, avoiding repeated conditional transforms for shared pilot rows and
    # enabling larger batched CUDA kernels.
    policy_counts: dict[str, np.ndarray] = {}
    grouped_by_policy: dict[str, GroupedPanelData] = {}
    for name in policies:
        probabilities = np.asarray(policy_probs[name], dtype=float)
        counts = largest_remainder_counts(probabilities, remaining)
        if int(counts.sum()) != remaining:
            raise RuntimeError("main panel allocation does not exhaust remaining budget")
        main_observations = []
        main_cursor = int(pilot_budget)
        for panel, count in zip(panels, counts):
            for row in target_full[main_cursor: main_cursor + int(count)]:
                main_observations.append((panel, row[list(panel)]))
            main_cursor += int(count)
        if int(pilot_counts.sum()) + int(counts.sum()) > int(budget):
            raise RuntimeError("allocated observations exceed budget")
        observations = pilot_observations + main_observations
        policy_counts[name] = counts
        grouped_by_policy[name] = GroupedPanelData.from_observations(observations)

    shared_score_views = None
    shared_score_meta = {"shared": False}
    time_shared_score_build = 0.0
    if (
        bool(share_score_basis)
        and str(score_backend) in {"fixed_qmc", "cached_qmc"}
        and not bool(score_direct_cuda)
        and len(grouped_by_policy) > 1
    ):
        shared_score_views, time_shared_score_build, shared_score_meta = build_shared_score_bases(
            mixture,
            grouped_by_policy,
            scale,
            order=int(score_qmc_order),
            seed=int(score_seed),
            dtype=str(hot_dtype),
            feature_fn=fn,
            storage_dir=score_storage_dir,
            build_on_cuda=bool(score_build_on_cuda),
            row_chunk=max(1, int(adaptive_chunk_rows)),
        )

    shared_adaptive_views = None
    shared_adaptive_meta = {"shared": False}
    time_shared_adaptive_build = 0.0
    if (
        bool(share_score_basis)
        and str(score_backend) == "exact_adaptive"
        and len(grouped_by_policy) > 1
    ):
        shared_adaptive_views, time_shared_adaptive_build, shared_adaptive_meta = build_shared_adaptive_score_bases(
            mixture,
            grouped_by_policy,
            scale,
            seed=int(score_seed),
            feature_fn=fn,
            start_order=int(adaptive_start_order),
            max_order=int(adaptive_max_order),
            atol=float(adaptive_atol),
            rtol=float(adaptive_rtol),
            scrambles=int(adaptive_scrambles),
            chunk_rows=int(adaptive_chunk_rows),
        )

    rows = []
    adaptive_information_cache = {}
    for name in policies:
        counts = policy_counts[name]
        grouped = grouped_by_policy[name]
        if shared_adaptive_views is not None:
            score_bases = shared_adaptive_views[name]
            time_score_build = float(time_shared_adaptive_build)
        elif str(score_backend) == "exact_adaptive":
            score_bases = {
                panel: AdaptiveConditionalMean(
                    mixture, grouped.rows(panel), panel, int(score_seed), scale,
                    feature_fn,
                    start_order=int(adaptive_start_order), max_order=int(adaptive_max_order),
                    atol=float(adaptive_atol), rtol=float(adaptive_rtol),
                    scrambles=int(adaptive_scrambles), chunk_rows=int(adaptive_chunk_rows),
                )
                for panel in grouped.panel_ids
            }
            time_score_build = 0.0
        elif str(score_backend) == "fixed_qmc" and bool(score_direct_cuda):
            score_bases = {
                panel: DirectConditionalMean(
                    mixture,
                    grouped.rows(panel),
                    panel,
                    int(score_seed),
                    scale,
                    feature_fn,
                    order=int(score_qmc_order),
                    row_chunk=max(1, int(adaptive_chunk_rows)),
                    cache_dtype=str(hot_dtype),
                )
                for panel in grouped.panel_ids
            }
            time_score_build = 0.0
        elif shared_score_views is not None:
            score_bases = shared_score_views[name]
            time_score_build = float(time_shared_score_build)
        elif str(score_backend) == "fixed_qmc":
            score_bases = {
                panel: FixedConditionalMean(
                    mixture,
                    grouped.rows(panel),
                    panel,
                    int(score_seed),
                    scale,
                    feature_fn,
                    order=int(score_qmc_order),
                    row_chunk=max(1, int(adaptive_chunk_rows)),
                    scrambles=max(1, int(score_scrambles)),
                    cache_dtype=str(hot_dtype),
                )
                for panel in grouped.panel_ids
            }
            time_score_build = 0.0
        else:
            score_bases, time_score_build = _build_score_bases(
                mixture, grouped, scale, score_qmc_order, score_seed, hot_dtype,
                feature_fn=fn,
                storage_dir=score_storage_dir,
            )
        beta_hat = np.asarray(beta_tilde, dtype=float).copy()
        diagnostics = [{
            "step": "pilot",
            "pilot_budget": int(pilot_counts.sum()),
            "beta_norm": float(np.linalg.norm(beta_hat)),
            "rho_min": float(np.min(pilot_rho[pilot_rho > 0])) if np.any(pilot_rho > 0) else 0.0,
            "pilot_beta_error_norm": float(np.linalg.norm(beta_hat - beta_true)),
        }]
        uses_gold = str(score_backend) == "exact_adaptive" or str(information_inner) == "exact_adaptive"
        gold_converged: bool | None = True if uses_gold else None
        gold_information_diagnostics = []
        gold_score_diagnostics = []

        def _info_at(beta, panels):
            nonlocal gold_converged
            if str(information_inner) == "exact_adaptive":
                infos, ok, info_diagnostics = gold_information_from_basis(
                    info_basis, mixture, beta, scale, panels, feature_fn=feature_fn,
                    start_order=int(adaptive_start_order), max_order=int(adaptive_max_order),
                    atol=float(adaptive_atol), rtol=float(adaptive_rtol),
                    scrambles=int(adaptive_scrambles), chunk_rows=int(adaptive_chunk_rows),
                    adaptive_cache=adaptive_information_cache,
                    return_diagnostics=True,
                )
                gold_converged = bool(ok) if gold_converged is None else gold_converged and bool(ok)
                gold_information_diagnostics.append({
                    "beta": np.asarray(beta, dtype=float).tolist(),
                    "converged": bool(ok),
                    "panels": info_diagnostics,
                })
                return infos
            return info_basis.information(beta, panels)

        info_library = _info_at(beta_hat, grouped.panel_ids)
        t_est = perf_counter()
        for update in range(int(scoring_steps)):
            if str(score_backend) == "exact_adaptive" or str(information_inner) == "exact_adaptive":
                print(
                    f"  gold scoring step {update}/{scoring_steps} "
                    f"n_panels={len(grouped.panel_ids)} n_obs={int(grouped.counts.sum())}",
                    flush=True,
                )
            if update > 0:
                info_library = _info_at(beta_hat, grouped.panel_ids)
            mu_gmm, _ = score_mu_cache.moments(beta_hat)
            beta_next, step_diag = final_rr_step(
                beta_hat, grouped, score_bases, info_library, ref_cache,
                max_step_norm=scoring_max_step_norm,
                norm_cap=theta_norm_cap,
                l1_cap=theta_l1_cap,
                mu=mu_gmm,
            )
            if str(score_backend) == "exact_adaptive":
                score_gold_ok = all(getattr(basis, "last_converged", True) for basis in score_bases.values())
                gold_converged = score_gold_ok if gold_converged is None else gold_converged and score_gold_ok
                gold_score_diagnostics.append({
                    "step": int(update),
                    "converged": bool(score_gold_ok),
                    "panels": [
                        {
                            "panel": [int(v) for v in panel],
                            "converged": bool(getattr(basis, "last_converged", True)),
                            "chunks": getattr(basis, "last_diagnostics", []),
                        }
                        for panel, basis in score_bases.items()
                    ],
                })
            diagnostics.append({"step": update, "step_norm": float(np.linalg.norm(beta_next - beta_hat)), **step_diag})
            beta_hat = beta_next
        time_est = perf_counter() - t_est
        kl = float(ref_cache.kl(beta_true, beta_hat))
        metrics = design_metrics(
            fisher, oracle_information, designs["oracle RR-GID"],
            probabilities, pilot_counts, counts,
        )
        risk = float(budget * kl / max(c_star, 1e-12))
        row = {
            "schema_version": "paper-result-v1",
            "experiment": "synthetic_main",
            "method": name,
            "policy": name,
            "budget": int(budget),
            "replication": int(seeds.get("replication", replication_seed)),
            "target_draw_seed": target_draw_seed,
            "design_seed": design_seed,
            "backend_seed": score_seed,
            "replication_seed": replication_seed,
            "pilot_or_design_seed": design_seed,
            "score_seed_root": score_seed,
            "information_seed_root": info_seed,
            "primary_metric": "risk_ratio_raw",
            "primary_loss": risk,
            "risk_ratio": risk,
            "risk_ratio_raw": risk,
            "kl_raw": kl,
            "B_kl_raw": float(budget * kl),
            "design_ratio_main": metrics["design_ratio_main"],
            "design_ratio_total": metrics["design_ratio_total_counts"],
            "design_ratio_total_counts": metrics["design_ratio_total_counts"],
            "phi_oracle": metrics["phi_oracle"],
            "c_star": c_star,
            "beta_hat": beta_hat.tolist(),
            "pilot_budget": int(pilot_counts.sum()),
            "scoring_steps": int(scoring_steps),
            "score_qmc_order": int(score_qmc_order),
            "score_mu_qmc_order": int(score_mu_order),
            "information_qmc_order": int(information_qmc_order),
            "score_backend": str(score_backend),
            "information_inner": str(information_inner),
            "gold_converged": gold_converged,
            "gold_information_diagnostics": gold_information_diagnostics,
            "gold_score_diagnostics": gold_score_diagnostics,
            "shared_score_basis": dict(shared_score_meta),
            "shared_adaptive_score_basis": dict(shared_adaptive_meta),
            "shared_adaptive_information_cache": bool(str(information_inner) == "exact_adaptive"),
            "allocated_observations": int(pilot_counts.sum() + counts.sum()),
            "main_counts": counts.tolist(),
            "pilot_counts": pilot_counts.tolist(),
            "target_draw_sha256": target_hash,
            "seed": replication_seed,
            "fw_gap": float(rr_gap) if name == "RR-GID" else float(prepared.get("oracle_constant", {}).get("fw_gap", np.nan)),
            "wall_seconds_total": float(time_target + time_pilot + time_info_build + time_fw + time_disc + time_score_build + time_est),
            "wall_seconds_basis_build": float(time_info_build + time_score_build),
            "wall_seconds_design": float(time_fw + time_disc),
            "wall_seconds_estimation": float(time_est),
            "generator_training_seconds": float(generator_training_seconds),
            "cache_hits": int(cache_counters().get("cache_hit", 0)),
            "cache_misses": int(cache_counters().get("cache_miss", 0)),
            "peak_gpu_memory_mb": _peak_memory().get("peak_gpu_alloc_mb"),
            "workload": workload_counters(),
            "update_diagnostics": diagnostics,
            "runtime": {
                "time_target_sampling": float(time_target),
                "time_reference_cache": float(time_ref),
                "time_pilot": float(time_pilot),
                "time_info_basis": float(time_info_build),
                "time_fw": float(time_fw),
                "time_disc": float(time_disc),
                "time_score_basis": float(time_score_build),
                "time_estimation": float(time_est),
                "device": _runtime_device(),
                "memory": _peak_memory(),
            },
        }
        if reproducibility:
            row.update(reproducibility)
        rows.append(row)
    return rows
