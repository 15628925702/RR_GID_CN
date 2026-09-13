"""Metrics for fail-closed validation of frozen conditional generators."""

from __future__ import annotations

import numpy as np

from .policies import frank_wolfe, objective, uniform_probabilities


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation without a SciPy dependency."""
    a = _average_ranks(np.asarray(left, dtype=float))
    b = _average_ranks(np.asarray(right, dtype=float))
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Spearman inputs must have the same length >= 2")
    a -= a.mean()
    b -= b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0.0 else 0.0


def weighted_covariance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Population weighted covariance with normalized non-negative weights."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 2 or w.shape != (len(x),):
        raise ValueError("invalid weighted covariance shapes")
    if np.any(w < 0.0) or not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("weighted covariance inputs must be finite and non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weighted covariance requires positive total weight")
    w = w / total
    centered = x - w @ x
    return (centered * w[:, None]).T @ centered


def panel_sensitivities(
    fisher: np.ndarray,
    informations: np.ndarray,
    probabilities: np.ndarray | None = None,
    costs: np.ndarray | None = None,
) -> np.ndarray:
    """Frank-Wolfe panel sensitivities at a specified allocation."""
    infos = np.asarray(informations, dtype=float)
    n_panels = len(infos)
    p = uniform_probabilities(n_panels) if probabilities is None else np.asarray(probabilities, dtype=float)
    c = np.ones(n_panels) if costs is None else np.asarray(costs, dtype=float)
    if p.shape != (n_panels,) or c.shape != (n_panels,):
        raise ValueError("allocation/cost shape does not match panel information")
    matrix = np.tensordot(p / c, infos, axes=(0, 0))
    matrix = 0.5 * (matrix + matrix.T)
    inverse = np.linalg.pinv(matrix, rcond=1e-10)
    f = 0.5 * (np.asarray(fisher, dtype=float) + np.asarray(fisher, dtype=float).T)
    return np.asarray(
        [np.trace(inverse @ f @ inverse @ info) / cost for info, cost in zip(infos, c)],
        dtype=float,
    )


def information_fidelity_report(
    fisher: np.ndarray,
    reference_information: np.ndarray,
    candidate_information: np.ndarray,
    *,
    spearman_min: float = 0.80,
    design_ratio_max: float = 1.25,
    top_k: int = 10,
    top_k_overlap_min: float = 0.50,
    fw_tolerance: float = 1e-4,
    fw_max_iter: int = 300,
) -> dict:
    """Compare a learned information basis with a validation reference basis.

    The downstream check is deliberately evaluated under the reference basis:
    optimize an allocation using the candidate matrices, then score that
    allocation with the reference matrices relative to the reference optimum.
    """
    truth = np.asarray(reference_information, dtype=float)
    learned = np.asarray(candidate_information, dtype=float)
    if truth.shape != learned.shape or truth.ndim != 3 or truth.shape[1] != truth.shape[2]:
        raise ValueError("reference and candidate information must be matching (S,r,r) arrays")
    if not np.isfinite(truth).all() or not np.isfinite(learned).all():
        raise ValueError("information arrays contain non-finite values")
    n_panels = len(truth)
    costs = np.ones(n_panels, dtype=float)
    safe = uniform_probabilities(n_panels)
    p_truth, truth_gap, truth_iterations = frank_wolfe(
        fisher, truth, costs, safe, tolerance=float(fw_tolerance), max_iter=int(fw_max_iter)
    )
    p_candidate, candidate_gap, candidate_iterations = frank_wolfe(
        fisher, learned, costs, safe, tolerance=float(fw_tolerance), max_iter=int(fw_max_iter)
    )
    truth_objective = objective(fisher, truth, p_truth, costs)
    candidate_on_truth = objective(fisher, truth, p_candidate, costs)
    design_ratio = float(candidate_on_truth / max(truth_objective, 1e-12))

    truth_scores = panel_sensitivities(fisher, truth, safe, costs)
    candidate_scores = panel_sensitivities(fisher, learned, safe, costs)
    spearman = spearman_correlation(truth_scores, candidate_scores)
    k = max(1, min(int(top_k), n_panels))
    truth_top = np.argsort(-truth_scores, kind="stable")[:k]
    candidate_top = np.argsort(-candidate_scores, kind="stable")[:k]
    overlap_count = len(set(map(int, truth_top)).intersection(map(int, candidate_top)))
    overlap = float(overlap_count / k)

    truth_norm = np.linalg.norm(truth.reshape(n_panels, -1), axis=1)
    rel_fro = np.linalg.norm((learned - truth).reshape(n_panels, -1), axis=1) / np.maximum(truth_norm, 1e-12)
    failed = []
    if spearman < float(spearman_min):
        failed.append("panel_ranking_spearman")
    if design_ratio > float(design_ratio_max):
        failed.append("validation_design_ratio")
    if overlap < float(top_k_overlap_min):
        failed.append("top_k_overlap")
    return {
        "passed": not failed,
        "failed_gates": failed,
        "panel_ranking_spearman": float(spearman),
        "validation_design_ratio": design_ratio,
        "top_k": k,
        "top_k_overlap_count": int(overlap_count),
        "top_k_overlap_fraction": overlap,
        "reference_top_panel_indices": [int(i) for i in truth_top],
        "candidate_top_panel_indices": [int(i) for i in candidate_top],
        "panel_relative_frobenius_median": float(np.median(rel_fro)),
        "panel_relative_frobenius_p95": float(np.quantile(rel_fro, 0.95)),
        "panel_relative_frobenius_max": float(rel_fro.max()),
        "reference_objective": float(truth_objective),
        "candidate_allocation_reference_objective": float(candidate_on_truth),
        "reference_fw_gap": float(truth_gap),
        "candidate_fw_gap": float(candidate_gap),
        "reference_fw_iterations": int(truth_iterations),
        "candidate_fw_iterations": int(candidate_iterations),
        "thresholds": {
            "panel_ranking_spearman_min": float(spearman_min),
            "validation_design_ratio_max": float(design_ratio_max),
            "top_k_overlap_min": float(top_k_overlap_min),
        },
    }


def conditional_reconstruction_report(
    reference_means: np.ndarray,
    candidate_means: np.ndarray,
    feature_scale: np.ndarray,
    *,
    normalized_rmse_max: float = 1.0,
    catastrophic_abs_z_max: float = 10.0,
) -> dict:
    """Summarize conditional-score reconstruction and reject catastrophic errors."""
    truth = np.asarray(reference_means, dtype=float)
    learned = np.asarray(candidate_means, dtype=float)
    scale = np.asarray(feature_scale, dtype=float)
    if truth.shape != learned.shape or truth.ndim != 2 or scale.shape != (truth.shape[1],):
        raise ValueError("invalid conditional reconstruction shapes")
    finite = bool(np.isfinite(truth).all() and np.isfinite(learned).all() and np.isfinite(scale).all())
    safe_scale = np.maximum(np.abs(scale), 1e-8)
    residual = (learned - truth) / safe_scale
    abs_z = np.abs(residual)
    rmse = float(np.sqrt(np.mean(residual ** 2))) if finite else float("inf")
    max_abs = float(abs_z.max()) if finite and abs_z.size else float("inf")
    failed = []
    if not finite:
        failed.append("nonfinite_conditional_reconstruction")
    if rmse > float(normalized_rmse_max):
        failed.append("conditional_reconstruction_rmse")
    if max_abs > float(catastrophic_abs_z_max):
        failed.append("catastrophic_conditional_score_outlier")
    return {
        "passed": not failed,
        "failed_gates": failed,
        "n_queries": int(len(truth)),
        "feature_dimension": int(truth.shape[1]),
        "normalized_rmse": rmse,
        "median_abs_z": float(np.median(abs_z)) if finite and abs_z.size else float("inf"),
        "p95_abs_z": float(np.quantile(abs_z, 0.95)) if finite and abs_z.size else float("inf"),
        "max_abs_z": max_abs,
        "thresholds": {
            "normalized_rmse_max": float(normalized_rmse_max),
            "catastrophic_abs_z_max": float(catastrophic_abs_z_max),
        },
    }


def sampling_health_report(
    *,
    importance_ess_fraction: float,
    unconditional_acceptance: float,
    conditional_acceptance: float,
    importance_ess_min: float = 0.10,
    unconditional_acceptance_min: float = 1e-3,
    conditional_acceptance_min: float = 1e-3,
) -> dict:
    """Check that reweighting and rejection samplers have not collapsed."""
    values = {
        "importance_ess_fraction": float(importance_ess_fraction),
        "unconditional_acceptance": float(unconditional_acceptance),
        "conditional_acceptance": float(conditional_acceptance),
    }
    failed = []
    if not all(np.isfinite(value) for value in values.values()):
        failed.append("nonfinite_sampling_diagnostic")
    if values["importance_ess_fraction"] < float(importance_ess_min):
        failed.append("importance_ess")
    if values["unconditional_acceptance"] < float(unconditional_acceptance_min):
        failed.append("unconditional_acceptance")
    if values["conditional_acceptance"] < float(conditional_acceptance_min):
        failed.append("conditional_acceptance")
    return {
        "passed": not failed,
        "failed_gates": failed,
        **values,
        "thresholds": {
            "importance_ess_min": float(importance_ess_min),
            "unconditional_acceptance_min": float(unconditional_acceptance_min),
            "conditional_acceptance_min": float(conditional_acceptance_min),
        },
    }


def secondary_diagnostics_report(
    rows: list[dict],
    *,
    expected_rows: int,
    c2st_auc_max: float = 0.70,
    importance_ess_min: float = 0.05,
    unconditional_acceptance_min: float = 1e-3,
    conditional_acceptance_min: float = 1e-3,
    lambda_min_min: float = 0.0,
    rr_fw_gap_max: float = 1e-3,
    primary_reproduction_atol: float = 1e-6,
) -> dict:
    """Evaluate completeness, numerical health, and family adequacy for RISK-3."""
    complete_failures = []
    scientific_failures = []
    if len(rows) != int(expected_rows):
        complete_failures.append("row_count")
    required = (
        "heldout_mean_rmse",
        "heldout_std_rmse",
        "c2st_auc",
        "importance_ess_fraction",
        "unconditional_acceptance",
        "conditional_acceptance",
        "lambda_min_M",
        "primary_loss_abs_delta",
    )
    for index, row in enumerate(rows):
        missing = [name for name in required if name not in row]
        if missing:
            complete_failures.append(f"row_{index}_missing_fields")
            continue
        if not all(np.isfinite(float(row[name])) for name in required):
            complete_failures.append(f"row_{index}_nonfinite")
            continue
        label = f"{row.get('campaign', index)}:{row.get('method', 'unknown')}"
        if float(row["primary_loss_abs_delta"]) > float(primary_reproduction_atol):
            complete_failures.append(f"{label}:primary_reproduction")
        if float(row["c2st_auc"]) > float(c2st_auc_max):
            scientific_failures.append(f"{label}:c2st_auc")
        if float(row["importance_ess_fraction"]) < float(importance_ess_min):
            scientific_failures.append(f"{label}:importance_ess")
        if float(row["unconditional_acceptance"]) < float(unconditional_acceptance_min):
            scientific_failures.append(f"{label}:unconditional_acceptance")
        if float(row["conditional_acceptance"]) < float(conditional_acceptance_min):
            scientific_failures.append(f"{label}:conditional_acceptance")
        if float(row["lambda_min_M"]) <= float(lambda_min_min):
            scientific_failures.append(f"{label}:lambda_min_M")
        if row.get("method") == "RR-GID":
            gap = row.get("fw_gap")
            if gap is None or not np.isfinite(float(gap)) or float(gap) > float(rr_fw_gap_max):
                scientific_failures.append(f"{label}:fw_gap")
    c2st_values = np.asarray([float(row["c2st_auc"]) for row in rows if "c2st_auc" in row], dtype=float)
    return {
        "diagnostic_complete": not complete_failures,
        "scientific_gate_passed": not complete_failures and not scientific_failures,
        "completion_failures": sorted(set(complete_failures)),
        "scientific_failures": sorted(set(scientific_failures)),
        "row_count": int(len(rows)),
        "expected_rows": int(expected_rows),
        "c2st_auc_min": float(c2st_values.min()) if len(c2st_values) else None,
        "c2st_auc_median": float(np.median(c2st_values)) if len(c2st_values) else None,
        "c2st_auc_max": float(c2st_values.max()) if len(c2st_values) else None,
        "thresholds": {
            "c2st_auc_max": float(c2st_auc_max),
            "importance_ess_min": float(importance_ess_min),
            "unconditional_acceptance_min": float(unconditional_acceptance_min),
            "conditional_acceptance_min": float(conditional_acceptance_min),
            "lambda_min_min": float(lambda_min_min),
            "rr_fw_gap_max": float(rr_fw_gap_max),
            "primary_reproduction_atol": float(primary_reproduction_atol),
        },
    }
