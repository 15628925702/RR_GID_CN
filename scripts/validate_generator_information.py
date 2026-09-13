"""RISK-2: fail-closed information-fidelity checks for frozen generators.

The Synthetic check compares a frozen learned generator with the exact mixture
oracle.  The Gas check uses a cross-fitted empirical conditional reference:
reference-train records are donors and disjoint reference-validation records
are outer/query observations.  In both cases the decisive downstream metric is
the design obtained from candidate information, evaluated under the reference
information matrices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
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
from rr_gid_cn.generator_validation import (
    conditional_reconstruction_report,
    information_fidelity_report,
    sampling_health_report,
    weighted_covariance,
)
from rr_gid_cn.synthetic_oracle import (
    all_pairs,
    make_frozen_mixture,
    reference_scale,
    tilted_conditional_mean_qmc,
    tilted_full_sample,
)
from rr_gid_cn.empirical_generator import (
    empirical_conditional_means,
    empirical_information_basis,
    load_gas_generator,
)
from rr_gid_cn.vaeac import (
    VAEACGenerator,
    imp_conditional_mean_from_generator,
    learned_information,
    load_vaeac_checkpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _git_diff_sha256() -> str:
    try:
        payload = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        return "unknown"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _stable_tilt_weights(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    logits = np.asarray(features, dtype=float) @ np.asarray(beta, dtype=float)
    weights = np.exp(logits - float(np.max(logits)))
    return weights / weights.sum()


def _thresholds(cfg: dict) -> dict:
    return dict(cfg.get("thresholds") or {})


def _information_report(fisher, truth, candidate, thresholds) -> dict:
    return information_fidelity_report(
        fisher,
        truth,
        candidate,
        spearman_min=float(thresholds["panel_ranking_spearman_min"]),
        design_ratio_max=float(thresholds["validation_design_ratio_max"]),
        top_k=int(thresholds["top_k"]),
        top_k_overlap_min=float(thresholds["top_k_overlap_min"]),
    )


def _conditional_report(truth, candidate, fisher, thresholds) -> dict:
    return conditional_reconstruction_report(
        truth,
        candidate,
        np.sqrt(np.maximum(np.diag(fisher), 1e-8)),
        normalized_rmse_max=float(thresholds["conditional_normalized_rmse_max"]),
        catastrophic_abs_z_max=float(thresholds["catastrophic_abs_z_max"]),
    )


def _sampling_report(generator, beta, base_pool, conditional_jobs, cfg, thresholds, seed) -> dict:
    sampling = cfg["sampling"]
    _, unconditional_acceptance, _ = generator.tilted_full_diagnostics(
        beta, int(sampling["tilted_draws"]), seed=seed
    )
    conditional_acceptances = []
    conditional_ess = []
    for index, (observed, panel) in enumerate(conditional_jobs):
        _, acceptance, ess = generator.tilted_conditional_diagnostics(
            beta,
            observed,
            panel,
            int(sampling["conditional_draws"]),
            seed=seed + 100 + index,
        )
        conditional_acceptances.append(float(acceptance))
        conditional_ess.append(float(ess))
    report = sampling_health_report(
        importance_ess_fraction=generator.importance_ess(beta, base_pool),
        unconditional_acceptance=float(unconditional_acceptance),
        conditional_acceptance=float(np.mean(conditional_acceptances)),
        importance_ess_min=float(thresholds["importance_ess_min"]),
        unconditional_acceptance_min=float(thresholds["unconditional_acceptance_min"]),
        conditional_acceptance_min=float(thresholds["conditional_acceptance_min"]),
    )
    report.update({
        "conditional_acceptance_min_observed": float(np.min(conditional_acceptances)),
        "conditional_acceptance_max_observed": float(np.max(conditional_acceptances)),
        "conditional_ess_mean": float(np.mean(conditional_ess)),
        "conditional_ess_min": float(np.min(conditional_ess)),
    })
    return report


def validate_synthetic(cfg: dict, device: str, output_dir: Path) -> dict:
    spec = cfg["synthetic"]
    thresholds = _thresholds(cfg)
    checkpoint = Path(spec["checkpoint"])
    artifact_path = Path(spec["oracle_artifact"])
    artifact = pickle.loads(artifact_path.read_bytes())
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    panels = tuple(all_pairs())
    beta = np.asarray(artifact["beta_true"], dtype=float)
    fisher = np.asarray(artifact["fisher"], dtype=float)
    truth_information = np.asarray(artifact["information"], dtype=float)
    model, payload = load_vaeac_checkpoint(checkpoint, device=device, expected_dim=16)
    generator = VAEACGenerator(
        model,
        payload.get("scale", scale),
        alpha=float(payload.get("alpha", 1.0)),
        device=device,
    )
    _, candidate_information = learned_information(
        generator,
        beta,
        panels,
        n_tilted=int(spec["n_tilted"]),
        n_conditional=int(spec["n_conditional"]),
        seed=202609210,
    )
    information = _information_report(fisher, truth_information, candidate_information, thresholds)
    selected = information["reference_top_panel_indices"][: int(spec["conditional_panels"])]
    q_per = int(spec["conditional_queries_per_panel"])
    outer = tilted_full_sample(mixture, beta, q_per, 202609211, scale)
    truth_means, candidate_means, conditional_jobs = [], [], []
    for rank, panel_index in enumerate(selected):
        panel = panels[int(panel_index)]
        observed = outer[:, list(panel)]
        truth = tilted_conditional_mean_qmc(
            mixture,
            beta,
            observed,
            panel,
            int(spec["conditional_truth_qmc_order"]),
            seed=202609300 + rank,
            scale=scale,
        )
        learned = imp_conditional_mean_from_generator(
            generator,
            beta,
            observed,
            panel,
            n=int(spec["conditional_generator_draws"]),
            seed=202609400 + rank,
            scale=scale,
        )
        truth_means.append(truth)
        candidate_means.append(learned)
        conditional_jobs.append((observed[0], panel))
    reconstruction = _conditional_report(
        np.concatenate(truth_means), np.concatenate(candidate_means), fisher, thresholds
    )
    base_pool = generator.sample_full(int(cfg["sampling"]["full_pool"]), seed=202609500)
    sampling = _sampling_report(
        generator, beta, base_pool, conditional_jobs, cfg, thresholds, seed=202609600
    )
    failed = [
        f"information:{name}" for name in information["failed_gates"]
    ] + [
        f"conditional:{name}" for name in reconstruction["failed_gates"]
    ] + [
        f"sampling:{name}" for name in sampling["failed_gates"]
    ]
    np.savez_compressed(
        output_dir / "synthetic_details.npz",
        reference_information=truth_information,
        candidate_information=candidate_information,
        reference_conditional_means=np.concatenate(truth_means),
        candidate_conditional_means=np.concatenate(candidate_means),
    )
    return {
        "environment": "synthetic",
        "passed": not failed,
        "failed_gates": failed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "oracle_artifact": str(artifact_path.resolve()),
        "oracle_artifact_sha256": _sha256(artifact_path),
        "beta": beta.tolist(),
        "information_fidelity": information,
        "conditional_reconstruction": reconstruction,
        "sampling_health": sampling,
    }


def validate_gas(cfg: dict, device: str, output_dir: Path) -> dict:
    spec = cfg["gas"]
    thresholds = _thresholds(cfg)
    checkpoint = Path(spec["checkpoint"])
    data_path = Path(spec["data"])
    data = np.load(data_path)
    mean, std, pcs = data["mean"], data["std"], data["pcs"]
    fn = lambda x: transform_features(x, mean, std, pcs)
    donor_x = np.asarray(data["ref_train"], dtype=float)
    donor_phi = fn(donor_x)
    validation_x = np.asarray(data["ref_val"], dtype=float)
    validation_phi = fn(validation_x)
    from run_gas_semisynthetic import calibrate_beta

    beta = calibrate_beta(validation_phi, seed=2026, target_ess=0.5)
    rng = np.random.default_rng(202609700)
    n_outer = min(len(validation_x), int(spec["empirical_outer_rows"]))
    outer_indices = rng.choice(len(validation_x), size=n_outer, replace=False)
    outer_x = validation_x[outer_indices]
    outer_phi = validation_phi[outer_indices]
    sensor_pairs = panel_library()
    panels = tuple(
        tuple(j for sensor in pair for j in range(sensor * 8, (sensor + 1) * 8))
        for pair in sensor_pairs
    )
    fisher, truth_information, empirical_means = empirical_information_basis(
        donor_x,
        donor_phi,
        outer_x,
        outer_phi,
        panels,
        beta,
        neighbors=int(spec["empirical_neighbors"]),
    )
    generator = load_gas_generator(spec, data, fn, device=device)
    if getattr(generator, "kind", "") == "empirical_knn":
        _, candidate_information, candidate_means_full = generator.panel_information(
            beta, panels, outer_x
        )
    else:
        _, candidate_information = learned_information(
            generator,
            beta,
            panels,
            n_tilted=int(spec["n_tilted"]),
            n_conditional=int(spec["n_conditional"]),
            seed=202609710,
            feature_fn=fn,
        )
        candidate_means_full = None
    information = _information_report(fisher, truth_information, candidate_information, thresholds)
    selected = information["reference_top_panel_indices"][: int(spec["conditional_panels"])]
    q_per = int(spec["conditional_queries_per_panel"])
    truth_means, candidate_means, conditional_jobs = [], [], []
    for rank, panel_index in enumerate(selected):
        panel_index = int(panel_index)
        panel = panels[panel_index]
        take = np.arange(q_per) + (rank * q_per)
        take %= n_outer
        observed = outer_x[take][:, list(panel)]
        truth = empirical_means[panel_index, take]
        if candidate_means_full is not None:
            learned = candidate_means_full[panel_index, take]
        else:
            learned = imp_conditional_mean_from_generator(
                generator,
                beta,
                observed,
                panel,
                n=int(spec["conditional_generator_draws"]),
                seed=202609800 + rank,
                feature_fn=fn,
            )
        truth_means.append(truth)
        candidate_means.append(learned)
        conditional_jobs.append((observed[0], panel))
    reconstruction = _conditional_report(
        np.concatenate(truth_means), np.concatenate(candidate_means), fisher, thresholds
    )
    base_pool = generator.sample_full(int(cfg["sampling"]["full_pool"]), seed=202609900)
    sampling = _sampling_report(
        generator, beta, base_pool, conditional_jobs, cfg, thresholds, seed=202610000
    )
    failed = [
        f"information:{name}" for name in information["failed_gates"]
    ] + [
        f"conditional:{name}" for name in reconstruction["failed_gates"]
    ] + [
        f"sampling:{name}" for name in sampling["failed_gates"]
    ]
    np.savez_compressed(
        output_dir / "gas_details.npz",
        reference_information=truth_information,
        candidate_information=candidate_information,
        reference_conditional_means=np.concatenate(truth_means),
        candidate_conditional_means=np.concatenate(candidate_means),
        outer_indices=outer_indices,
    )
    return {
        "environment": "gas",
        "passed": not failed,
        "failed_gates": failed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "data": str(data_path.resolve()),
        "data_sha256": _sha256(data_path),
        "beta": beta.tolist(),
        "information_fidelity": information,
        "conditional_reconstruction": reconstruction,
        "sampling_health": sampling,
        "reference_estimator": {
            "kind": "cross_fitted_tilt_weighted_knn",
            "donor_split": "ref_train",
            "outer_split": "ref_val",
            "outer_rows": int(n_outer),
            "neighbors": int(spec["empirical_neighbors"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/validation/generator_information_v1.yaml")
    )
    parser.add_argument("--scope", choices=("all", "synthetic", "gas"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = Path(cfg["output_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    provenance = {
        "code_commit": _git_commit(),
        "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
        "code_diff_sha256": _git_diff_sha256(),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda_runtime": torch.version.cuda if device == "cuda" else None,
    }
    started = perf_counter()
    environments = []
    try:
        if args.scope in ("all", "synthetic"):
            print("RISK-2 synthetic generator validation starting", flush=True)
            environments.append(validate_synthetic(cfg, device, output_dir))
            _write_json_atomic(output_dir / "synthetic_report.json", environments[-1])
            print(f"RISK-2 synthetic passed={environments[-1]['passed']}", flush=True)
        if args.scope in ("all", "gas"):
            print("RISK-2 Gas generator validation starting", flush=True)
            environments.append(validate_gas(cfg, device, output_dir))
            _write_json_atomic(output_dir / "gas_report.json", environments[-1])
            print(f"RISK-2 Gas passed={environments[-1]['passed']}", flush=True)
    except Exception as exc:
        report = {
            "schema_version": cfg["schema_version"],
            "passed": False,
            "failed_gates": ["execution_error"],
            "error": f"{type(exc).__name__}: {exc}",
            "environments": environments,
            "provenance": provenance,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": perf_counter() - started,
        }
        _write_json_atomic(output_dir / "report.json", report)
        raise
    failed = [f"{item['environment']}:{gate}" for item in environments for gate in item["failed_gates"]]
    report = {
        "schema_version": cfg["schema_version"],
        "evidence_level": "scientific-readiness-gate",
        "passed": not failed and len(environments) == (2 if args.scope == "all" else 1),
        "failed_gates": failed,
        "environments": environments,
        "provenance": provenance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": perf_counter() - started,
    }
    _write_json_atomic(output_dir / "report.json", report)
    print(json.dumps({"passed": report["passed"], "failed_gates": failed}, ensure_ascii=False), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
