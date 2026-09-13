"""E4: calibrate cached QMC against adaptive gold on the same queries / outer rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rr_gid_cn.conditional_backend import (
    AdaptiveConditionalMean,
    build_conditional_feature_basis,
    build_panel_information_basis,
    evaluate_conditional_basis,
    gold_information_from_basis,
)
from rr_gid_cn.paper_run import run_paper_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale, sample_full


def _relative_op(est: np.ndarray, gold: np.ndarray) -> float:
    gold = 0.5 * (np.asarray(gold) + np.asarray(gold).T)
    vals, vecs = np.linalg.eigh(gold)
    vals = np.maximum(vals, 1e-10)
    isqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    delta = 0.5 * (np.asarray(est) - gold + (np.asarray(est) - gold).T)
    return float(np.linalg.norm(isqrt @ delta @ isqrt, 2))


def _sample_beta(rng: np.random.Generator, dim: int = 12, scale: float = 0.35, cap: float = 2.0) -> np.ndarray:
    beta = rng.normal(0.0, scale, size=int(dim))
    nrm = float(np.linalg.norm(beta))
    if nrm > cap:
        beta *= cap / nrm
    return beta


def _adaptive_kwargs(cal: dict, args=None) -> dict:
    chunk = getattr(args, "adaptive_chunk_rows", None) if args is not None else None
    return dict(
        start_order=int(cal.get("adaptive_start_order", 8)),
        max_order=int(cal.get("adaptive_max_order", 16)),
        atol=float(cal.get("adaptive_atol", 2e-6)),
        rtol=float(cal.get("adaptive_rtol", 2e-5)),
        scrambles=int(cal.get("adaptive_scrambles", 4)),
        chunk_rows=int(chunk if chunk is not None else cal.get("adaptive_chunk_rows", 32)),
    )


def _info_order(cfg: dict, args) -> int:
    value = getattr(args, "information_qmc_order", None)
    return int(cfg.get("information_qmc_order", 6) if value is None else value)


def _score_order(cfg: dict, args) -> int:
    value = getattr(args, "score_qmc_order", None)
    return int(cfg.get("score_qmc_order", 8) if value is None else value)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(config_path: Path, prepared_path: Path) -> dict:
    """Record the exact source/config/artifact identity for every calibration run."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except Exception:
        dirty = True
    return {
        "code_commit": commit,
        "worktree_dirty": dirty,
        "calibration_script_sha256": _file_sha256(Path(__file__).resolve()),
        "config_sha256": _file_sha256(config_path.resolve()),
        "prepared_artifact_sha256": _file_sha256(prepared_path.resolve()) if prepared_path.exists() else None,
    }


def run_query(cfg: dict, cal: dict, mixture, scale, panels, args) -> dict:
    n_queries = int(args.queries or cal.get("query_count", 200))
    orders = [int(o) for o in (getattr(args, "orders", None) or cal.get("qmc_orders", [7, 8, 9]))]
    ak = _adaptive_kwargs(cal, args)
    rng = np.random.default_rng(20260902)
    full = sample_full(mixture, max(n_queries * 2, 64), seed=9)
    by_order = {order: {"rel": [], "coord_rel": [], "abs": [], "converged": []} for order in orders}
    for i in range(n_queries):
        panel = panels[int(rng.integers(len(panels)))]
        row = full[i % len(full)][list(panel)]
        beta = _sample_beta(rng)
        gold_mean = AdaptiveConditionalMean(
            mixture, np.atleast_2d(row), panel, seed=2000 + i, scale=scale, **ak,
        )
        gold = gold_mean.conditional_mean(beta)[0]
        gold_ok = bool(gold_mean.last_converged)
        gold_norm = float(np.linalg.norm(gold))
        for order in orders:
            basis = build_conditional_feature_basis(
                mixture, np.atleast_2d(row), panel, order=order, seed=1000 + i,
                scale=scale, dtype="float64",
            )
            cached = evaluate_conditional_basis(beta, basis)[0]
            vec_rel = float(np.linalg.norm(cached - gold) / max(gold_norm, 1e-8))
            coord_rel = float(np.max(np.abs(cached - gold) / np.maximum(np.abs(gold), 1e-8)))
            by_order[order]["rel"].append(vec_rel)
            by_order[order]["coord_rel"].append(coord_rel)
            by_order[order]["abs"].append(float(np.max(np.abs(cached - gold))))
            by_order[order]["converged"].append(gold_ok)
        if (i + 1) % 20 == 0 or i == 0:
            med8 = float(np.median(by_order[min(orders, key=lambda o: abs(o - 8))]["rel"]))
            print(f"query {i+1}/{n_queries} running_median_rel~{med8:.3e} gold_ok={gold_mean.last_converged}", flush=True)

    def summarize(order: int) -> dict:
        mask = np.asarray(by_order[order]["converged"], dtype=bool)
        rel_all = np.asarray(by_order[order]["rel"], dtype=float)
        abs_all = np.asarray(by_order[order]["abs"], dtype=float)
        coord_all = np.asarray(by_order[order]["coord_rel"], dtype=float)
        rel = rel_all[mask] if mask.any() else rel_all
        abs_coord = abs_all[mask] if mask.any() else abs_all
        ok = (
            int(mask.sum()) >= max(20, int(0.25 * len(mask)))
            and float(np.median(rel)) <= float(cal.get("median_relative_error", 1e-3))
            and float(np.quantile(rel, 0.95)) <= float(cal.get("p95_relative_error", 1e-2))
            and float(abs_coord.max()) <= float(cal.get("max_abs_coordinate_error", 2e-2))
        )
        return {
            "order": int(order),
            "n": int(len(rel_all)),
            "n_gold_converged": int(mask.sum()),
            "median_relative_error": float(np.median(rel)),
            "p95_relative_error": float(np.quantile(rel, 0.95)),
            "max_abs_coordinate_error": float(abs_coord.max()),
            "median_coordwise_relative_error": float(np.median(coord_all[mask] if mask.any() else coord_all)),
            "gold_converged_frac": float(mask.mean()),
            "ok": bool(ok),
        }

    order_reports = [summarize(order) for order in orders]
    passing = [item["order"] for item in order_reports if item["ok"]]
    frozen = int(min(passing)) if passing else None
    return {
        "n_queries": int(n_queries),
        "orders": order_reports,
        "frozen_order": frozen,
        "query_ok": bool(frozen is not None),
        "gold": "tilted_conditional_mean_exact",
    }


def run_information(cfg: dict, cal: dict, mixture, scale, panels, prepared, args) -> dict:
    n_panels = int(getattr(args, "information_panels", None) or cal.get("information_panels", 20))
    n_beta = int(getattr(args, "information_betas", None) or cal.get("information_betas", 5))
    op_tol = float(cal.get("information_op_tol", 0.05))
    rng = np.random.default_rng(20260903)
    allocation_kind = str(
        getattr(args, "information_allocation", None) or cal.get("information_allocation", "oracle")
    )
    if allocation_kind == "oracle":
        full_allocation = np.asarray(prepared["designs"]["oracle RR-GID"], dtype=float)
        selected_indices = np.argsort(-full_allocation)[: min(n_panels, len(panels))]
        allocation_coverage = float(full_allocation[selected_indices].sum())
        selected_allocation = full_allocation[selected_indices]
        selected_allocation /= selected_allocation.sum()
    elif allocation_kind == "uniform":
        selected_indices = rng.choice(len(panels), size=min(n_panels, len(panels)), replace=False)
        allocation_coverage = float(len(selected_indices) / len(panels))
        selected_allocation = np.full(len(selected_indices), 1.0 / len(selected_indices))
    else:
        raise ValueError(f"unsupported information allocation: {allocation_kind}")
    subset = tuple(panels[int(i)] for i in selected_indices)
    info_order = _info_order(cfg, args)
    basis = build_panel_information_basis(
        mixture, prepared["reference"], subset, scale=scale,
        outer_rows=int(getattr(args, "information_outer_rows", None) or cfg.get("information_outer_rows", 128)),
        order=info_order,
        seed=7, dtype="float64",
    )
    ak = _adaptive_kwargs(cal, args)
    aggregate_ops = []
    panel_relative_frobenius = []
    converged = []
    for k in range(n_beta):
        beta = _sample_beta(rng, scale=0.2)
        cached = basis.information(beta, subset)
        gold, ok = gold_information_from_basis(
            basis, mixture, beta, scale, subset, **ak,
        )
        cached_stack = np.stack([cached[panel] for panel in subset], axis=0)
        gold_stack = np.stack([gold[panel] for panel in subset], axis=0)
        cached_m = np.einsum("s,sij->ij", selected_allocation, cached_stack)
        gold_m = np.einsum("s,sij->ij", selected_allocation, gold_stack)
        aggregate_op = _relative_op(cached_m, gold_m)
        aggregate_ops.append(aggregate_op)
        for panel in subset:
            delta = np.asarray(cached[panel]) - np.asarray(gold[panel])
            panel_relative_frobenius.append(
                float(np.linalg.norm(delta, ord="fro") / max(np.linalg.norm(gold[panel], ord="fro"), 1e-12))
            )
            if np.linalg.eigvalsh(cached[panel]).min() < -1e-8:
                ok = False
        converged.append(bool(ok))
        print(
            f"info beta {k+1}/{n_beta} aggregate_op={aggregate_op:.4f} gold_ok={ok}",
            flush=True,
        )
    aggregate_ops = np.asarray(aggregate_ops, dtype=float)
    panel_relative_frobenius = np.asarray(panel_relative_frobenius, dtype=float)
    info_ok = bool(np.all(aggregate_ops <= op_tol) and all(converged))
    return {
        "n_panels": int(len(subset)),
        "n_betas": int(n_beta),
        "n_aggregate_matrices": int(len(aggregate_ops)),
        "mean_op": float(aggregate_ops.mean()),
        "max_op": float(aggregate_ops.max()),
        "p95_op": float(np.quantile(aggregate_ops, 0.95)),
        "panel_relative_frobenius_mean": float(panel_relative_frobenius.mean()),
        "panel_relative_frobenius_max": float(panel_relative_frobenius.max()),
        "tol": op_tol,
        "gold_converged": bool(all(converged)),
        "information_ok": info_ok,
        "same_outer": True,
        "cached_qmc_order": int(info_order),
        "gold": "adaptive_inner_on_frozen_outer",
        "comparison_object": "allocation-weighted aggregate M over selected panels",
        "allocation": allocation_kind,
        "allocation_coverage": allocation_coverage,
        "selected_allocation": selected_allocation.tolist(),
        "effective_information_panels": int(n_panels),
        "effective_information_betas": int(n_beta),
        "effective_information_outer_rows": int(getattr(args, "information_outer_rows", None) or cfg.get("information_outer_rows", 128)),
    }


def _restrict_e2e_panels(panels, prepared, top_n: int | None):
    """Return an auditable top-allocation subset for the bounded L2 e2e probe."""
    if top_n is None:
        return panels, prepared, None
    top_n = int(top_n)
    if top_n < 1 or top_n > len(panels):
        raise ValueError(f"--e2e-top-panels must be in [1, {len(panels)}]")
    allocation = np.asarray(prepared["designs"]["oracle RR-GID"], dtype=float)
    indices = np.argsort(-allocation, kind="stable")[:top_n]
    selected = tuple(panels[int(i)] for i in indices)
    subset = dict(prepared)
    subset["information"] = np.asarray(prepared["information"])[indices]
    subset["designs"] = {
        key: np.asarray(values, dtype=float)[indices]
        for key, values in prepared["designs"].items()
    }
    for key, values in subset["designs"].items():
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError(f"empty e2e design probabilities for {key}")
        subset["designs"][key] = values / total
    return selected, subset, {
        "selected_panel_indices": [int(i) for i in indices],
        "selected_panels": [list(panel) for panel in selected],
        "allocation_source": "prepared.designs.oracle RR-GID",
        "allocation_coverage": float(np.sum(allocation[indices])),
        "total_panels": int(len(panels)),
    }


def run_e2e(cfg: dict, cal: dict, mixture, scale, panels, prepared, args) -> dict:
    panels, prepared, panel_restriction = _restrict_e2e_panels(
        panels, prepared, getattr(args, "e2e_top_panels", None)
    )
    n_e2e = int(cfg.get("replications", 10))
    e2e_budget = int(getattr(args, "e2e_budget", None) or cfg.get("e2e_budget", 8000))
    score_order = _score_order(cfg, args)
    info_order = _info_order(cfg, args)
    out_dir = Path(args.out).parent
    basis_root = Path(getattr(args, "basis_root", None) or out_dir)
    jsonl = out_dir / "e2e.jsonl"
    done = {}
    if args.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["replication"])] = row
    ak = _adaptive_kwargs(cal, args)
    common = dict(
        scoring_steps=int(getattr(args, "e2e_scoring_steps", None) or cfg.get("e2e_scoring_steps", cfg.get("scoring_steps", 4))),
        scoring_max_step_norm=float(cfg.get("scoring_max_step_norm", 2.0)),
        information_qmc_order=info_order,
        information_outer_rows=int(cfg.get("information_outer_rows", 128)),
        information_scrambles=int(cfg.get("information_scrambles", 2)),
        hot_dtype=str(cfg.get("dtype", "float32")),
        policies=("oracle RR-GID",),
        pilot_schedule=cfg.get("pilot_schedule"),
        adaptive_start_order=ak["start_order"],
        adaptive_max_order=ak["max_order"],
        adaptive_atol=ak["atol"],
        adaptive_rtol=ak["rtol"],
        adaptive_scrambles=ak["scrambles"],
        adaptive_chunk_rows=ak["chunk_rows"],
    )
    t0 = perf_counter()
    print(
        f"e2e: {len(done)}/{n_e2e} done, score_order={score_order}, "
        f"info_order={info_order}, chunk_rows={ak['chunk_rows']}, jsonl={jsonl}",
        flush=True,
    )
    for rep in range(n_e2e):
        if args.max_hours is not None and (perf_counter() - t0) / 3600.0 >= float(args.max_hours):
            print(f"max-hours reached before e2e rep={rep}", flush=True)
            break
        if rep in done:
            print(f"e2e resume skip rep={rep}", flush=True)
            continue
        seed = 202609000 + 17 * (rep + 1)
        entry = {
            "replication": int(rep),
            "replication_seed": seed,
            "target_draw_seed": seed + 3,
            "pilot_or_design_seed": seed + 11,
            "score_seed_root": seed + 4,
            "information_seed_root": seed + 7000,
        }
        storage_dir = basis_root / "basis_cache" / f"rep{rep}"
        scaffold_t0 = perf_counter()
        # Fast and gold are paired evaluations on the same frozen outer rows.
        # Build that information scaffold once per replication and inject it
        # into both runs; rebuilding it in the gold call doubled the largest
        # allocation and made the full-panel preflight needlessly expensive.
        shared_info_basis = build_panel_information_basis(
            mixture, prepared["reference"], panels, scale=scale,
            outer_rows=int(cfg.get("information_outer_rows", 128)),
            order=int(info_order), seed=int(entry["information_seed_root"]),
            dtype=str(cfg.get("dtype", "float32")),
            scrambles=int(cfg.get("information_scrambles", 2)),
            storage_dir=storage_dir,
        )
        scaffold_seconds = perf_counter() - scaffold_t0
        fast_checkpoint = out_dir / f"e2e_rep{rep}_fast.json"
        if args.resume and fast_checkpoint.exists():
            fast = _load_json(fast_checkpoint)
            print(f"e2e rep={rep} reused cached checkpoint", flush=True)
        else:
            print(f"e2e rep={rep} starting cached", flush=True)
            fast = run_paper_replication(
                mixture, scale, panels, e2e_budget, seed, prepared, seed_manifest_entry=entry,
                score_qmc_order=score_order, score_backend="cached_qmc", information_inner="cached_qmc",
                information_storage_dir=storage_dir,
                info_basis=shared_info_basis,
                **common,
            )[0]
            _write_json_atomic(fast_checkpoint, fast)
        gold_checkpoint = out_dir / f"e2e_rep{rep}_gold.json"
        if args.resume and gold_checkpoint.exists():
            gold = _load_json(gold_checkpoint)
            print(f"e2e rep={rep} reused gold checkpoint", flush=True)
        else:
            print(
                f"e2e rep={rep} cached={fast['risk_ratio']:.3f} "
                f"s={fast.get('wall_seconds_total', 0):.1f} starting gold",
                flush=True,
            )
            gold = run_paper_replication(
                mixture, scale, panels, e2e_budget, seed, prepared, seed_manifest_entry=entry,
                score_qmc_order=score_order, score_backend="exact_adaptive", information_inner="exact_adaptive",
                information_storage_dir=storage_dir / "gold",
                info_basis=shared_info_basis,
                **common,
            )[0]
            _write_json_atomic(gold_checkpoint, gold)
        if fast["target_draw_sha256"] != gold["target_draw_sha256"]:
            raise RuntimeError("e2e pairing broke: target hashes differ")
        row = {
            "replication": int(rep),
            "fast": float(fast["risk_ratio"]),
            "gold": float(gold["risk_ratio"]),
            "abs_delta": float(abs(fast["risk_ratio"] - gold["risk_ratio"])),
            "signed_delta": float(fast["risk_ratio"] - gold["risk_ratio"]),
            "target_draw_sha256": fast["target_draw_sha256"],
            "gold_converged": bool(gold.get("gold_converged", False)),
            "fast_seconds": float(fast.get("wall_seconds_total", 0.0)),
            "gold_seconds": float(gold.get("wall_seconds_total", 0.0)),
            "score_qmc_order": int(score_order),
            "information_qmc_order": int(info_order),
            "information_basis_dtype": str(cfg.get("dtype", "float32")),
            "budget": int(e2e_budget),
            "evidence_level": "L2-e2e-preflight",
            "fast_backend": fast.get("score_backend"),
            "gold_backend": gold.get("score_backend"),
            "fast_runtime": fast.get("runtime"),
            "gold_runtime": gold.get("runtime"),
            "fast_workload": fast.get("workload"),
            "gold_workload": gold.get("workload"),
            "fast_kl_raw": fast.get("kl_raw"),
            "gold_kl_raw": gold.get("kl_raw"),
            "fast_beta_hat": fast.get("beta_hat"),
            "gold_beta_hat": gold.get("beta_hat"),
            "fast_design_ratio_main": fast.get("design_ratio_main"),
            "gold_design_ratio_main": gold.get("design_ratio_main"),
            "fast_update_diagnostics": fast.get("update_diagnostics"),
            "gold_update_diagnostics": gold.get("update_diagnostics"),
            "gold_information_diagnostics": gold.get("gold_information_diagnostics"),
            "gold_score_diagnostics": gold.get("gold_score_diagnostics"),
            "fast_cache_hits": fast.get("cache_hits"),
            "gold_cache_hits": gold.get("cache_hits"),
            "fast_cache_misses": fast.get("cache_misses"),
            "gold_cache_misses": gold.get("cache_misses"),
            "information_storage_dir": str(storage_dir),
            "shared_information_basis": True,
            "shared_information_basis_seconds": float(scaffold_seconds),
        }
        _write_json_atomic(
            out_dir / f"e2e_rep{rep}_details.json",
            {"fast": fast, "gold": gold, "comparison": row},
        )
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        done[rep] = row
        print(
            f"e2e rep={rep} fast={row['fast']:.3f} gold={row['gold']:.3f} "
            f"|d|={row['abs_delta']:.3f} gold_ok={row['gold_converged']}",
            flush=True,
        )
    rows = [done[k] for k in sorted(done)]
    if not rows:
        result = evaluate_e2e_gate([], n_e2e=n_e2e, cal=cal)
        result["n"] = 0
        if panel_restriction is not None:
            result["panel_restriction"] = panel_restriction
        return result
    result = evaluate_e2e_gate(rows, n_e2e=n_e2e, cal=cal)
    result.update({
        "n": int(len(rows)),
        "fast": [row["fast"] for row in rows],
        "gold": [row["gold"] for row in rows],
        "score_qmc_order": int(score_order),
        "information_qmc_order": int(info_order),
        "information_basis_dtype": str(cfg.get("dtype", "float32")),
        "budget": int(e2e_budget),
        "evidence_level": "L2-e2e-preflight",
        "publication_eligible": False,
        "basis_cache_root": str(basis_root.resolve()),
    })
    if panel_restriction is not None:
        result["panel_restriction"] = panel_restriction
    return result


def evaluate_e2e_gate(rows: list[dict], *, n_e2e: int, cal: dict) -> dict:
    """Evaluate absolute accuracy and paired signed bias separately.

    The first criterion bounds the mean absolute per-replication discrepancy.
    The paired Monte Carlo criterion applies to the magnitude of the signed
    mean discrepancy.  Comparing ``mean(abs(d))`` with ``SE(d)`` would make a
    two-replication preflight impossible by the triangle inequality.
    """
    signed = np.asarray(
        [row.get("signed_delta", row["fast"] - row["gold"]) for row in rows],
        dtype=float,
    )
    abs_delta = np.abs(signed)
    n = len(rows)
    signed_mean = float(signed.mean()) if n else float("nan")
    mean_abs = float(abs_delta.mean()) if n else float("nan")
    signed_se = float(signed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    converged = [bool(row.get("gold_converged", False)) for row in rows]
    converged_frac = float(np.mean(converged)) if n else 0.0
    tolerance = float(cal.get("end_to_end_mean_abs_delta", 0.15))
    failed_gates = []
    if n < int(n_e2e):
        failed_gates.append("minimum_replications")
    if not n or not all(converged):
        failed_gates.append("gold_convergence")
    if n and mean_abs > tolerance:
        failed_gates.append("mean_abs_delta_tolerance")
    if n and abs(signed_mean) > max(signed_se, 1e-12):
        failed_gates.append("signed_mean_delta_not_within_paired_se")
    return {
        "mean_abs_delta": mean_abs,
        "signed_mean_delta": signed_mean,
        "mean_signed_delta": signed_mean,
        "signed_paired_se": signed_se,
        "se": signed_se,
        "gold_converged_frac": converged_frac,
        "failed_gates": failed_gates,
        "ok": not failed_gates,
        "gate": (
            "all gold converged; mean |R_fast - R_gold| <= tolerance; "
            "|mean(R_fast - R_gold)| <= paired SE"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/oracle_calibration.yaml"))
    ap.add_argument("--prepared", type=Path, default=Path("experiments/paper/oracle_artifact.pkl"))
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--orders", type=int, nargs="+", default=None)
    ap.add_argument("--score-qmc-order", type=int, default=None)
    ap.add_argument("--information-qmc-order", type=int, default=None)
    ap.add_argument("--adaptive-chunk-rows", type=int, default=None)
    ap.add_argument("--information-panels", type=int, default=None)
    ap.add_argument("--information-betas", type=int, default=None)
    ap.add_argument("--information-outer-rows", type=int, default=None)
    ap.add_argument("--information-allocation", choices=("uniform", "oracle"), default=None)
    ap.add_argument("--stage", choices=("query", "information", "e2e", "all"), default="all")
    ap.add_argument("--skip-query", action="store_true")
    ap.add_argument("--skip-information", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument(
        "--e2e-top-panels", type=int, default=None,
        help="L2 only: restrict e2e to top oracle-allocation panels and report coverage",
    )
    ap.add_argument(
        "--e2e-budget", type=int, default=None,
        help="Budget for bounded L2 e2e preflight; does not alter registered paper budget",
    )
    ap.add_argument(
        "--e2e-scoring-steps", type=int, default=None,
        help="Scoring steps for bounded e2e preflight",
    )
    ap.add_argument("--out", type=Path, default=Path("results/paper/oracle_calibration/report.json"))
    ap.add_argument(
        "--basis-root", type=Path, default=None,
        help="Absolute or relative root for large e2e information basis caches; report remains under --out",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cal = cfg.get("calibration", {})
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    panels = all_pairs()
    report = {}
    provenance = _provenance(args.config, args.prepared)
    if args.stage != "all" or args.skip_query or args.skip_information or args.resume:
        report = _load_json(args.out)
    report["provenance"] = provenance
    stages = ("query", "information", "e2e") if args.stage == "all" else (args.stage,)

    if "query" in stages:
        if args.skip_query and report.get("query"):
            print("reused query calibration", flush=True)
        else:
            report["query"] = run_query(cfg, cal, mixture, scale, panels, args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"query": report["query"]}, indent=2), flush=True)

    prepared = None
    if any(stage in stages for stage in ("information", "e2e")):
        if not args.prepared.exists():
            raise SystemExit(f"missing oracle artifact: {args.prepared}")
        prepared = pickle.loads(args.prepared.read_bytes())

    if "information" in stages:
        if args.skip_information and report.get("information"):
            print("reused information calibration", flush=True)
        else:
            report["information"] = run_information(cfg, cal, mixture, scale, panels, prepared, args)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({"information": report["information"]}, indent=2), flush=True)

    if "e2e" in stages:
        report["end_to_end"] = run_e2e(cfg, cal, mixture, scale, panels, prepared, args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    query_ok = bool((report.get("query") or {}).get("query_ok", False))
    info_ok = bool((report.get("information") or {}).get("information_ok", False))
    e2e_ok = bool((report.get("end_to_end") or {}).get("ok", False))
    report["query_ok"] = query_ok
    report["information_ok"] = info_ok
    report["passed"] = bool(query_ok and info_ok and e2e_ok)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "query_ok": query_ok,
        "information_ok": info_ok,
        "e2e_ok": e2e_ok,
        "passed": report["passed"],
        "frozen_order": (report.get("query") or {}).get("frozen_order"),
        "information_max_op": (report.get("information") or {}).get("max_op"),
        "e2e_mean_abs_delta": (report.get("end_to_end") or {}).get("mean_abs_delta"),
    }, indent=2), flush=True)
    if args.stage == "all" and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
