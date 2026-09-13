"""Resumable synthetic main runner (paper E1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from rr_gid_cn.p4_integrity import load_seed_manifest, sha256_file, validate_experiment_mode
from rr_gid_cn.paper_run import PAPER_METHODS, run_paper_replication
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_state() -> tuple[bool, str]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return bool(status.strip()), hashlib.sha256(diff).hexdigest()
    except Exception:
        return True, "unknown"


def _runtime_environment() -> dict[str, str | bool]:
    """Collect the runtime identity once so result rows remain self-describing."""
    environment: dict[str, str | bool] = {"python_executable": sys.executable}
    try:
        import torch

        available = bool(torch.cuda.is_available())
        environment.update({
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "cuda_available": available,
            "gpu_name": torch.cuda.get_device_name(0) if available else "",
        })
    except Exception as exc:
        environment.update({"cuda_available": False, "runtime_probe_error": repr(exc)})
    return environment


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
        keys.add((str(row.get("method") or row.get("policy")), int(row["budget"]), int(row.get("replication", row.get("seed", 0)))))
    return keys


def seed_entry(budget: int, replication: int) -> dict:
    seed = 202609000 + int(budget) * 1000 + int(replication)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/synthetic_main.yaml"))
    ap.add_argument("--prepared", type=Path, default=Path("experiments/paper/oracle_artifact.pkl"))
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override config output_path (useful for isolated cloud shards).",
    )
    ap.add_argument(
        "--seed-manifest",
        type=Path,
        default=None,
        help="Override config seed_manifest for an isolated shard.",
    )
    ap.add_argument(
        "--certificate",
        type=Path,
        default=None,
        help="Override fixed-QMC equivalence certificate path.",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.output_path is not None:
        cfg["output_path"] = str(args.output_path)
    if args.seed_manifest is not None:
        cfg["seed_manifest"] = str(args.seed_manifest)
    if args.certificate is not None:
        cfg["fixed_qmc_equivalence_certificate"] = str(args.certificate)
    validate_experiment_mode(cfg, paper=True, formal=True)
    budgets = [int(args.budget)] if args.budget is not None else list(cfg["budgets"])
    replications = int(cfg["replications"])
    start, end = (0, replications) if args.rep_range is None else (int(args.rep_range[0]), int(args.rep_range[1]))
    methods = tuple(args.methods) if args.methods else tuple(cfg["methods"])
    manifest_path = Path(cfg["seed_manifest"])
    if not manifest_path.exists():
        write_manifest(manifest_path, cfg["budgets"], replications)
    manifest, _ = load_seed_manifest(manifest_path)
    if not args.prepared.exists():
        raise SystemExit(f"missing oracle artifact: {args.prepared}. Run scripts/build_oracle_artifact.py first.")
    prepared = pickle.loads(args.prepared.read_bytes())
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    panels = all_pairs()
    out_dir = Path(cfg["output_path"])
    out = out_dir / "rows.jsonl"
    done = _done_keys(out) if args.resume else set()
    commit = _git_commit()
    worktree_dirty, code_diff_sha = _git_state()
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    artifact_sha = sha256_file(args.prepared)
    cache_dir = cfg.get("information_storage_dir")
    runtime_environment = _runtime_environment()
    for budget in budgets:
        for rep in range(start, end):
            pending = [m for m in methods if (m, int(budget), int(rep)) not in done]
            if not pending:
                continue
            if (int(budget), int(rep)) not in manifest:
                entry = seed_entry(int(budget), int(rep))
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload.setdefault("rows", []).append(entry)
                manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                manifest[(int(budget), int(rep))] = {
                    name: int(entry[name])
                    for name in (
                        "replication_seed", "target_draw_seed", "pilot_or_design_seed",
                        "score_seed_root", "information_seed_root",
                    )
                }
            else:
                entry = dict(manifest[(int(budget), int(rep))])
            if cfg.get("information_seed_root") is not None:
                entry["information_seed_root"] = int(cfg["information_seed_root"])
            entry["replication"] = int(rep)
            score_storage_root = cfg.get("score_storage_dir")
            score_storage_dir = (
                str(Path(score_storage_root) / f"B{int(budget)}_rep{int(rep)}")
                if score_storage_root else None
            )
            rows = run_paper_replication(
                mixture, scale, panels, int(budget), int(entry["replication_seed"]), prepared,
                scoring_steps=int(cfg["scoring_steps"]),
                scoring_max_step_norm=float(cfg["scoring_max_step_norm"]),
                score_qmc_order=int(cfg["score_qmc_order"]),
                information_qmc_order=int(cfg["information_qmc_order"]),
                information_outer_rows=int(cfg["information_outer_rows"]),
                information_scrambles=int(cfg.get("information_scrambles", 2)),
                hot_dtype=str(cfg.get("dtype", "float32")),
                score_backend=str(cfg.get("score_backend", "cached_qmc")),
                information_inner=(
                    "cached_qmc"
                    if str(cfg.get("information_backend", "cached_qmc_cross")) == "cached_qmc_cross"
                    else str(cfg.get("information_backend"))
                ),
                information_storage_dir=cache_dir,
                information_reuse_existing=bool(cfg.get("information_reuse_existing", False)),
                score_storage_dir=score_storage_dir,
                adaptive_chunk_rows=int(cfg.get("adaptive_chunk_rows", 32)),
                score_scrambles=int(cfg.get("score_scrambles", 1)),
                score_mu_order=int(cfg.get("score_mu_qmc_order", cfg.get("score_qmc_order", 12))),
                score_mu_scrambles=int(cfg.get("score_mu_scrambles", 4)),
                share_score_basis=bool(cfg.get("share_score_basis", True)),
                score_build_on_cuda=bool(cfg.get("score_build_on_cuda", False)),
                score_direct_cuda=bool(cfg.get("score_direct_cuda", False)),
                adaptive_start_order=int(cfg.get("adaptive_start_order", 8)),
                adaptive_max_order=int(cfg.get("adaptive_max_order", 18)),
                adaptive_atol=float(cfg.get("adaptive_atol", 2e-5)),
                adaptive_rtol=float(cfg.get("adaptive_rtol", 2e-4)),
                adaptive_scrambles=int(cfg.get("adaptive_scrambles", 4)),
                policies=tuple(pending),
                pilot_schedule=cfg["pilot_schedule"],
                seed_manifest_entry=entry,
                reproducibility={
                    "code_commit": commit,
                    "dirty_worktree": worktree_dirty,
                    "code_diff_sha256": code_diff_sha,
                    "config_sha256": config_sha,
                    "config_path": str(args.config),
                    "artifact_sha256": artifact_sha,
                    "evidence_level": str(cfg.get("evidence_level", "L4")),
                    "experiment_mode": str(cfg.get("experiment_mode", "bulk")),
                    "environment": "synthetic",
                    "runtime_environment": runtime_environment,
                },
            )
            _append_jsonl(out, rows)
            if score_storage_dir and bool(cfg.get("score_storage_cleanup", True)):
                shutil.rmtree(score_storage_dir, ignore_errors=True)
            if args.profile:
                for row in rows:
                    rt = row["runtime"]
                    print(
                        f"{row['method']} B={budget} rep={rep} "
                        f"risk={row['risk_ratio']:.3f} "
                        f"total={rt.get('time_estimation', 0)+rt.get('time_score_basis', 0)+rt.get('time_info_basis', 0):.1f}s "
                        f"info={rt['time_info_basis']:.1f}s score={rt['time_score_basis']:.1f}s est={rt['time_estimation']:.1f}s",
                        flush=True,
                    )
            done.update((row["method"], int(budget), int(rep)) for row in rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
