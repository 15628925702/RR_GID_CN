"""Paper E2: nonlinearity mechanism at fixed B=8000, varying warp alpha."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from rr_gid_cn.p4_integrity import sha256_file, validate_experiment_mode
from rr_gid_cn.paper_run import run_paper_replication
from rr_gid_cn.s1_gate import prepare_s1_oracle
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


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


def _done_keys(path: Path) -> set[tuple[str, float, int, int]]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((
            str(row.get("method") or row.get("policy")),
            float(row["alpha"]),
            int(row["budget"]),
            int(row.get("replication", 0)),
        ))
    return keys


def seed_entry(alpha: float, budget: int, replication: int) -> dict:
    seed = 202609000 + int(round(float(alpha) * 100)) * 100000 + int(budget) * 1000 + int(replication)
    return {
        "alpha": float(alpha),
        "budget": int(budget),
        "replication": int(replication),
        "replication_seed": seed,
        "target_draw_seed": seed + 3,
        "pilot_or_design_seed": seed + 11,
        "score_seed_root": seed + 4,
        "information_seed_root": seed + 7000,
    }


def write_manifest(path: Path, alphas, budgets, replications: int) -> None:
    rows = [
        seed_entry(alpha, budget, rep)
        for alpha in alphas
        for budget in budgets
        for rep in range(int(replications))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def artifact_path(alpha: float, root: Path | None = None) -> Path:
    if root is not None:
        root = Path(root)
    if abs(float(alpha) - 1.0) < 1e-12:
        return (root / "oracle_artifact.pkl") if root is not None else Path("experiments/paper/oracle_artifact.pkl")
    tag = f"{float(alpha):.1f}".replace(".", "p")
    return (root / f"oracle_artifact_alpha_{tag}.pkl") if root is not None else Path(f"experiments/paper/oracle_artifact_alpha_{tag}.pkl")


def ensure_artifact(alpha: float, panels, artifact_root: Path | None = None) -> dict:
    path = artifact_path(alpha, artifact_root)
    if path.exists():
        return pickle.loads(path.read_bytes())
    source = artifact_path(alpha)
    if artifact_root is not None and source.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        source_meta = Path(str(source) + ".json")
        if source_meta.exists():
            Path(str(path) + ".json").write_bytes(source_meta.read_bytes())
        return pickle.loads(path.read_bytes())
    mixture = make_frozen_mixture(seed=2026, alpha=float(alpha))
    scale = reference_scale(mixture, n=6000, seed=2026)
    prepared = prepare_s1_oracle(
        mixture, scale, panels, seed=2026,
        reference_size=20000, large_reference_size=20000,
        information_samples=128, conditional_samples=32,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(prepared))
    Path(str(path) + ".json").write_text(
        json.dumps({"alpha": float(alpha), "c_star": prepared["oracle_constant"]["half_phi_oracle"]}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {path}", flush=True)
    return prepared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/nonlinearity.yaml"))
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--rep-range", type=int, nargs=2, default=None)
    ap.add_argument("--output-path", type=Path, default=None)
    ap.add_argument("--artifact-storage-dir", type=Path, default=None)
    ap.add_argument("--seed-manifest", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_experiment_mode(cfg, paper=True, formal=True)
    alphas = [float(args.alpha)] if args.alpha is not None else [float(a) for a in cfg["alphas"]]
    budgets = [int(b) for b in cfg["budgets"]]
    replications = int(cfg["replications"])
    start, end = (0, replications) if args.rep_range is None else (int(args.rep_range[0]), int(args.rep_range[1]))
    methods = tuple(cfg["methods"])
    artifact_root = Path(args.artifact_storage_dir or cfg["artifact_storage_dir"]) if (args.artifact_storage_dir or cfg.get("artifact_storage_dir")) else None
    manifest_path = Path(args.seed_manifest or cfg["seed_manifest"])
    if not manifest_path.exists():
        write_manifest(manifest_path, cfg["alphas"], budgets, replications)
    raw_rows = json.loads(manifest_path.read_text(encoding="utf-8")).get("rows", [])
    by_key = {}
    for row in raw_rows:
        by_key[(float(row.get("alpha", 1.0)), int(row["budget"]), int(row["replication"]))] = row
    panels = all_pairs()
    out = Path(args.output_path or cfg["output_path"]) / "rows.jsonl"
    done = _done_keys(out) if args.resume else set()
    commit = _git_commit()
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    for alpha in alphas:
        prepared = ensure_artifact(alpha, panels, artifact_root)
        mixture = make_frozen_mixture(seed=2026, alpha=float(alpha))
        scale = reference_scale(mixture, n=6000, seed=2026)
        artifact_sha = sha256_file(artifact_path(alpha, artifact_root))
        alpha_tag = f"{float(alpha):.1f}".replace(".", "p")
        info_storage_root = cfg.get("information_storage_dir")
        if info_storage_root:
            info_storage_dir = str(Path(info_storage_root) / f"alpha_{alpha_tag}")
        else:
            info_storage_dir = None
        score_storage_root = cfg.get("score_storage_dir")
        score_storage_dir = str(Path(score_storage_root) / f"alpha_{alpha_tag}") if score_storage_root else None
        for budget in budgets:
            for rep in range(start, end):
                pending = [m for m in methods if (m, float(alpha), int(budget), int(rep)) not in done]
                if not pending:
                    continue
                key = (float(alpha), int(budget), int(rep))
                entry = dict(by_key.get(key) or seed_entry(alpha, budget, rep))
                if cfg.get("information_seed_root") is not None:
                    entry["information_seed_root"] = int(cfg["information_seed_root"])
                entry["replication"] = int(rep)
                score_storage_dir_rep = (
                    str(Path(score_storage_dir) / f"B{int(budget)}_rep{int(rep)}")
                    if score_storage_dir else None
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
                    score_direct_cuda=bool(cfg.get("score_direct_cuda", False)),
                    score_build_on_cuda=bool(cfg.get("score_build_on_cuda", False)),
                    share_score_basis=bool(cfg.get("share_score_basis", True)),
                    adaptive_start_order=int(cfg.get("adaptive_start_order", 8)),
                    adaptive_max_order=int(cfg.get("adaptive_max_order", 18)),
                    adaptive_atol=float(cfg.get("adaptive_atol", 2e-5)),
                    adaptive_rtol=float(cfg.get("adaptive_rtol", 2e-4)),
                    adaptive_scrambles=int(cfg.get("adaptive_scrambles", 4)),
                    information_inner=(
                        "cached_qmc"
                        if str(cfg.get("information_backend", "cached_qmc_cross")) == "cached_qmc_cross"
                        else str(cfg.get("information_backend"))
                    ),
                    information_storage_dir=info_storage_dir,
                    information_reuse_existing=bool(cfg.get("information_reuse_existing", False)),
                    score_storage_dir=score_storage_dir_rep,
                    score_scrambles=int(cfg.get("score_scrambles", 1)),
                    score_mu_order=int(cfg.get("score_mu_qmc_order", cfg.get("score_qmc_order", 12))),
                    score_mu_scrambles=int(cfg.get("score_mu_scrambles", 4)),
                    policies=tuple(pending),
                    pilot_schedule=cfg["pilot_schedule"],
                    seed_manifest_entry=entry,
                    reproducibility={
                        "code_commit": commit,
                        "config_sha256": config_sha,
                        "artifact_sha256": artifact_sha,
                        "environment": "synthetic",
                    },
                )
                for row in rows:
                    row["experiment"] = "nonlinearity"
                    row["alpha"] = float(alpha)
                _append_jsonl(out, rows)
                if score_storage_dir_rep and bool(cfg.get("score_storage_cleanup", True)):
                    shutil.rmtree(score_storage_dir_rep, ignore_errors=True)
                if args.profile:
                    for row in rows:
                        print(
                            f"alpha={alpha} {row['method']} B={budget} rep={rep} "
                            f"risk={row['risk_ratio']:.3f}",
                            flush=True,
                        )
                done.update((row["method"], float(alpha), int(budget), int(rep)) for row in rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
