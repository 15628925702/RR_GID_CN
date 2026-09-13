"""Paper E3: frozen-generator reuse vs per-campaign discriminative OED."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
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

from rr_gid_cn.paper_run import run_paper_replication
from rr_gid_cn.p4_integrity import sha256_file, validate_experiment_mode
from rr_gid_cn.s1_gate import panel_information_cross, policy_designs
from rr_gid_cn.synthetic_oracle import (
    all_pairs,
    beta_direction_and_scale,
    feature_fn_from_dictionary,
    make_feature_dictionary,
    make_frozen_mixture,
    reference_scale,
    sample_feature_draw,
    tilted_moments,
)
from rr_gid_cn.vaeac import VAEACGenerator, learned_information, load_vaeac_checkpoint


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


def _last_cum(path: Path, sequence: int) -> tuple[float, float]:
    rr, disc = 0.0, 0.0
    if not path.exists():
        return rr, disc
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("sequence", -1)) != int(sequence):
            continue
        value = float(row.get("cumulative_method_seconds", 0.0))
        if row.get("method") == "RR-GID":
            rr = value
        elif row.get("method") == "Discriminative Score OED":
            disc = value
    return rr, disc


def _done_keys(path: Path) -> set[tuple[int, int]]:
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((int(row["sequence"]), int(row["campaign_index"])))
    return keys


class VAEACDesignBasis:
    def __init__(self, generator, n_tilted: int, n_cond: int, seed: int, feature_fn=None):
        self.generator = generator
        self.n_tilted = int(n_tilted)
        self.n_cond = int(n_cond)
        self.seed = int(seed)
        self.feature_fn = feature_fn
        self.last_seconds = 0.0

    def information(self, beta, panels):
        t0 = perf_counter()
        _, infos = learned_information(
            self.generator, beta, tuple(panels),
            n_tilted=self.n_tilted, n_conditional=self.n_cond,
            seed=self.seed, feature_fn=self.feature_fn,
        )
        self.last_seconds = perf_counter() - t0
        return {tuple(panel): np.asarray(infos[i], dtype=np.float64) for i, panel in enumerate(panels)}


def campaign_prepared(mixture, scale, panels, reference, seed, feature_fn):
    beta_true = beta_direction_and_scale(reference, int(seed), 0.5, scale, feature_fn=feature_fn)
    _, fisher = tilted_moments(beta_true, reference, scale, feature_fn=feature_fn)
    oracle_information = panel_information_cross(
        mixture, beta_true, panels, reference, scale,
        n_tilted=128, n_cond=32, seed=int(seed) + 1,
        feature_fn=feature_fn, conditional_method="qmc", qmc_order=8,
    )
    designs, certs = policy_designs(reference, panels, fisher, oracle_information, return_certificates=True)
    return {
        "reference": reference,
        "beta_true": beta_true,
        "fisher": fisher,
        "information": oracle_information,
        "designs": designs,
        "oracle_constant": {
            "phi_oracle": certs["oracle RR-GID"]["objective"],
            "half_phi_oracle": 0.5 * certs["oracle RR-GID"]["objective"],
            **certs["oracle RR-GID"],
        },
    }


def seed_entry(sequence: int, campaign: int) -> dict:
    seed = 202609000 + int(sequence) * 10000 + int(campaign)
    return {
        "replication": int(campaign),
        "replication_seed": seed,
        "target_draw_seed": seed + 3,
        "pilot_or_design_seed": seed + 11,
        "score_seed_root": seed + 4,
        "information_seed_root": seed + 7000,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/reuse.yaml"))
    ap.add_argument("--prepared", type=Path, default=Path("experiments/paper/oracle_artifact.pkl"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--sequence-range", type=int, nargs=2, default=None)
    ap.add_argument("--max-campaigns", type=int, default=None)
    ap.add_argument("--output-path", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_experiment_mode(cfg, paper=True, formal=True)
    ckpt_path = Path(args.checkpoint or cfg["generator_checkpoint"])
    if not ckpt_path.exists():
        raise SystemExit(f"missing VAEAC checkpoint: {ckpt_path}. Train with scripts/p6_train.py first.")
    if not args.prepared.exists():
        raise SystemExit(f"missing oracle artifact: {args.prepared}")
    artifact = pickle.loads(args.prepared.read_bytes())
    mixture = make_frozen_mixture(seed=2026, alpha=1.0)
    scale = reference_scale(mixture, n=6000, seed=2026)
    reference = artifact["reference"]
    dictionary = make_feature_dictionary()
    all_panels = all_pairs()
    n_seq = int(cfg["sequences"])
    n_campaigns = int(cfg["campaigns"][-1])
    start, end = (0, n_seq) if args.sequence_range is None else (int(args.sequence_range[0]), int(args.sequence_range[1]))
    if args.max_campaigns is not None:
        n_campaigns = min(n_campaigns, int(args.max_campaigns))
    budget = int(cfg["budgets"][0])
    out = Path(args.output_path or cfg["output_path"]) / "rows.jsonl"
    done = _done_keys(out) if args.resume else set()
    commit = _git_commit()
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    artifact_sha = sha256_file(args.prepared)
    ckpt_sha = sha256_file(ckpt_path)
    device = "cuda"
    try:
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    base_model, _ckpt = load_vaeac_checkpoint(str(ckpt_path), device=device, expected_dim=16)
    train_seconds = float(json.loads(Path(cfg["output_path"]).joinpath("vaeac_train.json").read_text())["train_seconds"]) if Path(cfg["output_path"]).joinpath("vaeac_train.json").exists() else 0.0
    prefixes = {int(t) for t in cfg["campaigns"]}
    t_wall = perf_counter()
    stop = False
    for seq in range(start, end):
        if stop:
            break
        cum_rr, cum_disc = _last_cum(out, seq) if args.resume else (0.0, 0.0)
        for camp in range(n_campaigns):
            if args.max_hours is not None and (perf_counter() - t_wall) / 3600.0 >= float(args.max_hours):
                print(f"max-hours {args.max_hours} reached at seq={seq} camp={camp}", flush=True)
                stop = True
                break
            if (seq, camp) in done:
                continue
            entry = seed_entry(seq, camp)
            rng = np.random.default_rng(entry["pilot_or_design_seed"])
            feats = sample_feature_draw(dictionary, entry["replication_seed"], r=int(cfg.get("r_features", 12)))
            fn = feature_fn_from_dictionary(feats, scale)
            idx = rng.choice(len(all_panels), size=int(cfg.get("n_panels", 40)), replace=False)
            panels = tuple(all_panels[int(i)] for i in idx)
            if args.profile:
                print(f"seq={seq} camp={camp} building campaign oracle...", flush=True)
            prepared = campaign_prepared(mixture, scale, panels, reference, entry["replication_seed"] + 17, fn)
            gen = VAEACGenerator(base_model, scale, alpha=1.0, device=device, feature_fn=fn)
            design_basis = VAEACDesignBasis(
                gen,
                n_tilted=int(cfg.get("gen_info_tilted", 128)),
                n_cond=int(cfg.get("gen_info_cond", 32)),
                seed=entry["information_seed_root"],
                feature_fn=fn,
            )
            rows = run_paper_replication(
                mixture, scale, panels, budget, int(entry["replication_seed"]), prepared,
                scoring_steps=int(cfg["scoring_steps"]),
                scoring_max_step_norm=float(cfg["scoring_max_step_norm"]),
                score_qmc_order=int(cfg["score_qmc_order"]),
                score_mu_order=int(cfg.get("score_mu_qmc_order", cfg.get("score_qmc_order", 12))),
                score_mu_scrambles=int(cfg.get("score_mu_scrambles", 4)),
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
                policies=tuple(cfg["methods"]),
                pilot_schedule=cfg["pilot_schedule"],
                seed_manifest_entry=entry,
                mlp_steps=int(cfg.get("mlp_steps", 100)),
                feature_fn=fn,
                design_info_basis=design_basis,
                generator_training_seconds=float(train_seconds) if camp == 0 else 0.0,
                reproducibility={
                    "code_commit": commit,
                    "config_sha256": config_sha,
                    "artifact_sha256": artifact_sha,
                    "checkpoint_sha256": ckpt_sha,
                    "environment": "synthetic",
                },
            )
            camp_rr = float(design_basis.last_seconds)
            camp_disc = 0.0
            for row in rows:
                rt = row.get("runtime") or {}
                if row["method"] == "RR-GID":
                    camp_rr += float(rt.get("time_fw", 0.0))
                    if camp == 0:
                        camp_rr += float(train_seconds)
                elif row["method"] == "Discriminative Score OED":
                    camp_disc += float(rt.get("time_disc", 0.0))
            cum_rr += camp_rr
            cum_disc += camp_disc
            for row in rows:
                row["experiment"] = "reuse"
                row["sequence"] = int(seq)
                row["campaign_index"] = int(camp)
                row["campaigns_so_far"] = int(camp) + 1
                row["report_prefix"] = (camp + 1) in prefixes
                if row["method"] == "RR-GID":
                    row["cumulative_method_seconds"] = float(cum_rr)
                    row["method_specific_seconds"] = float(camp_rr)
                else:
                    row["cumulative_method_seconds"] = float(cum_disc)
                    row["method_specific_seconds"] = float(camp_disc)
            _append_jsonl(out, rows)
            if args.profile:
                for row in rows:
                    print(
                        f"seq={seq} camp={camp} {row['method']} "
                        f"risk={row['risk_ratio']:.3f} cum={row['cumulative_method_seconds']:.1f}s",
                        flush=True,
                    )
            done.add((seq, camp))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
