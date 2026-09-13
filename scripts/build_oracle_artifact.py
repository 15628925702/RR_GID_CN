"""Build and freeze the synthetic oracle artifact used by paper experiments."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import yaml

from rr_gid_cn.p4_integrity import sha256_file
from rr_gid_cn.s1_gate import prepare_s1_oracle
from rr_gid_cn.synthetic_oracle import all_pairs, make_frozen_mixture, reference_scale


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/paper/oracle_calibration.yaml"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("experiments/paper/oracle_artifact.pkl"))
    ap.add_argument("--reference-size", type=int, default=20000)
    ap.add_argument("--large-reference-size", type=int, default=20000)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    mixture = make_frozen_mixture(seed=2026, alpha=float(args.alpha))
    scale = reference_scale(mixture, n=6000, seed=2026)
    panels = all_pairs()
    prepared = prepare_s1_oracle(
        mixture, scale, panels, seed=2026,
        reference_size=int(args.reference_size),
        large_reference_size=int(args.large_reference_size),
        information_samples=128, conditional_samples=32,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(pickle.dumps(prepared))
    meta = {
        "artifact": str(args.out),
        "sha256": sha256_file(args.out),
        "commit": _git_commit(),
        "config": str(args.config),
        "alpha": float(args.alpha),
        "c_star": prepared["oracle_constant"]["half_phi_oracle"],
        "phi_oracle": prepared["oracle_constant"]["phi_oracle"],
        "fw_gap": prepared["oracle_constant"]["fw_gap"],
        "artifact_metadata": prepared["artifact_metadata"],
    }
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
