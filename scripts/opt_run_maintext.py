"""Run the remaining opt-track main-text grids, then rebuild tables/figures.

1. E1 J=4, 20 paired replications, frozen E1 seeds, five PDF budgets.
2. R1 v6mix extra reps 20-30 on the same jsonl (paired CI).
3. Bundle + optional Hub upload to RR-GID-CN-opt.

Does not touch freeze jsonl or Hub freeze-20260914-v6rdv5.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(r"F:\Python\python.exe")
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
ENV["PYTHONUNBUFFERED"] = "1"
LOG = ROOT / "results" / "opt" / "maintext_run.log"


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(cmd: list[str]) -> int:
    log("RUN " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=ENV)
    log(f"exit {proc.returncode}")
    return int(proc.returncode)


def unique_e1(path: Path) -> int:
    if not path.exists():
        return 0
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((row.get("method") or row.get("policy"), int(row["budget"]), int(row.get("replication", 0))))
    return len(keys)


def unique_r1(path: Path) -> int:
    return unique_e1(path)


def upload_opt() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        log("HF_TOKEN missing, skip upload")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        repo = "Guaogua/RR-GID-CN-opt"
        mapping = [
            (ROOT / "results/opt/e1_j4_20/rows.jsonl", "results/opt/e1_j4_20/rows.jsonl"),
            (ROOT / "configs/opt/e1_j4_20.yaml", "configs/opt/e1_j4_20.yaml"),
            (ROOT / "results/opt/e5_r1_v6mix_20/rows.jsonl", "results/opt/e5_r1_v6mix_20/rows.jsonl"),
            (ROOT / "configs/opt/gas_r1_v6mix_20.yaml", "configs/opt/gas_r1_v6mix_20.yaml"),
            (ROOT / "configs/opt/gas_r1_v6mix_30.yaml", "configs/opt/gas_r1_v6mix_30.yaml"),
            (ROOT / "results/opt/e5_r2_v6mix_calahat_20/rows.jsonl", "results/opt/e5_r2_v6mix_calahat_20/rows.jsonl"),
            (ROOT / "configs/opt/gas_r2_v6mix_calahat_20.yaml", "configs/opt/gas_r2_v6mix_calahat_20.yaml"),
            (ROOT / "results/opt/maintext_summary.json", "results/opt/maintext_summary.json"),
            (ROOT / "figures/opt/fig3_gas.png", "figures/opt/fig3_gas.png"),
            (ROOT / "figures/opt/fig3_gas.pdf", "figures/opt/fig3_gas.pdf"),
            (ROOT / "figures/opt/fig1_synthetic_main_j4.png", "figures/opt/fig1_synthetic_main_j4.png"),
            (ROOT / "figures/opt/fig1_synthetic_main_j4.pdf", "figures/opt/fig1_synthetic_main_j4.pdf"),
            (ROOT / "核心文档/RR_GID_opt正文_三问题修完_20260915.md", "docs/RR_GID_opt_maintext_20260915.md"),
            (ROOT / "results/opt/HUB_README.md", "README.md"),
        ]
        for local, remote in mapping:
            if not local.exists():
                log("skip missing " + str(local))
                continue
            log("upload " + remote)
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type="dataset")
    except Exception as exc:
        log("upload failed: " + type(exc).__name__ + " " + str(exc)[:300])


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("maintext run start")
    e1_jsonl = ROOT / "results/opt/e1_j4_20/rows.jsonl"
    e1_cfg = ROOT / "configs/opt/e1_j4_20.yaml"
    if unique_e1(e1_jsonl) < 400:
        code = run([
            str(PY), str(ROOT / "scripts/run_synthetic_main.py"),
            "--config", str(e1_cfg), "--resume", "--profile",
        ])
        if code != 0:
            log("E1 failed")
            sys.exit(code)
    log(f"E1 unique={unique_e1(e1_jsonl)}/400")

    r1_jsonl = ROOT / "results/opt/e5_r1_v6mix_20/rows.jsonl"
    r1_cfg = ROOT / "configs/opt/gas_r1_v6mix_30.yaml"
    manifest = ROOT / "results/opt/e5_r1_v6mix_20/seed_manifest.json"
    if not manifest.exists():
        sys.path.insert(0, str(ROOT / "scripts"))
        import run_gas_semisynthetic as r1mod
        r1mod.write_manifest(manifest, [400, 800, 1600, 3200], 30)
        log("wrote " + str(manifest))
    if unique_r1(r1_jsonl) < 480:
        code = run([
            str(PY), str(ROOT / "scripts/run_gas_semisynthetic.py"),
            "--config", str(r1_cfg),
            "--data", str(ROOT / "data/gas/processed/gas_processed.npz"),
            "--resume", "--profile", "--rep-range", "20", "30",
        ])
        if code != 0:
            log("R1 extra failed")
            sys.exit(code)
    log(f"R1 unique={unique_r1(r1_jsonl)}")

    run([str(PY), str(ROOT / "scripts/make_opt_maintext_bundle.py")])
    upload_opt()
    log("maintext run finished")


if __name__ == "__main__":
    main()
