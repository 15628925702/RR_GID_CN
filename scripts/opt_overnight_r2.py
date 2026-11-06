"""Overnight R2 opt track: finish empahat 20, archive, follow-R1 20, PDF-calibrated A-hat.

Do not touch frozen E1 / e5_r1_v6rd / e5_r2_v5.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = Path(r"F:\Python\python.exe")
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "scripts")
ENV["PYTHONUNBUFFERED"] = "1"
LOG = ROOT / "results" / "opt" / "overnight_r2.log"


def log(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def unique_rows(path: Path) -> int:
    if not path.exists():
        return 0
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add(
            (
                row.get("campaign"),
                row.get("method") or row.get("policy"),
                int(row["budget"]),
                int(row.get("replication", 0)),
            )
        )
    return len(keys)


def run_grid(config: Path, extra: list[str] | None = None) -> int:
    cmd = [str(PY), str(ROOT / "scripts" / "run_gas_natural.py"), "--config", str(config), "--resume", "--profile"]
    if extra:
        cmd.extend(extra)
    log("RUN " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT), env=ENV)
    log(f"exit {proc.returncode} config={config.name}")
    return int(proc.returncode)


def summarize(jsonl: Path) -> dict:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {}
    for row in rows:
        by[(row["campaign"], row["method"], int(row["budget"]), int(row["replication"]))] = row
    methods = ["Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"]
    out = {
        "n_lines": len(rows),
        "n_unique": len(by),
        "rows_sha256": __import__("hashlib").sha256(jsonl.read_bytes()).hexdigest(),
        "means": {},
    }
    camps = sorted({k[0] for k in by})
    budgets = sorted({k[2] for k in by})
    for camp in camps:
        out["means"][camp] = {}
        for budget in budgets:
            out["means"][camp][str(budget)] = {}
            for method in methods:
                xs = [
                    by[(camp, method, budget, rep)]["projection_loss"]
                    for rep in range(20)
                    if (camp, method, budget, rep) in by
                ]
                if not xs:
                    continue
                arr = np.asarray(xs, float)
                se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
                out["means"][camp][str(budget)][method] = {
                    "mean": float(arr.mean()),
                    "ci_halfwidth": float(1.96 * se),
                    "n": int(len(arr)),
                }
    return out


def archive_and_upload(name: str, jsonl: Path, yaml_path: Path) -> None:
    summary = summarize(jsonl)
    sp = jsonl.parent / "summary.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    hdir = Path(r"H:/RR_GID_CN_data/active/opt_20260914") / name
    hdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jsonl, hdir / "rows.jsonl")
    shutil.copy2(sp, hdir / "summary.json")
    shutil.copy2(yaml_path, hdir / yaml_path.name)
    log(f"archived {name} unique={summary['n_unique']} sha={summary['rows_sha256'][:12]}")
    token = os.environ.get("HF_TOKEN")
    if not token:
        log("HF_TOKEN missing, skip upload")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        repo = "Guaogua/RR-GID-CN-opt"
        mapping = [
            (jsonl, f"results/opt/{name}/rows.jsonl"),
            (sp, f"results/opt/{name}/summary.json"),
            (yaml_path, f"configs/opt/{yaml_path.name}"),
        ]
        for local, remote in mapping:
            log("upload " + remote)
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type="dataset")
    except Exception as exc:
        log("upload failed: " + type(exc).__name__ + " " + str(exc)[:200])


def rr_beats_uniform(jsonl: Path) -> bool:
    rows = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    rr = np.mean([r["projection_loss"] for r in rows if r["method"] == "RR-GID"])
    uni = np.mean([r["projection_loss"] for r in rows if r["method"] == "Uniform SQD"])
    log(f"smoke RR={rr:.4f} Uni={uni:.4f}")
    return float(rr) < float(uni)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return str(pid) in out
    except Exception:
        return False


def wait_empahat() -> None:
    jsonl = ROOT / "results" / "opt" / "e5_r2_v6mix_empahat_20" / "rows.jsonl"
    cfg = ROOT / "configs" / "opt" / "gas_r2_v6mix_empahat_20.yaml"
    writer = int(os.environ.get("EMPAHAT_PID") or "0")
    while unique_rows(jsonl) < 720:
        n = unique_rows(jsonl)
        alive = pid_alive(writer)
        log(f"empahat unique={n}/720 writer_alive={alive} pid={writer}")
        if not alive:
            log("empahat writer gone, resume")
            code = run_grid(cfg)
            writer = 0
            if unique_rows(jsonl) >= 720:
                break
            if code != 0:
                log("resume failed, retry in 60s")
        time.sleep(60)
    log(f"empahat done unique={unique_rows(jsonl)}")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("overnight start")
    wait_empahat()
    archive_and_upload(
        "e5_r2_v6mix_empahat_20",
        ROOT / "results" / "opt" / "e5_r2_v6mix_empahat_20" / "rows.jsonl",
        ROOT / "configs" / "opt" / "gas_r2_v6mix_empahat_20.yaml",
    )

    smoke_cfg = ROOT / "configs" / "opt" / "gas_r2_v6mix_calahat_smoke.yaml"
    smoke_jsonl = ROOT / "results" / "opt" / "e5_r2_v6mix_calahat_smoke" / "rows.jsonl"
    run_grid(
        smoke_cfg,
        extra=["--campaign", "batches8_9", "--budget", "1600", "--rep-range", "0", "5"],
    )
    calahat_ok = smoke_jsonl.exists() and unique_rows(smoke_jsonl) >= 20 and rr_beats_uniform(smoke_jsonl)
    log(f"calahat smoke ok={calahat_ok}")

    follow_cfg = ROOT / "configs" / "opt" / "gas_r2_v6mix_follow_r1_20.yaml"
    run_grid(follow_cfg)
    follow_jsonl = ROOT / "results" / "opt" / "e5_r2_v6mix_follow_r1_20" / "rows.jsonl"
    if unique_rows(follow_jsonl) >= 720:
        archive_and_upload("e5_r2_v6mix_follow_r1_20", follow_jsonl, follow_cfg)

    if calahat_ok:
        cal_cfg = ROOT / "configs" / "opt" / "gas_r2_v6mix_calahat_20.yaml"
        run_grid(cal_cfg)
        cal_jsonl = ROOT / "results" / "opt" / "e5_r2_v6mix_calahat_20" / "rows.jsonl"
        if unique_rows(cal_jsonl) >= 720:
            archive_and_upload("e5_r2_v6mix_calahat_20", cal_jsonl, cal_cfg)
    else:
        log("skip calahat 20-rep: smoke did not beat Uniform")
    log("overnight finished")


if __name__ == "__main__":
    main()
