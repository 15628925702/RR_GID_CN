#!/usr/bin/env python3
"""Submit v6sp 20-rep full grid + hat-collapse smokes. No personal names."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260913_rr_gid_e5_r1_hatfix"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402

SH = f"{ROOT}/scripts/run_cloud_e5_r1_smoke_shard.sh"
REP_SLICES = ((0, 10, "a"), (10, 20, "b"))
R1_BUDGETS = (400, 800, 1600, 3200)
SMOKES = ("v6np", "v6sc", "v6s12", "v6j4", "v6rd", "v6spnp")


def submit(service, attempts, name, out, entry, description):
    body = payload(
        name=name, entrypoint=entry, gpu_num=1, cpu=13.0, memory=234.0, deadline=7200,
        description=description,
    )
    obj = call(service, "CreateCustomTask", "2021-10-01", body)
    code, message = err(obj)
    rec = {
        "task_id": extract_id(obj),
        "name": name,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "Failed" if code else "submitted",
        "error_code": code,
        "error_message": (message or "")[:300] if message else None,
    }
    attempts.append(rec)
    print(json.dumps({"ok": not bool(code), **{k: rec[k] for k in ("task_id", "name", "status")}}, ensure_ascii=False), flush=True)
    Path(f"{RUN}/attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
    if code:
        print(f"WARN {name}: {code} {message}", flush=True)


def main() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = ""
    Path(RUN).mkdir(parents=True, exist_ok=True)
    service = make_service()
    attempts = []
    cfg20 = f"{ROOT}/configs/paper/gas_semisynthetic_empirical_v6sp_20_cloud.yaml"
    for budget in R1_BUDGETS:
        for start, end, tag in REP_SLICES:
            shard = f"b{budget}-{tag}"
            out = f"{ROOT}/results/e5_r1_v6sp/shards/shard_{shard}"
            entry = (
                f"bash -lc 'mkdir -p {out}; "
                f"bash {SH} {cfg20} {budget} {start} {end} {out} > {out}/cmd.log 2>&1'"
            )
            submit(
                service, attempts, f"rr-gid-e5r1-v6sp-b{budget}-{tag}", out, entry,
                f"E5-R1 two-stage I_S 20-rep B={budget} reps {start}-{end}",
            )
    for tag in SMOKES:
        cfg = f"{ROOT}/configs/paper/gas_semisynthetic_empirical_{tag}_cloud.yaml"
        out = f"{ROOT}/results/e5_r1_{tag}_smoke/shards/shard_b1600-s"
        entry = (
            f"bash -lc 'mkdir -p {out}; "
            f"bash {SH} {cfg} 1600 0 5 {out} > {out}/cmd.log 2>&1'"
        )
        submit(
            service, attempts, f"rr-gid-e5r1-{tag}-s1600", out, entry,
            f"E5-R1 hatfix smoke {tag} B=1600 reps 0-5",
        )
    print(json.dumps({"ok": True, "n_tasks": len(attempts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
