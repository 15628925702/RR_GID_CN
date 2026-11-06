#!/usr/bin/env python3
"""Submit E5-v3 shards: uncapped design + I_S(0) ablation. No personal names."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260913_rr_gid_e5_v3"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402

R1_BUDGETS = (400, 800, 1600, 3200)
R2_CAMPAIGNS = ("batch7", "batches8_9", "batch10")
R2_BUDGETS = (800, 1600, 3200)
CAMP_TAG = {"batch7": "b7", "batches8_9": "b89", "batch10": "b10"}
REP_SLICES = ((0, 10, "a"), (10, 20, "b"))


def add_job(service, attempts, *, name, entry, description):
    body = payload(
        name=name,
        entrypoint=entry,
        gpu_num=1,
        cpu=13.0,
        memory=234.0,
        deadline=7200,
        description=description,
    )
    obj = call(service, "CreateCustomTask", "2021-10-01", body)
    code, message = err(obj)
    task_id = extract_id(obj)
    rec = {
        "task_id": task_id,
        "name": name,
        "entrypoint": entry,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "Failed" if code else "submitted",
        "error_code": code,
        "error_message": (message or "")[:300] if message else None,
    }
    attempts.append(rec)
    print(json.dumps({"ok": not bool(code), **{k: rec[k] for k in ("task_id", "name", "status", "error_code")}}, ensure_ascii=False), flush=True)
    Path(f"{RUN}/attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
    if code:
        print(f"WARN submit failed for {name}: {code} {message}", flush=True)


def main() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = ""
    Path(RUN).mkdir(parents=True, exist_ok=True)
    service = make_service()
    attempts = []

    for budget in R1_BUDGETS:
        for start, end, tag in REP_SLICES:
            shard = f"b{budget}-{tag}"
            out = f"{ROOT}/results/e5_r1_v3/shards/shard_{shard}"
            entry = (
                f"bash -lc 'mkdir -p {out}; "
                f"bash {ROOT}/scripts/run_cloud_e5_r1_v3_shard.sh {budget} {start} {end} {shard} "
                f"> {out}/cmd.log 2>&1'"
            )
            add_job(
                service, attempts,
                name=f"rr-gid-e5r1v3-b{budget}-{tag}",
                entry=entry,
                description=f"E5-R1 v3 uncapped design B={budget} reps {start}-{end}",
            )

    for camp in R2_CAMPAIGNS:
        for budget in R2_BUDGETS:
            for start, end, tag in REP_SLICES:
                shard = f"{CAMP_TAG[camp]}-{budget}-{tag}"
                out = f"{ROOT}/results/e5_r2_v3/shards/shard_{shard}"
                entry = (
                    f"bash -lc 'mkdir -p {out}; "
                    f"bash {ROOT}/scripts/run_cloud_e5_r2_v3_shard.sh {camp} {budget} {start} {end} {shard} "
                    f"> {out}/cmd.log 2>&1'"
                )
                add_job(
                    service, attempts,
                    name=f"rr-gid-e5r2v3-{CAMP_TAG[camp]}-{budget}-{tag}",
                    entry=entry,
                    description=f"E5-R2 v3 uncapped {camp} B={budget} reps {start}-{end}",
                )

    for budget in R1_BUDGETS:
        for start, end, tag in REP_SLICES:
            shard = f"b{budget}-{tag}"
            out = f"{ROOT}/results/e5_r1_v3_info0/shards/shard_{shard}"
            entry = (
                f"bash -lc 'mkdir -p {out}; "
                f"bash {ROOT}/scripts/run_cloud_e5_r1_v3_info0_shard.sh {budget} {start} {end} {shard} "
                f"> {out}/cmd.log 2>&1'"
            )
            add_job(
                service, attempts,
                name=f"rr-gid-e5r1i0-b{budget}-{tag}",
                entry=entry,
                description=f"E5-R1 v3 I_S(0) B={budget} reps {start}-{end}",
            )

    Path(f"{RUN}/attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
    Path(f"{RUN}/status.json").write_text(
        json.dumps(
            {
                "n_tasks": len(attempts),
                "task_ids": [a["task_id"] for a in attempts],
                "latest_status": "submitted",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "n_tasks": len(attempts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
