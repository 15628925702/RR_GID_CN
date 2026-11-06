#!/usr/bin/env python3
"""Submit 9 one-GPU E5-R2 v2 shards. Names must not contain personal names."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260913_rr_gid_e5_r2_v2"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402

CAMPAIGNS = ("batch7", "batches8_9", "batch10")
BUDGETS = (800, 1600, 3200)
CAMP_TAG = {"batch7": "b7", "batches8_9": "b89", "batch10": "b10"}


def main() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = ""
    Path(RUN).mkdir(parents=True, exist_ok=True)
    service = make_service()
    attempts = []
    for camp in CAMPAIGNS:
        for budget in BUDGETS:
            tag = f"{CAMP_TAG[camp]}-{budget}"
            name = f"rr-gid-e5r2v2-{tag}"
            shard = tag
            out = f"{ROOT}/results/e5_r2_v2/shards/shard_{shard}"
            entry = (
                f"bash -lc 'mkdir -p {out}; "
                f"bash {ROOT}/scripts/run_cloud_e5_r2_v2_shard.sh {camp} {budget} 0 20 {shard} "
                f"> {out}/cmd.log 2>&1'"
            )
            body = payload(
                name=name,
                entrypoint=entry,
                gpu_num=1,
                cpu=13.0,
                memory=234.0,
                deadline=7200,
                description=f"E5-R2 v2 {camp} B={budget} 20rep empirical-knn",
            )
            obj = call(service, "CreateCustomTask", "2021-10-01", body)
            code, message = err(obj)
            task_id = extract_id(obj)
            rec = {
                "task_id": task_id,
                "name": name,
                "campaign": camp,
                "budget": budget,
                "shard": shard,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "status": "Failed" if code else "submitted",
                "error_code": code,
                "error_message": (message or "")[:300] if message else None,
            }
            attempts.append(rec)
            print(json.dumps({"ok": not bool(code), **rec}, ensure_ascii=False), flush=True)
            if code:
                Path(f"{RUN}/attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
                raise SystemExit(f"submit failed for {name}: {code} {message}")
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


if __name__ == "__main__":
    main()
