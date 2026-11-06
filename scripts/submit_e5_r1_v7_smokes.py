#!/usr/bin/env python3
"""Submit E5-R1 5-rep smokes at B=1600. No personal names."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260913_rr_gid_e5_r1_v7_smokes"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402

JOBS = (
    ("v7", "knn-init I_S(knn) mix0.1 cap3"),
    ("v7z", "knn-init I_S(0) mix0.1 cap3"),
    ("v6m05", "knn-pilot mix0.05"),
    ("v6m00", "knn-pilot mix0"),
    ("v6c2", "knn-pilot score-cap2"),
    ("v7c2", "knn-init I_S(knn) cap2"),
    ("v6r", "knn-pilot radius1.25"),
)


def main() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = ""
    Path(RUN).mkdir(parents=True, exist_ok=True)
    service = make_service()
    attempts = []
    for tag, desc in JOBS:
        name = f"rr-gid-e5r1-{tag}-s1600"
        cfg = f"{ROOT}/configs/paper/gas_semisynthetic_empirical_{tag}_cloud.yaml"
        out = f"{ROOT}/results/e5_r1_{tag}_smoke/shards/shard_b1600-s"
        entry = (
            f"bash -lc 'mkdir -p {out}; "
            f"bash {ROOT}/scripts/run_cloud_e5_r1_smoke_shard.sh {cfg} 1600 0 5 {out} "
            f"> {out}/cmd.log 2>&1'"
        )
        body = payload(
            name=name,
            entrypoint=entry,
            gpu_num=1,
            cpu=13.0,
            memory=234.0,
            deadline=7200,
            description=f"E5-R1 smoke B=1600 reps 0-5 {desc}",
        )
        obj = call(service, "CreateCustomTask", "2021-10-01", body)
        code, message = err(obj)
        rec = {
            "task_id": extract_id(obj),
            "name": name,
            "tag": tag,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "status": "Failed" if code else "submitted",
            "error_code": code,
            "error_message": (message or "")[:300] if message else None,
        }
        attempts.append(rec)
        print(json.dumps({"ok": not bool(code), **{k: rec[k] for k in ("task_id", "name", "tag", "status")}}, ensure_ascii=False), flush=True)
        Path(f"{RUN}/attempts.json").write_text(json.dumps(attempts, indent=2), encoding="utf-8")
        if code:
            print(f"WARN {name}: {code} {message}", flush=True)
    print(json.dumps({"ok": True, "n_tasks": len(attempts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
