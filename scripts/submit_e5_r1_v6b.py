#!/usr/bin/env python3
"""Submit E5-R1 v6b: I_S(0) + v4 mix/share cap. No personal names."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260913_rr_gid_e5_r1_v6b"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402

R1_BUDGETS = (400, 800, 1600, 3200)
REP_SLICES = ((0, 10, "a"), (10, 20, "b"))


def main() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = ""
    Path(RUN).mkdir(parents=True, exist_ok=True)
    service = make_service()
    attempts = []
    for budget in R1_BUDGETS:
        for start, end, tag in REP_SLICES:
            shard = f"b{budget}-{tag}"
            name = f"rr-gid-e5r1v6b-b{budget}-{tag}"
            out = f"{ROOT}/results/e5_r1_v6b/shards/shard_{shard}"
            entry = (
                f"bash -lc 'mkdir -p {out}; "
                f"bash {ROOT}/scripts/run_cloud_e5_r1_v6b_shard.sh {budget} {start} {end} {shard} "
                f"> {out}/cmd.log 2>&1'"
            )
            body = payload(
                name=name, entrypoint=entry, gpu_num=1, cpu=13.0, memory=234.0, deadline=7200,
                description=f"E5-R1 v6b I_S(0) mix-cap B={budget} reps {start}-{end}",
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
    print(json.dumps({"ok": True, "n_tasks": len(attempts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
