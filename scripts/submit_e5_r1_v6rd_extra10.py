#!/usr/bin/env python3
"""v6rd extra 10 paired reps at B=400 only. Does not overwrite e5_r1_v6rd n=20."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260914_rr_gid_e5_r1_v6rd_extra"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402


def main() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = ""
    Path(RUN).mkdir(parents=True, exist_ok=True)
    service = make_service()
    cfg = f"{ROOT}/configs/paper/gas_semisynthetic_empirical_v6rd_20_cloud.yaml"
    sh = f"{ROOT}/scripts/run_cloud_e5_r1_smoke_shard.sh"
    out = f"{ROOT}/results/e5_r1_v6rd_extra/shards/shard_b400-c"
    name = "rr-gid-e5r1-v6rd-b400-c"
    entry = (
        f"bash -lc 'mkdir -p {out}; "
        f"bash {sh} {cfg} 400 20 30 {out} > {out}/cmd.log 2>&1'"
    )
    body = payload(
        name=name, entrypoint=entry, gpu_num=1, cpu=13.0, memory=234.0, deadline=7200,
        description="E5-R1 v6rd extra reps 20-30 B=400",
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
    Path(f"{RUN}/attempts.json").write_text(json.dumps([rec], indent=2), encoding="utf-8")
    print(json.dumps({"ok": not bool(code), **{k: rec[k] for k in ("task_id", "name", "status")}}, ensure_ascii=False), flush=True)
    if code:
        raise SystemExit(f"{code} {message}")


if __name__ == "__main__":
    main()
