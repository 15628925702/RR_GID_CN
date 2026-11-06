#!/usr/bin/env python3
"""Submit Gas VAEAC opt retraining as one CustomTask. No personal names."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260914_rr_gid_vaeac_opt_v1"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402


def main() -> None:
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[key] = ""
    Path(RUN).mkdir(parents=True, exist_ok=True)
    out = f"{ROOT}/results/opt/vaeac_gas_opt_v1"
    name = "rr-gid-vaeac-gas-opt-v1"
    entry = (
        f"bash -lc 'mkdir -p {out}; "
        f"http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= "
        f"bash {ROOT}/scripts/run_cloud_vaeac_opt.sh "
        f"> {out}/cmd.log 2>&1'"
    )
    body = payload(
        name=name,
        entrypoint=entry,
        gpu_num=1,
        cpu=13.0,
        memory=234.0,
        deadline=14400,
        description="Opt-track Gas VAEAC retrain empty-mask 40 percent, not paper freeze ckpt",
    )
    service = make_service()
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
    print(json.dumps({"ok": not bool(code), **rec}, ensure_ascii=False), flush=True)
    if code:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
