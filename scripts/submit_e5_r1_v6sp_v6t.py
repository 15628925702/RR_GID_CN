#!/usr/bin/env python3
"""Submit two-stage I_S smoke and I_S(beta*) diagnostic 20-rep. No personal names."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
RUN = f"{WD}/runs/20260913_rr_gid_e5_r1_v6sp_v6t"
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err, extract_id, make_service, payload  # noqa: E402


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
    sh = f"{ROOT}/scripts/run_cloud_e5_r1_smoke_shard.sh"
    sp_cfg = f"{ROOT}/configs/paper/gas_semisynthetic_empirical_v6sp_cloud.yaml"
    out = f"{ROOT}/results/e5_r1_v6sp_smoke/shards/shard_b1600-s"
    submit(
        service, attempts, "rr-gid-e5r1-v6sp-s1600", out,
        f"bash -lc 'mkdir -p {out}; bash {sh} {sp_cfg} 1600 0 5 {out} > {out}/cmd.log 2>&1'",
        "E5-R1 two-stage I_S from pilot Newton B=1600 reps 0-5",
    )
    t_cfg = f"{ROOT}/configs/paper/gas_semisynthetic_empirical_v6t_20_cloud.yaml"
    for budget, start, end, tag in (
        (1600, 0, 10, "a"),
        (1600, 10, 20, "b"),
        (3200, 0, 10, "a"),
        (3200, 10, 20, "b"),
    ):
        shard = f"b{budget}-{tag}"
        out = f"{ROOT}/results/e5_r1_v6t/shards/shard_{shard}"
        submit(
            service, attempts, f"rr-gid-e5r1-v6t-b{budget}-{tag}", out,
            f"bash -lc 'mkdir -p {out}; bash {sh} {t_cfg} {budget} {start} {end} {out} > {out}/cmd.log 2>&1'",
            f"E5-R1 diagnostic I_S(beta-star) B={budget} reps {start}-{end}",
        )
    print(json.dumps({"ok": True, "n_tasks": len(attempts)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
