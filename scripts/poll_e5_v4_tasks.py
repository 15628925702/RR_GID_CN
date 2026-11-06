#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

WD = "/kairos_vepfs_volc/autodrive/manlichen/intern/ivan"
ROOT = f"{WD}/rr_gid_cloud_20260912"
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ[key] = ""
sys.path.insert(0, WD)
sys.path.insert(0, os.path.join(WD, ".sdk_cache", "volcsdk"))
sys.path.insert(0, f"{ROOT}/scripts")
from submit_custom_task import call, err  # noqa: E402
from vemlp_queue_query import make_service  # noqa: E402

path = Path(sys.argv[1] if len(sys.argv) > 1 else f"{WD}/runs/20260913_rr_gid_e5_v4_e3/attempts.json")
rows = json.loads(path.read_text(encoding="utf-8"))
svc = make_service()
counts: Counter[str] = Counter()
for rec in rows:
    tid = rec.get("task_id")
    if not tid or rec.get("status") == "Failed":
        state = rec.get("error_code") or "submit_failed"
        counts[str(state)] += 1
        print(rec.get("name"), tid, state, flush=True)
        continue
    obj = call(svc, "GetCustomTask", "2021-10-01", {"Id": tid})
    code, msg = err(obj)
    res = obj.get("Result") or {}
    state = res.get("State") or res.get("Status") or res.get("state") or code
    counts[str(state)] += 1
    print(rec.get("name"), tid, state, (msg or "")[:120], flush=True)
print("COUNTS", dict(counts), flush=True)
