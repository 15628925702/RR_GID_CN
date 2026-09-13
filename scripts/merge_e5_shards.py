"""Merge E5 cloud shard jsonl into a single paper rows file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def key(row: dict) -> tuple:
    return (
        str(row.get("campaign") or row.get("experiment") or ""),
        str(row.get("method") or row.get("policy") or ""),
        int(row.get("budget", -1)),
        int(row.get("replication", -1)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    by_key: dict[tuple, dict] = {}
    shard_files = sorted(args.shards_dir.glob("**/rows.jsonl"))
    for path in shard_files:
        if "smoke" in path.as_posix():
            continue
        for row in load_jsonl(path):
            by_key[key(row)] = row
    rows = [by_key[k] for k in sorted(by_key)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    budgets: dict[int, int] = {}
    methods: dict[str, int] = {}
    campaigns: dict[str, int] = {}
    for camp, method, budget, _rep in by_key:
        budgets[budget] = budgets.get(budget, 0) + 1
        methods[method] = methods.get(method, 0) + 1
        campaigns[camp] = campaigns.get(camp, 0) + 1
    print(json.dumps({
        "out": str(args.out),
        "n_rows": len(rows),
        "n_shards": len(shard_files),
        "per_budget": budgets,
        "per_method": methods,
        "per_campaign": campaigns,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
