"""Merge local E1 rows with cloud shard jsonl. Dedup by (method, budget, replication)."""

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


def key(row: dict) -> tuple[str, int, int]:
    return (str(row.get("method")), int(row.get("budget", -1)), int(row.get("replication", -1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-rows", type=Path, required=True)
    ap.add_argument("--shards-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_key: dict[tuple[str, int, int], dict] = {}
    sources: dict[tuple[str, int, int], str] = {}
    for row in load_jsonl(args.local_rows):
        k = key(row)
        by_key[k] = row
        sources[k] = "local"
    shard_files = sorted(args.shards_dir.glob("**/rows.jsonl"))
    for path in shard_files:
        if "smoke" in path.as_posix():
            continue
        for row in load_jsonl(path):
            k = key(row)
            by_key[k] = row
            sources[k] = path.as_posix()
    rows = [by_key[k] for k in sorted(by_key, key=lambda x: (x[1], x[2], x[0]))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts: dict[int, int] = {}
    for method, budget, _rep in by_key:
        counts[budget] = counts.get(budget, 0) + 1
    print(json.dumps({
        "out": str(args.out),
        "n_rows": len(rows),
        "n_shards": len(shard_files),
        "per_budget": counts,
        "n_local_kept": sum(1 for k, src in sources.items() if src == "local"),
        "n_cloud": sum(1 for k, src in sources.items() if src != "local"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
