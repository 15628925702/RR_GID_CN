"""Merge E5-R2 v5 shards into a single rows.jsonl. Does not touch v3."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARDS = ROOT / "tmp" / "e5_r2_v5_pull" / "extract"
OUT = ROOT / "results" / "e5_r2_v5"


def load_rows(root: Path) -> list[dict]:
    rows = []
    for p in sorted(root.rglob("rows.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(row: dict) -> tuple:
    return (
        row.get("campaign"),
        row.get("method") or row.get("policy"),
        int(row.get("budget", -1)),
        int(row.get("replication", -1)),
    )


def main() -> None:
    src = DEFAULT_SHARDS
    rows = load_rows(src)
    uniq = {}
    dups = 0
    for row in rows:
        k = key(row)
        if k in uniq:
            dups += 1
            continue
        uniq[k] = row
    merged = [uniq[k] for k in sorted(uniq)]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "rows.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for row in merged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for row in merged:
        counts[(row.get("campaign"), int(row.get("budget", -1)))] += 1
    print("src_rows", len(rows), "merged", len(merged), "dups", dups)
    print("wrote", dest)
    for camp, B in sorted(counts):
        print(camp, B, counts[(camp, B)])


if __name__ == "__main__":
    main()
