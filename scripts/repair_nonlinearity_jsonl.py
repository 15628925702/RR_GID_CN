import json
import pathlib

root = pathlib.Path(r"H:\RR_GID_CN_data\active\paper_runs\nonlinearity_v1")
source = (root / "rows.jsonl").read_text(encoding="utf-8")
parts = [part for part in source.split("\\n") if part.strip()]
rows = [json.loads(part) for part in parts]
seen = set()
canonical = []
duplicates = 0
for row in rows:
    key = (float(row["alpha"]), int(row["budget"]), int(row["replication"]), row["method"])
    if key in seen:
        duplicates += 1
        continue
    seen.add(key)
    canonical.append(row)
(root / "rows_repaired_from_stream.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in canonical),
    encoding="utf-8",
)
print({"stream_records": len(rows), "canonical": len(canonical), "duplicates": duplicates})
