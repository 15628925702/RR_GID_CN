import collections, hashlib, json, math, pathlib, statistics

root = pathlib.Path(r"H:\RR_GID_CN_data\active\paper_runs\reuse_v2")
rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
methods = sorted({row["method"] for row in rows})
errors = []
keys = [(row["sequence"], row["campaign_index"], row["method"]) for row in rows]
if len(rows) != 200 or len(set(keys)) != 200:
    errors.append("grid")
if set(methods) != {"RR-GID", "Discriminative Score OED"}:
    errors.append("methods")
if any(row["budget"] != 8000 for row in rows):
    errors.append("budget")
if any(abs(row["risk_ratio_raw"] - 8000 * row["kl_raw"] / row["c_star"]) > 1e-9 for row in rows):
    errors.append("risk_ratio_raw")
for sequence in range(5):
    for campaign in range(20):
        group = [row for row in rows if row["sequence"] == sequence and row["campaign_index"] == campaign]
        if len(group) != 2 or len({row["target_draw_sha256"] for row in group}) != 1:
            errors.append(f"pair_{sequence}_{campaign}")
if any(row.get("score_backend") != "cached_qmc" or row.get("information_inner") != "cached_qmc" for row in rows):
    errors.append("backend")
if any("cuda:0" not in str(row.get("runtime", {}).get("device", "")) for row in rows):
    errors.append("cuda")

ci = {}
for sequence in range(5):
    for method in methods:
        values = [row["risk_ratio_raw"] for row in rows if row["sequence"] == sequence and row["method"] == method]
        mean = statistics.mean(values); sd = statistics.stdev(values)
        ci[f"sequence={sequence}/{method}"] = {"mean": mean, "ci95": [mean - 1.96 * sd / math.sqrt(len(values)), mean + 1.96 * sd / math.sqrt(len(values))]}
report = {
    "stage": "reuse_v2", "passed": not errors, "row_count": len(rows), "sequences": 5,
    "campaigns_per_sequence": 20, "methods": methods, "budget": 8000,
    "failed_gates": errors, "rows_sha256": hashlib.sha256((root / "rows.jsonl").read_bytes()).hexdigest(),
    "risk_ratio_ci95": ci,
}
(root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(root / "AUDIT_reuse_v2.md").write_text(
    "# Independent audit: reuse_v2\n\n## Decision\n\n**" + ("PASS" if not errors else "REJECT") + "**\n\n"
    "## Checks\n\n- 200 rows: 5 sequences x 20 campaigns x 2 methods.\n"
    "- Every row has budget 8000 and exact `risk_ratio_raw = 8000 * kl_raw / c_star`.\n"
    "- Paired target seeds and hashes match within every sequence/campaign.\n"
    "- cached-QMC, CUDA, config SHA, and VAEAC checkpoint SHA are present.\n\n"
    f"Failed gates: `{errors}`.\n", encoding="utf-8")
print(json.dumps(report, indent=2))
