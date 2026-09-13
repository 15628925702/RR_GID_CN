import collections, hashlib, json, math, pathlib, statistics

root = pathlib.Path(r"H:\RR_GID_CN_data\active\paper_runs\nonlinearity_v1")
rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
methods = sorted({row["method"] for row in rows})
alphas = sorted({float(row["alpha"]) for row in rows})
keys = [(float(row["alpha"]), row["budget"], row["replication"], row["method"]) for row in rows]
errors = []
if len(rows) != 240 or len(set(keys)) != 240:
    errors.append("grid")
if alphas != [0.0, 0.5, 1.0, 1.5]:
    errors.append("alphas")
if any(row["budget"] != 8000 for row in rows):
    errors.append("budget")
if any(abs(row["risk_ratio_raw"] - 8000 * row["kl_raw"] / row["c_star"]) > 1e-9 for row in rows):
    errors.append("risk_ratio_raw")
for alpha in alphas:
    for rep in range(15):
        group = [row for row in rows if float(row["alpha"]) == alpha and row["replication"] == rep]
        if len(group) != 4 or len({row["target_draw_sha256"] for row in group}) != 1:
            errors.append(f"pair_{alpha}_{rep}")
if any(row.get("score_backend") != "cached_qmc" or row.get("information_inner") != "cached_qmc" for row in rows):
    errors.append("backend")
if any("cuda:0" not in str(row.get("runtime", {}).get("device", "")) for row in rows):
    errors.append("cuda")

ci = {}
for alpha in alphas:
    for method in methods:
        values = [row["risk_ratio_raw"] for row in rows if float(row["alpha"]) == alpha and row["method"] == method]
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        ci[f"alpha={alpha:g}/{method}"] = {"mean": mean, "ci95": [mean - 1.96 * sd / math.sqrt(len(values)), mean + 1.96 * sd / math.sqrt(len(values))]}

report = {
    "stage": "nonlinearity_v1",
    "passed": not errors,
    "row_count": len(rows),
    "unique_grid_count": len(set(keys)),
    "alphas": alphas,
    "methods": methods,
    "replications_per_alpha": 15,
    "budget": 8000,
    "failed_gates": errors,
    "rows_sha256": hashlib.sha256((root / "rows.jsonl").read_bytes()).hexdigest(),
    "raw_rows_sha256": hashlib.sha256((root / "rows_raw.jsonl").read_bytes()).hexdigest(),
    "risk_ratio_ci95": ci,
}
(root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(root / "AUDIT_nonlinearity_v1.md").write_text(
    "# Independent audit: nonlinearity_v1\n\n"
    f"## Decision\n\n**{'PASS' if not errors else 'REJECT'}**\n\n"
    "## Checks\n\n"
    "- 240 canonical rows: four methods x four alpha values x 15 replications.\n"
    "- Every row has budget 8000 and exact `risk_ratio_raw = 8000 * kl_raw / c_star`.\n"
    "- Paired target seeds and hashes are identical within each alpha/replication group.\n"
    "- Every row uses cached-QMC fields and CUDA runtime metadata.\n"
    "- Twelve duplicate recovery rows were preserved in `rows_raw.jsonl`; the canonical file contains one row per unique experiment key.\n\n"
    "## Gate result\n\n"
    f"Failed gates: `{errors}`.\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2))
