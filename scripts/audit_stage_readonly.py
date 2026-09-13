"""Read-only structural audit for a completed synthetic/Gas stage.

Same checks as ``audit_b32000.py`` but never overwrites the stage's existing
``report.json``: results are written under the requested output prefix.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--replications", type=int, default=20)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--audit-md", type=pathlib.Path, required=True)
    args = ap.parse_args()

    root = args.root
    budget = int(args.budget)
    rows = [json.loads(x) for x in (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    methods = sorted({r["method"] for r in rows})
    reps: dict = collections.defaultdict(list)
    for r in rows:
        reps[r["replication"]].append(r)

    errors = []
    if len(rows) != 4 * args.replications:
        errors.append("row_count")
    if any(r["budget"] != budget for r in rows):
        errors.append("budget")
    residual = max(abs(r["risk_ratio_raw"] - budget * r["kl_raw"] / r["c_star"]) for r in rows)
    if residual > 1e-9:
        errors.append("risk_ratio_raw")
    for key, group in reps.items():
        if len(group) != len(methods):
            errors.append(f"rep_size_{key}")
        if len({x["target_draw_sha256"] for x in group}) != 1:
            errors.append(f"pair_hash_{key}")
        if len({x["target_draw_seed"] for x in group}) != 1:
            errors.append(f"pair_seed_{key}")
    if any(r.get("score_backend") != "cached_qmc" for r in rows):
        errors.append("score_backend")
    if any(r.get("information_inner") != "cached_qmc" for r in rows):
        errors.append("information_backend")
    if any("cuda:0" not in str(r.get("runtime", {}).get("device", "")) for r in rows):
        errors.append("cuda")
    duplicate_keys = len(rows) - len({(r["replication"], r["method"]) for r in rows})
    if duplicate_keys:
        errors.append("duplicate_keys")

    ci = {}
    for m in methods:
        xs = [r["risk_ratio_raw"] for r in rows if r["method"] == m]
        mean = statistics.mean(xs)
        sd = statistics.stdev(xs)
        half = 1.96 * sd / math.sqrt(len(xs))
        ci[m] = {"mean": mean, "ci95": [mean - half, mean + half], "n": len(xs)}

    report = {
        "stage": args.stage,
        "passed": not errors,
        "row_count": len(rows),
        "methods": methods,
        "replications": len(reps),
        "budget": budget,
        "risk_ratio_residual_max": residual,
        "duplicate_keys": duplicate_keys,
        "failed_gates": errors,
        "rows_sha256": hashlib.sha256((root / "rows.jsonl").read_bytes()).hexdigest(),
        "risk_ratio_ci95": ci,
    }
    report_path = root / f"{args.out_prefix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    verdict = "PASS" if not errors else "REJECT"
    lines = [
        f"# Independent read-only re-audit: {args.stage} (B={budget})",
        "",
        "Date: 2026-09-10",
        "",
        "## Decision",
        "",
        f"**{verdict}**",
        "",
        "## Checks",
        "",
        f"- {len(rows)} rows: {len(methods)} methods x {len(reps)} paired replications.",
        f"- Budget exactly {budget} on every row.",
        f"- `risk_ratio_raw = B * kl_raw / c_star` residual max `{residual:.3e}`.",
        "- Paired target seeds and hashes identical within every replication.",
        "- Every row records `cached_qmc` score and information backends on CUDA.",
        "- No duplicate `(replication, method)` keys.",
        "",
        "## 95% CI",
        "",
    ]
    lines += [
        f"- {m}: mean={v['mean']:.6f}, CI95=[{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]"
        for m, v in ci.items()
    ]
    lines += ["", "## Failed gates", "", f"`{errors}`", ""]
    args.audit_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
