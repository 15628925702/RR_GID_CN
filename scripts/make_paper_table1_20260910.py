"""Table 1: maximum-budget summary from the audited staging tree.

Per the 2026-09-02 replan, Table 1 reports, at the largest budget only:

  * primary loss;
  * improvement relative to Uniform SQD;
  * design ratio;
  * method-specific compute;
  * 95% paired bootstrap CI on the paired difference vs Uniform SQD.

The bootstrap unit is the replication, and every method in a replication shares
the same target draw, so the resampling is paired.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

METHODS = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")
BASELINE = "Uniform SQD"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 20000, seed: int = 20260910) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(rows: list[dict], value_key: str, time_key: str, budget: int) -> list[dict]:
    subset = [r for r in rows if int(r["budget"]) == budget]
    by_method: dict[str, dict[int, dict]] = {}
    for row in subset:
        method = row.get("method") or row.get("policy")
        by_method.setdefault(method, {})[int(row["replication"])] = row
    if BASELINE not in by_method:
        raise KeyError(f"baseline {BASELINE!r} missing at budget {budget}")

    base = by_method[BASELINE]
    out = []
    for method in METHODS:
        if method not in by_method:
            continue
        shared = sorted(set(by_method[method]) & set(base))
        vals = np.asarray([float(by_method[method][r][value_key]) for r in shared])
        base_vals = np.asarray([float(base[r][value_key]) for r in shared])
        diffs = base_vals - vals
        lo, hi = bootstrap_ci(diffs)
        times = np.asarray([float(by_method[method][r].get(time_key) or 0.0) for r in shared])
        design = np.asarray([
            float(by_method[method][r]["design_ratio_main"])
            for r in shared
            if by_method[method][r].get("design_ratio_main") is not None
        ])
        out.append({
            "method": method,
            "budget": int(budget),
            "n_paired_replications": len(shared),
            "primary_loss_mean": float(vals.mean()),
            "primary_loss_se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "improvement_vs_uniform_abs": float(diffs.mean()),
            "improvement_vs_uniform_pct": float(100.0 * diffs.mean() / base_vals.mean()),
            "paired_diff_ci95_low": lo,
            "paired_diff_ci95_high": hi,
            "design_ratio_mean": float(design.mean()) if design.size else None,
            "method_seconds_median": float(np.median(times)) if times.size else None,
            "method_seconds_p90": float(np.percentile(times, 90)) if times.size else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", type=Path, default=Path("results/paper_audited_20260910"))
    ap.add_argument("--out", type=Path, default=Path("paper_tables"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    syn = load_jsonl(args.staged / "synthetic_main" / "rows.jsonl")
    gas = load_jsonl(args.staged / "gas_semisynthetic" / "rows.jsonl")
    tables = {
        "Table 1a synthetic max budget (B=32000)": summarise(
            syn, "risk_ratio_raw", "wall_seconds_total", 32000),
        "Table 1b gas semi-synthetic max budget (B=3200)": summarise(
            gas, "primary_loss", "wall_seconds_total", 3200),
    }

    for name, rows in tables.items():
        slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace(",", "")
        with (args.out / f"{slug}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        lines = ["\\begin{table}[t]", "\\centering", "\\small",
                 f"\\caption{{{name}.}}", f"\\label{{tab:{slug}}}",
                 "\\begin{tabular}{lrrrrrr}", "\\toprule",
                 "method & loss & $\\Delta$ vs Uniform & CI95 low & CI95 high & $R_{design}$ & median s \\\\",
                 "\\midrule"]
        for row in rows:
            lines.append(
                f"{row['method']} & {row['primary_loss_mean']:.4f} & "
                f"{row['improvement_vs_uniform_pct']:.1f}\\% & "
                f"{row['paired_diff_ci95_low']:.4f} & {row['paired_diff_ci95_high']:.4f} & "
                f"{row['design_ratio_mean']:.3f} & {row['method_seconds_median']:.1f} \\\\"
            )
        lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
        (args.out / f"{slug}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out / (slug + '.csv')}")

    (args.out / "table1.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    print(json.dumps(tables, indent=2))


if __name__ == "__main__":
    main()
