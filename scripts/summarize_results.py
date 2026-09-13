"""Aggregate paper jsonl into Table 1 and a machine-readable summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")
ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None, "halfwidth": None}
    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = values[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "halfwidth": float(0.5 * (hi - lo)),
    }


def paired_delta_ci(method_vals: np.ndarray, uniform_vals: np.ndarray, n_boot: int = 2000, seed: int = 1):
    method_vals = np.asarray(method_vals, dtype=float)
    uniform_vals = np.asarray(uniform_vals, dtype=float)
    if method_vals.size == 0 or method_vals.size != uniform_vals.size:
        return {"mean": None, "lo": None, "hi": None, "halfwidth": None, "rel_vs_uniform": None}
    diffs = method_vals - uniform_vals
    stats = bootstrap_mean_ci(diffs, n_boot=n_boot, seed=seed)
    u = float(uniform_vals.mean())
    stats["rel_vs_uniform"] = None if abs(u) < 1e-15 else float((u - float(method_vals.mean())) / u)
    return stats


def _index_by_rep(rows: list[dict], value_key: str) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        method = row.get("method") or row.get("policy")
        key = int(row.get("replication", row.get("campaign_index", 0)))
        if "campaign" in row:
            tag = {"batch7": 1, "batches8_9": 2, "batch10": 3}.get(row["campaign"], 0)
            key = tag * 1000 + int(row.get("replication", 0))
        out[method][key] = float(row[value_key])
    return out


def aligned(index: dict[str, dict[int, float]], method: str, reference: str = "Uniform SQD"):
    keys = sorted(set(index.get(reference, {})) & set(index.get(method, {})))
    return np.array([index[method][k] for k in keys], dtype=float), np.array(
        [index[reference][k] for k in keys], dtype=float
    )


def method_specific_seconds(row: dict) -> float | None:
    name = row.get("method") or row.get("policy")
    if "method_specific_seconds" in row:
        return float(row["method_specific_seconds"])
    runtime = row.get("runtime") or {}
    if name == "RR-GID":
        return float(runtime.get("time_info_basis", 0.0)) + float(runtime.get("time_fw", 0.0)) + float(
            row.get("generator_training_seconds", 0.0)
        )
    if name == "Discriminative Score OED":
        return float(runtime.get("time_disc", 0.0))
    if name == "A-OSQD":
        return float(runtime.get("time_fw", 0.0))
    if runtime:
        return 0.0
    return None


def summarize_cell(rows: list[dict], value_key: str) -> dict:
    index = _index_by_rep(rows, value_key)
    out = {}
    for method in METHODS:
        if method not in index:
            continue
        vals = np.array(list(index[method].values()), dtype=float)
        stats = bootstrap_mean_ci(vals, seed=1000 + METHODS.index(method))
        if "Uniform SQD" in index:
            m_al, u_al = aligned(index, method)
            paired = paired_delta_ci(m_al, u_al)
        else:
            paired = {"mean": None, "lo": None, "hi": None, "halfwidth": None, "rel_vs_uniform": None}
        design = [float(r["design_ratio_main"]) for r in rows if (r.get("method") or r.get("policy")) == method]
        compute = [s for r in rows if (r.get("method") or r.get("policy")) == method for s in [method_specific_seconds(r)] if s is not None]
        out[method] = {
            "primary": stats,
            "paired_delta_vs_uniform": paired,
            "design_ratio_mean": float(np.mean(design)) if design else None,
            "method_specific_seconds_mean": float(np.mean(compute)) if compute else None,
        }
    return out


def max_budget_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    bmax = max(int(r["budget"]) for r in rows)
    return [r for r in rows if int(r["budget"]) == bmax]


def curve(rows: list[dict], value_key: str, extra: str | None = None) -> list[dict]:
    groups: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        method = row.get("method") or row.get("policy")
        key = (method, int(row["budget"]))
        if extra is not None:
            key = (method, int(row["budget"]), row[extra])
        groups[key].append(float(row[value_key]))
    out = []
    for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
        arr = np.asarray(vals, dtype=float)
        rec = {
            "method": key[0],
            "budget": key[1],
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        }
        if extra is not None:
            rec[extra] = key[2]
        out.append(rec)
    return out


def reuse_frontier(rows: list[dict], prefixes=(1, 5, 10, 20)) -> list[dict]:
    out = []
    for t in prefixes:
        for method in ("RR-GID", "Discriminative Score OED"):
            snapshot = [r for r in rows if r.get("method") == method and int(r.get("campaigns_so_far", -1)) == int(t)]
            history = [r for r in rows if r.get("method") == method and int(r.get("campaigns_so_far", 10**9)) <= int(t)]
            if not snapshot:
                continue
            risks = defaultdict(list)
            seconds = {}
            for row in history:
                risks[int(row["sequence"])].append(float(row["risk_ratio"]))
            for row in snapshot:
                seconds[int(row["sequence"])] = float(row["cumulative_method_seconds"])
            seqs = sorted(set(risks) & set(seconds))
            mean_risk = float(np.mean([np.mean(risks[s]) for s in seqs]))
            mean_sec = float(np.mean([seconds[s] for s in seqs]))
            out.append({
                "campaigns": int(t),
                "method": method,
                "mean_risk_ratio": mean_risk,
                "mean_cumulative_seconds": mean_sec,
                "n_sequences": int(len(seqs)),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/paper")
    ap.add_argument("--results", type=Path, default=ROOT / "results/paper")
    args = ap.parse_args()
    results = args.results
    syn = load_jsonl(results / "synthetic_main" / "rows.jsonl")
    nonlin = load_jsonl(results / "nonlinearity" / "rows.jsonl")
    reuse = load_jsonl(results / "reuse" / "rows.jsonl")
    r1 = load_jsonl(results / "gas_semisynthetic" / "rows.jsonl")
    r2 = load_jsonl(results / "gas_natural" / "rows.jsonl")
    secondary_path = results / "gas_natural" / "secondary_max_budget.json"
    secondary = json.loads(secondary_path.read_text(encoding="utf-8")) if secondary_path.exists() else None
    calib_path = results / "oracle_calibration" / "report.json"
    calibration = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else None

    table = {
        "synthetic_main": summarize_cell(max_budget_rows(syn), "risk_ratio"),
        "gas_semisynthetic": summarize_cell(max_budget_rows(r1), "primary_loss"),
        "gas_natural": {},
    }
    for camp in ("batch7", "batches8_9", "batch10"):
        camp_rows = [r for r in max_budget_rows(r2) if r.get("campaign") == camp]
        table["gas_natural"][camp] = summarize_cell(camp_rows, "primary_loss")

    summary = {
        "synthetic_main_curve": curve(syn, "risk_ratio"),
        "nonlinearity_curve": curve(nonlin, "risk_ratio", extra="alpha"),
        "nonlinearity_design": curve(nonlin, "design_ratio_main", extra="alpha"),
        "reuse_frontier": reuse_frontier(reuse),
        "gas_semisynthetic_curve": curve(r1, "primary_loss"),
        "gas_natural_curve": curve(r2, "primary_loss", extra="campaign"),
        "table1_max_budget": table,
        "secondary_max_budget": secondary,
        "oracle_calibration": calibration,
        "counts": {
            "synthetic_main": len(syn),
            "nonlinearity": len(nonlin),
            "reuse": len(reuse),
            "gas_semisynthetic": len(r1),
            "gas_natural": len(r2),
        },
    }
    out_json = results / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = results / "table1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "block", "campaign", "method", "n", "primary_mean", "ci95_lo", "ci95_hi",
            "halfwidth", "rel_vs_uniform", "design_ratio", "method_seconds",
        ])
        def emit(block, campaign, payload):
            for method, cell in payload.items():
                p = cell["primary"]
                d = cell["paired_delta_vs_uniform"]
                writer.writerow([
                    block, campaign, method, p["n"], p["mean"], p["lo"], p["hi"], p["halfwidth"],
                    d.get("rel_vs_uniform"), cell["design_ratio_mean"], cell["method_specific_seconds_mean"],
                ])
        emit("synthetic_main", "", table["synthetic_main"])
        emit("gas_semisynthetic", "", table["gas_semisynthetic"])
        for camp, payload in table["gas_natural"].items():
            emit("gas_natural", camp, payload)

    print(f"wrote {out_json}")
    print(f"wrote {csv_path}")
    for block, payload in (("E1 B=32000", table["synthetic_main"]), ("E5 R1 B=3200", table["gas_semisynthetic"])):
        print(block)
        for method, cell in payload.items():
            p = cell["primary"]
            rel = cell["paired_delta_vs_uniform"].get("rel_vs_uniform")
            rel_s = "--" if rel is None else f"{100 * rel:+.1f}%"
            print(f"  {method:28s} mean={p['mean']:.4g}  n={p['n']}  |half|/mean={abs(p['halfwidth'])/max(abs(p['mean']),1e-12):.3f}  vsU={rel_s}")
    if calibration:
        query = calibration.get("query") or {}
        info = calibration.get("information") or {}
        e2e = calibration.get("end_to_end") or {}
        print("E4 calibration")
        print(f"  query_ok={query.get('query_ok')} frozen_order={query.get('frozen_order')}")
        for item in query.get("orders") or []:
            print(
                f"    order {item['order']}: median_rel={item.get('median_relative_error'):.4g} "
                f"p95={item.get('p95_relative_error'):.4g} max_abs={item.get('max_abs_coordinate_error'):.4g} "
                f"ok={item.get('ok')}"
            )
        print(
            f"  info_ok={info.get('information_ok')} mean_op={info.get('mean_op')} "
            f"max_op={info.get('max_op')} same_outer={info.get('same_outer')}"
        )
        print(
            f"  e2e_ok={e2e.get('ok')} n={e2e.get('n')} "
            f"mean_abs_delta={e2e.get('mean_abs_delta')} se={e2e.get('se')}"
        )
    if secondary:
        print("E5 R2 secondary (max B, 1 rep/campaign)")
        for row in secondary.get("rows", []):
            print(
                f"  {row['campaign']:12s} {row['method']:28s} "
                f"loss={row['projection_loss']:.3f} rmse={row['heldout_mean_rmse']:.3f} "
                f"c2st={row['c2st_auc']:.3f} ess={row['generator_ess']:.3f}"
            )


if __name__ == "__main__":
    main()
