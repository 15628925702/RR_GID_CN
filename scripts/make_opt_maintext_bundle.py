"""Build opt-track main-text tables, figures, and SHA summaries.

Does not write into the frozen paper-results canonical paths.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "results" / "opt"
FIG = ROOT / "figures" / "opt"
HOPT = Path(r"H:/RR_GID_CN_data/active/opt_20260914")
METHODS = ("Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID")
COLORS = {
    "Uniform SQD": "#7A7A7A",
    "A-OSQD": "#D62728",
    "Discriminative Score OED": "#1F77B4",
    "RR-GID": "#2CA02C",
}
MARKERS = {
    "Uniform SQD": "o",
    "A-OSQD": "s",
    "Discriminative Score OED": "^",
    "RR-GID": "D",
}
PATHS = {
    "e1": OPT / "e1_j4_20" / "rows.jsonl",
    "r1": OPT / "e5_r1_v6mix_20" / "rows.jsonl",
    "r2": OPT / "e5_r2_v6mix_calahat_20" / "rows.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_map(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple, dict]:
    out = {}
    for row in rows:
        key = tuple(
            int(row[k]) if k in {"budget", "replication"} else (row.get("method") or row.get("policy") if k == "method" else row.get(k))
            for k in keys
        )
        out[key] = row
    return out


def se(xs: list[float]) -> float:
    return statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def mean_se(rows: list[dict], metric: str, method: str, extra: dict | None = None) -> dict | None:
    extra = extra or {}
    vals = []
    for row in rows:
        if (row.get("method") or row.get("policy")) != method:
            continue
        if any(row.get(k) != v for k, v in extra.items()):
            continue
        vals.append(float(row[metric]))
    if not vals:
        return None
    mu = float(statistics.mean(vals))
    s = se(vals)
    return {"n": len(vals), "mean": mu, "se": s, "ci_halfwidth": 1.96 * s}


def paired_diff(rows: list[dict], metric: str, group: tuple[str, ...], a="RR-GID", b="Uniform SQD") -> dict:
    buckets: dict[tuple, dict] = defaultdict(dict)
    for row in rows:
        key = tuple(int(row[k]) if k in {"budget", "replication"} else row.get(k) for k in group)
        buckets[key][row.get("method") or row.get("policy")] = float(row[metric])
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for key, mets in buckets.items():
        if a in mets and b in mets:
            grouped[key[:-1]].append(mets[a] - mets[b])
    out = {}
    for key, xs in grouped.items():
        mu = float(statistics.mean(xs))
        s = se(xs)
        hw = 1.96 * s
        out[str(key)] = {
            "n": len(xs),
            "mean_diff": mu,
            "ci_low": mu - hw,
            "ci_high": mu + hw,
            "excludes_zero": abs(mu) > hw,
        }
    return out


def table_rows(rows: list[dict], metric: str, extra_keys: tuple[str, ...] = ()) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        method = row.get("method") or row.get("policy")
        key = (method, int(row["budget"])) + tuple(row[k] for k in extra_keys)
        groups[key].append(row)
    out = []
    for key, chunk in sorted(groups.items(), key=lambda kv: (str(kv[0][2:]) if extra_keys else 0, kv[0][1], METHODS.index(kv[0][0]) if kv[0][0] in METHODS else 99)):
        method, budget = key[0], key[1]
        extras = dict(zip(extra_keys, key[2:]))
        stats = mean_se(chunk, metric, method)
        if stats is None:
            continue
        rec = {"method": method, "budget": budget, **extras, **stats}
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_ax(ax):
    ax.grid(True, which="both", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def errorbar(ax, xs, ys, ses, method):
    ax.errorbar(
        xs, ys, yerr=ses, marker=MARKERS[method], color=COLORS[method],
        label=method, capsize=3, linewidth=1.7, markersize=6,
    )


def fig1(syn: list[dict], out: Path) -> None:
    if not syn:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in METHODS:
        stats = {}
        for budget in sorted({int(r["budget"]) for r in syn}):
            rec = mean_se(syn, "risk_ratio", method, {"budget": budget})
            if rec:
                stats[budget] = rec
        if not stats:
            continue
        xs = list(stats)
        errorbar(ax, xs, [stats[b]["mean"] for b in xs], [stats[b]["se"] for b in xs], method)
    ax.axhline(1.0, ls="--", color="#444444", lw=1.0, label=r"$R_{\mathrm{risk}}=1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"target budget $B$")
    ax.set_ylabel(r"$R_{\mathrm{risk}}=B\cdot\mathrm{KL}/C^\star$")
    ax.set_title("Figure 1: synthetic main (J=4, 20-rep opt)")
    style_ax(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig3(r1: list[dict], r2: list[dict], out: Path) -> None:
    if not r1 or not r2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharey=False)
    panels = [
        (axes[0, 0], r1, None, "R1 semi-synthetic"),
        (axes[0, 1], r2, "batch7", "R2: batch 7"),
        (axes[1, 0], r2, "batches8_9", "R2: batches 8–9"),
        (axes[1, 1], r2, "batch10", "R2: batch 10"),
    ]
    for ax, rows, camp, title in panels:
        subset = rows if camp is None else [r for r in rows if r.get("campaign") == camp]
        for method in METHODS:
            stats = {}
            for budget in sorted({int(r["budget"]) for r in subset}):
                rec = mean_se(subset, "primary_loss", method, {"budget": budget})
                if rec:
                    stats[budget] = rec
            if not stats:
                continue
            xs = list(stats)
            errorbar(ax, xs, [stats[b]["mean"] for b in xs], [stats[b]["se"] for b in xs], method)
        ax.set_xscale("log")
        ax.set_xlabel(r"budget $B$")
        ax.set_ylabel(r"family-projection loss $D_{\widehat A}$")
        ax.set_title(title)
        style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def e1_status(rows: list[dict]) -> dict:
    expected = 5 * 20 * 4
    uniq = unique_map(rows, ("method", "budget", "replication"))
    rr32000 = mean_se(rows, "risk_ratio", "RR-GID", {"budget": 32000})
    uni32000 = mean_se(rows, "risk_ratio", "Uniform SQD", {"budget": 32000})
    return {
        "n_unique": len(uniq),
        "expected": expected,
        "complete": len(uniq) >= expected,
        "rr_B32000": rr32000,
        "uniform_B32000": uni32000,
    }


def archive(src: Path, dest_dir: Path) -> None:
    if not src.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (OPT / "paper_tables").mkdir(parents=True, exist_ok=True)
    e1 = load_jsonl(PATHS["e1"])
    r1 = load_jsonl(PATHS["r1"])
    r2 = load_jsonl(PATHS["r2"])
    payload = {
        "canonical": {
            "e1": "opt_e1_j4_20",
            "e5_r1": "e5_r1_v6mix_20",
            "e5_r2": "e5_r2_v6mix_calahat_20",
            "generator": "vaeac_gas_opt_v6mix.pt",
            "generator_sha256": "5bb8d4818bfbf7e6792e9494e2de9bc06eeafa84f795a86221fad03d622583da",
            "risk2_spearman": 0.8024029446489341,
            "risk2_top10": "5/10",
            "risk2_design_ratio": 1.163301486846778,
            "c2st_auc_ref_val": 0.684724849721975,
        },
        "sha256": {name: sha256_file(path) for name, path in PATHS.items() if path.exists()},
        "e1": {
            "status": e1_status(e1),
            "by_budget": table_rows(e1, "risk_ratio"),
            "paired_rr_minus_uniform": paired_diff(e1, "risk_ratio", ("budget", "replication")) if e1 else {},
        },
        "r1": {
            "n": len(unique_map(r1, ("method", "budget", "replication"))),
            "hit_norm_cap": {
                method: sum(1 for r in r1 if (r.get("method") or r.get("policy")) == method and bool(r.get("hit_norm_cap")))
                for method in METHODS
            },
            "by_budget": table_rows(r1, "primary_loss"),
            "paired_rr_minus_uniform": paired_diff(r1, "primary_loss", ("budget", "replication")),
            "paired_rr_minus_disc": paired_diff(r1, "primary_loss", ("budget", "replication"), "RR-GID", "Discriminative Score OED"),
        },
        "r2": {
            "n": len(unique_map(r2, ("campaign", "method", "budget", "replication"))),
            "by_campaign": table_rows(r2, "primary_loss", ("campaign",)),
            "paired_rr_minus_uniform": paired_diff(r2, "primary_loss", ("campaign", "budget", "replication")),
        },
    }
    summary_path = OPT / "maintext_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(OPT / "paper_tables" / "Table_1a_opt_e1_j4.csv", payload["e1"]["by_budget"])
    write_csv(OPT / "paper_tables" / "Table_1b_opt_e5_r1_v6mix.csv", payload["r1"]["by_budget"])
    write_csv(OPT / "paper_tables" / "Table_1c_opt_e5_r2_calahat.csv", payload["r2"]["by_campaign"])
    fig1(e1, FIG / "fig1_synthetic_main_j4.png")
    fig3(r1, r2, FIG / "fig3_gas.png")
    archive(summary_path, HOPT / "maintext")
    archive(PATHS["r1"], HOPT / "e5_r1_v6mix_20")
    archive(PATHS["r2"], HOPT / "e5_r2_v6mix_calahat_20")
    if PATHS["e1"].exists():
        archive(PATHS["e1"], HOPT / "e1_j4_20")
    for fig in FIG.glob("fig*"):
        archive(fig, HOPT / "maintext" / "figures")
        archive(fig, OPT / "figures")
    print(json.dumps({
        "summary": str(summary_path),
        "e1_complete": payload["e1"]["status"]["complete"],
        "e1_unique": payload["e1"]["status"]["n_unique"],
        "r1_n": payload["r1"]["n"],
        "r2_n": payload["r2"]["n"],
        "fig3": str(FIG / "fig3_gas.png"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
