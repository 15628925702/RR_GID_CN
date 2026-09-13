"""Paper Figures 1–3 from results/paper jsonl (guide Phase H)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
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
CAMPAIGN_LABELS = {
    "batch7": "batch 7",
    "batches8_9": "batches 8–9",
    "batch10": "batch 10",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def grouped_mean_se(rows: list[dict], value_key: str, by: tuple[str, ...]):
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] if k != "method" else (row.get("method") or row.get("policy")) for k in by)
        buckets[key].append(float(row[value_key]))
    out = {}
    for key, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        out[key] = (float(arr.mean()), se, int(len(arr)))
    return out


def style_ax(ax):
    ax.grid(True, which="both", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def errorbar(ax, xs, ys, ses, method, **kwargs):
    ax.errorbar(
        xs, ys, yerr=ses, marker=MARKERS[method], color=COLORS[method],
        label=method, capsize=3, linewidth=1.7, markersize=6, **kwargs,
    )


def fig1(syn: list[dict], out: Path) -> None:
    stats = grouped_mean_se(syn, "risk_ratio", ("method", "budget"))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in METHODS:
        budgets = sorted({b for (m, b) in stats if m == method})
        xs = [b for b in budgets]
        ys = [stats[(method, b)][0] for b in budgets]
        ses = [stats[(method, b)][1] for b in budgets]
        errorbar(ax, xs, ys, ses, method)
    ax.axhline(1.0, ls="--", color="#444444", lw=1.0, label=r"$R_{\mathrm{risk}}=1$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"target budget $B$")
    ax.set_ylabel(r"$R_{\mathrm{risk}}=B\cdot\mathrm{KL}/C^\star$")
    ax.set_title("Figure 1: synthetic main")
    style_ax(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig2(nonlin: list[dict], reuse: list[dict], out: Path) -> None:
    risk = grouped_mean_se(nonlin, "risk_ratio", ("method", "alpha"))
    design = grouped_mean_se(nonlin, "design_ratio_main", ("method", "alpha"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.5))

    alphas = sorted({a for (m, a) in risk})
    for method in METHODS:
        ys = [risk[(method, a)][0] for a in alphas]
        ses = [risk[(method, a)][1] for a in alphas]
        errorbar(ax1, alphas, ys, ses, method)
    ax1.axhline(1.0, ls="--", color="#444444", lw=1.0)
    ax1.set_xlabel(r"nonlinear strength $\alpha$")
    ax1.set_ylabel(r"$R_{\mathrm{risk}}$")
    ax1.set_title("Figure 2(a): nonlinearity")
    style_ax(ax1)
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")

    inset = ax1.inset_axes([0.52, 0.52, 0.44, 0.42])
    for method in METHODS:
        ys = [design[(method, a)][0] for a in alphas]
        inset.plot(alphas, ys, marker=MARKERS[method], color=COLORS[method], linewidth=1.3, markersize=4)
    inset.axhline(1.0, ls="--", color="#888888", lw=0.8)
    inset.set_title(r"$R_{\mathrm{design}}$", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(True, alpha=0.25)

    prefixes = (1, 5, 20, 50)
    for method in ("RR-GID", "Discriminative Score OED"):
        xs, ys = [], []
        for t in prefixes:
            snapshot = [r for r in reuse if r.get("method") == method and int(r.get("campaigns_so_far", -1)) == t]
            history = [r for r in reuse if r.get("method") == method and int(r.get("campaigns_so_far", 10**9)) <= t]
            if not snapshot:
                continue
            risks = defaultdict(list)
            seconds = {}
            for row in history:
                risks[int(row["sequence"])].append(float(row["risk_ratio"]))
            for row in snapshot:
                seconds[int(row["sequence"])] = float(row["cumulative_method_seconds"])
            seqs = sorted(set(risks) & set(seconds))
            xs.append(float(np.mean([seconds[s] for s in seqs])))
            ys.append(float(np.mean([np.mean(risks[s]) for s in seqs])))
        ax2.plot(xs, ys, marker=MARKERS[method], color=COLORS[method], label=method, linewidth=1.7, markersize=6)
        for x, y, t in zip(xs, ys, prefixes):
            ax2.annotate(f"T={t}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("cumulative method-specific compute (s)")
    ax2.set_ylabel(r"mean $R_{\mathrm{risk}}$ across campaigns")
    ax2.set_title("Figure 2(b): generator reuse")
    style_ax(ax2)
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def fig3(r1: list[dict], r2: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharey=False)
    panels = [
        (axes[0, 0], r1, None, "R1 semi-synthetic"),
        (axes[0, 1], r2, "batch7", "R2: batch 7"),
        (axes[1, 0], r2, "batches8_9", "R2: batches 8–9"),
        (axes[1, 1], r2, "batch10", "R2: batch 10"),
    ]
    for ax, rows, camp, title in panels:
        subset = rows if camp is None else [r for r in rows if r.get("campaign") == camp]
        stats = grouped_mean_se(subset, "primary_loss", ("method", "budget"))
        for method in METHODS:
            budgets = sorted({b for (m, b) in stats if m == method})
            if not budgets:
                continue
            ys = [stats[(method, b)][0] for b in budgets]
            ses = [stats[(method, b)][1] for b in budgets]
            errorbar(ax, budgets, ys, ses, method)
        ax.set_xscale("log")
        ax.set_xlabel(r"budget $B$")
        ax.set_ylabel(r"family-projection loss $D_{\widehat A}$")
        ax.set_title(title)
        style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/paper")
    ap.add_argument("--results", type=Path, default=ROOT / "results/paper")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures/paper")
    args = ap.parse_args()
    results = args.results
    figures = args.figures
    figures.mkdir(parents=True, exist_ok=True)
    syn = load_jsonl(results / "synthetic_main" / "rows.jsonl")
    nonlin = load_jsonl(results / "nonlinearity" / "rows.jsonl")
    reuse = load_jsonl(results / "reuse" / "rows.jsonl")
    r1 = load_jsonl(results / "gas_semisynthetic" / "rows.jsonl")
    r2 = load_jsonl(results / "gas_natural" / "rows.jsonl")
    fig1(syn, figures / "fig1_synthetic_main.png")
    fig2(nonlin, reuse, figures / "fig2_mechanisms.png")
    fig3(r1, r2, figures / "fig3_gas.png")
    print(f"wrote {figures / 'fig1_synthetic_main.png'}")
    print(f"wrote {figures / 'fig2_mechanisms.png'}")
    print(f"wrote {figures / 'fig3_gas.png'}")


if __name__ == "__main__":
    main()
