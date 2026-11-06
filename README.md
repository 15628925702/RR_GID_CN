# Generator-Aware Measurement Design (GAMD)

Code and audited results for the ICLR 2027 submission 10759, *Generator-Aware Measurement Design for Budgeted Distribution Recovery*.

In code and in the result files the method is called `RR-GID`. It is the method called GAMD in the paper.

## Paper item → result files

All numbers in the paper are recomputed from the files below by `results/paper_10759/check_pdf_numbers.py` (60/60 match).

| Paper | Files under `results/paper_10759/` | Rows |
|---|---|---:|
| Fig. 1(a), budget sweep, 20 paired reps | `fig1a_e1_budget_sweep/rows.jsonl` | 400 |
| Fig. 1(b), nonlinearity α ∈ {0, 0.5, 1.5}, 15 paired reps | `fig1b_e2_nonlinearity/rows.jsonl` | 240 |
| Fig. 1(b), α = 1 point | reuses Fig. 1(a) at B = 8000 (20 reps) | — |
| Table 1, Gas controlled shift, 30 paired reps | `table1_gas_controlled_shift/rows.jsonl` | 480 |
| Fig. 2, Gas natural drift, batch 7 and batches 8–9, 20 paired reps | `fig2_gas_natural_drift/rows.jsonl` | 720 |
| Table 2, numerical validation | `table2_numerical_validation/risk1_order14_10rep_report.json` | — |
| Gas frozen VAEAC (128 → 512, latent 64) | `gas_generator/vaeac_gas_opt_v6mix.pt` | — |

`fig2_gas_natural_drift/rows.jsonl` also contains a batch-10 campaign. The paper reports only batch 7 and batches 8–9. `fig1b_e2_nonlinearity/rows.jsonl` also contains an α = 1 condition with 15 reps; the paper plots the α = 1 point from Fig. 1(a) instead.

SHA256 of every file is in `results/paper_10759/MANIFEST.sha256`.

## Provenance

Every row records `code_commit = 82034c5` with `dirty_worktree = true`. The runs used the code in this commit, which is 82034c5 plus those local changes. `results/paper_10759/run_configs/` holds the exact YAML files that were run (their SHA256 equals the `config_sha256` stored in the rows). The copies in `configs/opt/` have the same contents with different line endings.

| Paper item | Run config |
|---|---|
| Fig. 1(a) | `run_configs/e1_j4_20_ftilde.yaml` |
| Fig. 1(b) | `run_configs/e2_ftilde_15.yaml` |
| Table 1, reps 0–19 | `run_configs/gas_r1_v6mix_20.yaml` |
| Table 1, reps 20–29 | `run_configs/gas_r1_v6mix_30.yaml` |
| Fig. 2 | `run_configs/gas_r2_v6mix_calahat_20.yaml` |

GAMD's design Fisher is evaluated at the initialization estimate β̃ (`rr_design_fisher: plugin_tilde`), never at the true β*.

Conditional scores use validated fixed QMC of order 14 (`score_backend: fixed_qmc` in every synthetic row). `table2_numerical_validation/fixed_cached_equivalence_order14.json` certifies it agrees with the cached order-14 backend to 6.0 × 10⁻⁹.

## Environment

Python 3.10, PyTorch 2.6.0 + CUDA 12.4, NumPy 2.2, SciPy 1.15. Results were produced on an RTX 4050 laptop GPU.

```text
pip install -e .
```

The Gas data are the UCI Gas Sensor Array Drift dataset (`data/gas/batch*.dat`). The processed file used by every Gas run is included as `data/gas/processed/gas_processed.npz` (SHA256 `e87cddfa88ac690de88281747cae1e5198e110ad19c51c69974513328d4d8ffa`); `scripts/p8_preprocess.py` rebuilds it.

## Reproduce

Paths inside the run configs point to the original workstation. Edit `seed_manifest`, `generator_checkpoint`, `fixed_qmc_equivalence_certificate` and `output_path` before running.

```text
python scripts/run_synthetic_main.py   --config results/paper_10759/run_configs/e1_j4_20_ftilde.yaml --resume
python scripts/run_nonlinearity.py     --config results/paper_10759/run_configs/e2_ftilde_15.yaml --resume
python scripts/run_gas_semisynthetic.py --config results/paper_10759/run_configs/gas_r1_v6mix_30.yaml --data data/gas/processed/gas_processed.npz --resume
python scripts/run_gas_natural.py      --config results/paper_10759/run_configs/gas_r2_v6mix_calahat_20.yaml --data data/gas/processed/gas_processed.npz --resume
python results/paper_10759/check_pdf_numbers.py
```

## Layout

| Path | Contents |
|---|---|
| `src/rr_gid_cn/` | GAMD, baselines (Uniform SQD, A-OSQD, Discriminative Score OED), QMC backends, VAEAC |
| `scripts/` | Experiment runners and Gas generator training |
| `configs/` | Experiment configs; paper runs are in `configs/opt/` |
| `experiments/paper/` | Synthetic oracle artifacts used by Fig. 1 |
| `results/paper_10759/` | Paper results, run configs, numerical validation, Gas generator |
| `paper_results/` | Earlier 2026-09-13 freeze, superseded by `results/paper_10759/` |
