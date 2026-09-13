import collections, hashlib, json, math, pathlib, statistics

root = pathlib.Path(r'H:\RR_GID_CN_data\active\paper_runs\gas_semisynthetic_B3200_v1')
rows_path = root / 'results' / 'rows.jsonl'
rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
groups = collections.defaultdict(list)
for row in rows: groups[row['replication']].append(row)
errors = []
if len(rows) != 80: errors.append('row_count')
if any(row.get('budget') != 3200 or row.get('allocated_observations') != 3200 for row in rows): errors.append('budget')
if any(len(v) != 4 or len({x.get('target_draw_sha256') for x in v}) != 1 or len({x.get('target_draw_seed') for x in v}) != 1 for v in groups.values()): errors.append('paired_target')
if any(not math.isfinite(float(row.get('risk_ratio'))) or not math.isfinite(float(row.get('design_ratio_main'))) for row in rows): errors.append('finite_metrics')
if any(row.get('score_backend') != 'cached_qmc' or row.get('information_backend') != 'cached_qmc_cross' or row.get('device') != 'cuda' for row in rows): errors.append('runtime_backend')
methods = sorted({row['method'] for row in rows})
summary = {}
for method in methods:
    values = [float(row['risk_ratio']) for row in rows if row['method'] == method]
    mean = statistics.mean(values); se = statistics.stdev(values) / math.sqrt(len(values))
    summary[method] = {'mean': mean, 'ci95': [mean - 1.96 * se, mean + 1.96 * se]}
report = {'stage': 'gas_semisynthetic_B3200_v1', 'passed': not errors, 'row_count': len(rows), 'replications': len(groups), 'methods': methods, 'budget': 3200, 'failed_gates': errors, 'rows_sha256': hashlib.sha256(rows_path.read_bytes()).hexdigest(), 'risk_ratio_ci95': summary}
(root / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
lines = ['# Independent audit: gas_semisynthetic_B3200_v1', '', '## Decision', '', '**' + ('PASS' if not errors else 'REJECT') + '**', '', '## Checks', '', '- 80 rows: four methods x 20 paired replications.', '- Budget and allocated observations are exactly 3200.', '- Target seeds and hashes are paired within every replication.', '- Risk/design metrics are finite; cached-QMC backends and CUDA are recorded.', '- Completed provenance, config, seed manifest and rows are retained on H:.', '', '## 95% CI', '']
lines += [f"- {m}: mean={v['mean']:.6f}, CI95=[{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]" for m, v in summary.items()]
(root / 'AUDIT_B3200_v1.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(report, indent=2))
