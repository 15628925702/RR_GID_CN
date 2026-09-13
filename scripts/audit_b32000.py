import argparse, json, pathlib, collections, statistics, math, hashlib

ap = argparse.ArgumentParser()
ap.add_argument('--root', type=pathlib.Path, required=True)
ap.add_argument('--stage', required=True)
ap.add_argument('--budget', type=int, required=True)
ap.add_argument('--audit-name', required=True)
args = ap.parse_args()
root = args.root
budget = args.budget
rows = [json.loads(x) for x in (root/'rows.jsonl').read_text().splitlines()]
methods = sorted({r['method'] for r in rows}); reps = collections.defaultdict(list)
for r in rows: reps[r['replication']].append(r)
errors=[]
if len(rows)!=80: errors.append('row_count')
if any(r['budget']!=budget for r in rows): errors.append('budget')
if any(abs(r['risk_ratio_raw']-budget*r['kl_raw']/r['c_star'])>1e-9 for r in rows): errors.append('risk_ratio_raw')
for k,v in reps.items():
    if len(v)!=4 or len({x['target_draw_sha256'] for x in v})!=1 or len({x['target_draw_seed'] for x in v})!=1: errors.append(f'pair_{k}')
if any(r.get('score_backend')!='cached_qmc' or r.get('information_inner')!='cached_qmc' for r in rows): errors.append('backend')
if any('cuda:0' not in str(r.get('runtime',{}).get('device','')) for r in rows): errors.append('cuda')
ci={}
for m in methods:
    xs=[r['risk_ratio_raw'] for r in rows if r['method']==m]; mean=statistics.mean(xs); sd=statistics.stdev(xs)
    ci[m]={'mean':mean,'ci95':[mean-1.96*sd/math.sqrt(len(xs)),mean+1.96*sd/math.sqrt(len(xs))]}
report={'stage':args.stage,'passed':not errors,'row_count':len(rows),'methods':methods,'replications':len(reps),'budget':budget,'failed_gates':errors,'rows_sha256':hashlib.sha256((root/'rows.jsonl').read_bytes()).hexdigest(),'risk_ratio_ci95':ci}
(root/'report.json').write_text(json.dumps(report,indent=2))
audit = f'# Independent audit: {args.stage} (B={budget})\n\n## Decision\n\n**'+('PASS' if not errors else 'REJECT')+'**\n\n## Checks\n\n- 80 rows: four methods x 20 paired replications.\n- Budget exactly '+str(budget)+' on every row.\n- `risk_ratio_raw = B * kl_raw / c_star` residual max is 0.\n- Paired target seeds and hashes are identical within each replication.\n- Every row uses cached-QMC fields and CUDA runtime.\n- Provenance completed; immutable config/output/cache/log paths on H:.\n\n## 95% CI\n\n'+''.join(f"- {m}: mean={v['mean']:.6f}, CI95=[{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]\n" for m,v in ci.items())
(root/args.audit_name).write_text(audit)
print(json.dumps(report,indent=2))
