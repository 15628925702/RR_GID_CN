"""Recompute every number quoted in submission 10759 from the result files next to this script."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = ["Uniform SQD", "A-OSQD", "Discriminative Score OED", "RR-GID"]
SHORT = {"Uniform SQD": "Unif", "A-OSQD": "AOSQD", "Discriminative Score OED": "Disc", "RR-GID": "GAMD"}
checks: list[tuple[str, str, str, bool]] = []


def load(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ms(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n), n


def check(label: str, pdf: str, got: float, nd: int | None) -> None:
    s = f"{got:.3e}" if nd is None else f"{got:.{nd}f}"
    checks.append((label, pdf, s, s == pdf))


def table(rows, key):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[key(r)][r["method"]].append(r["primary_loss"])
    return out


def lowest(cell) -> bool:
    return ms(cell["RR-GID"])[0] == min(ms(cell[m])[0] for m in M)


# Figure 1(a)
e1 = load(HERE / "fig1a_e1_budget_sweep/rows.jsonl")
t1 = table(e1, lambda r: int(r["budget"]))
for m, v in [("RR-GID", "1.012"), ("Uniform SQD", "2.986"), ("A-OSQD", "5.671"), ("Discriminative Score OED", "5.960")]:
    check(f"Fig1a B=32000 {SHORT[m]}", v, ms(t1[32000][m])[0], 3)
ok = all(lowest(t1[B]) for B in t1)
checks.append(("Fig1a GAMD lowest at every B", "yes", "yes" if ok else "no", ok))
n1 = sorted({ms(t1[B][m])[2] for B in t1 for m in M})
checks.append(("Fig1a n per cell", "[20]", str(n1), n1 == [20]))
pilot = {B: sorted({int(r["pilot_budget"]) for r in e1 if int(r["budget"]) == B})[0] for B in sorted(t1)}
got = ",".join(str(pilot[B]) for B in sorted(pilot))
checks.append(("Init sizes synthetic", "200,283,400,566,800", got, got == "200,283,400,566,800"))

# Figure 1(b): alpha in {0, 0.5, 1.5} from the nonlinearity grid; alpha = 1 is Fig. 1(a) at B = 8000
e2 = load(HERE / "fig1b_e2_nonlinearity/rows.jsonl")
t2 = table(e2, lambda r: float(r["alpha"]))
check("Fig1b a=1.5 A-OSQD", "12.854", ms(t2[1.5]["A-OSQD"])[0], 3)
check("Fig1b a=1.5 GAMD", "2.787", ms(t2[1.5]["RR-GID"])[0], 3)
ok = all(lowest(t2[a]) for a in (0.0, 0.5, 1.5)) and lowest(t1[8000])
checks.append(("Fig1b GAMD lowest at every alpha", "yes", "yes" if ok else "no", ok))
n2 = sorted({ms(t2[a][m])[2] for a in (0.0, 0.5, 1.5) for m in M})
checks.append(("Fig1b n per cell (alpha != 1)", "[15]", str(n2), n2 == [15]))
p2 = sorted({int(r["pilot_budget"]) for r in e2})
checks.append(("Init size nonlinearity", "[400]", str(p2), p2 == [400]))

# Table 1
r1 = load(HERE / "table1_gas_controlled_shift/rows.jsonl")
t3 = table(r1, lambda r: int(r["budget"]))
pdf_t1 = {
    400: ["0.1166", "0.0041", "0.1185", "0.0040", "0.1160", "0.0040", "0.1147", "0.0042"],
    800: ["0.1060", "0.0038", "0.1085", "0.0035", "0.1062", "0.0036", "0.1045", "0.0038"],
    1600: ["0.1066", "0.0020", "0.1087", "0.0020", "0.1057", "0.0021", "0.1033", "0.0026"],
    3200: ["0.1071", "0.0023", "0.1093", "0.0022", "0.1081", "0.0024", "0.1063", "0.0028"],
}
for B, vals in pdf_t1.items():
    for i, m in enumerate(M):
        mean, se, n = ms(t3[B][m])
        check(f"Table1 B={B} {SHORT[m]} mean (n={n})", vals[2 * i], mean, 4)
        check(f"Table1 B={B} {SHORT[m]} SE", vals[2 * i + 1], se, 4)
p3 = sorted({int(r["pilot_budget"]) for r in r1})
checks.append(("Init sizes Gas", "[80, 114, 160, 227]", str(p3), p3 == [80, 114, 160, 227]))

# Figure 2 (batch 7 and batches 8-9 only)
r2 = load(HERE / "fig2_gas_natural_drift/rows.jsonl")
t4 = table(r2, lambda r: (r["campaign"], int(r["budget"])))
check("Fig2 batch7 B=3200 Unif", "0.0506", ms(t4[("batch7", 3200)]["Uniform SQD"])[0], 4)
check("Fig2 batch7 B=3200 GAMD", "0.0374", ms(t4[("batch7", 3200)]["RR-GID"])[0], 4)
for m, v in [("RR-GID", "0.1095"), ("Uniform SQD", "0.1288"), ("A-OSQD", "0.1194"), ("Discriminative Score OED", "0.1218")]:
    check(f"Fig2 batches8_9 B=3200 {SHORT[m]}", v, ms(t4[("batches8_9", 3200)][m])[0], 4)
cells = [(c, B) for c in ("batch7", "batches8_9") for B in (800, 1600, 3200)]
ok = all(lowest(t4[k]) for k in cells)
checks.append(("Fig2 GAMD lowest both campaigns every B", "yes", "yes" if ok else "no", ok))
n4 = sorted({ms(t4[k][m])[2] for k in cells for m in M})
checks.append(("Fig2 n per cell", "[20]", str(n4), n4 == [20]))

# Table 2
rep = json.loads((HERE / "table2_numerical_validation/risk1_order14_10rep_report.json").read_text(encoding="utf-8"))
q = rep["query"]["orders"][0]
check("Table2 median rel cond-mean error", "9.488e-04", q["median_relative_error"], None)
check("Table2 p95 rel cond-mean error", "2.162e-03", q["p95_relative_error"], None)
check("Table2 max abs coordinate error", "2.236e-03", q["max_abs_coordinate_error"], None)
check("Table2 mean info operator error", "1.421e-03", rep["information"]["mean_op"], None)
check("Table2 max info operator error", "1.977e-03", rep["information"]["max_op"], None)
check("Table2 mean abs R-risk difference", "0.038105", rep["end_to_end"]["mean_abs_delta"], 6)
checks.append(("Table2 report passed", "True", str(rep["passed"]), rep["passed"] is True))

bad = 0
for label, pdf, got, ok in checks:
    bad += not ok
    print(("OK  " if ok else "FAIL"), f"{label}: paper={pdf} got={got}")
print(f"\n{len(checks) - bad}/{len(checks)} match")
raise SystemExit(1 if bad else 0)
