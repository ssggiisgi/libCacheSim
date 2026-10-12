#!/usr/bin/env python3
"""
Fixed-rho sweep for GDSF-GCA: find the best fixed rho per (trace, cache size).

The best fixed rho is an upper bound on what online rho learning can gain, so
this is the first thing to check before redesigning the learning rule.

For every trace it runs
    gdsf                       (baseline, should equal rho=1)
    gdsf-gca -e "rho=<r>"      for every r in --rhos
and writes all results to a CSV plus a per-(trace, size) summary.

    python3 scripts/gca_rho_sweep.py \
        --trace "data/cloudPhysicsIO.vscsi|vscsi" \
        --trace "data/twitter_cluster52.csv|csv|time-col=1,obj-id-col=2,obj-size-col=3,obj-id-is-num=1" \
        --sizes 0.001,0.005,0.01,0.05,0.1,0.2 \
        --out result/rho_sweep.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

DEFAULT_RHOS = "0,1e-5,1e-4,1e-3,0.01,0.1,0.5,1,2,4"

# <trace> <algo> cache size <size>, <n> req, miss ratio <mr>[, byte miss ratio <bmr>]
RESULT_RE = re.compile(
    r"cache size\s+(?P<size>\S+?),\s+\d+ req, miss ratio (?P<mr>\d+\.\d+)"
    r"(?:, byte miss ratio (?P<bmr>\d+\.\d+))?"
)


def parse_trace_arg(s):
    parts = s.split("|")
    if len(parts) not in (2, 3):
        sys.exit(f'--trace must be "PATH|TYPE" or "PATH|TYPE|TRACE_PARAMS", got "{s}"')
    return {"path": parts[0], "type": parts[1], "params": parts[2] if len(parts) == 3 else ""}


def run_one(cachesim, trace, algo, eparams, sizes, ignore_obj_size, num_thread):
    cmd = [cachesim, trace["path"], trace["type"], algo, sizes,
           "--ignore-obj-size", "1" if ignore_obj_size else "0",
           "--num-thread", str(num_thread)]
    if trace["params"]:
        cmd += ["-t", trace["params"]]
    if eparams:
        cmd += ["-e", eparams]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    rows = []
    for line in p.stdout.splitlines():
        m = RESULT_RE.search(line)
        if m:
            rows.append((m["size"], float(m["mr"]), float(m["bmr"]) if m["bmr"] else None))
    if p.returncode != 0 or not rows:
        print(f"[WARN] failed: {' '.join(cmd)}\n{p.stderr[-2000:]}", file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", action="append", required=True,
                    help='"PATH|TYPE" or "PATH|TYPE|TRACE_PARAMS", repeatable')
    ap.add_argument("--sizes", default="0.001,0.005,0.01,0.05,0.1,0.2")
    ap.add_argument("--rhos", default=DEFAULT_RHOS)
    ap.add_argument("--ignore-obj-size", action="store_true")
    ap.add_argument("--cachesim", default="./_build/bin/cachesim")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4),
                    help="number of cachesim processes run in parallel")
    ap.add_argument("--out", default="result/rho_sweep.csv")
    args = ap.parse_args()

    traces = [parse_trace_arg(t) for t in args.trace]
    rhos = [r.strip() for r in args.rhos.split(",")]
    n_sizes = len(args.sizes.split(","))

    # one job = one (trace, algo config); cachesim itself runs all sizes in parallel
    jobs = []
    for t in traces:
        jobs.append((t, "gdsf", "", "GDSF"))
        for r in rhos:
            jobs.append((t, "gdsf-gca", f"rho={r}", r))

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, args.cachesim, t, algo, ep, args.sizes,
                          args.ignore_obj_size, n_sizes) for (t, algo, ep, _) in jobs]
        results = [f.result() for f in futs]

    # results[(trace, size)][config] = (mr, bmr)
    table = defaultdict(dict)
    size_order = defaultdict(list)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace", "ignore_obj_size", "config", "cache_size", "miss_ratio", "byte_miss_ratio"])
        for (t, _, _, label), rows in zip(jobs, results):
            name = os.path.basename(t["path"])
            for size, mr, bmr in rows:
                w.writerow([name, int(args.ignore_obj_size), label, size, mr, "" if bmr is None else bmr])
                table[(name, size)][label] = (mr, bmr)
                if size not in size_order[name]:
                    size_order[name].append(size)

    # summary: best fixed rho per (trace, size), by object miss ratio
    print(f"\n{'trace':<28} {'size':>10} {'GDSF':>7} {'rho=0':>7} {'best rho':>9} {'best':>7} {'gain vs GDSF':>13}  rho=1==GDSF")
    for name in dict.fromkeys(os.path.basename(t["path"]) for t in traces):
        for size in size_order[name]:
            cell = table[(name, size)]
            if "GDSF" not in cell:
                continue
            gdsf = cell["GDSF"][0]
            cands = [(cell[r][0], r) for r in rhos if r in cell]
            best_mr = min(mr for mr, _ in cands)
            # ties: report every rho that reaches the best value
            best_rhos = "/".join(r for mr, r in cands if mr == best_mr)
            gain = (gdsf - best_mr) / gdsf * 100 if gdsf > 0 else 0.0
            same = "ok" if cell.get("1", (None,))[0] == gdsf else "MISMATCH"
            r0 = cell.get("0", (float("nan"),))[0]
            print(f"{name:<28} {size:>10} {gdsf:>7.4f} {r0:>7.4f} {best_rhos:>9} {best_mr:>7.4f} {gain:>12.2f}%  {same}")
    print(f"\nall results: {args.out}")


if __name__ == "__main__":
    main()
