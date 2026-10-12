#!/usr/bin/env python3
"""
Compare GDSF-GCA (fixed rho sweep) against every other eviction algorithm
on one trace over a range of cache sizes.

Each (algo, config) is one cachesim process that runs all sizes in parallel.
Writes one CSV row per (algo, config, cache size); failed / timed-out runs are
recorded with status != ok so they show up in the report instead of vanishing.
Rows are appended as each config finishes, and --resume skips configs that are
already in the CSV. Each cachesim gets an address-space limit (--mem-limit-gb)
so an algorithm that over-allocates fails by itself instead of triggering the
system OOM killer (e.g. wtinyLFU sizes its sketch by cache bytes).

example (from the repo root):
    python3 scripts/gca_compare_algos.py \
        --trace "data/cloudPhysicsIO.vscsi.lcs|lcs" \
        --out result/compare_cloudPhysics.csv
"""

import argparse
import csv
import os
import re
import resource
import subprocess
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# the comparison set for GDSF-GCA (gdsf-gca itself is added from --rhos)
DEFAULT_ALGOS = "lru,arc,lfu,gdsf,cacheus,lecar,belady,beladySize"
DEFAULT_RHOS = "0,1e-6,1e-5,1e-4,1e-3,0.01,0.1,1,4,16"
OFFLINE = {"belady", "beladysize"}

# <trace> <ALGO-NAME> cache size <label>, <n> req, miss ratio <mr>, byte miss ratio <bmr>
RESULT_RE = re.compile(
    r"\s(?P<name>\S+) cache size\s+(?P<label>\S+?),\s+\d+ req, "
    r"miss ratio (?P<mr>\d+\.\d+), byte miss ratio (?P<bmr>\d+\.\d+)")


def parse_trace_arg(s):
    parts = s.split("|")
    if len(parts) not in (2, 3):
        sys.exit(f'--trace must be "PATH|TYPE" or "PATH|TYPE|TRACE_PARAMS", got "{s}"')
    return {"path": parts[0], "type": parts[1], "params": parts[2] if len(parts) == 3 else ""}


def run_cachesim(args, trace, algo, eparams, sizes):
    """one cachesim process over the given sizes; returns (status, rows)"""
    n = len(sizes.split(","))
    cmd = [args.cachesim, trace["path"], trace["type"], algo, sizes,
           "--ignore-obj-size", "1" if args.ignore_obj_size else "0",
           "--num-thread", str(n)]
    if trace["params"]:
        cmd += ["-t", trace["params"]]
    if eparams:
        cmd += ["-e", eparams]
    if algo.lower() in args.hashpower_for:
        cmd += ["--hashpower", str(args.hashpower_for[algo.lower()])]
    limit = int(args.mem_limit_gb * (1 << 30))
    set_limit = lambda: resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=args.timeout, preexec_fn=set_limit)
        out, rc, status = p.stdout, p.returncode, "ok"
    except subprocess.TimeoutExpired as e:
        out, rc, status = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), -1, "timeout"
    # cachesim prints results in the order the sizes were given; the printed
    # label is rounded differently by the single- and multi-size code paths,
    # so rows are keyed by the requested fraction, not by the label
    fracs = sizes.split(",")
    rows = [(fracs[i] if i < n else "", m["label"], float(m["mr"]), float(m["bmr"]))
            for i, m in enumerate(RESULT_RE.finditer(out))]
    if status == "ok" and (rc != 0 or len(rows) != n):
        status = f"failed(rc={rc}, {len(rows)}/{n} sizes)"
    return status, rows


def run_one(args, trace, algo, eparams, n_sizes):
    t0 = time.time()
    status, rows = run_cachesim(args, trace, algo, eparams, args.sizes)
    print(f"[{status:>8}] {algo:<12} {eparams:<10} {time.time() - t0:6.1f}s", file=sys.stderr)
    return status, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help='"PATH|TYPE" or "PATH|TYPE|TRACE_PARAMS"')
    ap.add_argument("--algos", default=DEFAULT_ALGOS)
    ap.add_argument("--rhos", default=DEFAULT_RHOS, help="fixed rho values for gdsf-gca")
    ap.add_argument("--sizes", default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.4")
    ap.add_argument("--ignore-obj-size", action="store_true")
    ap.add_argument("--cachesim", default="./_build/bin/cachesim")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per cachesim process")
    ap.add_argument("--jobs", type=int, default=1,
                    help="cachesim processes run in parallel")
    ap.add_argument("--hashpower-for", default="cacheus=20",
                    help="per-algorithm initial hash table size (log2), e.g. cacheus=20. "
                         "cacheus keeps 7 internal hash tables, ~740MB per cache size "
                         "at the default hashpower 24, and cachesim allocates every "
                         "size up front; tables grow on demand, so a smaller start "
                         "only saves memory (none of the default algos sample "
                         "through the hash table)")
    ap.add_argument("--mem-limit-gb", type=float, default=4.0,
                    help="address-space limit per cachesim process")
    ap.add_argument("--resume", action="store_true", help="skip configs already in --out")
    ap.add_argument("--out", default="result/compare.csv")
    args = ap.parse_args()
    args.hashpower_for = {k.strip().lower(): int(v) for k, v in
                          (kv.split("=") for kv in args.hashpower_for.split(",") if kv.strip())}

    trace = parse_trace_arg(args.trace)
    n_sizes = len(args.sizes.split(","))
    jobs = [(a.strip(), "", a.strip()) for a in args.algos.split(",") if a.strip()]
    jobs += [("gdsf-gca", f"rho={r.strip()}", f"gdsf-gca(rho={r.strip()})") for r in args.rhos.split(",") if r.strip()]

    header = ["trace", "algo", "config", "offline", "status", "cache_size",
              "size_frac", "miss_ratio", "byte_miss_ratio"]
    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            done = {r["config"] for r in csv.DictReader(f)}
        jobs = [j for j in jobs if j[2] not in done]
        print(f"resume: {len(done)} configs already done, {len(jobs)} to run", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    f = open(args.out, "a" if done else "w", newline="")
    w = csv.writer(f)
    if not done:
        w.writerow(header)
    lock = threading.Lock()
    name = os.path.basename(trace["path"])

    def run_and_write(job):
        algo, eparams, label = job
        status, rows = run_one(args, trace, algo, eparams, n_sizes)
        offline = int(algo.lower() in OFFLINE)
        with lock:
            if not rows:
                w.writerow([name, algo, label, offline, status, "", "", "", ""])
            for frac, size, mr, bmr in sorted(rows, key=lambda r: float(r[0])):
                w.writerow([name, algo, label, offline, status, size, frac, mr, bmr])
            f.flush()
        return status, rows

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(run_and_write, jobs))
    f.close()
    bad = [(j[2], r[0]) for j, r in zip(jobs, results) if r[0] != "ok"]
    print(f"\n{len(jobs) - len(bad)}/{len(jobs)} configs ok; written to {args.out}", file=sys.stderr)
    for label, status in bad:
        print(f"  {label}: {status}", file=sys.stderr)


if __name__ == "__main__":
    main()
