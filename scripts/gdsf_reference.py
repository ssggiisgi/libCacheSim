#!/usr/bin/env python3
"""
Independent GDSF reference, used to check gdsf and gdsf-gca (rho=1, learn=0).

It shares no code with libCacheSim and follows the same rules:
    H = L + freq * cost / size, L = priority of the last evicted object
    a hit increments freq and resets H; a miss inserts with freq = 1 first,
    then evicts while over capacity
    ties are broken by the older last request (FIFO), as in abstractRank.hpp
    an object larger than the cache is never inserted

cost:
    uniform  cost = 1e6                                  (= libCacheSim gdsf)
    latency  cost = 46.73 + 12.30 * size_MiB  [ms]       (= GDSF-Latency of
             cloud-olap-cache-experiments, the LARA paper's L(o) = a + b*S(o))

Reads requests printed by tracePrint (time,obj_id,obj_size,...) on stdin and
prints "<cache bytes> <miss ratio> <byte miss ratio>" per cache size. Compare
with cachesim run on the same byte sizes:

    ./_build/bin/tracePrint /home/gain/work/olap_traces/MTDDS_20tenants.lcs lcs \
        | python3 scripts/gdsf_reference.py --cost latency 1000000000 4000000000
    ./_build/bin/cachesim /home/gain/work/olap_traces/MTDDS_20tenants.lcs lcs \
        gdsf-gca 1000000000,4000000000 --ignore-obj-size 0 -e "rho=1,cost=latency"
"""

import argparse
import heapq
import sys


def cost_fn(kind, alpha, beta):
    if kind == "uniform":
        return lambda sz: 1.0e6
    return lambda sz: alpha + beta * (sz / 1024 / 1024)


def run(reqs, cap, cost):
    L = 0.0
    occ = 0
    cur = {}  # obj_id -> (priority, last request vtime, size, freq)
    heap = []  # (priority, vtime, obj_id); stale entries are skipped on pop
    miss = miss_bytes = total_bytes = 0
    for vt, (oid, sz) in enumerate(reqs, 1):
        total_bytes += sz
        if oid in cur:
            _, _, osz, freq = cur[oid]
            freq += 1
            p = L + freq * cost(osz) / osz
            cur[oid] = (p, vt, osz, freq)
            heapq.heappush(heap, (p, vt, oid))
            continue
        miss += 1
        miss_bytes += sz
        if sz > cap:
            continue
        p = L + cost(sz) / sz
        cur[oid] = (p, vt, sz, 1)
        heapq.heappush(heap, (p, vt, oid))
        occ += sz
        while occ > cap:
            p, v, o = heapq.heappop(heap)
            if o in cur and cur[o][1] == v:
                L = p
                occ -= cur[o][2]
                del cur[o]
    return miss / len(reqs), miss_bytes / total_bytes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost", choices=["uniform", "latency"], default="uniform")
    ap.add_argument("--lat-alpha", type=float, default=46.73, help="ms per request")
    ap.add_argument("--lat-beta", type=float, default=12.30, help="ms per MiB")
    ap.add_argument("sizes", nargs="+", type=int, help="cache sizes in bytes")
    args = ap.parse_args()
    cost = cost_fn(args.cost, args.lat_alpha, args.lat_beta)
    reqs = []
    for line in sys.stdin:
        if line[:1].isdigit():
            f = line.split(",")
            reqs.append((int(f[1]), int(f[2])))
    for cap in args.sizes:
        mr, bmr = run(reqs, cap, cost)
        print(f"{cap} {mr:.4f} {bmr:.4f}")


if __name__ == "__main__":
    main()
