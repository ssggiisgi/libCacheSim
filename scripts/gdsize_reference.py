#!/usr/bin/env python3
"""
Independent GD-Size reference, used to check libCacheSim's gdsize (GDSize.cpp).

It shares no code with libCacheSim and follows the same rules:
    H = L + 1e6 / size, L = priority of the last evicted object
    a hit resets H; a miss inserts first, then evicts while over capacity
    ties are broken by the older last request (FIFO), as in abstractRank.hpp
    an object larger than the cache is never inserted

Reads requests printed by tracePrint (time,obj_id,obj_size,...) on stdin and
prints "<cache bytes> <miss ratio> <byte miss ratio>" per cache size. Compare
with cachesim run on the same byte sizes:

    ./_build/bin/tracePrint data/twitter_cluster52.lcs lcs \
        | python3 scripts/gdsize_reference.py 30000 300000 3000000
    ./_build/bin/cachesim data/twitter_cluster52.lcs lcs gdsize \
        30000,300000,3000000 --ignore-obj-size 0
"""

import heapq
import sys


def run(reqs, cap):
    L = 0.0
    occ = 0
    cur = {}  # obj_id -> (priority, last request vtime, size)
    heap = []  # (priority, vtime, obj_id); stale entries are skipped on pop
    miss = miss_bytes = total_bytes = 0
    for vt, (oid, sz) in enumerate(reqs, 1):
        total_bytes += sz
        if oid in cur:
            p = L + 1e6 / cur[oid][2]
            cur[oid] = (p, vt, cur[oid][2])
            heapq.heappush(heap, (p, vt, oid))
            continue
        miss += 1
        miss_bytes += sz
        if sz > cap:
            continue
        p = L + 1e6 / sz
        cur[oid] = (p, vt, sz)
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
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    reqs = []
    for line in sys.stdin:
        if line[:1].isdigit():
            f = line.split(",")
            reqs.append((int(f[1]), int(f[2])))
    for cap in map(int, sys.argv[1:]):
        mr, bmr = run(reqs, cap)
        print(f"{cap} {mr:.4f} {bmr:.4f}")


if __name__ == "__main__":
    main()
