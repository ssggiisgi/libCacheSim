#!/usr/bin/env python3
"""Independent GDSF with pin/unpin, following the LARA simulator's rules.

Checks bin/pinsim (GDSF-GCA, pin=1, rho=1) on a cloud OLAP request log
("request_type,obj_id,size", 0 = access, 1 = unpin):
  access hit  -> freq + 1, K = L + freq * 1e6 / size, pin + 1
  access miss -> insert with freq 1, pin 1 and K = L + 1e6 / size (L before this miss),
                 then evict unpinned objects with the smallest K (ties: least recently
                 accessed first) until the object fits;
                 L = max(L, largest evicted K) once per miss (GDSF with the delta clamp)
  unpin       -> pin - 1
  objects larger than 0.9 x cache are always misses and never cached.
If every remaining object is pinned, eviction stops and the cache stays over capacity.

usage: gdsf_pin_reference.py <trace.csv> <cache size in bytes>
prints: n_req miss_ratio byte_miss_ratio n_evict_blocked
"""
import heapq
import sys

path, cache = sys.argv[1], int(sys.argv[2])
limit = 0.9 * cache
key, freq, size, pins, last = {}, {}, {}, {}, {}
heap, used, L = [], 0, 0.0
n = miss = req_b = miss_b = blocked = 0
with open(path) as f:
    next(f)
    for line in f:
        t, o, s = line.rstrip().split(",")
        s = int(s)
        if s > limit:
            if t == "0":
                n += 1; miss += 1; req_b += s; miss_b += s
            continue
        if t == "1":
            pins[o] -= 1
            continue
        n += 1; req_b += s
        if o in key:
            freq[o] += 1
            k_new = L + freq[o] * 1e6 / size[o]
        else:
            miss += 1; miss_b += s
            freq[o], size[o], pins[o] = 1, s, 0
            k_new = L + 1e6 / s  # K uses the clock before this miss's evictions
            used += s
            top, held = None, []
            while used > cache:
                while heap:
                    k, tv, v = heapq.heappop(heap)
                    if v in key and key[v] == k and last[v] == tv:
                        if pins[v] > 0:
                            held.append((k, tv, v))
                            continue
                        break
                else:
                    blocked += 1
                    break
                top = k if top is None or k > top else top
                used -= size[v]
                del key[v], freq[v], size[v], last[v], pins[v]
            for h in held:
                heapq.heappush(heap, h)
            if top is not None:
                L = max(L, top)
        pins[o] += 1
        key[o], last[o] = k_new, n
        heapq.heappush(heap, (key[o], n, o))
print(n, f"{miss / n:.6f}", f"{miss_b / req_b:.6f}", blocked)
