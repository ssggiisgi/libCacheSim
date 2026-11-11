#!/usr/bin/env python3
"""
Check GDSF-GCA frequency resurrection (eq. 14) against an event log.

For every ghost hit, the object's next insert must have
    freq0 = max(1, round_half_up(1 + exp(-turns / w_res) * fr_evict))
where fr_evict is the freq in the object's last evict row, and every
other insert must have freq = 1.

usage (from the repo root):
    ./_build/bin/cachesim data/cloudPhysicsIO.vscsi.lcs lcs gdsf-gca 0.01 \
        --ignore-obj-size 0 -e "learn=1,resurrect=1,event-log=/tmp/ev"
    python3 scripts/gca_check_resurrection.py /tmp/ev_<cache bytes>.csv 1.0
"""

import csv
import math
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, w_res = sys.argv[1], float(sys.argv[2])
    last_evict_freq = {}
    pending = None  # (obj_id, expected freq0)
    n_ghost = n_ok = n_bad = 0
    for r in csv.DictReader(open(path)):
        ev = r["event"]
        if ev == "evict":
            last_evict_freq[r["obj_id"]] = int(r["freq"])
        elif ev == "ghost_hit":
            n_ghost += 1
            v = 1 + math.exp(-float(r["turns"]) / w_res) * last_evict_freq[r["obj_id"]]
            pending = (r["obj_id"], max(1, math.floor(v + 0.5)))  # llround: half away from 0
        elif ev == "insert":
            want = pending[1] if pending and pending[0] == r["obj_id"] else 1
            if int(r["freq"]) == want:
                n_ok += 1
            else:
                n_bad += 1
                if n_bad <= 5:
                    print(f"mismatch at n_req {r['n_req']}: obj {r['obj_id']} freq {r['freq']}, want {want}")
            pending = None
    print(f"{n_ghost} ghost hits, {n_ok} inserts ok, {n_bad} wrong")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
