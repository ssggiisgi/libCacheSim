#!/usr/bin/env python3
"""Check GDSF-GCA against the hand trace of data/toy_gca.csv.

The hand trace (gishin: gdsf-gca/docs/toy-trace.md) fixes which object is
admitted, evicted or dropped from the ghost, and in which order. Python only
does the arithmetic of eqs (2)-(14). The test runs cachesim with event-log and
compares every logged event, in order, with the hand trace.

Setup: cache 10 B, ghost 10 B, c0 = 1e6, gamma = 1, learn=1, rho=2, eta=0.5,
rho-max=2.5, resurrect=1, W_mem = W_res = 1, W_burst = 0.25.

usage (repo root, after building _build/bin/cachesim):
    python3 scripts/test_gca_toy.py [--out DIR]     # default DIR: result/toy_test/

Writes to DIR: event_log.csv (cachesim's raw log), comparison.csv (one row per
checked value: logged, expected, match) and summary.txt (date, commit, result).
"""

import argparse
import csv
import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = "learn=1,rho=2,eta=0.5,rho-max=2.5,resurrect=1"
CACHE, ETA, W_MEM, W_BURST, W_RES = 10, 0.5, 1.0, 0.25, 1.0
LOG_RHO_MIN, LOG_RHO_MAX = math.log(1e-6), math.log(2.5)

A, B, C, D, E, F, G, X = 1, 2, 3, 4, 5, 6, 7, 9
SIZE = {A: 4, B: 4, C: 2, D: 5, E: 1, F: 2, G: 4, X: 11}


def V(obj, freq):
    return freq * 1e6 / SIZE[obj]


class HandTrace:
    """Cache state and the expected event log, advanced one hand-traced step at a time."""

    def __init__(self):
        self.events = []
        self.clock, self.log_rho, self.admitted = 0.0, math.log(2.0), 0
        self.key, self.freq = {}, {}
        self.ghost = {}  # obj -> (slack, bytes admitted at eviction, freq at eviction)

    def emit(self, n, event, obj, **fields):
        if obj in SIZE:
            fields["obj_size"] = SIZE[obj]
        self.events.append((n, event, obj, fields))

    def hit(self, n, obj):
        self.freq[obj] += 1
        self._set_key(obj)
        self.emit(n, "hit", obj, freq=self.freq[obj], key=self.key[obj],
                  value=V(obj, self.freq[obj]), clock=self.clock)

    def miss(self, n, obj):
        """Algorithm 1, Cases 3-4: ghost hit (if any), then admit."""
        freq0 = self._ghost_hit(n, obj) if obj in self.ghost else 1
        self.freq[obj] = freq0
        self._set_key(obj)
        self.admitted += SIZE[obj]
        self.emit(n, "insert", obj, freq=freq0, key=self.key[obj], value=V(obj, freq0),
                  clock=self.clock)

    def bypass(self, n, obj):
        self.emit(n, "bypass", obj)

    def evict(self, n, *victims, ghost_drops=()):
        """Algorithm 3: evict the given victims in order, then update the clock once."""
        base = self.clock
        for v in victims:
            key, freq = self.key.pop(v), self.freq.pop(v)
            slack = (key - base) / V(v, freq)
            self.ghost[v] = (slack, self.admitted, freq)
            self.emit(n, "evict", v, freq=freq, key=key, clock_base=base,
                      value=V(v, freq), slack=slack)
        for d in ghost_drops:
            del self.ghost[d]
            self.emit(n, "ghost_drop", d)
        rho, top = math.exp(self.log_rho), key  # the last victim has the largest key
        self.clock = base + rho * max(0.0, top - base)
        self.emit(n, "clock", -1, key=top, clock_base=base, rho=rho, clock=self.clock)

    def _set_key(self, obj):
        self.key[obj] = self.clock + V(obj, self.freq[obj])

    def _ghost_hit(self, n, obj):
        """Algorithm 2: update log rho and return the resurrected freq."""
        slack, admitted_at_evict, freq_evict = self.ghost.pop(obj)
        turns = (self.admitted - admitted_at_evict) / CACHE
        s = min(1.0, max(0.0, slack))
        g_raw = 2 * s - 1
        g = g_raw * math.exp(-turns / W_BURST) if g_raw > 0 else g_raw
        w = math.exp(-turns / W_MEM)
        raw = self.log_rho + ETA * w * g
        new = min(LOG_RHO_MAX, max(LOG_RHO_MIN, raw))
        self.emit(n, "ghost_hit", obj, slack=slack, turns=turns, s=s, g_raw=g_raw, w=w, g=g,
                  dlog_rho=new - self.log_rho, rho=math.exp(new), clamped=int(raw != new))
        self.log_rho = new
        return max(1, math.floor(1.5 + math.exp(-turns / W_RES) * freq_evict))


def expected():
    t = HandTrace()
    t.miss(1, A); t.miss(2, B); t.hit(3, A); t.miss(4, C)  # 4 + 4 + 2 = 10 B, full
    t.miss(5, D); t.evict(5, D)                 # D has the smallest key: evicts itself, slack 1
    t.miss(6, D); t.evict(6, B, A)              # turns 0, raise -> clamped; A, C tie, A older
    t.miss(7, E)                                # fits, no eviction
    t.miss(8, B); t.evict(8, C)                 # turns 0.1, lower; clock delta clamped at 0
    t.bypass(9, X)                              # 11 B > cache
    t.miss(10, A); t.evict(10, D)               # turns 0.5, freq_evict 2
    t.miss(11, F); t.evict(11, B, ghost_drops=[C])  # B, A tie, B older; ghost 11 B > 10
    t.miss(12, G); t.evict(12, A, ghost_drops=[D])
    t.miss(13, B); t.evict(13, F)               # turns 0.4, gated raise
    t.hit(14, B)
    return t.events


def run_cachesim(event_log):
    """Run cachesim on the toy trace, keep its event log at event_log, and return the rows."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "ev")
        subprocess.run(
            [f"{ROOT}/_build/bin/cachesim", f"{ROOT}/data/toy_gca.csv", "csv", "gdsf-gca",
             str(CACHE), "--ignore-obj-size", "0",
             "-t", "time-col=1,obj-id-col=2,obj-size-col=3,has-header=true,obj-id-is-num=1",
             "-e", f"{PARAMS},event-log={prefix}"],
            check=True, capture_output=True)
        shutil.copy(glob.glob(prefix + "_*.csv")[0], event_log)
    with open(event_log) as f:
        return list(csv.DictReader(f))


def compare(want, got):
    """One row per checked value: (i, n_req, event, obj_id, field, logged, expected, match)."""
    rows = []
    for i, ((n, event, obj, fields), row) in enumerate(zip(want, got)):
        logged = (int(row["n_req"]), row["event"], int(row["obj_id"] or -1))
        if logged != (n, event, obj):
            rows.append((i, n, event, obj, "event", " ".join(map(str, logged)),
                         f"{n} {event} {obj}", False))
            continue
        for k, v in fields.items():
            ok = row[k] != "" and math.isclose(float(row[k]), v, rel_tol=1e-9, abs_tol=1e-9)
            rows.append((i, n, event, obj, k, row[k], repr(v), ok))
    return rows


def commit():
    r = subprocess.run(["git", "-C", ROOT, "describe", "--always", "--dirty"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def main():
    ap = argparse.ArgumentParser(description="Check GDSF-GCA against the toy hand trace.")
    ap.add_argument("--out", default=f"{ROOT}/result/toy_test",
                    help="where event_log.csv, comparison.csv and summary.txt are written")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)

    want, got = expected(), run_cachesim(f"{out}/event_log.csv")
    rows = compare(want, got)
    with open(f"{out}/comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "n_req", "event", "obj_id", "field", "logged", "expected", "match"])
        w.writerows(r[:-1] + (int(r[-1]),) for r in rows)

    bad = [f"event {i} (req {n} {ev} obj {obj}): {k} = {got_v}, want {want_v}"
           for i, n, ev, obj, k, got_v, want_v, ok in rows if not ok]
    if len(got) != len(want):
        bad.append(f"event count: {len(got)} logged, {len(want)} expected")
    summary = [
        "GDSF-GCA toy trace test (scripts/test_gca_toy.py, data/toy_gca.csv)",
        f"date:    {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"commit:  {commit()}",
        f"params:  cache {CACHE} B, {PARAMS}",
        f"result:  {'FAIL' if bad else 'PASS'} - {len(want)} events expected, {len(got)} logged, "
        f"{len(rows)} values checked, {len(bad)} mismatches",
        *bad,
    ]
    with open(f"{out}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"written to {out}/")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
