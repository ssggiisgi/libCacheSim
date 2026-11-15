#!/usr/bin/env python3
"""Verify converted SAP HANA traces against the original .csv.zst files.

For each trace, checks that
  1. <conv>/<t>.csv is exactly the decompressed <src>/<t>.csv.zst,
  2. <conv>/<t>.lcs holds the accesses (request_type 0) of the CSV, in order:
     clock_time = request index, same obj_id and obj_size,
  3. next_access_vtime is the 1-based index of the next access to the same
     object (INT64_MAX if none), as traceConv writes it.

Decompression uses the system libzstd through ctypes (no zstd CLI needed).

usage (repo root):
    python3 scripts/verify_olap_trace.py <src dir with .csv.zst> <dir with .csv/.lcs> [trace ...]
"""

import ctypes
import ctypes.util
import struct
import sys

TRACES = ["CAB_MT", "CAB_MT_Variant2", "CAB_MT_Variant3", "CAB_ST", "MTDDS_20tenants",
          "RedBench_Cloud", "lineitem_regions"]
LCS_HEADER, NEVER = 8192, 2**63 - 1

zstd = ctypes.CDLL(ctypes.util.find_library("zstd"))
zstd.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
zstd.ZSTD_decompress.restype = ctypes.c_size_t
zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]


def unzstd(path):
    src = open(path, "rb").read()
    n = zstd.ZSTD_getFrameContentSize(src, len(src))
    if n >= 2**64 - 2:
        sys.exit(f"{path}: frame has no content size")
    dst = ctypes.create_string_buffer(n)
    got = zstd.ZSTD_decompress(dst, n, src, len(src))
    if zstd.ZSTD_isError(got) or got != n:
        sys.exit(f"{path}: decompression failed")
    return dst.raw


def verify(t, src_dir, conv_dir):
    raw = unzstd(f"{src_dir}/{t}.csv.zst")
    errors = []
    if raw != open(f"{conv_dir}/{t}.csv", "rb").read():
        errors.append("csv differs from the decompressed zst")
    lines = raw.decode().splitlines()
    if lines[0] != "request_type,obj_id,size":
        errors.append(f"unexpected header {lines[0]!r}")
    rows = [tuple(map(int, l.split(","))) for l in lines[1:]]
    access = [(o, s) for ty, o, s in rows if ty == 0]
    if any(ty not in (0, 1) for ty, _, _ in rows):
        errors.append("request_type other than 0/1")

    body = open(f"{conv_dir}/{t}.lcs", "rb").read()[LCS_HEADER:]
    if len(body) % 24:
        errors.append("lcs body is not a whole number of v1 records")
    recs = list(struct.iter_unpack("<IQIq", body[: len(body) // 24 * 24]))
    if len(recs) != len(access):
        errors.append(f"{len(recs)} lcs records, {len(access)} accesses")
    bad = sum((ct, o, s) != (i, *a) for i, ((ct, o, s, _), a) in enumerate(zip(recs, access)))
    nxt, bad_next = {}, 0
    for i in range(len(recs) - 1, -1, -1):
        o = recs[i][1]
        bad_next += recs[i][3] != nxt.get(o, NEVER)
        nxt[o] = i + 1
    if bad:
        errors.append(f"{bad} records differ in time/id/size")
    if bad_next:
        errors.append(f"{bad_next} records have a wrong next_access_vtime")

    sizes = {}
    size_changes = sum(sizes.setdefault(o, s) != s for o, s in access)
    print(f"{t:17s} {'OK  ' if not errors else 'FAIL'} rows {len(rows):>9,}  accesses {len(access):>9,}"
          f"  objects {len(sizes):>6,}  size changes per id {size_changes}")
    for e in errors:
        print(f"    {e}")
    return not errors


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ok = [verify(t, sys.argv[1], sys.argv[2]) for t in sys.argv[3:] or TRACES]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
