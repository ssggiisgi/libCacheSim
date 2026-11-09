#!/usr/bin/env bash
# Convert a cloud OLAP request log (cloud-olap-cache-experiments,
# benchmarks/cloud_benchmark_traces/*.csv after zstd -d) to lcs.
#
# Input CSV: "request_type,obj_id,size" with a header, where request_type is
#   0 = access (hit or miss), 1 = unpin (the object may be evicted again).
# libCacheSim has no pinning, so only the accesses (type 0) are kept, in order;
# the request index becomes the timestamp. The traces carry no miss cost.
# traceConv then adds next_access_vtime, so Belady runs on the result.
#
# usage (from the repo root):
#   scripts/convert_olap_trace.sh <in.csv> <out.lcs>

set -euo pipefail
if [ $# -ne 2 ]; then
  echo "usage: $0 <in.csv> <out.lcs>" >&2
  exit 1
fi
in=$1
out=$2
tmp=$(mktemp --suffix=.csv)
trap 'rm -f "$tmp"' EXIT

awk -F, 'NR > 1 && $1 == 0 { print n++ "," $2 "," $3 }' "$in" > "$tmp"
"$(dirname "$0")/../_build/bin/traceConv" "$tmp" csv -o "$out" \
  -t "time-col=1,obj-id-col=2,obj-size-col=3,delimiter=,,has-header=false,obj-id-is-num=1"
echo "$in -> $out ($(wc -l < "$tmp") requests)"
