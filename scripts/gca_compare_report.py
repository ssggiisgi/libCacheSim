#!/usr/bin/env python3
"""
Turn one or more CSVs from gca_compare_algos.py into a self-contained HTML
report: a cross-trace summary of GDSF-GCA (best fixed rho) vs the other
algorithms, a rho-range analysis for choosing rho-min / rho-max, and per-trace
curves, ranking table, and rho sensitivity heatmap.

example:
    python3 scripts/gca_compare_report.py result/compare_*.csv \
        --out result/gca_report.html
"""

import argparse
import csv
import json
import os

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gca_compare_report.template.html")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--title", default="GDSF-GCA 고정 ρ 비교")
    ap.add_argument("--rho-range", default="1e-6,4",
                    help="rho-min,rho-max defaults in GDSF_GCA.cpp, outlined in the range matrix")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for path in args.csv:
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append({
                    "trace": r["trace"], "algo": r["algo"], "config": r["config"],
                    "offline": int(r["offline"]), "status": r["status"],
                    "label": r["cache_size"],
                    "frac": float(r["size_frac"]) if r["size_frac"] else None,
                    "mr": float(r["miss_ratio"]) if r["miss_ratio"] else None,
                    "bmr": float(r["byte_miss_ratio"]) if r["byte_miss_ratio"] else None,
                })

    fracs = sorted({r["frac"] for r in rows if r["frac"] is not None})
    html = open(TEMPLATE, encoding="utf-8").read()
    # </ in JSON would close the inline <script>; escape it
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = (html.replace("__TITLE__", args.title)
                .replace("__FRACS__", json.dumps(fracs))
                .replace("__RANGE__", json.dumps([x.strip() for x in args.rho_range.split(",")]))
                .replace("__DATA__", data))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"written to {args.out} ({len(rows)} rows from {len(args.csv)} csv)")


if __name__ == "__main__":
    main()
