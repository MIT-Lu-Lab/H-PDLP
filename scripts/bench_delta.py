#!/usr/bin/env python3
"""Diff the per-instance iterations and time of two cuHSDLP result dates for one suite."""
import argparse
import csv
from pathlib import Path

from bench_compare import load_hsd
from common import RESULT_DIR


def ratio(a, b):
    return f"{a / b:.3f}" if b else ""


def run(suite, new_date, old_date, output=None):
    new, _ = load_hsd(suite, new_date)
    old, _ = load_hsd(suite, old_date)
    insts = sorted(set(new) & set(old))

    out = output or RESULT_DIR / f"delta_HSD_{suite}_{new_date}_vs_{old_date}.csv"
    fields = ["instance", "new_status", "old_status", "new_iters", "old_iters",
              "iters_delta", "iters_ratio", "new_time", "old_time",
              "time_delta", "time_ratio"]

    n_status_change = n_faster = n_slower = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in insts:
            n, o = new[i], old[i]
            di, dt = n["iters"] - o["iters"], n["time"] - o["time"]
            if n["status"] != o["status"]:
                n_status_change += 1
            if o["time"] > 1e-9:
                if n["time"] < o["time"] * 0.97:
                    n_faster += 1
                elif n["time"] > o["time"] * 1.03:
                    n_slower += 1
            w.writerow({
                "instance": i,
                "new_status": n["status"], "old_status": o["status"],
                "new_iters": n["iters"], "old_iters": o["iters"],
                "iters_delta": di, "iters_ratio": ratio(n["iters"], o["iters"]),
                "new_time": f"{n['time']:.4f}", "old_time": f"{o['time']:.4f}",
                "time_delta": f"{dt:+.4f}", "time_ratio": ratio(n["time"], o["time"]),
            })

    # Aggregates cover the instances OPTIMAL on both dates.
    both_opt = [i for i in insts
                if new[i]["status"] == "OPTIMAL" and old[i]["status"] == "OPTIMAL"]
    nt = sum(new[i]["time"] for i in both_opt)
    ot = sum(old[i]["time"] for i in both_opt)
    ni = sum(new[i]["iters"] for i in both_opt)
    oi = sum(old[i]["iters"] for i in both_opt)
    print(f"{suite}: {len(insts)} shared instances -> {out}")
    print(f"  status changes: {n_status_change}   "
          f"faster(>3%): {n_faster}   slower(>3%): {n_slower}")
    if both_opt:
        print(f"  both OPTIMAL ({len(both_opt)}): "
              f"iters {ni:,} vs {oi:,} (x{ni/max(oi,1):.3f}), "
              f"time {nt:.1f}s vs {ot:.1f}s (x{nt/max(ot,1e-9):.3f})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", required=True, help="e.g. miplib, mittelmann, netlib_infeas")
    p.add_argument("--new", required=True, help="newer date, e.g. 20260605")
    p.add_argument("--old", required=True, help="older date, e.g. 20260602")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    run(args.suite, args.new, args.old, args.output)


if __name__ == "__main__":
    main()
