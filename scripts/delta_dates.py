#!/usr/bin/env python3
"""Diff two cuHSDLP result dates for one benchmark suite.

Reads summary files from result/{suite}_{new} and result/{suite}_{old} and writes
result/delta_HSD_{suite}_{new}_vs_{old}.csv with per-instance iters/time deltas.

Usage:
    python delta_dates.py --suite miplib --new 20260605 --old 20260602
"""

import argparse
import csv
import os
import sys

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")


def load(suite, date):
    """Parse {suite}_{date} summaries into {instance: {status, iters, time}}."""
    folder = os.path.join(RESULT_DIR, f"{suite}_{date}")
    if not os.path.isdir(folder):
        sys.exit(f"Error: {folder} not found")
    out = {}
    for fname in os.listdir(folder):
        if not fname.endswith("_summary.txt"):
            continue
        inst = fname[: -len("_summary.txt")]
        raw = {}
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    raw[k.strip()] = v.strip()
        out[inst] = {
            "status": raw.get("Termination Reason", ""),
            "iters": int(raw.get("Iterations Count", 0)),
            "time": float(raw.get("Runtime (sec)", 0.0)),
        }
    return out


def ratio(a, b):
    return f"{a / b:.3f}" if b else ""


def main():
    p = argparse.ArgumentParser(description="Diff two cuHSDLP result dates.")
    p.add_argument("--suite", required=True, help="e.g. miplib, mittelmann, netlib_infeas")
    p.add_argument("--new", required=True, help="newer date, e.g. 20260605")
    p.add_argument("--old", required=True, help="older date, e.g. 20260602")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    new = load(args.suite, args.new)
    old = load(args.suite, args.old)
    insts = sorted(set(new) & set(old))

    out = args.output or os.path.join(
        RESULT_DIR, f"delta_HSD_{args.suite}_{args.new}_vs_{args.old}.csv")
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

    # Aggregate over instances OPTIMAL in both dates (the comparable subset)
    both_opt = [i for i in insts
                if new[i]["status"] == "OPTIMAL" and old[i]["status"] == "OPTIMAL"]
    nt = sum(new[i]["time"] for i in both_opt)
    ot = sum(old[i]["time"] for i in both_opt)
    ni = sum(new[i]["iters"] for i in both_opt)
    oi = sum(old[i]["iters"] for i in both_opt)
    print(f"{args.suite}: {len(insts)} shared instances -> {out}")
    print(f"  status changes: {n_status_change}   "
          f"faster(>3%): {n_faster}   slower(>3%): {n_slower}")
    if both_opt:
        print(f"  both OPTIMAL ({len(both_opt)}): "
              f"iters {ni:,} vs {oi:,} (x{ni/max(oi,1):.3f}), "
              f"time {nt:.1f}s vs {ot:.1f}s (x{nt/max(ot,1e-9):.3f})")


if __name__ == "__main__":
    main()
