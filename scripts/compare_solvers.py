#!/usr/bin/env python3
"""Compare cuHSDLP and cuPDLPx benchmark results on MIPLIB instances."""

import argparse
import csv
import math
import os
import sys

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "result")

# Per-benchmark configuration: HSD result folder prefix, cuPDLPx CSV, config name prefix
BENCHMARKS = {
    "small":      {"folder_prefix": "miplib_small", "pdlpx_csv": "cuPDLPx_result_miplib.csv",     "config_prefix": "miplib"},
    "miplib":     {"folder_prefix": "miplib",       "pdlpx_csv": "cuPDLPx_result_miplib.csv",     "config_prefix": "miplib"},
    "mittelmann": {"folder_prefix": "mittelmann",   "pdlpx_csv": "cuPDLPx_result_mittelmann.csv", "config_prefix": "mittelmann"},
}

# Fields extracted from each solver's output
METRIC_KEYS = ["status", "iters", "time", "primal_obj", "dual_obj",
               "rel_gap", "rel_pr_res", "rel_du_res"]


# Parse command-line arguments
def parse_args():
    p = argparse.ArgumentParser(description="Compare cuHSDLP vs cuPDLPx results.")
    p.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="miplib",
                   help="benchmark suite (small, miplib, mittelmann)")
    p.add_argument("--date", default="20260503", help="cuHSDLP result folder date (e.g. 20260503)")
    p.add_argument("--tol", default="1e-4", help="tolerance level (e.g. 1e-4, 1e-8)")
    p.add_argument("--output", default=None, help="output CSV path")
    return p.parse_args()


def load_hsd_results(folder_prefix, date_suffix):
    """Parse cuHSDLP summary files into {instance: metrics}."""
    # Look for summary files in the specified result folder
    folder = os.path.join(RESULT_DIR, f"{folder_prefix}_{date_suffix}")
    if not os.path.isdir(folder):
        sys.exit(f"Error: {folder} not found")

    # Map summary file keys to our metric keys
    key_map = {
        "Termination Reason": "status",
        "Iterations Count": "iters",
        "Runtime (sec)": "time",
        "Primal Objective Value": "primal_obj",
        "Dual Objective Value": "dual_obj",
        "Relative Objective Gap": "rel_gap",
        "Relative Primal Residual": "rel_pr_res",
        "Relative Dual Residual": "rel_du_res",
    }

    # Parse each summary file into a metrics dictionary
    results = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith("_summary.txt"):
            continue
        instance = fname.replace("_summary.txt", "")
        raw = {}
        with open(os.path.join(folder, fname)) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    raw[k.strip()] = v.strip()
        results[instance] = {
            mk: int(raw[fk]) if mk == "iters"
            else raw[fk] if mk == "status"
            else float(raw[fk])
            for fk, mk in key_map.items() if fk in raw
        }
    return results


def load_pdlpx_results(csv_path, config):
    """Parse cuPDLPx CSV into {instance: metrics}, filtered by config."""
    # Look for the cuPDLPx CSV file
    if not os.path.isfile(csv_path):
        sys.exit(f"Error: {csv_path} not found")

    # Map CSV column names to our metric keys
    key_map = {
        "termination_reason": "status",
        "total_count": "iters",
        "time_sec": "time",
        "primal_obj": "primal_obj",
        "dual_objective_value": "dual_obj",
        "relative_objective_gap": "rel_gap",
        "relative_primal_residual": "rel_pr_res",
        "relative_dual_residual": "rel_du_res",
    }

    # 
    results = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["config"] != config:
                continue
            d = {}
            for ck, mk in key_map.items():
                v = row.get(ck, "")
                if mk == "status":
                    d[mk] = v
                elif mk == "iters":
                    d[mk] = int(float(v)) if v else 0
                else:
                    d[mk] = float(v) if v else 0.0
            results[row["instance"]] = d
    return results


# Safe ratio function to handle near-zero denominators
def safe_ratio(a, b):
    """Return a/b as formatted string, or empty if b ≈ 0."""
    return f"{a / b:.2f}" if b > 1e-12 else ""


# Shifted geometric mean (Achterberg/SCIP convention, used by the Mittelmann
# benchmarks): SGM_s(x) = exp(mean(log(xi + s))) - s.  Standard shifts are
# s = 10 for time (seconds) and s = 100 for iterations / nodes.  The shift
# softens near-zero values so small instances no longer dominate or get dropped.
def shifted_geomean(values, shift):
    """Return the shifted geometric mean of `values` with the given shift."""
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift

# Write combined CSV
def write_csv(hsd, pdlpx, path):
    """Write side-by-side comparison CSV with HSD/PDLPx columns interleaved."""
    # Only include instances that HSD has results for
    instances = sorted(i for i in hsd if i in pdlpx)

    # Interleave HSD/PDLPx columns per metric for easy comparison
    fields = ["instance"]
    for k in METRIC_KEYS:
        fields += [f"HSD_{k}", f"PDLPx_{k}"]
    fields += ["iter_ratio", "time_ratio"]

    # Helper to format metric values for CSV output
    def fmt(val, key):
        """Format a metric value for CSV output."""
        if val is None:
            return ""
        if key == "status":
            return val
        if key == "iters":
            return str(val)
        if key == "time":
            return f"{val:.4f}"
        return f"{val:.6e}"

    # Write the CSV file
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for inst in instances:
            h, p = hsd.get(inst, {}), pdlpx.get(inst, {})
            row = {"instance": inst}
            for prefix, src in [("HSD", h), ("PDLPx", p)]:
                for k in METRIC_KEYS:
                    row[f"{prefix}_{k}"] = fmt(src.get(k), k)
            row["iter_ratio"] = safe_ratio(h.get("iters", 0), p.get("iters", 0)) if h and p else ""
            row["time_ratio"] = safe_ratio(h.get("time", 0), p.get("time", 0)) if h and p else ""
            w.writerow(row)
    return instances


# Print summary statistics
def print_summary(hsd, pdlpx, instances):
    """Print summary statistics to stdout."""
    # Consider only instances that both solvers attempted
    shared = [i for i in instances if i in hsd and i in pdlpx]
    
    # Count how many were solved to OPTIMAL by each solver
    h_opt = sum(1 for i in shared if hsd[i]["status"] == "OPTIMAL")
    p_opt = sum(1 for i in shared if pdlpx[i]["status"] == "OPTIMAL")
    both_opt = [i for i in shared
                if hsd[i]["status"] == "OPTIMAL" and pdlpx[i]["status"] == "OPTIMAL"]
    print(f"Shared instances: {len(shared)}")
    print(f"OPTIMAL: HSD {h_opt}/{len(shared)},  PDLPx {p_opt}/{len(shared)}")
    print()

    # Both-OPTIMAL intersection: reference only.  Restricting the SGM to instances
    # both solvers happen to solve silently drops any instance one solver fails,
    # which flatters whichever solver fails more.
    if both_opt:
        ht = sum(hsd[i]["time"] for i in both_opt)
        pt = sum(pdlpx[i]["time"] for i in both_opt)
        hi = sum(hsd[i]["iters"] for i in both_opt)
        pi = sum(pdlpx[i]["iters"] for i in both_opt)
        print(f"Both OPTIMAL ({len(both_opt)}) [reference, intersection only]:")
        print(f"  Iters: HSD {hi:>12,}  PDLPx {pi:>12,}  ratio {hi / max(pi, 1):.2f}x")
        print(f"  Time:  HSD {ht:>11.1f}s  PDLPx {pt:>11.1f}s  ratio {ht / max(pt, 1):.2f}x")
        h_time_sgm = shifted_geomean([hsd[i]["time"] for i in both_opt], 10)
        p_time_sgm = shifted_geomean([pdlpx[i]["time"] for i in both_opt], 10)
        print(f"  SGM time  (s=10):  ratio {h_time_sgm / max(p_time_sgm, 1e-12):.2f}x")
        print()

    # Headline SGM (Achterberg/Mittelmann and the cuPDLPx PSLP paper convention):
    # include every genuine optimization run -- at least one solver reached OPTIMAL,
    # or some solver hit the wall-clock limit -- and assign a failed solver the time
    # limit.  Instances both solvers agree are infeasible/unbounded carry no optimum
    # to time, so they are excluded.  The wall-clock limit is read off the timed-out
    # runs themselves (they record the limit as their time); a non-OPTIMAL solver is
    # charged that limit.  Iterations use the recorded count: these runs are
    # time-limited, so a timed-out solver's iteration count is real work performed.
    sgm_pop = [i for i in shared
               if hsd[i]["status"] == "OPTIMAL" or pdlpx[i]["status"] == "OPTIMAL"
               or hsd[i]["status"] == "TIME_LIMIT" or pdlpx[i]["status"] == "TIME_LIMIT"]
    timeouts = ([hsd[i]["time"] for i in sgm_pop if hsd[i]["status"] == "TIME_LIMIT"]
                + [pdlpx[i]["time"] for i in sgm_pop if pdlpx[i]["status"] == "TIME_LIMIT"])
    time_limit = max(timeouts) if timeouts else None

    def charged_time(d):
        return d["time"] if d["status"] == "OPTIMAL" or time_limit is None else time_limit

    if sgm_pop:
        h_time = [charged_time(hsd[i]) for i in sgm_pop]
        p_time = [charged_time(pdlpx[i]) for i in sgm_pop]
        h_iter = [hsd[i]["iters"] for i in sgm_pop]
        p_iter = [pdlpx[i]["iters"] for i in sgm_pop]
        excluded = len(shared) - len(sgm_pop)
        limit_note = f", timeouts charged {time_limit:.0f}s" if time_limit else ""
        print(f"SGM over {len(sgm_pop)} instances "
              f"({excluded} infeasible/unbounded excluded{limit_note}):")
        for s in (1, 10):
            ratio = shifted_geomean(h_time, s) / max(shifted_geomean(p_time, s), 1e-12)
            print(f"  SGM time  (s={s:<2}): ratio {ratio:.2f}x")
        ratio_i = shifted_geomean(h_iter, 100) / max(shifted_geomean(p_iter, 100), 1e-12)
        print(f"  SGM iters (s=100): ratio {ratio_i:.2f}x")
        print()

    # Exclusive solves
    for label, cond in [("HSD solved, PDLPx didn't",
                         lambda i: hsd[i]["status"] == "OPTIMAL" and pdlpx[i]["status"] != "OPTIMAL"),
                        ("PDLPx solved, HSD didn't",
                         lambda i: hsd[i]["status"] != "OPTIMAL" and pdlpx[i]["status"] == "OPTIMAL")]:
        wins = [i for i in shared if cond(i)]
        if wins:
            print(f"{label} ({len(wins)}):")
            for i in wins:
                print(f"  {i}")
            print()

# Main function to orchestrate loading, comparing, and output
def main():
    # Parse arguments
    args = parse_args()
    bench = BENCHMARKS[args.benchmark]
    config = f"{bench['config_prefix']}_{args.tol}_presolve"
    pdlpx_csv = os.path.join(RESULT_DIR, bench["pdlpx_csv"])
    output = args.output or os.path.join(RESULT_DIR, f"comparison_cuHSDLP_vs_cuPDLPx_{args.benchmark}.csv")

    # Load results from both solvers
    hsd = load_hsd_results(bench["folder_prefix"], args.date)
    pdlpx = load_pdlpx_results(pdlpx_csv, config)
    instances = write_csv(hsd, pdlpx, output)

    # Print summary statistics
    print(f"Output: {output}\n")
    print_summary(hsd, pdlpx, instances)


if __name__ == "__main__":
    main()
