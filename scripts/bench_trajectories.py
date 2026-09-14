#!/usr/bin/env python3.12
"""Plot the τ and ω trajectories of every instance, one panel per benchmark suite."""
import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from common import FIGURE_DIR, RESULT_DIR, read_summary

RC = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans"],
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
    "axes.linewidth":   0.9,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "savefig.dpi":      200,
    "figure.dpi":       130,
}

# Termination-status colours; unknown reasons fall back to C_OTHER.
REASON_COLOR = {
    "OPTIMAL":           "#1f5fa6",
    "PRIMAL_INFEASIBLE": "#e07b00",
    "DUAL_INFEASIBLE":   "#b5179e",
    "TIME_LIMIT":        "#cc2a36",
    "ITERATION_LIMIT":   "#7b1f8a",
}
C_OTHER = "#888888"

TRACE_SUFFIX = "_tau_trace.csv"
DOWNSAMPLE_THRESHOLD = 50000   # traces longer than this are log-downsampled to N_KEEP rows
N_KEEP = 5000
ALPHA_UNIFORM = 0.35
LW_UNIFORM    = 1.2

Y_CEIL          = 1e6        # τ axis ceiling
# τ is exactly 0 on infeasible rays, so a symlog linear band below the smallest positive τ (8.17e-9) keeps 0 plottable.
TAU_LINTHRESH   = 1e-9

OMEGA_Y_FLOOR    = 1e-10         # axis floor
OMEGA_Y_CEIL     = 1e14          # axis ceiling (data max ~4.4e13)
OMEGA_DATA_FLOOR = OMEGA_Y_FLOOR * 10.0   # nudge above axis so floor lines visible

# trace column -> (keep only positive values for a log axis, clip applied before plotting)
COLUMNS = {
    "tau":   (False, lambda vals: np.minimum(vals, Y_CEIL)),
    "omega": (True,  lambda vals: np.clip(vals, OMEGA_DATA_FLOOR, OMEGA_Y_CEIL)),
}


def iso(date):
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"


def list_instances(bench_dir):
    """Instance stems that have a τ-trace CSV in this benchmark's result dir."""
    return sorted(p.name[:-len(TRACE_SUFFIX)] for p in (RESULT_DIR / bench_dir).glob(f"*{TRACE_SUFFIX}"))


def read_reason(inst, bench_dir):
    """Termination Reason from {inst}_summary.txt, or 'UNKNOWN' if absent."""
    path = RESULT_DIR / bench_dir / f"{inst}_summary.txt"
    return read_summary(path).get("Termination Reason", "UNKNOWN") if path.exists() else "UNKNOWN"


def downsample(its, vals, label):
    """(iters, values, iter_at_max, max_value) without torn rows, log-downsampled past the threshold keeping the peak."""
    its, vals = np.asarray(its), np.asarray(vals)
    # A torn NFS write can drop a leading digit of iter, so keep only rows at or above the running max.
    if its.size > 1 and np.any(np.diff(its) < 0):
        mask = its >= np.maximum.accumulate(its)
        sys.stderr.write(f"[torn-trace] {label}: drop {int((~mask).sum())} corrupt row(s)\n")
        its, vals = its[mask], vals[mask]
    peak = int(np.argmax(vals))
    it_max, max_value = float(its[peak]), float(vals[peak])
    if len(its) > DOWNSAMPLE_THRESHOLD:
        keep = set(np.unique(np.round(np.logspace(0, math.log10(len(its) - 1), N_KEEP)).astype(int)))
        keep.update({peak, max(0, peak - 1), min(len(its) - 1, peak + 1)})
        sel = np.array(sorted(keep))
        its, vals = its[sel], vals[sel]
    return its, vals, it_max, max_value


def load_traces(inst, bench_dir):
    """Every COLUMNS trace of one instance from a single read of its CSV, None for a column without rows."""
    raw = {column: ([], []) for column in COLUMNS}
    with (RESULT_DIR / bench_dir / f"{inst}{TRACE_SUFFIX}").open() as h:
        for row in csv.DictReader(h):
            for column, (its, vals) in raw.items():
                try:
                    it, v = float(row["iter"]), float(row[column])
                except (ValueError, KeyError, TypeError):
                    continue
                if v > 0 or not COLUMNS[column][0]:
                    its.append(it); vals.append(v)
    return {column: downsample(its, vals, f"{bench_dir}/{inst}") if its else None for column, (its, vals) in raw.items()}


def plot_panel(axes, bench_dir):
    """Draw every instance's traces on axes[column] coloured by termination status; returns each column's peaks and legend."""
    peaks = {column: [] for column in axes}
    legend = {column: {} for column in axes}
    for inst in list_instances(bench_dir):
        traces = load_traces(inst, bench_dir)
        if all(trace is None for trace in traces.values()):
            continue
        reason = read_reason(inst, bench_dir)
        color = REASON_COLOR.get(reason, C_OTHER)
        for column, ax in axes.items():
            if traces[column] is None:
                continue
            its, vals, it_max, max_value = traces[column]
            # Non-OPTIMAL runs are drawn above the dense OPTIMAL bundle.
            ax.plot(its, COLUMNS[column][1](vals), color=color, lw=LW_UNIFORM, alpha=ALPHA_UNIFORM,
                    zorder=2 if reason == "OPTIMAL" else 5)
            peaks[column].append((it_max, max_value, color))
            legend[column][reason] = color
    return peaks, legend


def status_legend(ax, legend):
    """Add a legend mapping the termination statuses present in this panel."""
    if not legend:
        return
    handles = [mpatches.Patch(color=c, label=r) for r, c in sorted(legend.items())]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92)


def style_tau_panel(ax, title, peaks, legend):
    ax.set_xscale("log")
    # linscale=3 makes the linear band one 3-decade cell tall, and the -linthresh floor adds one more cell below 0.
    ax.set_yscale("symlog", linthresh=TAU_LINTHRESH, linscale=3.0)
    ax.set_ylim(-TAU_LINTHRESH, Y_CEIL)
    ax.set_yticks([0, 1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
    ax.axhline(0.0, color="0.45", lw=0.8, ls=":", alpha=0.7)  # τ→0 reference
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.set_title(f"{title}  (n={len(peaks)})", pad=8, loc="left")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\tau$")
    ax.grid(True, alpha=0.25, which="both")
    # Spikes above Y_CEIL are labelled with just their value, staggered along y.
    spikes = sorted((peak for peak in peaks if peak[1] > Y_CEIL), key=lambda peak: peak[0])
    for k, (it_max, mx, color) in enumerate(spikes):
        ax.annotate(f"{mx:.0e}",
                    xy=(it_max, Y_CEIL),
                    xytext=(0, -4 - 14 * (k % 3)), textcoords="offset points",
                    fontsize=9, ha="center", va="top", color=color,
                    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=color, lw=0.6, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.6))
    status_legend(ax, legend)


def style_omega_panel(ax, title, peaks, legend):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(OMEGA_Y_FLOOR, OMEGA_Y_CEIL)
    ax.set_title(f"{title}  (n={len(peaks)})", pad=8, loc="left")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\omega$  (primal weight)")
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.25, which="both")
    status_legend(ax, legend)


def save(fig, stem, suptitle, footer):
    """Title the figure and save it as {stem}.pdf and {stem}.png."""
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.5, 0.005, footer, ha="center", fontsize=9.5, color="grey")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    sys.stderr.write(f"Saved {stem}.[pdf,png]\n")
    plt.close(fig)


def run(date, infeas_date=None, output_dir=FIGURE_DIR):
    infeas_date = infeas_date or date
    # (result subdir, title) in display order; suites without a trace for their date are dropped.
    panels = [p for p in [
        (f"miplib_{date}",               "MIPLIB"),
        (f"mittelmann_{date}",           "Mittelmann"),
        (f"netlib_infeas_{infeas_date}", "Netlib (infeasible)"),
        (f"miplib_infeas_{infeas_date}", "MIPLIB (infeasible)"),
    ] if list_instances(p[0])]
    sys.stderr.write(f"[trajectories] date={date}  infeas_date={infeas_date}  panels={len(panels)}\n")
    if not panels:
        return
    label = iso(date) if infeas_date == date else f"{iso(date)} (infeasible suites: {iso(infeas_date)})"
    footer = f"cuHSDLP experiment {label}  ·  default solver parameters"
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        figures = {column: plt.figure(figsize=(11.5, 4.2 * len(panels))) for column in COLUMNS}
        grids = {column: fig.add_gridspec(len(panels), 1, hspace=0.42, left=0.10, right=0.96, top=0.92, bottom=0.04)
                 for column, fig in figures.items()}
        for i, (bench_dir, title) in enumerate(panels):
            axes = {column: figures[column].add_subplot(grids[column][i]) for column in COLUMNS}
            peaks, legend = plot_panel(axes, bench_dir)
            style_tau_panel(axes["tau"], title, peaks["tau"], legend["tau"])
            style_omega_panel(axes["omega"], title, peaks["omega"], legend["omega"])
        save(figures["tau"], output_dir / f"tau_regime_triptych_{date}",
             r"$\tau$-trajectory by benchmark suite", footer)
        save(figures["omega"], output_dir / f"omega_regime_triptych_{date}",
             r"$\omega$-trajectory by benchmark suite", f"{footer}  ·  $\\omega$ = restart primal weight, as traced")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, help="date of the MIPLIB and Mittelmann runs, e.g. 20260712")
    p.add_argument("--infeas-date", default=None, help="date of the infeasible-suite runs, default --date")
    p.add_argument("--output-dir", type=Path, default=FIGURE_DIR)
    args = p.parse_args()
    run(args.date, args.infeas_date, args.output_dir)


if __name__ == "__main__":
    main()
