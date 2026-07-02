#!/usr/bin/env python3.12
"""Trace-based τ + ω trajectory visualisation, one panel per benchmark suite.

Two figures (both rendered together, PDF and PNG):
  tau_regime_triptych_{DATE}.{pdf,png}    — τ trajectory per benchmark
  omega_regime_triptych_{DATE}.{pdf,png}  — ω trajectory per benchmark

Each figure stacks one panel per benchmark suite present for {DATE} — MIPLIB,
Mittelmann, Netlib (infeasible), MIPLIB-infeas (1e-8) — overlaying every
instance's trajectory in that suite, coloured by the instance's termination
status (OPTIMAL / PRIMAL_INFEASIBLE / TIME_LIMIT / …).  Suites with no trace for
{DATE} are dropped, so the main suites (one date) and the infeas sweep (another)
each render their own figure.

The infeas sweep is often run on a different date than the main suites; pass a
second date to source the MIPLIB-infeas (1e-8) panel from it while the main
panels stay on the first date — all four land on one page.

Usage:
    python3.12 tau_story.py                       # default: today (all panels)
    python3.12 tau_story.py 20260622              # main + infeas both on this date
    python3.12 tau_story.py 20260622 20260623     # main on 0622, infeas on 0623
    DATE=20260622 INFEAS_DATE=20260623 python3.12 tau_story.py   # via env

Reads from   result/{bench}_{DATE}/{instance}_tau_trace.csv   (per-instance trace)
             result/{bench}_{DATE}/{instance}_summary.txt      (termination status)
Writes to    result/figures/{tau,omega}_regime_triptych_{DATE}.{pdf,png}
"""
import csv, math, os, sys, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
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
})

PROJ = Path(__file__).resolve().parent.parent
DATE = (sys.argv[1] if len(sys.argv) > 1
        else os.environ.get("DATE")
        or datetime.date.today().strftime("%Y%m%d"))
# The infeas sweep often lands on its own date; default it to the main DATE.
INFEAS_DATE = (sys.argv[2] if len(sys.argv) > 2
               else os.environ.get("INFEAS_DATE") or DATE)
FIG_DIR = PROJ / "result" / "figures"                  # shared global figures
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Panels, in display order: (key, result-subdir, title).  The MIPLIB-infeas
# (1e-8) panel is the eps-sweep infeasibility experiment.  Panels whose result
# dir holds no trace are dropped, so a date with only some suites present still
# renders (e.g. the main suites on one date, the infeas sweep on another).
BENCH_PANELS = [
    ("miplib",            f"miplib_{DATE}",                   "MIPLIB"),
    ("mittelmann",        f"mittelmann_{DATE}",               "Mittelmann"),
    ("netlib_infeas",     f"netlib_infeas_{DATE}",                   "Netlib (infeasible)"),
    ("miplib_infeas_1e8", f"miplib_infeas_pbb_eps1e8_{INFEAS_DATE}", "MIPLIB-infeas (1e-8)"),
]
BENCH_PANELS = [p for p in BENCH_PANELS
                if any((PROJ / "result" / p[1]).glob("*_tau_trace.csv"))]
N_PANELS = len(BENCH_PANELS)
sys.stderr.write(f"[tau_story] DATE={DATE}  INFEAS_DATE={INFEAS_DATE}  PROJ={PROJ}  panels={N_PANELS}\n")

def _iso(d):
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"
DATE_LABEL = _iso(DATE) if INFEAS_DATE == DATE else f"{_iso(DATE)} (infeas 1e-8: {_iso(INFEAS_DATE)})"

# Termination-status colours (legend keys).  Unknown reasons fall back to grey.
C_OPT, C_PINF, C_DINF, C_TL, C_ITL = "#1f5fa6", "#e07b00", "#b5179e", "#cc2a36", "#7b1f8a"
REASON_COLOR = {
    "OPTIMAL":           C_OPT,
    "PRIMAL_INFEASIBLE": C_PINF,
    "DUAL_INFEASIBLE":   C_DINF,
    "TIME_LIMIT":        C_TL,
    "ITERATION_LIMIT":   C_ITL,
}
C_OTHER = "#888888"

def reason_color(reason):
    return REASON_COLOR.get(reason, C_OTHER)

Y_CEIL          = 1e6        # τ axis ceiling
# τ collapses to EXACTLY 0 on infeasible rays (82% of netlib_infeas rows are
# pdhg_tau==0).  A pure log axis can't render 0, so we use symlog: a linear
# band [0, TAU_LINTHRESH] lets collapsed trajectories land on a true 0 baseline,
# while everything above stays ordinary log (smallest positive τ is 8.17e-9, so
# linthresh=1e-9 keeps all real data in the log region — only zeros drop linear).
TAU_LINTHRESH   = 1e-9
ALPHA_UNIFORM = 0.35
LW_UNIFORM    = 1.2

# ---------- helpers --------------------------------------------------------
def list_instances(bench_dir):
    """Instance stems that have a τ-trace CSV in this benchmark's result dir."""
    d = PROJ / "result" / bench_dir
    if not d.is_dir():
        return []
    suffix = "_tau_trace.csv"
    return sorted(p.name[:-len(suffix)] for p in d.glob(f"*{suffix}"))

def read_reason(inst, bench_dir):
    """Termination Reason from {inst}_summary.txt, or 'UNKNOWN' if absent."""
    f = PROJ / "result" / bench_dir / f"{inst}_summary.txt"
    if not f.exists():
        return "UNKNOWN"
    with f.open() as h:
        for line in h:
            if line.startswith("Termination Reason"):
                return line.split(":", 1)[1].strip()
    return "UNKNOWN"

# ---------- streamed loaders with peak detection ---------------------------
DOWNSAMPLE_THRESHOLD = 50000   # load full trace below this; downsample above

def stream_trace(inst, bench_dir, n_keep=5000):
    """Return (iter_arr, tau_arr, iter_at_max, max_tau).
    Single pass — load ALL rows for traces ≤ DOWNSAMPLE_THRESHOLD; for huge
    traces downsample but force the max-τ row into the keep set so brief
    spikes survive."""
    f = PROJ / "result" / bench_dir / f"{inst}_tau_trace.csv"
    if not f.exists(): return None, None, None, None
    all_iters = []; all_taus = []
    with f.open() as h:
        r = csv.DictReader(h)
        for row in r:
            try:
                all_iters.append(float(row["iter"]))
                all_taus .append(float(row["pdhg_tau"]))
            except (ValueError, KeyError):
                continue
    if not all_iters: return None, None, None, None
    its_arr  = np.asarray(all_iters)
    taus_arr = np.asarray(all_taus)
    # A torn trace (an NFS write dropping a leading digit, e.g. 367957200 ->
    # 67957200) leaves a row whose iter falls below the running max.  On a log
    # x-axis that paints a backward-jumping garbage line — it is why long
    # TIME_LIMIT instances (e.g. Linf_520c) plotted as noise.  Keep only rows
    # at or above the running max so the single real trajectory survives.
    if its_arr.size > 1 and np.any(np.diff(its_arr) < 0):
        mask = its_arr >= np.maximum.accumulate(its_arr)
        sys.stderr.write(f"[torn-trace] {bench_dir}/{inst}: drop {int((~mask).sum())} corrupt row(s)\n")
        its_arr = its_arr[mask]; taus_arr = taus_arr[mask]
    max_row_idx = int(np.argmax(taus_arr))
    max_tau     = float(taus_arr[max_row_idx])
    iter_at_max = float(its_arr [max_row_idx])
    n_total     = len(its_arr)
    if n_total <= DOWNSAMPLE_THRESHOLD:
        return its_arr, taus_arr, iter_at_max, max_tau
    # Only huge traces get downsampled — force max-row inclusion
    keep_idx = set(np.unique(np.round(
        np.logspace(0, math.log10(n_total-1), n_keep)).astype(int)))
    keep_idx.update({max_row_idx, max(0, max_row_idx-1),
                     min(n_total-1, max_row_idx+1)})
    keep_sorted = np.array(sorted(keep_idx))
    return its_arr[keep_sorted], taus_arr[keep_sorted], iter_at_max, max_tau

def stream_omega_trace(inst, bench_dir, n_keep=5000):
    """Load full ω = sqrt(step_d/step_p) trajectory from a tau_trace CSV.
    Same downsampling rule as stream_trace; force max-row inclusion."""
    f = PROJ / "result" / bench_dir / f"{inst}_tau_trace.csv"
    if not f.exists(): return None, None, None, None
    iters = []; omegas = []
    with f.open() as h:
        r = csv.DictReader(h)
        for row in r:
            try:
                sp = float(row["step_size_primal"])
                sd = float(row["step_size_dual"])
                if sp > 0 and sd > 0:
                    iters.append(float(row["iter"]))
                    omegas.append(math.sqrt(sd / sp))
            except (ValueError, KeyError, TypeError):
                continue
    if not iters: return None, None, None, None
    its  = np.asarray(iters)
    oms  = np.asarray(omegas)
    if its.size > 1 and np.any(np.diff(its) < 0):     # drop torn-trace rows (see stream_trace)
        mask = its >= np.maximum.accumulate(its)
        sys.stderr.write(f"[torn-trace] {bench_dir}/{inst}: drop {int((~mask).sum())} corrupt row(s)\n")
        its = its[mask]; oms = oms[mask]
    max_row_idx = int(np.argmax(oms))
    max_om      = float(oms[max_row_idx])
    iter_at_max = float(its[max_row_idx])
    n_total = len(its)
    if n_total <= DOWNSAMPLE_THRESHOLD:
        return its, oms, iter_at_max, max_om
    keep = set(np.unique(np.round(np.logspace(0, math.log10(n_total-1), n_keep)).astype(int)))
    keep.update({max_row_idx, max(0, max_row_idx-1), min(n_total-1, max_row_idx+1)})
    sel = np.array(sorted(keep))
    return its[sel], oms[sel], iter_at_max, max_om


def status_legend(ax, reasons_seen):
    """Add a legend mapping the termination statuses present in this panel."""
    if not reasons_seen:
        return
    handles = [mpatches.Patch(color=c, label=r)
               for r, c in sorted(reasons_seen.items())]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.92)


# =============================================================================
# FIG 1 -- τ trajectory, one panel per benchmark
# =============================================================================
def overlay_tau_panel(ax, bench_dir, panel_title):
    """Overlay every instance's τ trajectory, coloured by termination status.
    Annotate upward spikes that exceed Y_CEIL with their actual max value."""
    n_plotted = 0
    up_spikes = []          # (iter_at_max, max_tau, color)
    reasons_seen = {}
    for inst in list_instances(bench_dir):
        its, taus, it_max, mx = stream_trace(inst, bench_dir)
        if its is None: continue
        reason = read_reason(inst, bench_dir)
        color  = reason_color(reason)
        reasons_seen[reason] = color
        # Only clip the top (spikes annotated separately); zeros plot natively
        # in the symlog linear band — no lower clip, so τ=0 shows as true 0.
        # OPTIMAL runs pile into a dense bundle at τ≈1; draw non-OPTIMAL on top
        # (zorder only, same width/alpha) so they aren't hidden under the bundle.
        tau_clip = np.minimum(taus, Y_CEIL)
        ax.plot(its, tau_clip, color=color, lw=LW_UNIFORM, alpha=ALPHA_UNIFORM,
                zorder=2 if reason == "OPTIMAL" else 5)
        n_plotted += 1
        if mx > Y_CEIL:
            up_spikes.append((it_max, mx, color))
    ax.set_xscale("log")
    # linscale=3 → the linear band [0, linthresh] spans one 3-decade "cell",
    # matching the log tick spacing (1e-9→1e-6→…) so every grid interval is the
    # same visual height.  Floor at -linthresh adds one more identical cell below
    # zero, so the τ=0 collapse baseline floats one cell above the axis frame.
    ax.set_yscale("symlog", linthresh=TAU_LINTHRESH, linscale=3.0)
    ax.set_ylim(-TAU_LINTHRESH, Y_CEIL)
    ax.set_yticks([0, 1e-9, 1e-6, 1e-3, 1e0, 1e3, 1e6])
    ax.axhline(0.0, color="0.45", lw=0.8, ls=":", alpha=0.7)  # τ→0 reference
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.set_title(f"{panel_title}  (n={n_plotted})", pad=8, loc="left")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\tau$")
    ax.grid(True, alpha=0.25, which="both")

    # Upward spike labels — just the number, staggered along y
    up_spikes.sort(key=lambda t: t[0])
    for k, (it_max, mx, color) in enumerate(up_spikes):
        ax.annotate(f"{mx:.0e}",
                    xy=(it_max, Y_CEIL),
                    xytext=(0, -4 - 14 * (k % 3)), textcoords="offset points",
                    fontsize=9, ha="center", va="top", color=color,
                    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec=color, lw=0.6, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7, alpha=0.6))
    status_legend(ax, reasons_seen)
    return n_plotted

fig = plt.figure(figsize=(11.5, 4.2 * N_PANELS))
gs  = fig.add_gridspec(N_PANELS, 1, hspace=0.42, left=0.10, right=0.96, top=0.92, bottom=0.04)
for i, (_key, dir_, title) in enumerate(BENCH_PANELS):
    overlay_tau_panel(fig.add_subplot(gs[i]), dir_, title)

fig.suptitle(r"$\tau$-trajectory by benchmark suite",
             fontsize=16, fontweight="bold", y=0.985)
fig.text(0.5, 0.005,
         f"cuHSDLP experiment {DATE_LABEL}  ·  default solver parameters",
         ha="center", fontsize=9.5, color="grey")
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"tau_regime_triptych_{DATE}.{ext}", bbox_inches="tight")
sys.stderr.write(f"Saved {FIG_DIR/f'tau_regime_triptych_{DATE}.[pdf,png]'}\n")
plt.close(fig)


# =============================================================================
# FIG 2 -- ω trajectory, one panel per benchmark (parallel to the τ figure)
# =============================================================================
OMEGA_Y_FLOOR    = 1e-10         # axis floor
OMEGA_Y_CEIL     = 1e14          # axis ceiling (data max ~4.4e13; old 1e8 truncated it)
OMEGA_DATA_FLOOR = OMEGA_Y_FLOOR * 10.0   # nudge above axis so floor lines visible

def overlay_omega_panel(ax, bench_dir, panel_title):
    """Overlay every instance's ω trajectory, coloured by termination status."""
    n_plotted = 0
    reasons_seen = {}
    for inst in list_instances(bench_dir):
        its, oms, _, _ = stream_omega_trace(inst, bench_dir)
        if its is None: continue
        reason = read_reason(inst, bench_dir)
        color  = reason_color(reason)
        reasons_seen[reason] = color
        oms_clip = np.clip(oms, OMEGA_DATA_FLOOR, OMEGA_Y_CEIL)
        ax.plot(its, oms_clip, color=color, lw=LW_UNIFORM, alpha=ALPHA_UNIFORM,
                zorder=2 if reason == "OPTIMAL" else 5)
        n_plotted += 1
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(OMEGA_Y_FLOOR, OMEGA_Y_CEIL)
    ax.set_title(f"{panel_title}  (n={n_plotted})", pad=8, loc="left")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\omega$  (primal weight)")
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.5)
    ax.grid(True, alpha=0.25, which="both")
    status_legend(ax, reasons_seen)
    return n_plotted

fig = plt.figure(figsize=(11.5, 4.2 * N_PANELS))
gs  = fig.add_gridspec(N_PANELS, 1, hspace=0.42, left=0.10, right=0.96, top=0.92, bottom=0.04)
for i, (_key, dir_, title) in enumerate(BENCH_PANELS):
    overlay_omega_panel(fig.add_subplot(gs[i]), dir_, title)

fig.suptitle(r"$\omega$-trajectory by benchmark suite",
             fontsize=16, fontweight="bold", y=0.985)
fig.text(0.5, 0.005,
         f"cuHSDLP experiment {DATE_LABEL}  ·  default solver parameters  ·  $\\omega = \\sqrt{{\\sigma_d / \\eta}}$ recovered per row",
         ha="center", fontsize=9.5, color="grey")
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"omega_regime_triptych_{DATE}.{ext}", bbox_inches="tight")
sys.stderr.write(f"Saved {FIG_DIR/f'omega_regime_triptych_{DATE}.[pdf,png]'}\n")
plt.close(fig)

print("DONE")
