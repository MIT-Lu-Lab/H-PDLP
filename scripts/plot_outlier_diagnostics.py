import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT = os.path.join(os.path.dirname(__file__), "..", "result")
RUN = os.path.join(RESULT, "mittelmann_20260629")
INSTANCES = ["ns1688926", "Dual2_5000", "Linf_520c"]

def load(inst):
    path = os.path.join(RUN, f"{inst}_tau_trace.csv")
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def omega(row):
    sp = float(row["step_size_primal"]); sd = float(row["step_size_dual"])
    return math.sqrt(sd / sp) if sp > 0 and sd > 0 else float("nan")

METRICS = [
    ("tau",   lambda r: float(r["tau"]),               "tau",        False),
    ("omega", omega,                                    "omega",      True),
    ("rel_p", lambda r: float(r["rel_primal_res"]),     "rel primal res", True),
    ("gap",   lambda r: float(r["rel_gap"]),            "rel gap",    True),
]

fig, axes = plt.subplots(len(METRICS), len(INSTANCES),
                         figsize=(16, 11), sharex="col")

for j, inst in enumerate(INSTANCES):
    rows = load(inst)
    iters_rows = [r for r in rows if r["event"] in ("iter", "init", "final")]
    restarts = [int(r["iter"]) for r in rows if r["event"] == "restart"]
    x = [max(int(r["iter"]), 1) for r in iters_rows]
    for i, (key, fn, ylabel, logy) in enumerate(METRICS):
        ax = axes[i][j]
        y = []
        for r in iters_rows:
            try:
                v = fn(r)
            except Exception:
                v = float("nan")
            if logy and (v is None or v <= 0 or not math.isfinite(v)):
                v = float("nan")
            y.append(v)
        ax.plot(x, y, lw=0.7, color="tab:blue")
        for rs in restarts:
            ax.axvline(max(rs, 1), color="red", ls="--", lw=0.5, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(left=200)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="major", alpha=0.3)
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=11)
        if i == 0:
            ax.set_title(f"{inst}  ({len(restarts)} restarts)", fontsize=12, fontweight="bold")
        if i == len(METRICS) - 1:
            ax.set_xlabel("iteration", fontsize=10)

fig.suptitle("Outlier diagnostics: tau / omega / rel_primal_res / rel_gap  (red dashed = restart)",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(RESULT, "figures", "outlier_diagnostics_20260629.png")
fig.savefig(out, dpi=130)
print("wrote", out)
