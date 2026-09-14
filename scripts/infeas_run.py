#!/usr/bin/env python3.12
"""Run one baseline solver over a benchmark dataset, resumable and shardable across Slurm tasks."""
import argparse
import csv
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from common import RESULT_DIR, read_csv_rows
from infeas_datasets import (DATASETS, DatasetSpec, dataset_files, expected_tags, instance_tag,
                             instance_time_limit, manifest_tags, read_nonempty_lines)
from infeas_solvers import SOLVERS, THREADS, SolveResult, solve

# A claim older than TL + slack belongs to a task Slurm killed; a finished sibling's claim stays as its done-marker.
CLAIM_STALE_SLACK = 3600.0
# An ERROR status carries the exception text, which may span several lines.
STATUS = re.compile(r"^(?:[A-Za-z][A-Za-z_]*|code_\d+|ERROR:.*)$", re.DOTALL)


@dataclass(frozen=True)
class RunConfig:
    solver: str
    dataset: str
    spec: DatasetSpec
    mode: str
    eps: float | None
    date: str
    task: int
    ntasks: int
    remaining_file: Path | None
    base: str                  # result-file stem shared by the merged CSV, the task files and the claims
    status_col: str
    fieldnames: list[str]
    output_csv: Path
    merged_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", nargs="?", default=time.strftime("%Y%m%d"))
    parser.add_argument("--solver", required=True, choices=sorted(SOLVERS))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--mode", choices=["raw", "pslp"], default="raw")
    # Without --eps every solver keeps its own termination criteria, which are not commensurable across solvers.
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--remaining_file", default=None, help="file of MPS paths to process before sharding")
    return parser.parse_args()


def build_config(solver: str, dataset: str, date: str, mode: str = "raw", eps: float | None = None,
                 task: int = 1, ntasks: int = 1, remaining_file: str | None = None) -> RunConfig:
    if ntasks < 1:
        raise SystemExit("--ntasks must be >= 1")
    if not 1 <= task <= ntasks:
        raise SystemExit("--task must satisfy 1 <= task <= ntasks")

    spec = DATASETS[dataset]
    suffix = "pslp" if mode == "pslp" else "nopre"
    base = result_base(solver, dataset, suffix, eps)
    status_col = f"{solver}_status"
    # "elapsed" is total wall clock, "solve_time" the optimize call alone as the GPU solvers report it, "node" the machine.
    fieldnames = ["instance", status_col, "iters", "elapsed", "solve_time"]
    fieldnames += ["objective", "node"] if spec.kind == "opt" else ["node"]
    return RunConfig(
        solver=solver, dataset=dataset, spec=spec, mode=mode, eps=eps, date=date,
        task=task, ntasks=ntasks, remaining_file=Path(remaining_file) if remaining_file else None,
        base=base, status_col=status_col, fieldnames=fieldnames,
        output_csv=output_csv_path(base, date, task, ntasks), merged_csv=RESULT_DIR / f"{base}.csv",
    )


def eps_tag(eps: float | None) -> str:
    if eps is None:
        return "default"
    return "eps" + f"{eps:.0e}".replace("e-0", "e").replace("e-", "e")   # 1e-4 -> eps1e4


# Result-file stem: infeas_<solver>_<dataset>_<suffix>[_eps<tag>], or opt_<solver>_<dataset>_<suffix>_eps<tag>.
def result_base(solver: str, dataset: str, suffix: str, eps: float | None) -> str:
    if DATASETS[dataset].kind == "opt":
        return f"opt_{solver}_{dataset}_{suffix}_{eps_tag(eps)}"
    # An explicit --eps gets its own files, or the run would resume from the default-tolerance results.
    return f"infeas_{solver}_{dataset}_{suffix}" + ("" if eps is None else f"_{eps_tag(eps)}")


# Task files carry the Slurm job id: concurrent arrays repeat task indices, and NFS appends from two nodes tear rows.
def output_csv_path(base: str, date: str, task: int, ntasks: int) -> Path:
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID", "")
    if ntasks == 1 and not job:
        return RESULT_DIR / f"{base}.csv"
    stem = f"{base}_{date}_task{task}of{ntasks}"
    return RESULT_DIR / (f"{stem}_j{job}.csv" if job else f"{stem}.csv")


# Task files of one experiment across dates; _oomfail/_crash/_walltimeout rows are recorded after a task was killed.
def task_output_files(base: str) -> list[Path]:
    patterns = ("_*_task*of*.csv", "_*_oomfail.csv", "_*_crash.csv", "_*_walltimeout.csv")
    # An explicit --eps extends the stem with _eps<tag>, so those files belong to another experiment.
    return sorted({path for pattern in patterns for path in RESULT_DIR.glob(base + pattern)
                   if not path.name[len(base):].startswith("_eps")})


# Every result file of the experiment: the merged CSV when present, then the task files.
def result_files(config: RunConfig) -> list[Path]:
    return ([config.merged_csv] if config.merged_csv.exists() else []) + task_output_files(config.base)


# Instances with a row in any result file, which a resume skips.
def load_done_keys(config: RunConfig) -> set[str]:
    return {row["instance"] for path in result_files(config) for row in read_csv_rows(path)}


def bad_row(row: dict[str, str], status_col: str, tags: set[str]) -> str | None:
    """Why a result row is malformed, or None when it is sound."""
    status = row.get(status_col)
    if status is None or not STATUS.match(status):
        return "status"
    instance = (row.get("instance") or "").strip()
    if not instance:
        return "instance"
    # A line torn by a concurrent NFS append keeps NUL padding inside the instance name.
    if any(ch < " " for ch in instance):
        return "torn instance name"
    if instance not in tags:
        return "unknown instance"
    if any(k is None for k in row):        # more fields on the line than the header
        return "extra fields"
    for numeric in ("iters", "elapsed"):
        if (row.get(numeric) or "").strip():
            try:
                float(row[numeric])
            except ValueError:
                return numeric
    return None


# One row per instance from result CSVs read oldest first, so the latest run wins; malformed rows are skipped.
def merged_rows(paths: list[Path], config: RunConfig, tags: set[str]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in sorted(paths, key=lambda p: p.stat().st_mtime):
        for row in read_csv_rows(path):
            if bad_row(row, config.status_col, tags):
                print(f"  skipping malformed row in {path.name}: {str(row.get('instance'))[:40]!r}")
                continue
            rows[row["instance"]] = row
    return rows


# Unfinished instances of either the provided list or the whole arm, in this task's sweep order.
def select_shard(config: RunConfig) -> tuple[list[Path], int]:
    if config.remaining_file:
        files = [Path(line) for line in read_nonempty_lines(config.remaining_file)]
    else:
        files = dataset_files(config.spec, config.mode)
    done = load_done_keys(config)
    remaining = [path for path in files if instance_tag(path) not in done]
    return sweep_order(remaining, config.task, config.ntasks), len(files)


# Sweep the whole remaining list from a staggered start, odd tasks forward and even tasks backward.
def sweep_order(files: list[Path], task: int, ntasks: int) -> list[Path]:
    start = (task - 1) * len(files) // ntasks
    if task % 2 == 1:
        return files[start:] + files[:start]
    return files[start::-1] + files[:start:-1]


# Per-date claim dir; mkdir is atomic (also on NFS), so exactly one task wins.
def claim_path(config: RunConfig, tag: str) -> Path:
    return RESULT_DIR / f".claim_{config.base}_{config.date}" / tag


def try_claim(config: RunConfig, tag: str) -> bool:
    path = claim_path(config, tag)
    try:
        path.mkdir(parents=True)
        return True
    except FileExistsError:
        return steal_stale_claim(config, tag, path)


def steal_stale_claim(config: RunConfig, tag: str, path: Path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        age = 0.0
    if age <= instance_time_limit(config.spec, tag) + CLAIM_STALE_SLACK or tag in load_done_keys(config):
        return False
    print(f"  {tag}: stealing stale claim (age {age / 3600:.1f}h)")
    path.touch()   # take ownership so siblings do not steal it at the same time
    return True


# Only called when no row was written, so a later task can retry.
def release_claim(config: RunConfig, tag: str) -> None:
    try:
        claim_path(config, tag).rmdir()
    except OSError:
        pass


def make_row(config: RunConfig, instance: str, result: SolveResult) -> dict[str, object]:
    row: dict[str, object] = {
        "instance": instance,
        config.status_col: result.status,
        "iters": result.iters,
        "elapsed": result.elapsed,
        "solve_time": "" if result.solve_elapsed is None else result.solve_elapsed,
        "node": os.environ.get("SLURMD_NODENAME", ""),
    }
    if "objective" in config.fieldnames:
        row["objective"] = "" if result.objective is None else f"{result.objective:.12g}"
    return row


def process_shard(config: RunConfig, shard: list[Path]) -> int:
    """Append a row per claimed instance to this task's result file and return the file's row count."""
    done = {row["instance"] for row in read_csv_rows(config.output_csv)} if config.output_csv.exists() else set()
    if done:
        print(f"  loaded {len(done)} existing results")
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.output_csv.open("a" if done else "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=config.fieldnames)
        if not done:
            writer.writeheader()
        if config.mode == "pslp" and config.task == 1:
            seed_manifest_rows(config, done, writer)
            fh.flush()

        for mps_path in shard:
            tag = instance_tag(mps_path)
            if tag in done or not try_claim(config, tag):
                continue   # already in this file, or a sibling task owns (or already solved) it
            try:
                # Flushed now so a solver dying on a signal still leaves the instance name in the log.
                print(f"  {tag}: solving", flush=True)
                result = solve(config.solver, mps_path, instance_time_limit(config.spec, tag), config.eps)
                writer.writerow(make_row(config, tag, result))
                fh.flush()
            except BaseException:
                release_claim(config, tag)
                raise
            done.add(tag)
            solve_label = "" if result.solve_elapsed is None else f" solve={result.solve_elapsed:.2f}s"
            obj_label = "" if result.objective is None else f"  obj={result.objective:.10g}"
            print(f"  {tag}: {result.status}  iters={result.iters}  t={result.elapsed:.2f}s{solve_label}{obj_label}")
    return len(done)


# PSLP presolve already settled these instances, so they are seeded instead of solved.
def seed_manifest_rows(config: RunConfig, done: set[str], writer: csv.DictWriter) -> None:
    before = len(done)
    for tag, status in manifest_tags(config.spec):
        if tag not in done:
            writer.writerow(make_row(config, tag, SolveResult(status, 0)))
            done.add(tag)
    print(f"  wrote {len(done) - before} presolve-detected rows")


def maybe_merge_outputs(config: RunConfig) -> None:
    """Rewrite the merged CSV from every result file once each expected instance has a row."""
    # Only a run that wrote straight into the merged file has nothing to merge; ntasks=1 jobs may write task files.
    if config.output_csv == config.merged_csv:
        return
    task_files = task_output_files(config.base)
    if not task_files:
        return
    tags = expected_tags(config.spec, config.mode)
    rows = merged_rows(result_files(config), config, tags)
    if len(rows) < len(tags):
        print(f"merge pending: {len(rows)}/{len(tags)} rows across {len(task_files)} files")
        return
    with config.merged_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=config.fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda row: row["instance"]))
    print(f"merged -> {config.merged_csv}  ({len(rows)} rows)")


def print_run_summary(config: RunConfig, shard_size: int, total_size: int) -> None:
    tl_label = "class-based" if config.spec.time_limit is None else f"{config.spec.time_limit:g}s"
    eps_label = "solver default" if config.eps is None else f"{config.eps:g}"
    print(f"{config.solver} {config.mode} {config.dataset} | task {config.task}/{config.ntasks} | "
          f"{shard_size}/{total_size} instances | TL={tl_label} | threads={THREADS} | eps={eps_label}")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    # Turn Slurm's SIGTERM into SystemExit so the in-flight claim is released.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(143))
    config = build_config(**vars(parse_args()))
    shard, total = select_shard(config)
    print_run_summary(config, len(shard), total)
    rows = process_shard(config, shard)
    print(f"wrote {config.output_csv}  ({rows} rows)")
    maybe_merge_outputs(config)


if __name__ == "__main__":
    main()
