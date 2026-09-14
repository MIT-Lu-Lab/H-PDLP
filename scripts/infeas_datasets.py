"""Benchmark datasets of the infeasibility pipeline: instance files, PSLP manifests and time limits."""
import glob
from dataclasses import dataclass
from pathlib import Path

LP_ROOT = Path("/nfs/sloanlab007/projects/pdopt_proj/instances/lp")
MIPLIB_INFEAS_ROOT = Path("/home/botang/src/miplib-infeas")
INSTANCE_LISTS = Path("/home/botang/src/cuHSDLP/instances")


@dataclass(frozen=True)
class DatasetSpec:
    raw_root: Path
    pslp_root: Path
    pattern: str = "*.mps.gz"            # "*/*.mps.gz" pools the kind subdirectories
    kind: str = "infeas"                 # "infeas" or "opt" (feasible benchmark)
    time_limit: float | None = 3600.0    # None means MIPLIB class-based limits
    instance_list: Path | None = None    # restrict to the names in this file


DATASETS = {
    # Status detection gets a flat 3600s, the large instances included.
    "miplib": DatasetSpec(MIPLIB_INFEAS_ROOT / "dataset", MIPLIB_INFEAS_ROOT / "dataset_reduced_pslp",
                          pattern="*/*.mps.gz"),
    "netlib": DatasetSpec(LP_ROOT / "netlib_infeas", LP_ROOT / "netlib_infeas_reduced_pslp"),
    # Feasible suites use the instance lists and time limits of benchmark_c_code_pibest.sh.
    "mittelmann": DatasetSpec(LP_ROOT / "mittelmann", LP_ROOT / "mittelmann_reduced_pslp", kind="opt",
                              time_limit=15000.0, instance_list=INSTANCE_LISTS / "mittelmann_instance.txt"),
    "miplib_opt": DatasetSpec(LP_ROOT / "miplib2017", LP_ROOT / "miplib2017_reduced_pslp", kind="opt",
                              time_limit=None, instance_list=INSTANCE_LISTS / "miplib_instance.txt"),
}
INFEAS_DATASETS = tuple(name for name, spec in DATASETS.items() if spec.kind == "infeas")


def read_nonempty_lines(path: Path) -> list[str]:
    with path.open() as fh:
        return [line.strip() for line in fh if line.strip()]


# Tags are file stems and unique across the whole pool (<name>_if / <name>_ub).
def instance_tag(mps_path: Path) -> str:
    return mps_path.name.removesuffix(".mps.gz")


def is_unbounded(tag: str) -> bool:
    return tag.endswith("_ub")


# Class-based limits are looked up by the source MIPLIB name, without the _if/_ub sample suffix.
def instance_time_limit(spec: DatasetSpec, tag: str) -> float:
    if spec.time_limit is None:
        from lp_utils import instance_tl
        return instance_tl(tag.removesuffix("_if").removesuffix("_ub"), True)
    return spec.time_limit


# The suite's instance names, the first token per line (miplib lists carry class/nnz too), or None for every file.
def instance_names(spec: DatasetSpec) -> set[str] | None:
    if spec.instance_list is None:
        return None
    return {line.split()[0] for line in read_nonempty_lines(spec.instance_list)}


def dataset_files(spec: DatasetSpec, mode: str) -> list[Path]:
    """Instance files of the raw or PSLP-reduced arm, limited to the suite's instance list."""
    root = spec.pslp_root if mode == "pslp" else spec.raw_root
    files = sorted(Path(path) for path in glob.glob(str(root / spec.pattern)))
    names = instance_names(spec)
    return files if names is None else [path for path in files if instance_tag(path) in names]


# Feasible-suite manifests do not say which side presolve proved, so they seed INFEASIBLE_OR_UNBOUNDED.
def seeded_status(spec: DatasetSpec, tag: str) -> str:
    if spec.kind == "opt":
        return "INFEASIBLE_OR_UNBOUNDED"
    return "DUAL_INFEASIBLE" if is_unbounded(tag) else "PRIMAL_INFEASIBLE"


def manifest_tags(spec: DatasetSpec) -> list[tuple[str, str]]:
    """(tag, status) of the instances PSLP settled in presolve, from the manifests beside the reduced files."""
    names = instance_names(spec)
    seeded = []
    # Only the feasible suites can have presolve-solved instances.
    for outcome in ("infeasible", "solved") if spec.kind == "opt" else ("infeasible",):
        pattern = spec.pattern.replace("*.mps.gz", f"_manifest_{outcome}.txt")
        for manifest in sorted(glob.glob(str(spec.pslp_root / pattern))):
            for tag in read_nonempty_lines(Path(manifest)):
                if names is None or tag in names:
                    seeded.append((tag, "OPTIMAL" if outcome == "solved" else seeded_status(spec, tag)))
    return seeded


# The instance files plus, in pslp mode, the instances PSLP settled without leaving a reduced file.
def expected_tags(spec: DatasetSpec, mode: str) -> set[str]:
    tags = {instance_tag(path) for path in dataset_files(spec, mode)}
    if mode == "pslp":
        tags |= {tag for tag, _ in manifest_tags(spec)}
    return tags
