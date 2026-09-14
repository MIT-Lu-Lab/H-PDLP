"""Paths and parsers shared by the result pipelines."""
import csv
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parents[1] / "result"
FIGURE_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_summary(path):
    """Key/value pairs of a cuHSDLP *_summary.txt, values kept as strings."""
    with open(path) as f:
        return {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in f if ":" in line)}
