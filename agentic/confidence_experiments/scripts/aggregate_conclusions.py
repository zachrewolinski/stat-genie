#!/usr/bin/env python3
"""
Aggregate confidence experiment results into one CSV.

Each run directory contains both conclusion.txt and confidence.txt.
This script reads both, renames their explanation fields to avoid
collision, and writes a single row per run.

Expected layout:
    outputs/{dataset}/{distribution}/{perturbation}/run{N}/
        conclusion.txt   ->  {"response": N, "explanation": "..."}
        confidence.txt   ->  {"confidence": N, "explanation": "..."}
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

# Import parsing utilities from the scalar_experiments aggregation script.
_SCALAR_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scalar_experiments" / "scripts"
sys.path.insert(0, str(_SCALAR_SCRIPTS))
from aggregate_conclusions import _parse_conclusion  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR
RUN_DIR_PATTERN = re.compile(r"^run(\d+)$", re.IGNORECASE)

BASE_COLUMNS = ["dataset", "distribution", "perturbation", "run_id"]
CONCLUSION_SCHEMA: list[tuple[str, str]] = [
    ("response", "score_0_100"),
    ("explanation", "string"),
]
CONFIDENCE_SCHEMA: list[tuple[str, str]] = [
    ("confidence", "score_0_100"),
    ("explanation", "string"),
]
REQUIRED_COLUMNS = [
    "response",
    "response_explanation",
    "confidence",
    "confidence_explanation",
]


def _discover_runs(
    outputs_dir: Path,
) -> list[tuple[str, str, str, int, Path]]:
    """Walk outputs/{dataset}/{distribution}/{perturbation}/run{N}/."""
    runs: list[tuple[str, str, str, int, Path]] = []
    if not outputs_dir.exists():
        return runs

    for dataset_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        for distribution_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            # Skip pve directories -- no confidence files there yet.
            if distribution_dir.name == "pve":
                continue
            for perturbation_dir in sorted(
                p for p in distribution_dir.iterdir() if p.is_dir()
            ):
                for run_dir in sorted(
                    p for p in perturbation_dir.iterdir() if p.is_dir()
                ):
                    match = RUN_DIR_PATTERN.match(run_dir.name)
                    if not match:
                        continue
                    runs.append((
                        dataset_dir.name,
                        distribution_dir.name,
                        perturbation_dir.name,
                        int(match.group(1)),
                        run_dir,
                    ))
    return runs


def _read_and_parse(
    path: Path,
    schema: list[tuple[str, str]],
) -> dict[str, Any] | None:
    """Read a JSON file and validate it against the given schema."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_conclusion(raw, schema=schema)


def _parse_and_merge_run(
    run_dir: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse both conclusion.txt and confidence.txt, merge into one row.

    Returns (merged_dict, None) on success, or (None, reason) on failure.
    """
    conclusion_path = run_dir / "conclusion.txt"
    confidence_path = run_dir / "confidence.txt"

    if not conclusion_path.exists():
        return None, "conclusion.txt missing"
    if not confidence_path.exists():
        return None, "confidence.txt missing"

    conclusion = _read_and_parse(conclusion_path, CONCLUSION_SCHEMA)
    if conclusion is None:
        return None, "conclusion.txt invalid"

    confidence = _read_and_parse(confidence_path, CONFIDENCE_SCHEMA)
    if confidence is None:
        return None, "confidence.txt invalid"

    merged: dict[str, Any] = {
        "response": conclusion["response"],
        "response_explanation": conclusion["explanation"],
        "confidence": confidence["confidence"],
        "confidence_explanation": confidence["explanation"],
    }
    return merged, None


def _write_csv(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    extra_columns: list[str] = sorted({
        key
        for row in rows
        for key in row
        if key not in BASE_COLUMNS and key not in REQUIRED_COLUMNS
    })
    fieldnames = BASE_COLUMNS + REQUIRED_COLUMNS + extra_columns

    output_path = output_dir / "aggregated_results" / "aggregated_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def aggregate_all_runs(outputs_dir: Path, output_dir: Path) -> tuple[int, int]:
    runs = _discover_runs(outputs_dir)
    if not runs:
        print(f"No run directories found in {outputs_dir}.")
        return 0, 0

    valid_rows: list[dict[str, Any]] = []
    invalid_entries: list[tuple[Path, str]] = []

    for dataset, distribution, perturbation, run_id, run_dir in runs:
        merged, reason = _parse_and_merge_run(run_dir)
        if merged is None:
            invalid_entries.append((run_dir, reason))
            continue

        valid_rows.append({
            "dataset": dataset,
            "distribution": distribution,
            "perturbation": perturbation,
            "run_id": run_id,
            **merged,
        })

    valid_rows.sort(
        key=lambda r: (r["dataset"], r["distribution"], r["perturbation"], r["run_id"])
    )

    if valid_rows:
        output_path = _write_csv(valid_rows, output_dir)
        print(f"Wrote {len(valid_rows)} rows to {output_path}")
    else:
        print("No valid runs found.")

    if invalid_entries:
        print(f"\nSkipped {len(invalid_entries)} invalid runs:")
        for run_dir, reason in sorted(invalid_entries, key=lambda e: str(e[0])):
            print(f"  {run_dir}: {reason}")

    return len(valid_rows), len(invalid_entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate confidence experiment results into a single CSV."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Path to outputs directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the aggregated CSV file is written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    total_valid, total_invalid = aggregate_all_runs(
        outputs_dir=args.outputs_dir,
        output_dir=args.output_dir,
    )
    return 1 if total_valid == 0 and total_invalid == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
