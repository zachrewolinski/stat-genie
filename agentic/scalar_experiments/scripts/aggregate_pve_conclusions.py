#!/usr/bin/env python3
"""
Aggregate PVE experiment conclusions into one CSV.

Expected layout:
    outputs/{dataset}/pve/pve_{X}/{perturbation}/run{N}/conclusion.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

from aggregate_conclusions import (
    CONCLUSION_SCHEMA,
    _load_conclusion_json,
    _parse_conclusion,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR

PVE_LEVEL_PATTERN = re.compile(r"^pve_([\d.]+)$")
RUN_DIR_PATTERN = re.compile(r"^run(\d+)$", re.IGNORECASE)

BASE_COLUMNS = ["dataset", "pve_level", "perturbation", "run_id"]


def _discover_pve_runs(
    outputs_dir: Path,
) -> list[tuple[str, str, str, int, Path]]:
    """Walk outputs/{dataset}/pve/pve_{X}/{perturbation}/run{N}/conclusion.txt."""
    runs: list[tuple[str, str, str, int, Path]] = []
    if not outputs_dir.exists():
        return runs

    for dataset_dir in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        pve_dir = dataset_dir / "pve"
        if not pve_dir.is_dir():
            continue
        for level_dir in sorted(p for p in pve_dir.iterdir() if p.is_dir()):
            level_match = PVE_LEVEL_PATTERN.match(level_dir.name)
            if not level_match:
                continue
            pve_level = level_match.group(1)
            for pert_dir in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                for run_dir in sorted(p for p in pert_dir.iterdir() if p.is_dir()):
                    run_match = RUN_DIR_PATTERN.match(run_dir.name)
                    if not run_match:
                        continue
                    conclusion_path = run_dir / "conclusion.txt"
                    if conclusion_path.exists():
                        runs.append((
                            dataset_dir.name,
                            pve_level,
                            pert_dir.name,
                            int(run_match.group(1)),
                            conclusion_path,
                        ))
    return runs


def aggregate_pve_runs(
    outputs_dir: Path, output_dir: Path
) -> tuple[int, int]:
    runs = _discover_pve_runs(outputs_dir)
    if not runs:
        print(f"No PVE conclusions found in {outputs_dir}.")
        return 0, 0

    valid_rows: list[dict[str, Any]] = []
    n_invalid = 0

    for dataset, pve_level, perturbation, run_id, conclusion_path in runs:
        try:
            raw = conclusion_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  skip (unreadable): {conclusion_path} -- {exc}")
            n_invalid += 1
            continue

        parsed = _parse_conclusion(raw, schema=CONCLUSION_SCHEMA)
        if parsed is None:
            print(f"  skip (invalid): {conclusion_path}")
            n_invalid += 1
            continue

        valid_rows.append({
            "dataset": dataset,
            "pve_level": pve_level,
            "perturbation": perturbation,
            "run_id": run_id,
            **parsed,
        })

    valid_rows.sort(
        key=lambda r: (r["dataset"], r["pve_level"], r["perturbation"], r["run_id"])
    )

    if valid_rows:
        conclusion_columns = [name for name, _ in CONCLUSION_SCHEMA]
        extra_columns = sorted({
            k for row in valid_rows for k in row
            if k not in BASE_COLUMNS and k not in conclusion_columns
        })
        fieldnames = BASE_COLUMNS + conclusion_columns + extra_columns

        output_path = output_dir / "aggregated_results" / "aggregated_pve_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_rows)
        print(f"Wrote {len(valid_rows)} rows to {output_path}")
    else:
        print("No valid PVE conclusions found.")

    if n_invalid:
        print(f"Skipped {n_invalid} invalid conclusions.")

    return len(valid_rows), n_invalid


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate PVE experiment conclusions into a single CSV."
    )
    parser.add_argument(
        "--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR,
        help="Path to outputs directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where the aggregated CSV is written.",
    )
    args = parser.parse_args()
    n_valid, n_invalid = aggregate_pve_runs(args.outputs_dir, args.output_dir)
    return 1 if n_valid == 0 and n_invalid == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
