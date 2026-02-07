#!/usr/bin/env python3
"""
Collect conclusion.txt from each run, check they're valid scalars in [-100, 100],
and write one CSV (run_id, perturbation, dataset, conclusion).
"""

import csv
import re
import sys
from pathlib import Path
from typing import Optional, Tuple


script_dir = Path(__file__).resolve().parent.parent
outputs_dir = script_dir / "outputs"
output_csv_path = script_dir / "aggregated_results.csv"


def validate_conclusion(content: str) -> Optional[float]:
    """Return the scalar if it's a single number in [-100, 100], else None."""
    content = content.strip()
    if not content:
        return None
    # one number only, no extra punctuation or text
    if not re.match(r"^-?\d+(\.\d+)?$", content):
        return None
    try:
        value = float(content)
    except ValueError:
        return None
    if value < -100 or value > 100:
        return None
    return value


def discover_all_runs() -> list[Tuple[str, str, int, Path]]:
    """Find every run dir under outputs/ that has a conclusion.txt."""
    runs = []
    if not outputs_dir.exists():
        return runs
    # outputs/dataset/perturbation/runN
    for dataset_dir in outputs_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        
        for perturbation_dir in dataset_dir.iterdir():
            if not perturbation_dir.is_dir():
                continue
            
            perturbation_name = perturbation_dir.name
            
            for run_dir in perturbation_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                m = re.match(r"^run(\d+)$", run_dir.name, re.IGNORECASE)
                if not m:
                    continue
                run_number = int(m.group(1))
                conclusion_path = run_dir / "conclusion.txt"
                if conclusion_path.exists():
                    runs.append((dataset_name, perturbation_name, run_number, conclusion_path))
    
    return runs


def aggregate_to_csv(output_path: Path) -> Tuple[int, int]:
    """Collect valid conclusions and write CSV. Returns (valid_count, invalid_count)."""
    runs = discover_all_runs()
    if not runs:
        print(f"No conclusion.txt files found in {outputs_dir}.")
        return 0, 0

    valid_entries = []
    invalid_files = []
    for dataset, perturbation, run_number, conclusion_path in runs:
        try:
            raw = conclusion_path.read_text(encoding="utf-8")
            conclusion_value = validate_conclusion(raw)
        except Exception:
            conclusion_value = None
        if conclusion_value is not None:
            valid_entries.append({
                "run_id": run_number,
                "perturbation": perturbation,
                "dataset": dataset,
                "conclusion": conclusion_value
            })
        else:
            invalid_files.append(conclusion_path)

    valid_entries.sort(key=lambda x: (x["dataset"], x["perturbation"], x["run_id"]))
    invalid_files.sort()

    if valid_entries:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "perturbation", "dataset", "conclusion"])
            writer.writeheader()
            writer.writerows(valid_entries)
        print(f"Wrote {len(valid_entries)} rows to {output_path}")
    else:
        print("No valid conclusions found.")

    if invalid_files:
        print(f"\nSkipped {len(invalid_files)} invalid or missing:")
        for p in invalid_files:
            print(f"  {p}")

    return len(valid_entries), len(invalid_files)


def main() -> int:
    valid_count, invalid_count = aggregate_to_csv(output_csv_path)
    if valid_count == 0 and invalid_count == 0:
        return 1 
    return 0


if __name__ == "__main__":
    sys.exit(main())
