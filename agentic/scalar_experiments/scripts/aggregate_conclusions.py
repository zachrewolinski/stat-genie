#!/usr/bin/env python3
"""
Aggregate scalar conclusions into one CSV.

Expected layout:
    outputs/dataset/distribution/perturbation/runN/conclusion.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR
RUN_DIR_PATTERN = re.compile(r"^run(\d+)$", re.IGNORECASE)

BASE_COLUMNS = ["dataset", "distribution", "perturbation", "run_id"]
CONCLUSION_SCHEMA: list[tuple[str, str]] = [
    ("response", "score_0_100"),
    ("explanation", "string"),
]


def _is_valid_score_0_100(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and 0 <= numeric <= 100


def _normalize_for_csv(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def _coerce_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip()


def _load_conclusion_json(raw_text: str) -> dict[str, Any] | None:
    text = raw_text.strip()
    if not text:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Some runs emit doubly-escaped quotes inside explanation strings
        # (e.g., \\"No\\"). Collapse to standard JSON escaping (\").
        repaired_text = text.replace("\\\\\"", "\\\"")
        try:
            data = json.loads(repaired_text)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    return data


def _validate_field(field_name: str, field_type: str, value: Any) -> tuple[bool, Any]:
    if field_type == "score_0_100":
        if not _is_valid_score_0_100(value):
            return False, None
        return True, _normalize_for_csv(value)

    if field_type == "string":
        return True, _coerce_to_string(value)

    raise ValueError(f"Unsupported field_type={field_type} for field={field_name}")


def _parse_conclusion(
    raw_text: str,
    schema: list[tuple[str, str]],
) -> dict[str, Any] | None:
    data = _load_conclusion_json(raw_text)
    if data is None:
        return None

    row: dict[str, Any] = {}
    for field_name, field_type in schema:
        if field_name not in data:
            return None
        ok, normalized_value = _validate_field(field_name, field_type, data[field_name])
        if not ok:
            return None
        row[field_name] = normalized_value

    # Carry through any additional conclusion fields.
    for field_name, raw_value in data.items():
        if field_name in row:
            continue
        row[field_name] = _normalize_for_csv(raw_value)

    return row


def _discover_runs(outputs_dir: Path) -> list[tuple[str, str, str, int, Path]]:
    runs: list[tuple[str, str, str, int, Path]] = []
    if not outputs_dir.exists():
        return runs

    for dataset_dir in sorted(path for path in outputs_dir.iterdir() if path.is_dir()):
        for distribution_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for perturbation_dir in sorted(path for path in distribution_dir.iterdir() if path.is_dir()):
                for run_dir in sorted(path for path in perturbation_dir.iterdir() if path.is_dir()):
                    match = RUN_DIR_PATTERN.match(run_dir.name)
                    if not match:
                        continue
                    conclusion_path = run_dir / "conclusion.txt"
                    if conclusion_path.exists() and conclusion_path.is_file():
                        runs.append(
                            (
                                dataset_dir.name,
                                distribution_dir.name,
                                perturbation_dir.name,
                                int(match.group(1)),
                                conclusion_path,
                            )
                        )
    return runs


def _write_csv(
    rows: list[dict[str, Any]],
    schema: list[tuple[str, str]],
    output_dir: Path,
) -> Path:
    required_conclusion_columns = [field_name for field_name, _ in schema]
    extra_columns: list[str] = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in BASE_COLUMNS and key not in required_conclusion_columns
        }
    )
    fieldnames = BASE_COLUMNS + required_conclusion_columns + extra_columns

    output_path = output_dir / "aggregated_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _print_invalid_conclusions(invalid_entries: list[dict[str, Any]]) -> None:
    if not invalid_entries:
        return

    print(f"Skipped {len(invalid_entries)} invalid conclusions:")
    for idx, entry in enumerate(invalid_entries, start=1):
        path = entry["path"]
        reason = entry.get("reason", "").strip()
        content = entry.get("content")

        print(f"  [{idx}] path: {path}")
        if reason:
            print(f"      reason: {reason}")
        print("      conclusion:")

        if content is None:
            print("          <unreadable file>")
        else:
            stripped = content.strip("\n")
            if stripped == "":
                print("          <empty>")
            else:
                for line in stripped.splitlines():
                    print(f"          {line}")
        print()


def aggregate_all_runs(outputs_dir: Path, output_dir: Path) -> tuple[int, int]:
    runs = _discover_runs(outputs_dir)
    if not runs:
        print(f"No conclusions found in {outputs_dir}.")
        return 0, 0

    valid_rows: list[dict[str, Any]] = []
    invalid_entries: list[dict[str, Any]] = []
    schema = CONCLUSION_SCHEMA

    for dataset, distribution, perturbation, run_id, conclusion_path in runs:
        try:
            raw_conclusion = conclusion_path.read_text(encoding="utf-8")
        except OSError as exc:
            invalid_entries.append(
                {
                    "path": conclusion_path,
                    "reason": f"failed to read file ({exc})",
                    "content": None,
                }
            )
            continue

        parsed = _parse_conclusion(
            raw_conclusion,
            schema=schema,
        )

        if parsed is None:
            invalid_entries.append(
                {
                    "path": conclusion_path,
                    "reason": "invalid JSON or failed conclusion schema validation",
                    "content": raw_conclusion,
                }
            )
            continue

        valid_rows.append(
            {
                "dataset": dataset,
                "distribution": distribution,
                "perturbation": perturbation,
                "run_id": run_id,
                **parsed,
            }
        )

    valid_rows.sort(
        key=lambda row: (row["dataset"], row["distribution"], row["perturbation"], row["run_id"])
    )
    invalid_entries.sort(key=lambda entry: str(entry["path"]))

    if valid_rows:
        output_path = _write_csv(valid_rows, schema, output_dir)
        print(f"Wrote {len(valid_rows)} rows to {output_path}")
    else:
        print("No valid conclusions found.")

    _print_invalid_conclusions(invalid_entries)

    return len(valid_rows), len(invalid_entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate scalar experiment conclusions into a single CSV."
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
