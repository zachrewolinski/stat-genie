#!/usr/bin/env python3
"""
Collect run dirs (each with extracted_analysis.json + extracted_final_conclusion.txt)
into one BLADE-style dir: multirun_analyses.json, final_conclusion_0.txt, etc.
Usable as a single analysis_results_path for the BLADE judge.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RUN_DIR_RE = re.compile(r"^run(\d+)$", re.IGNORECASE)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _infer_dataset_and_perturbation(input_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    # .../outputs/<dataset>/<perturbation>
    perturbation = input_dir.name if input_dir.name else None
    dataset = input_dir.parent.name if input_dir.parent and input_dir.parent.name else None
    return dataset, perturbation


def _discover_run_dirs(input_dir: Path) -> List[Tuple[int, Path]]:
    runs: List[Tuple[int, Path]] = []
    for child in input_dir.iterdir():
        if not child.is_dir():
            continue
        m = RUN_DIR_RE.match(child.name)
        if not m:
            continue
        runs.append((int(m.group(1)), child))
    runs.sort(key=lambda t: t[0])
    return runs


def _extract_analysis_entry(extracted_analysis_json: Dict[str, Any]) -> Dict[str, Any]:
    if "extracted_analysis" not in extracted_analysis_json:
        raise ValueError("Expected key 'extracted_analysis' in extracted_analysis.json payload.")
    entry = extracted_analysis_json["extracted_analysis"]
    if not isinstance(entry, dict):
        raise ValueError("'extracted_analysis' must be an object.")
    for k in ("cvars", "analysis_code"):
        if k not in entry:
            raise ValueError(f"'extracted_analysis' missing required key '{k}'.")
    return {
        "cvars": entry["cvars"],
        "analysis_code": entry["analysis_code"],
    }


def aggregate_runs(
    *,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    require_extracted: bool,
) -> None:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found or not a directory: {input_dir}")

    runs = _discover_run_dirs(input_dir)
    if not runs:
        raise FileNotFoundError(f"No run directories matching 'run<N>' found under: {input_dir}")

    dataset_guess, perturbation_guess = _infer_dataset_and_perturbation(input_dir)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                "Pass --overwrite to replace its contents."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses: Dict[str, Dict[str, Any]] = {}
    conclusions: Dict[str, str] = {}

    dataset_name_from_runs: Optional[str] = None

    for idx, (run_number, run_path) in enumerate(runs):
        extracted_analysis_path = run_path / "extracted_analysis.json"
        extracted_conclusion_path = run_path / "extracted_final_conclusion.txt"

        if require_extracted and (not extracted_analysis_path.exists() or not extracted_conclusion_path.exists()):
            missing = []
            if not extracted_analysis_path.exists():
                missing.append(str(extracted_analysis_path))
            if not extracted_conclusion_path.exists():
                missing.append(str(extracted_conclusion_path))
            raise FileNotFoundError(
                "Missing extracted files for run directory. "
                "Run extract_single_run.py first.\n- " + "\n- ".join(missing)
            )

        extracted_payload = _read_json(extracted_analysis_path)
        entry = _extract_analysis_entry(extracted_payload)

        run_dataset_name = extracted_payload.get("dataset_name")
        if isinstance(run_dataset_name, str) and run_dataset_name.strip():
            if dataset_name_from_runs is None:
                dataset_name_from_runs = run_dataset_name.strip()
            elif dataset_name_from_runs != run_dataset_name.strip():
                raise ValueError(
                    f"Inconsistent dataset_name across runs: '{dataset_name_from_runs}' vs '{run_dataset_name}'."
                )

        conc_text = _read_text(extracted_conclusion_path).strip()
        try:
            json.loads(conc_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Conclusion file is not valid JSON: {extracted_conclusion_path}\n"
                f"JSON error: {e}\n\nContent:\n{conc_text}"
            ) from e

        analyses[str(idx)] = entry
        conclusions[str(idx)] = conc_text

    dataset_name = dataset_name_from_runs or dataset_guess or "unknown_dataset"

    multirun = {
        "dataset_name": dataset_name,
        "n": len(analyses),
        "analyses": analyses,
    }
    (output_dir / "multirun_analyses.json").write_text(
        json.dumps(multirun, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for i, conc in conclusions.items():
        (output_dir / f"final_conclusion_{i}.txt").write_text(conc.strip() + "\n", encoding="utf-8")

    provenance = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "dataset_guess": dataset_guess,
        "perturbation_guess": perturbation_guess,
        "dataset_name_used": dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {
                "analysis_index": idx,
                "run_number": run_number,
                "run_dir": str(run_path),
                "extracted_analysis_json": str(run_path / "extracted_analysis.json"),
                "extracted_final_conclusion_txt": str(run_path / "extracted_final_conclusion.txt"),
            }
            for idx, (run_number, run_path) in enumerate(runs)
        ],
    }
    (output_dir / "source_runs.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-run extractions into a BLADE-style multirun dir.")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing run<N>/ subdirectories (e.g. agentic/experiments/outputs/affairs/anonymize).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory to write BLADE-style files. "
            "Default: agentic/experiments/outputs_extracted/<dataset>/<perturbation>_output"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output dir.")
    parser.add_argument("--no-require-extracted", action="store_true", help="Don't require extracted_* in each run dir.")

    args = parser.parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()

    dataset_guess, perturbation_guess = _infer_dataset_and_perturbation(input_dir)
    default_output_dir = (
        Path(__file__).resolve().parent.parent
        / "outputs_extracted"
        / (dataset_guess or "unknown_dataset")
        / f"{(perturbation_guess or 'unknown_perturbation')}_output"
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir

    aggregate_runs(
        input_dir=input_dir,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
        require_extracted=not bool(args.no_require_extracted),
    )

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

