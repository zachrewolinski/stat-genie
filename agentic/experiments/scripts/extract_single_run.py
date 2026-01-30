#!/usr/bin/env python3
"""
Per-run extractor for coding-agent outputs.

Given a run directory containing `analysis.py`, `info.json`, and `conclusion.txt`,
this script uses the same LLM infrastructure as the BLADE pipeline to produce:

- `extracted_analysis.json`: extracted `cvars` plus full `analysis_code` (analysis.py contents)
- `extracted_final_conclusion.txt`: JSON like BLADE `final_conclusion_{i}.txt`

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from stat_genie.blade_pipeline.llms.config import llm


def _get_allowed_columns_from_info(info: Dict[str, Any]) -> Set[str]:
    """Return the set of canonical column names from info.json (data_desc.fields[].column)."""
    allowed: Set[str] = set()
    data_desc = info.get("data_desc") or {}
    fields = data_desc.get("fields") or []
    for f in fields:
        if isinstance(f, dict) and "column" in f:
            allowed.add(str(f["column"]))
    return allowed


def _filter_cvars_to_info_json_schema(cvars: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter cvars so every "columns" list contains only names from info.json (data_desc.fields[].column).
    Removes or maps derived variable names from analysis.py to the canonical schema.
    Raises ValueError if a required field (e.g. dv.columns) becomes empty after filtering.
    """
    allowed = _get_allowed_columns_from_info(info)
    if not allowed:
        raise ValueError(
            "info.json has no data_desc.fields with 'column'; cannot filter cvars."
        )

    def filter_columns(columns: Any) -> List[str]:
        if not isinstance(columns, list):
            return []
        return [c for c in columns if isinstance(c, str) and c in allowed]

    out: Dict[str, Any] = {}

    # ivs: list of {description, columns}
    ivs_raw = cvars.get("ivs")
    if isinstance(ivs_raw, list):
        out["ivs"] = []
        for item in ivs_raw:
            if not isinstance(item, dict):
                out["ivs"].append(item)
                continue
            filtered = dict(item)
            cols = filter_columns(item.get("columns"))
            filtered["columns"] = cols
            out["ivs"].append(filtered)
    else:
        out["ivs"] = ivs_raw if ivs_raw is not None else []

    # dv: single object {description, columns}
    dv_raw = cvars.get("dv")
    if isinstance(dv_raw, dict):
        out["dv"] = dict(dv_raw)
        out["dv"]["columns"] = filter_columns(dv_raw.get("columns"))
        if not out["dv"]["columns"]:
            raise ValueError(
                "After filtering to info.json schema, dv.columns is empty; "
                "the dependent variable must refer to at least one column from info.json."
            )
    else:
        out["dv"] = dv_raw

    # controls: list of {description, is_moderator, moderator_on, columns}
    controls_raw = cvars.get("controls")
    if isinstance(controls_raw, list):
        out["controls"] = []
        for item in controls_raw:
            if not isinstance(item, dict):
                out["controls"].append(item)
                continue
            filtered = dict(item)
            filtered["columns"] = filter_columns(item.get("columns"))
            out["controls"].append(filtered)
    else:
        out["controls"] = controls_raw if controls_raw is not None else []

    return out


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    # Common cases: ```json ... ``` / ``` ... ```
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    return t


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_research_question(info: Dict[str, Any]) -> str:
    rq = info.get("research_questions")
    if isinstance(rq, list) and rq:
        return str(rq[0]).strip()
    if isinstance(rq, str):
        return rq.strip()
    return ""


def _infer_dataset_name_from_run_dir(run_dir: Path) -> Optional[str]:
    # Prefer a single CSV name like affairs.csv (but do not read it).
    csvs = sorted([p for p in run_dir.iterdir() if p.is_file() and p.suffix == ".csv"])
    if len(csvs) == 1:
        return csvs[0].stem
    # If multiple, pick the one that isn't obviously auxiliary.
    for p in csvs:
        if p.name.lower() not in {"data.csv", "dataset.csv"}:
            return p.stem
    return csvs[0].stem if csvs else None


def _parse_claude_conclusion_txt(raw: str) -> Tuple[Optional[str], str]:
    """
    Parse the coding agent's "Yes/No + short justification" convention.
    """
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
    if not lines:
        return None, ""
    first = lines[0].lower()
    if first in {"yes", "no"}:
        answer = "Yes" if first == "yes" else "No"
        justification = " ".join(lines[1:]).strip()
        return answer, justification
    # Fall back: no explicit yes/no line
    return None, " ".join(lines).strip()


@dataclass(frozen=True)
class ExtractPaths:
    run_dir: Path
    analysis_py: Path
    info_json: Path
    conclusion_txt: Path


def _resolve_paths(run_dir: Path) -> ExtractPaths:
    analysis_py = run_dir / "analysis.py"
    info_json = run_dir / "info.json"
    conclusion_txt = run_dir / "conclusion.txt"
    missing = [str(p) for p in (analysis_py, info_json, conclusion_txt) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in run directory:\n- " + "\n- ".join(missing)
        )
    return ExtractPaths(
        run_dir=run_dir, analysis_py=analysis_py, info_json=info_json, conclusion_txt=conclusion_txt
    )


def _llm_extract_analysis_entry(
    *,
    llm_provider: str,
    llm_model: str,
    use_cache: bool,
    research_question: str,
    info_json: Dict[str, Any],
    analysis_code: str,
    raw_conclusion_txt: str,
) -> Dict[str, Any]:
    """
    Return a dict with keys: cvars, analysis_code.
    cvars come from the LLM; analysis_code is the full analysis.py file content.
    """
    assistant = llm(provider=llm_provider, model=llm_model, use_cache=use_cache)

    system_prompt = (
        "You are a careful data-science assistant. Read the analysis script and return a structured "
        "summary of the variables used (IVs, DV, controls).\n\n"
        "Return valid JSON only (no markdown, no extra text)."
    )

    user_prompt = f"""
Research question:
{research_question}

Dataset metadata (info.json):
{json.dumps(info_json, indent=2)}

The data scientist produced this analysis script (analysis.py):
<analysis.py>
{analysis_code}
</analysis.py>

The data scientist produced this conclusion (conclusion.txt):
<conclusion.txt>
{raw_conclusion_txt}
</conclusion.txt>

Extract the following field and return JSON ONLY with this exact top-level schema:
{{
  "cvars": {{
    "ivs": [{{"description": "...", "columns": ["..."]}}],
    "dv": {{"description": "...", "columns": ["..."]}},
    "controls": [{{"description": "...", "is_moderator": false, "moderator_on": null, "columns": ["..."]}}]
  }}
}}

Definitions (use these to classify variables from the research question and analysis):
- IV (independent variable): the predictor(s) or treatment(s) the research question asks about; the "X" whose effect on the outcome is of interest.
- DV (dependent variable): the outcome or response the research question asks about; the "Y" that is predicted or explained.
- Controls: covariates included to hold constant (e.g. demographics, confounders); not the main IV(s) or DV.

Constraints:
- "columns" MUST contain ONLY column names that appear in info.json (data_desc.fields[].column).
  Do NOT use derived variable names from analysis.py (e.g. dummy columns, renamed columns, or
  computed columns). For each variable used in the analysis, identify the underlying canonical
  column name from info.json and use that. If analysis.py uses a derived name (e.g. "children_yes"
  from feature6), map it back to the info.json column (e.g. "feature6").
- Use the research question and info.json + analysis.py usage to assign each variable to IV(s), DV, or controls.
- If there are no controls, return [].
- If moderator info is unclear, set is_moderator=false and moderator_on=null.
"""

    result = assistant.generate(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    text = result.text[0].content if hasattr(result, "text") else str(result)
    clean = _strip_code_fences(str(text))
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM analysis extraction did not return valid JSON. "
            f"JSON error: {e}\n\nRaw response:\n{clean}"
        ) from e

    if not isinstance(parsed, dict):
        raise ValueError("LLM analysis extraction must return a JSON object at top-level.")
    if "cvars" not in parsed:
        raise ValueError("LLM analysis extraction missing required key: cvars")
    cvars = parsed["cvars"]
    if not isinstance(cvars, dict):
        raise ValueError("LLM analysis extraction 'cvars' must be an object.")
    for k in ("ivs", "dv", "controls"):
        if k not in cvars:
            raise ValueError(f"LLM analysis extraction 'cvars' missing required key: {k}")

    return {
        "cvars": cvars,
        "analysis_code": analysis_code,
    }


def _llm_normalize_conclusion(
    *,
    llm_provider: str,
    llm_model: str,
    use_cache: bool,
    research_question: str,
    parsed_answer: Optional[str],
    parsed_justification: str,
    raw_conclusion_txt: str,
) -> str:
    """
    Returns a JSON string like: {"answer": "...", "justification": "..."}
    matching BLADE final_conclusion_{i}.txt style.
    """
    assistant = llm(provider=llm_provider, model=llm_model, use_cache=use_cache)

    system_prompt = (
        "Normalize the conclusion into a consistent JSON format for downstream evaluation. "
        "Return valid JSON only (no markdown, no extra text)."
    )

    # Even if we can parse a clean Yes/No, use the LLM to rewrite a tight, consistent justification.
    user_prompt = f"""
Research question:
{research_question}

Raw conclusion (conclusion.txt):
{raw_conclusion_txt}

Parsed from conclusion.txt (may be empty/unknown):
- parsed_answer: {parsed_answer}
- parsed_justification: {parsed_justification}

Return JSON ONLY with exactly these keys:
{{
  "answer": "Yes|No|Not enough information",
  "justification": "..."
}}

Rules:
- "answer" must be exactly one of: "Yes", "No", "Not enough information".
- If the raw conclusion is missing/ambiguous, use "Not enough information".
- The justification should be 1-3 sentences, concise, and consistent with the raw conclusion.
"""

    result = assistant.generate(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    text = result.text[0].content if hasattr(result, "text") else str(result)
    clean = _strip_code_fences(str(text))
    try:
        obj = json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM conclusion normalization did not return valid JSON. "
            f"JSON error: {e}\n\nRaw response:\n{clean}"
        ) from e

    if not isinstance(obj, dict) or "answer" not in obj or "justification" not in obj:
        raise ValueError(
            f"LLM conclusion normalization must return keys 'answer' and 'justification'. Got: {obj}"
        )

    # Re-serialize to canonical JSON (pretty, like existing BLADE files).
    return json.dumps(obj, indent=2, ensure_ascii=False)


def extract_single_run(
    *,
    run_dir: Path,
    llm_provider: str,
    llm_model: str,
    use_cache: bool,
    out_analysis_json: Path,
    out_conclusion_json_txt: Path,
) -> None:
    paths = _resolve_paths(run_dir)

    analysis_code = _read_text(paths.analysis_py)
    info = _read_json(paths.info_json)
    research_question = _get_research_question(info)
    raw_conclusion_txt = _read_text(paths.conclusion_txt)

    dataset_name = _infer_dataset_name_from_run_dir(run_dir)
    parsed_answer, parsed_justification = _parse_claude_conclusion_txt(raw_conclusion_txt)

    extracted = _llm_extract_analysis_entry(
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_cache=use_cache,
        research_question=research_question,
        info_json=info,
        analysis_code=analysis_code,
        raw_conclusion_txt=raw_conclusion_txt,
    )
    extracted["cvars"] = _filter_cvars_to_info_json_schema(extracted["cvars"], info)

    normalized_conclusion = _llm_normalize_conclusion(
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_cache=use_cache,
        research_question=research_question,
        parsed_answer=parsed_answer,
        parsed_justification=parsed_justification,
        raw_conclusion_txt=raw_conclusion_txt,
    )

    payload = {
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "research_question": research_question,
        "extracted_analysis": extracted,  # contains cvars and analysis_code (full analysis.py)
        "extraction_metadata": {
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "use_cache": use_cache,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "analysis_py": str(paths.analysis_py),
                "info_json": str(paths.info_json),
                "conclusion_txt": str(paths.conclusion_txt),
            },
        },
    }

    out_analysis_json.parent.mkdir(parents=True, exist_ok=True)
    out_conclusion_json_txt.parent.mkdir(parents=True, exist_ok=True)

    out_analysis_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_conclusion_json_txt.write_text(normalized_conclusion + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract BLADE-like fields from a single coding-agent run directory.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a coding-agent run directory (contains analysis.py, info.json, conclusion.txt).",
    )
    parser.add_argument("--llm-provider", default="openai", help="LLM provider (e.g. openai, anthropic).")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="LLM model name (as in config/llm_config.yml).")
    parser.add_argument("--use-cache", action="store_true", help="Enable LLM cache (default: disabled).")
    parser.add_argument(
        "--out-analysis-json",
        default=None,
        help="Output path for extracted analysis JSON (default: <run-dir>/extracted_analysis.json).",
    )
    parser.add_argument(
        "--out-conclusion-json",
        default=None,
        help="Output path for normalized conclusion JSON text (default: <run-dir>/extracted_final_conclusion.txt).",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir is not a directory: {run_dir}")

    out_analysis_json = (
        Path(args.out_analysis_json).expanduser().resolve()
        if args.out_analysis_json
        else run_dir / "extracted_analysis.json"
    )
    out_conclusion_json = (
        Path(args.out_conclusion_json).expanduser().resolve()
        if args.out_conclusion_json
        else run_dir / "extracted_final_conclusion.txt"
    )

    # Keep default outputs inside the run directory.
    if not str(out_analysis_json).startswith(str(run_dir)) and args.out_analysis_json is None:
        raise ValueError("Default output path must be inside the run directory.")
    if not str(out_conclusion_json).startswith(str(run_dir)) and args.out_conclusion_json is None:
        raise ValueError("Default output path must be inside the run directory.")

    extract_single_run(
        run_dir=run_dir,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        use_cache=bool(args.use_cache),
        out_analysis_json=out_analysis_json,
        out_conclusion_json_txt=out_conclusion_json,
    )

    print(f"Wrote: {out_analysis_json}")
    print(f"Wrote: {out_conclusion_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

