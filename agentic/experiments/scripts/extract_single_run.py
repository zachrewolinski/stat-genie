#!/usr/bin/env python3
"""
Per-run extractor for Claude Code outputs.

Given a run directory containing `analysis.py`, `info.json`, and `conclusion.txt`,
this script uses the same LLM infrastructure as the BLADE pipeline to produce:

- `extracted_analysis.json`: extracted `cvars`/`transform_code`/`m_code`
- `extracted_final_conclusion.txt`: JSON like BLADE `final_conclusion_{i}.txt`

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from stat_genie.blade_pipeline.llms.config import llm


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
    Parse the Claude Code "Yes/No + short justification" convention.
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
    Return a dict with keys: cvars, transform_code, m_code.
    """
    assistant = llm(provider=llm_provider, model=llm_model, use_cache=use_cache)

    system_prompt = (
        "You are a careful data-science assistant. Read the analysis script and return a structured "
        "summary of variables and modeling choices.\n\n"
        "Return valid JSON only (no markdown, no extra text)."
    )

    # Keep the schema tightly aligned with BLADE multirun_analyses.json.
    user_prompt = f"""
Research question:
{research_question}

Dataset metadata (info.json):
{json.dumps(info_json, indent=2)}

Claude Code run produced this analysis script (analysis.py):
<analysis.py>
{analysis_code}
</analysis.py>

Claude Code run produced this conclusion (conclusion.txt):
<conclusion.txt>
{raw_conclusion_txt}
</conclusion.txt>

Extract the following fields and return JSON ONLY with this exact top-level schema:
{{
  "cvars": {{
    "ivs": [{{"description": "...", "columns": ["..."]}}],
    "dv": {{"description": "...", "columns": ["..."]}},
    "controls": [{{"description": "...", "is_moderator": false, "moderator_on": null, "columns": ["..."]}}]
  }},
  "transform_code": "...",
  "m_code": "..."
}}

Constraints:
- "columns" must match names used in analysis.py exactly (e.g. "children_yes", "feature6").
- Use info.json + analysis.py usage patterns to decide DV vs IV(s) vs controls.
- If there are no controls, return [].
- If moderator info is unclear, set is_moderator=false and moderator_on=null.
- transform_code / m_code should be best-effort code excerpts (strings), not lists.
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

    # Minimal structural validation (single-pass; no auto-fix loop).
    if not isinstance(parsed, dict):
        raise ValueError("LLM analysis extraction must return a JSON object at top-level.")
    for k in ("cvars", "transform_code", "m_code"):
        if k not in parsed:
            raise ValueError(f"LLM analysis extraction missing required key: {k}")
    cvars = parsed.get("cvars")
    if not isinstance(cvars, dict):
        raise ValueError("LLM analysis extraction 'cvars' must be an object.")
    for k in ("ivs", "dv", "controls"):
        if k not in cvars:
            raise ValueError(f"LLM analysis extraction 'cvars' missing required key: {k}")

    return parsed


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

Raw Claude Code conclusion.txt:
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
        "extracted_analysis": extracted,  # contains cvars/transform_code/m_code
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
    parser = argparse.ArgumentParser(description="Extract BLADE-like fields from a single Claude Code run directory.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a Claude Code run directory (contains analysis.py, info.json, conclusion.txt).",
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

