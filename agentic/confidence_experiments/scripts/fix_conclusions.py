#!/usr/bin/env python3
"""Repair malformed JSON in confidence-experiment conclusion and confidence files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import importlib.util

# Both modules use sys.path tricks that cause circular imports when loaded
# as regular imports from the same directory. Load them explicitly by path.
_SCALAR_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scalar_experiments" / "scripts"
_LOCAL_SCRIPTS = Path(__file__).resolve().parent


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Scalar module must be loaded first (the local one depends on it).
_scalar_agg = _load_module(
    "scalar_aggregate_conclusions",
    _SCALAR_SCRIPTS / "aggregate_conclusions.py",
)
_parse_conclusion = _scalar_agg._parse_conclusion

_local_agg = _load_module(
    "confidence_aggregate_conclusions",
    _LOCAL_SCRIPTS / "aggregate_conclusions.py",
)
CONCLUSION_SCHEMA = _local_agg.CONCLUSION_SCHEMA
CONFIDENCE_SCHEMA = _local_agg.CONFIDENCE_SCHEMA
_discover_runs = _local_agg._discover_runs

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"

PVE_LEVEL_PATTERN = re.compile(r"^pve_([\d.]+)$")
RUN_DIR_PATTERN = re.compile(r"^run(\d+)$", re.IGNORECASE)


def _discover_pve_runs(
    outputs_dir: Path,
) -> list[tuple[str, str, str, int, Path]]:
    """Walk outputs/{dataset}/pve/pve_{X}/{perturbation}/run{N}/."""
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
                            run_dir,
                        ))
    return runs

AZURE_ENDPOINT = "https://fxdata-eastus2.openai.azure.com/openai"
AZURE_API_VERSION = "2025-04-01-preview"

_OPENAI_DEFAULT_MODEL = "gpt-5"
_AZURE_DEFAULT_MODEL = "gpt-5"


# ---------------------------------------------------------------------------
# Programmatic fix helpers
# ---------------------------------------------------------------------------


def _strip_outer_junk(text: str) -> str:
    """Strip characters before the first '{' and after the last '}'."""
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        return text
    return text[first : last + 1]


def _escape_literal_whitespace(text: str) -> str:
    """Replace literal newlines/tabs inside JSON string values with escapes."""
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        return text

    inner = text[first + 1 : last]
    inner = re.sub(r'(?<!\\)\n', r'\\n', inner)
    inner = re.sub(r'(?<!\\)\t', r'\\t', inner)
    inner = re.sub(r'(?<!\\)\r', r'\\r', inner)
    return text[: first + 1] + inner + text[last:]


def _fix_unescaped_quotes(text: str) -> str:
    """Escape unescaped double-quotes inside JSON string values.

    Finds colon-quote boundaries that delimit value strings, then escapes
    any interior quotes that aren't already escaped.
    """
    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        colon_match = re.search(r':\s*"', text[i:])
        if not colon_match:
            result.append(text[i:])
            break

        value_start = i + colon_match.end()
        result.append(text[i:value_start])

        j = value_start
        while j < n:
            if text[j] == '\\':
                j += 2
                continue
            if text[j] == '"':
                rest = text[j + 1:].lstrip()
                if not rest or rest[0] in (',', '}'):
                    break
                else:
                    result.append('\\"')
                    j += 1
                    continue
            result.append(text[j])
            j += 1

        if j < n:
            result.append('"')
            i = j + 1
        else:
            i = n

    return ''.join(result)


def _try_programmatic_fix(raw: str) -> str | None:
    """Apply programmatic fixes and return valid JSON string, or None."""
    text = raw.strip()

    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    text = _strip_outer_junk(text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    text = _escape_literal_whitespace(text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    text = _fix_unescaped_quotes(text)
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    text2 = text.replace('\\\\"', '\\"')
    try:
        json.loads(text2)
        return text2
    except json.JSONDecodeError:
        pass

    # raw_decode: handles trailing junk that contains '}' characters
    for candidate in (text, text2):
        first = candidate.find("{")
        if first == -1:
            continue
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate, idx=first)
            return json.dumps(obj)
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------


def _llm_fix(raw: str, schema: list[tuple[str, str]], deployment: str | None) -> str | None:
    """Use OpenAI (or Azure OpenAI) to recover valid JSON from malformed text.

    Provider selection uses env vars: OPENAI_API_KEY takes priority over
    AZURE_OPENAI_API_KEY. Returns None when no key is available.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if not openai_key and not azure_key:
        return None

    try:
        import openai as _openai_mod  # noqa: F811
    except ImportError:
        print("openai package not installed; skipping LLM fallback", file=sys.stderr)
        return None

    if openai_key:
        client = _openai_mod.OpenAI(api_key=openai_key)
        model = deployment or _OPENAI_DEFAULT_MODEL
        extra_kwargs: dict[str, Any] = {"max_completion_tokens": 4096}
    else:
        client = _openai_mod.AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION,
        )
        model = deployment or _AZURE_DEFAULT_MODEL
        extra_kwargs = {"temperature": 0, "max_completion_tokens": 4096}

    # Build key description from schema
    key_descs = []
    for key, kind in schema:
        if kind == "score_0_100":
            key_descs.append(f'"{key}" (integer 0-100)')
        else:
            key_descs.append(f'"{key}" (string)')
    keys_str = " and ".join(key_descs)

    prompt = (
        f"The following text should be a JSON object with keys {keys_str}. "
        "Extract the intended JSON and return only the valid JSON object, "
        "with no extra text.\n\n"
        f"---\n{raw}\n---"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **extra_kwargs,
        )
    except Exception as exc:
        print(f"LLM request failed: {exc}", file=sys.stderr)
        return None

    choice = response.choices[0]
    content = choice.message.content
    if not content:
        reason = choice.finish_reason
        refusal = getattr(choice.message, "refusal", None)
        print(
            f"LLM returned empty response (finish_reason={reason}, refusal={refusal})",
            file=sys.stderr,
        )
        return None

    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)

    try:
        json.loads(content)
        return content
    except json.JSONDecodeError as exc:
        preview = content[:300].replace("\n", "\\n")
        print(f"LLM returned invalid JSON: {exc}", file=sys.stderr)
        print(f"LLM response preview: {preview}", file=sys.stderr)
        return None


def _fix_files(
    labeled_paths: list[tuple[str, Path]],
    schema: list[tuple[str, str]],
    *,
    dry_run: bool = False,
    verbose: bool = False,
    llm_deployment: str | None = None,
) -> dict[str, int]:
    """Try to repair each file in *labeled_paths*.

    Each entry is (human-readable label, path to the JSON file).
    *schema* is passed to _parse_conclusion and _llm_fix for validation.
    """
    stats = {
        "scanned": 0,
        "valid": 0,
        "fixed_programmatic": 0,
        "fixed_llm": 0,
        "still_broken": 0,
    }

    for label, file_path in labeled_paths:
        stats["scanned"] += 1

        try:
            raw = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            if verbose:
                print(f"skip {label}: unreadable ({exc})")
            stats["still_broken"] += 1
            continue

        parsed = _parse_conclusion(raw, schema=schema)
        if parsed is not None:
            stats["valid"] += 1
            continue

        # Try programmatic fixes
        fixed = _try_programmatic_fix(raw)
        if fixed is not None:
            check = _parse_conclusion(fixed, schema=schema)
            if check is not None:
                if verbose or dry_run:
                    print(f"fixed programmatically: {label}")
                if not dry_run:
                    file_path.write_text(fixed, encoding="utf-8")
                stats["fixed_programmatic"] += 1
                continue

        if verbose or dry_run:
            print(f"trying LLM fallback: {label}")
        llm_fixed = _llm_fix(raw, schema, llm_deployment)
        if llm_fixed is not None:
            check = _parse_conclusion(llm_fixed, schema=schema)
            if check is not None:
                if verbose or dry_run:
                    print(f"fixed with LLM: {label}")
                if not dry_run:
                    file_path.write_text(llm_fixed, encoding="utf-8")
                stats["fixed_llm"] += 1
                continue

        if verbose or dry_run:
            preview = raw[:200].replace("\n", "\\n")
            print(f"still broken {label}: {preview}")
        stats["still_broken"] += 1

    return stats


def _merge_stats(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: a.get(k, 0) + b.get(k, 0) for k in a}


def _print_stats(label: str, stats: dict[str, int], dry_run: bool) -> None:
    mode = " (DRY RUN)" if dry_run else ""
    print(f"\n{label}{mode}:")
    print(f"  scanned: {stats['scanned']}")
    print(f"  already valid: {stats['valid']}")
    print(f"  fixed programmatically: {stats['fixed_programmatic']}")
    print(f"  fixed with LLM: {stats['fixed_llm']}")
    print(f"  still broken: {stats['still_broken']}")


def fix_conclusions(
    outputs_dir: Path,
    *,
    pve: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    llm_deployment: str | None = None,
) -> dict[str, int]:
    runs = _discover_runs(outputs_dir)

    # Build separate labeled-path lists for each file type
    conclusion_paths: list[tuple[str, Path]] = []
    confidence_paths: list[tuple[str, Path]] = []
    for ds, dist, pert, rid, run_dir in runs:
        label = f"{ds}/{dist}/{pert}/run{rid}"
        conclusion_file = run_dir / "conclusion.txt"
        confidence_file = run_dir / "confidence.txt"
        if conclusion_file.exists():
            conclusion_paths.append((label, conclusion_file))
        if confidence_file.exists():
            confidence_paths.append((label, confidence_file))

    conclusion_stats = _fix_files(
        conclusion_paths,
        CONCLUSION_SCHEMA,
        dry_run=dry_run,
        verbose=verbose,
        llm_deployment=llm_deployment,
    )
    confidence_stats = _fix_files(
        confidence_paths,
        CONFIDENCE_SCHEMA,
        dry_run=dry_run,
        verbose=verbose,
        llm_deployment=llm_deployment,
    )

    _print_stats("conclusion.txt", conclusion_stats, dry_run)
    _print_stats("confidence.txt", confidence_stats, dry_run)

    combined = _merge_stats(conclusion_stats, confidence_stats)

    if pve:
        pve_paths: list[tuple[str, Path]] = []
        for ds, lvl, pert, rid, run_dir in _discover_pve_runs(outputs_dir):
            label = f"{ds}/pve/pve_{lvl}/{pert}/run{rid}"
            conclusion_file = run_dir / "conclusion.txt"
            if conclusion_file.exists():
                pve_paths.append((label, conclusion_file))

        pve_stats = _fix_files(
            pve_paths,
            CONCLUSION_SCHEMA,
            dry_run=dry_run,
            verbose=verbose,
            llm_deployment=llm_deployment,
        )
        _print_stats("PVE conclusion.txt", pve_stats, dry_run)
        combined = _merge_stats(combined, pve_stats)

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix invalid JSON in confidence experiment conclusion and confidence files."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Path to outputs directory (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report issues without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print details of each fix.",
    )
    parser.add_argument(
        "--pve",
        action="store_true",
        help="Also scan PVE output directories.",
    )
    parser.add_argument(
        "--llm-deployment",
        default=None,
        help=(
            "Model/deployment for LLM fallback. "
            f"Default: {_OPENAI_DEFAULT_MODEL} (OpenAI) or {_AZURE_DEFAULT_MODEL} (Azure)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if os.environ.get("OPENAI_API_KEY"):
        provider = "OpenAI"
        default_model = _OPENAI_DEFAULT_MODEL
    elif os.environ.get("AZURE_OPENAI_API_KEY"):
        provider = "Azure OpenAI"
        default_model = _AZURE_DEFAULT_MODEL
    else:
        provider = None
        default_model = None

    if provider:
        model = args.llm_deployment or default_model
        print(f"LLM fallback enabled: {provider} ({model})")
    else:
        print("LLM fallback disabled: no API key found", file=sys.stderr)

    stats = fix_conclusions(
        args.outputs_dir,
        pve=args.pve,
        dry_run=args.dry_run,
        verbose=args.verbose,
        llm_deployment=args.llm_deployment,
    )

    return 1 if stats["still_broken"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
