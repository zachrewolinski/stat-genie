#!/usr/bin/env python3
"""
Module providing a utility to extract a final answer / key statistics from a
model output object.

Function:
- extract_final_answer(model_output)
    - Inspects model_output (dict-like or list/tuple of dicts) and extracts
      common statistics: estimate/coefficient, standard error, p-value,
      confidence interval (lower/upper), and an exponentiated estimate (odds
      ratio / exp(coef)) when available.
    - Returns a dictionary with keys:
        - "object": a dict of extracted numeric values (missing values are NaN)
        - "description": a brief human-readable explanation of what was returned
"""

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import math


def _as_float(value: Any) -> float:
    """Try to convert value to float, otherwise return NaN."""
    try:
        if value is None:
            return float("nan")
        # If it's already a float or int, cast directly
        if isinstance(value, (float, int)):
            return float(value)
        # Strings that represent numbers should be converted
        return float(str(value))
    except Exception:
        return float("nan")


def _get_first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    """Return the first present (non-None) value from mapping for keys, or None."""
    for k in keys:
        if k in mapping:
            val = mapping.get(k)
            if val is not None:
                return val
    return None


def extract_final_answer(model_output: Any) -> Dict[str, Any]:
    """
    Extracts key statistics from a model_output object.

    Parameters
    ----------
    model_output : Any
        Expected to be a mapping (dict-like) or a sequence whose first element
        is a mapping. The function looks for common keys for estimates, p-values,
        standard errors and confidence intervals.

    Returns
    -------
    Dict[str, Any]
        {
            "object": {
                "estimate": float or NaN,
                "std_error": float or NaN,
                "p_value": float or NaN,
                "ci_lower": float or NaN,
                "ci_upper": float or NaN,
                "exp_estimate": float or NaN  # if available
            },
            "description": str  # brief explanation of the above values
        }
    """
    # Normalize model_output to a mapping if possible
    info: Mapping[str, Any]
    if isinstance(model_output, Mapping):
        info = model_output
    elif isinstance(model_output, (list, tuple)) and len(model_output) > 0 and isinstance(model_output[0], Mapping):
        info = model_output[0]
    else:
        # If it's some other object, try to use its __dict__ (best effort)
        info = getattr(model_output, "__dict__", {}) or {}

    # Common candidate keys for different statistics
    estimate_keys = [
        "estimate", "coef", "coefficient", "beta", "estimate_value", "value", "mean"
    ]
    se_keys = ["std_error", "std_err", "se", "sigma", "stderr"]
    pvalue_keys = ["p_value", "pvalue", "p", "P>|t|", "pval"]
    ci_lower_keys = [
        "ci_lower", "ci_l", "conf_int_low", "conf_int_lower", "lower",
        "95ci_lower", "exp_ci_95_lower", "ci_2.5%", "ci_lower_95"
    ]
    ci_upper_keys = [
        "ci_upper", "ci_u", "conf_int_high", "conf_int_upper", "upper",
        "95ci_upper", "exp_ci_95_upper", "ci_97.5%", "ci_upper_95"
    ]
    exp_estimate_keys = ["exp_coef", "odds_ratio", "exp_estimate", "exp"]

    # Extract values
    raw_estimate = _get_first(info, estimate_keys)
    raw_se = _get_first(info, se_keys)
    raw_p = _get_first(info, pvalue_keys)
    raw_ci_lo = _get_first(info, ci_lower_keys)
    raw_ci_hi = _get_first(info, ci_upper_keys)
    raw_exp = _get_first(info, exp_estimate_keys)

    # If exponentiated estimate not present but keys imply exponentiated CI exist,
    # try to get them as exp_estimate as well.
    if raw_exp is None:
        # Sometimes the info provides "exp_coef" under slightly different keys already attempted above.
        raw_exp = _get_first(info, ["exp(coef)", "exp_coef", "exp_beta"])

    estimate = _as_float(raw_estimate)
    std_error = _as_float(raw_se)
    p_value = _as_float(raw_p)
    ci_lower = _as_float(raw_ci_lo)
    ci_upper = _as_float(raw_ci_hi)
    exp_estimate = _as_float(raw_exp)

    # If CI missing but we have estimate and std_error, compute approximate 95% CI
    if math.isnan(ci_lower) or math.isnan(ci_upper):
        if not math.isnan(estimate) and not math.isnan(std_error):
            z = 1.96
            ci_lower = estimate - z * std_error
            ci_upper = estimate + z * std_error

    extracted = {
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "exp_estimate": exp_estimate,
    }

    # Build a short description
    desc_parts = []
    if not math.isnan(estimate):
        desc_parts.append(f"Estimate = {estimate}")
    if not math.isnan(std_error):
        desc_parts.append(f"SE = {std_error}")
    if not math.isnan(ci_lower) and not math.isnan(ci_upper):
        desc_parts.append(f"95% CI = [{ci_lower}, {ci_upper}]")
    if not math.isnan(p_value):
        desc_parts.append(f"p-value = {p_value}")
    if not math.isnan(exp_estimate):
        desc_parts.append(f"Exp(estimate) = {exp_estimate}")

    description = " | ".join(desc_parts) if desc_parts else "No numeric statistics could be extracted."

    return {"object": extracted, "description": description}