def extract_final_answer(model_output):
    """
    Extract interpretable statistics from a fitted model output.

    The function attempts to extract the coefficient, p-value, and 95% confidence
    interval for the parameter named 'name_z'. If 'name_z' is not present, the
    function will instead:
      - If any parameters prefixed with 'name_' exist, return a mapping of those
        parameter names to their coefficients and p-values.
      - Otherwise, return coefficients and p-values for the first up to 10
        parameters available in the model output.

    Returns a dictionary with two keys:
      - "object": the extracted numeric values (either a single-parameter dict or
        a mapping of parameter names to their stats).
      - "description": a short explanation of what is being returned.
    """
    # Helper to safely get a numeric value from model_output attributes
    def _safe_get(attr_container, key, index_lookup=None):
        try:
            return float(attr_container[key])
        except Exception:
            # try lookup by position if key not directly indexable
            if index_lookup is not None:
                try:
                    idx = index_lookup.index(key)
                    return float(attr_container[idx])
                except Exception:
                    return None
            return None

    # Ensure model_output has params
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not have a 'params' attribute")

    params = model_output.params

    # Determine parameter names
    try:
        param_names = list(params.index)
    except Exception:
        try:
            param_names = list(params.keys())
        except Exception:
            # fallback to numeric indices as strings
            try:
                param_names = [str(i) for i in range(len(params))]
            except Exception:
                param_names = []

    target = "name_z"

    # If the exact target exists, return its stats
    if target in param_names:
        coef = _safe_get(params, target, index_lookup=param_names)
        pvalue = None
        if hasattr(model_output, "pvalues"):
            pvalue = _safe_get(model_output.pvalues, target, index_lookup=param_names)
        ci_lower = ci_upper = None
        try:
            # conf_int may be a DataFrame-like or ndarray
            ci = model_output.conf_int()
            if hasattr(ci, "loc") and target in getattr(ci, "index", []):
                # DataFrame-like: columns 0 and 1
                ci_lower = float(ci.loc[target, 0])
                ci_upper = float(ci.loc[target, 1])
            else:
                # ndarray-like: align by param_names
                if target in param_names:
                    idx = param_names.index(target)
                    ci_lower = float(ci[idx, 0])
                    ci_upper = float(ci[idx, 1])
        except Exception:
            # If conf_int unavailable or extraction failed, leave as None
            pass

        return {
            "object": {
                "parameter": target,
                "coef": coef,
                "pvalue": pvalue,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            },
            "description": f"Estimated coefficient, p-value, and 95% CI for parameter '{target}' (CI may be None if not available).",
        }

    # If target not present, collect all 'name_' parameters if any
    name_params = [n for n in param_names if n.startswith("name_")]

    if name_params:
        result = {}
        for n in name_params:
            coef = _safe_get(params, n, index_lookup=param_names)
            pvalue = None
            if hasattr(model_output, "pvalues"):
                pvalue = _safe_get(model_output.pvalues, n, index_lookup=param_names)
            result[n] = {"coef": coef, "pvalue": pvalue}
        return {
            "object": result,
            "description": (
                f"Parameter 'name_z' not found. Returning coefficients and p-values for the {len(name_params)} "
                "parameters beginning with 'name_'. Values may be None if not available."
            ),
        }

    # As a final fallback, return the first up to 10 parameters with their coef and p-value
    fallback_count = min(10, len(param_names))
    fallback = {}
    for n in param_names[:fallback_count]:
        coef = _safe_get(params, n, index_lookup=param_names)
        pvalue = None
        if hasattr(model_output, "pvalues"):
            pvalue = _safe_get(model_output.pvalues, n, index_lookup=param_names)
        fallback[n] = {"coef": coef, "pvalue": pvalue}

    return {
        "object": fallback,
        "description": (
            "Parameter 'name_z' not found and no 'name_' parameters present. Returning coefficients and p-values "
            f"for the first {fallback_count} parameters available in the model output."
        ),
    }