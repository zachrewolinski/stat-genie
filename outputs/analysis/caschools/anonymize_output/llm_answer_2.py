import numpy as np

def extract_final_answer(model_output):
    """
    Extracts statistics for the StuTeacherRatio coefficient from a statsmodels results object
    (including robust-covariance results returned by get_robustcov_results).
    Returns a dictionary with keys:
      - "object": dict with numeric summary (coefficient, std err, t, p-value, 95% CI)
      - "description": plain-language interpretation about whether a lower student-teacher
                       ratio is associated with higher academic performance.
    The function is robust to cases where model_output.params (and related attributes)
    may be numpy arrays rather than pandas Series/DataFrame by using model_output.model.exog_names
    when available, or falling back to numeric indices.
    """
    var = "StuTeacherRatio"

    # Helper to obtain parameter names in order
    def _get_param_names(params, model_output):
        # If params has an index (e.g., pandas Series), use it
        if hasattr(params, "index"):
            return list(params.index)
        # Try model.exog_names (common in statsmodels results)
        if hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
            try:
                return list(model_output.model.exog_names)
            except Exception:
                pass
        # Fallback: numeric string names based on length
        try:
            arr = np.asarray(params)
            return [str(i) for i in range(arr.shape[0])]
        except Exception:
            return []

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("Provided model_output does not appear to be a statsmodels results object with 'params'.")

    params_raw = model_output.params
    param_names = _get_param_names(params_raw, model_output)
    params_arr = np.asarray(params_raw)

    if len(param_names) != params_arr.shape[0]:
        # If counts mismatch, still try to proceed using available names or indices
        # Create default names matching params_arr length
        param_names = param_names[: params_arr.shape[0]] if param_names else [str(i) for i in range(params_arr.shape[0])]

    # Build a mapping name -> value
    params_map = {}
    for i, name in enumerate(param_names):
        try:
            params_map[name] = float(params_arr[i])
        except Exception:
            # If conversion fails, attempt to get value from params_raw by label
            try:
                params_map[name] = float(params_raw[name])
            except Exception:
                params_map[name] = float("nan")

    if var not in params_map:
        raise ValueError(f"Variable '{var}' not found in model parameters. Available params: {param_names}")

    coef = params_map[var]

    # Helper to extract a numeric attribute (bse, tvalues, pvalues)
    def _get_attr_value(attr_name, var, param_names, model_output):
        if hasattr(model_output, attr_name):
            attr = getattr(model_output, attr_name)
            # If attr has index/labels (pandas Series)
            if hasattr(attr, "index"):
                try:
                    return float(attr[var])
                except Exception:
                    # fallback: try positional access if lengths align
                    try:
                        idx = param_names.index(var)
                        return float(np.asarray(attr)[idx])
                    except Exception:
                        return float("nan")
            else:
                # assume numpy array-like; index by position
                try:
                    idx = param_names.index(var)
                    return float(np.asarray(attr)[idx])
                except Exception:
                    return float("nan")
        return float("nan")

    se = _get_attr_value("bse", var, param_names, model_output)
    tval = _get_attr_value("tvalues", var, param_names, model_output)
    pval = _get_attr_value("pvalues", var, param_names, model_output)

    # If se is NaN but tval is available, compute se = coef / tval
    if np.isnan(se) and (not np.isnan(tval)) and tval != 0:
        try:
            se = float(coef / tval)
        except Exception:
            se = float("nan")

    # If tval is NaN but se is available, compute tval = coef / se
    if np.isnan(tval) and (not np.isnan(se)) and se != 0:
        try:
            tval = float(coef / se)
        except Exception:
            tval = float("nan")

    # Confidence interval extraction
    ci_low, ci_high = float("nan"), float("nan")
    try:
        ci_raw = model_output.conf_int(alpha=0.05)
        # If DataFrame-like with .loc
        if hasattr(ci_raw, "loc"):
            if var in ci_raw.index:
                row = ci_raw.loc[var]
                # row could be length-2
                ci_low, ci_high = float(row.iloc[0]), float(row.iloc[1])
            else:
                # fallback to positional by matching param_names
                idx = param_names.index(var)
                row = np.asarray(ci_raw)
                ci_low, ci_high = float(row[idx, 0]), float(row[idx, 1])
        else:
            # assume ndarray-like with shape (k_params, 2)
            ci_arr = np.asarray(ci_raw)
            idx = param_names.index(var)
            # Handle shapes like (2, k) by checking second dim
            if ci_arr.ndim == 2 and ci_arr.shape[0] == len(param_names):
                ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            elif ci_arr.ndim == 2 and ci_arr.shape[1] == len(param_names):
                ci_low, ci_high = float(ci_arr[0, idx]), float(ci_arr[1, idx])
            else:
                # Unexpected shape
                ci_low, ci_high = float("nan"), float("nan")
    except Exception:
        ci_low, ci_high = float("nan"), float("nan")

    # Interpretation:
    # - The model coefficient is the change in AvgScore for a one-unit increase in StuTeacherRatio
    # - Because lower StuTeacherRatio means fewer students per teacher, a negative coefficient
    #   would mean that increasing the ratio (more students per teacher) is associated with
    #   lower scores; equivalently, a lower ratio (smaller class sizes) is associated with
    #   higher scores.
    if (not np.isnan(pval)) and (pval < 0.05):
        sig_text = "statistically significant (p < 0.05)"
    elif not np.isnan(pval):
        sig_text = "not statistically significant (p >= 0.05)"
    else:
        sig_text = "p-value not available"

    if coef < 0:
        direction_text = (
            "The estimated association is negative: higher student-teacher ratios (more students per teacher) "
            "are associated with lower average scores. Equivalently, lower student-teacher ratios (smaller class sizes) "
            "are associated with higher academic performance."
        )
    elif coef > 0:
        direction_text = (
            "The estimated association is positive: higher student-teacher ratios (more students per teacher) "
            "are associated with higher average scores (i.e., lower ratios associated with lower performance)."
        )
    else:
        direction_text = "The estimated association is essentially zero."

    # Safe formatting for numeric values that may be NaN
    def _fmt(val, fmt="{:.4f}"):
        try:
            if val is None:
                return "NA"
            if isinstance(val, (int, float)) and np.isnan(val):
                return "NA"
            return fmt.format(val)
        except Exception:
            return str(val)

    conclusion = (
        f"{direction_text} This effect is {sig_text}. "
        f"Estimated effect: a one-unit increase in StuTeacherRatio changes AvgScore by {_fmt(coef)} points "
        f"(SE = {_fmt(se)}, t = {_fmt(tval, fmt='{:.3f}')}, p = {_fmt(pval, fmt='{:.3g}')}, 95% CI = [{_fmt(ci_low)}, {_fmt(ci_high)}])."
    )

    result_object = {
        "variable": var,
        "coefficient": coef,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }

    return {
        "object": result_object,
        "description": conclusion,
    }