def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels OLS RegressionResultsWrapper
    to answer how Age, Sex, and ReceivedHelp influence LogEfficiency.

    Returns a dictionary with keys:
      - "object": dict of extracted numeric results (coefficients, p-values,
                  CIs, simple slopes for Age and Sex with/without help, etc.)
      - "description": brief interpretation of what these numbers mean.

    The function expects parameter names in the model to include:
      'Age', 'SexBinary', 'ReceivedHelp', 'Age:ReceivedHelp', 'SexBinary:ReceivedHelp'
    as in the model formula used to fit the object.
    """
    import numpy as np

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # Basic validation
    if model_output is None:
        raise ValueError("model_output is None")
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels result (missing .params)")

    # Extract coefficients, p-values
    params = model_output.params
    pvalues = model_output.pvalues

    # Confidence intervals: use numeric-safe extraction
    ci_array = model_output.conf_int()

    try:
        param_names = list(params.index)
    except Exception:
        try:
            param_names = list(params.keys())
        except Exception:
            param_names = []

    # Helper to robustly extract CI for a parameter name or by position
    def get_ci_for_name(name, idx):
        # If ci_array is a pandas DataFrame and has the param as an index label
        try:
            if hasattr(ci_array, "loc") and name in getattr(ci_array, "index", []):
                row = ci_array.loc[name]
                # row may be a Series; take first two entries as lower and upper
                lower = safe_float(row.iloc[0])
                upper = safe_float(row.iloc[1])
                return [lower, upper]
        except Exception:
            pass

        # Fallback: treat ci_array as numpy-like and index by position
        try:
            arr = np.asarray(ci_array)
            if arr.ndim == 2 and idx < arr.shape[0]:
                return [safe_float(arr[idx, 0]), safe_float(arr[idx, 1])]
        except Exception:
            pass

        return [None, None]

    ci_dict = {name: get_ci_for_name(name, i) for i, name in enumerate(param_names)}

    # Helper to get scalar param/pvalue/ci (if missing, set None)
    def get_param_info(name):
        est = safe_float(params[name]) if name in getattr(params, "index", []) else None
        p = safe_float(pvalues[name]) if name in getattr(pvalues, "index", []) else None
        ci = ci_dict.get(name, [None, None])
        sig = (p is not None) and (p < 0.05)
        return {"estimate": est, "pvalue": p, "ci_95": ci, "significant_p_lt_0.05": sig}

    results = {}
    # Core main effects and interactions
    for nm in ["Age", "SexBinary", "ReceivedHelp", "Age:ReceivedHelp", "SexBinary:ReceivedHelp"]:
        results[nm] = get_param_info(nm)

    # Compute simple slopes using linear contrasts:
    # - Effect of Age when ReceivedHelp = 0 -> coefficient 'Age'
    # - Effect of Age when ReceivedHelp = 1 -> Age + Age:ReceivedHelp (use t_test)
    # - Effect of Sex (SexBinary) when ReceivedHelp = 0 -> 'SexBinary'
    # - Effect of Sex when ReceivedHelp = 1 -> SexBinary + SexBinary:ReceivedHelp
    def contrast_result(expr):
        """
        Run t_test on a linear combination expression like 'Age + Age:ReceivedHelp'
        Return dict with estimate, pvalue, ci, significant flag.
        """
        try:
            ct = model_output.t_test(expr)
            eff_arr = np.asarray(ct.effect).ravel()
            est = safe_float(eff_arr[0]) if eff_arr.size > 0 else None
            pv_arr = np.asarray(ct.pvalue).ravel()
            pv = safe_float(pv_arr[0]) if pv_arr.size > 0 else None
            # ct.conf_int() may return an array-like 2x2 or similar
            ci_raw = ct.conf_int()
            ci_raw_arr = np.asarray(ci_raw)
            if ci_raw_arr.ndim == 2:
                # take first row
                lower = safe_float(ci_raw_arr[0, 0])
                upper = safe_float(ci_raw_arr[0, 1])
            else:
                lower = upper = None
            sig = (pv is not None) and (pv < 0.05)
            return {"estimate": est, "pvalue": pv, "ci_95": [lower, upper], "significant_p_lt_0.05": sig}
        except Exception as e:
            return {"estimate": None, "pvalue": None, "ci_95": [None, None], "significant_p_lt_0.05": False, "error": str(e)}

    # Age simple slopes
    results["Age_no_help"] = results.get("Age")  # equivalent to Age when ReceivedHelp=0
    results["Age_with_help"] = contrast_result("Age + Age:ReceivedHelp")

    # Sex simple slopes
    results["SexBinary_no_help"] = results.get("SexBinary")
    results["SexBinary_with_help"] = contrast_result("SexBinary + SexBinary:ReceivedHelp")

    # ReceivedHelp main effect (interpreted at Age=0, SexBinary=0 in this parameterization)
    results["ReceivedHelp_main"] = results.get("ReceivedHelp")

    # Model-level summaries
    model_stats = {
        "nobs": int(model_output.nobs) if hasattr(model_output, "nobs") else None,
        "rsquared": safe_float(getattr(model_output, "rsquared", None)),
        "rsquared_adj": safe_float(getattr(model_output, "rsquared_adj", None)),
        "aic": safe_float(getattr(model_output, "aic", None)),
        "bic": safe_float(getattr(model_output, "bic", None)),
    }

    # Build return object; safely convert dict values to floats where possible
    coeffs = {}
    for k, v in getattr(params, "to_dict", lambda: {})().items():
        coeffs[k] = safe_float(v)

    pvals_out = {}
    for k, v in getattr(pvalues, "to_dict", lambda: {})().items():
        pvals_out[k] = safe_float(v)

    summary_text = None
    try:
        if hasattr(model_output, "summary"):
            summ = model_output.summary()
            # summary() may return an object with as_text()
            summary_text = summ.as_text() if hasattr(summ, "as_text") else str(summ)
    except Exception:
        summary_text = None

    return_object = {
        "coefficients": coeffs,
        "pvalues": pvals_out,
        "conf_int_95": ci_dict,
        "simple_slopes_and_contrasts": results,
        "model_stats": model_stats,
        "model_summary_text": summary_text
    }

    # Short description to interpret the key numbers:
    description_lines = [
        "Returned: coefficients, p-values, 95% CIs, and simple slopes for Age and Sex with and without ReceivedHelp.",
        "Interpretation notes:",
        "- Coefficients are on the log(nuts/sec) scale. A unit change in Age corresponds to coefficient change in log-efficiency.",
        "- SexBinary = 1 means Male; its coefficient is the (log) difference Male minus Female when ReceivedHelp = 0.",
        "- ReceivedHelp coefficient is the difference (ReceivedHelp=1 vs 0) at Age=0 and SexBinary=0; interactions show how Age or Sex effects change when help is received.",
        "- Use 'simple_slopes_and_contrasts' entries to see whether Age or Sex effects differ significantly when help is received (p < 0.05 flagged).",
    ]
    description = " ".join(description_lines)

    return {"object": return_object, "description": description}