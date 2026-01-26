def extract_final_answer(model_output):
    """
    Extracts the coefficient and related statistics for the 'student_teacher_ratio'
    variable from a fitted statsmodels RegressionResults object.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, pvalue, 95% CI, nobs, significant)
      - "description": brief interpretation in context (direction and significance)
    """
    res = model_output

    # Basic validation
    if not hasattr(res, "params"):
        return {
            "object": None,
            "description": "The provided model_output does not appear to be a fitted statsmodels results object (no .params)."
        }

    var = "student_teacher_ratio"
    try:
        params_index = list(res.params.index)
    except Exception:
        # params might be a plain array; cannot locate by name
        params_index = []

    if var not in params_index:
        return {
            "object": None,
            "description": f"The model does not include the variable '{var}'. Available parameters: {params_index}"
        }

    # Extract main statistics, robust to different shapes of conf_int output
    try:
        coef = float(res.params[var])
    except Exception:
        coef = None

    try:
        se = float(res.bse[var]) if hasattr(res, "bse") else None
    except Exception:
        se = None

    try:
        pvalue = float(res.pvalues[var]) if hasattr(res, "pvalues") else None
    except Exception:
        pvalue = None

    # Confidence interval (95%)
    ci_lower, ci_upper = None, None
    try:
        ci = res.conf_int(alpha=0.05)
        try:
            # If ci is pandas-like and has the variable as an index
            if hasattr(ci, "loc") and var in getattr(ci, "index", []):
                row = ci.loc[var]
                # row may be a Series-like
                ci_lower = float(row.iloc[0]) if hasattr(row, "iloc") else float(row[0])
                ci_upper = float(row.iloc[1]) if hasattr(row, "iloc") else float(row[1])
            else:
                # fallback: assume numpy array or DataFrame without var index
                idx = params_index.index(var)
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])
        except Exception:
            ci_lower, ci_upper = None, None
    except Exception:
        ci_lower, ci_upper = None, None

    # Sample size if available
    try:
        nobs = int(res.nobs)
    except Exception:
        nobs = None

    # Significance at conventional levels
    significant = None
    if pvalue is not None:
        significant = {
            "p_lt_0.01": pvalue < 0.01,
            "p_lt_0.05": pvalue < 0.05,
            "p_lt_0.10": pvalue < 0.10
        }

    # Prepare formatted strings for interpretation (avoid formatting None with numeric specifiers)
    coef_str = f"{coef:.4f}" if coef is not None else "None"
    se_str = f"{se:.4f}" if se is not None else "None"
    ci_lower_str = f"{ci_lower:.4f}" if ci_lower is not None else "None"
    ci_upper_str = f"{ci_upper:.4f}" if ci_upper is not None else "None"

    # Interpretation: sign of coef -> relationship direction
    if coef is None:
        interpretation = "Could not extract coefficient value for interpretation."
    else:
        if coef < 0:
            direction = ("A negative coefficient means that a higher student-teacher ratio "
                         "(more students per teacher) is associated with lower district AvgScore. "
                         "Equivalently, a lower student-teacher ratio (fewer students per teacher) "
                         "is associated with higher AvgScore.")
        elif coef > 0:
            direction = ("A positive coefficient means that a higher student-teacher ratio "
                         "(more students per teacher) is associated with higher district AvgScore. "
                         "Equivalently, a lower student-teacher ratio (fewer students per teacher) "
                         "is associated with lower AvgScore.")
        else:
            direction = "The coefficient is exactly zero (no linear association detected)."

        sig_text = ""
        if significant is not None:
            if significant["p_lt_0.05"]:
                sig_text = " The effect is statistically significant at the 5% level."
            elif significant["p_lt_0.10"]:
                sig_text = " The effect is marginally significant at the 10% level."
            else:
                sig_text = " The effect is not statistically significant at conventional levels."

        interpretation = (
            f"Estimated effect: each additional student per teacher is associated with a change of {coef_str} "
            f"points in AvgScore (SE={se_str}). 95% CI = [{ci_lower_str}, {ci_upper_str}]."
            + " " + direction + sig_text
        )

    result_object = {
        "variable": var,
        "coefficient": coef,
        "std_error": se,
        "p_value": pvalue,
        "conf_int_95": [ci_lower, ci_upper],
        "nobs": nobs,
        "significant": significant
    }

    return {
        "object": result_object,
        "description": interpretation
    }