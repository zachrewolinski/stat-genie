def extract_final_answer(model_output):
    """
    Extract statistics for the effect of StudentTeacherRatio from a fitted statsmodels OLS results object.
    Returns a dictionary with keys:
      - "object": a dict of extracted numeric results (coef, se, t, p, 95% CI, nobs, r_squared, conclusion)
      - "description": a short interpretation of what the numbers mean for the question:
                       "Is a lower student-teacher ratio associated with higher academic performance?"
    """
    # Name of the variable of interest
    var = 'StudentTeacherRatio'
    
    # Basic checks
    if model_output is None:
        return {
            "object": None,
            "description": "No model_output provided."
        }
    if not hasattr(model_output, 'params'):
        return {
            "object": None,
            "description": "Provided model_output does not look like a fitted statsmodels results object (missing .params)."
        }
    try:
        params_index = list(model_output.params.index)
    except Exception:
        params_index = None

    if params_index is None or var not in params_index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in the fitted model's parameters. Cannot extract its effect."
        }

    # Extract coefficient and statistics
    try:
        coef = float(model_output.params[var])
    except Exception:
        coef = None
    try:
        se = float(model_output.bse[var]) if hasattr(model_output, 'bse') else None
    except Exception:
        se = None
    try:
        tval = float(model_output.tvalues[var]) if hasattr(model_output, 'tvalues') else None
    except Exception:
        tval = None
    try:
        pval = float(model_output.pvalues[var]) if hasattr(model_output, 'pvalues') else None
    except Exception:
        pval = None

    # 95% confidence interval (attempt robust-aware conf_int)
    ci_lower = ci_upper = None
    try:
        ci_all = model_output.conf_int()
        # conf_int may return a DataFrame or ndarray
        if hasattr(ci_all, 'loc'):
            ci_lower, ci_upper = map(float, ci_all.loc[var])
        else:
            # assume numpy array in same order as params
            idx = params_index.index(var)
            ci_lower, ci_upper = map(float, ci_all[idx])
    except Exception:
        ci_lower = ci_upper = None

    # Additional info
    try:
        nobs = int(model_output.nobs)
    except Exception:
        nobs = None
    try:
        r_squared = float(model_output.rsquared)
    except Exception:
        r_squared = None

    # Interpretation relative to the research question:
    # - StudentTeacherRatio is students per teacher. Lower ratio = fewer students per teacher.
    # - A negative coef means that increasing StudentTeacherRatio (more students per teacher)
    #   is associated with lower AvgScore; equivalently, a lower ratio is associated with higher AvgScore.
    # - We judge "associated" if p < 0.05 (two-sided).
    conclusion = ""
    if pval is None:
        conclusion = "Could not determine statistical significance (p-value unavailable)."
    else:
        if pval < 0.05:
            if coef is not None and coef < 0:
                conclusion = (
                    "Yes — statistically significant association: the coefficient is negative "
                    f"({coef:.4f}, p = {pval:.3g}), meaning that a lower student–teacher ratio "
                    "is associated with higher average district academic performance. "
                    f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]."
                )
            elif coef is not None and coef > 0:
                conclusion = (
                    "Statistically significant association, but the coefficient is positive "
                    f"({coef:.4f}, p = {pval:.3g}), meaning that a lower student–teacher ratio "
                    "is associated with lower average performance (the opposite of the hypothesized direction). "
                    f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]."
                )
            else:
                conclusion = f"Statistically significant (p = {pval:.3g}) but coefficient unavailable."
        else:
            # Not significant
            if coef is not None:
                direction = "negative" if coef < 0 else ("positive" if coef > 0 else "zero")
                conclusion = (
                    "No — the association is not statistically significant at the 0.05 level. "
                    f"Coefficient = {coef:.4f} ({direction}), p = {pval:.3g}. "
                    "This provides no strong evidence that a lower student–teacher ratio is associated with higher AvgScore."
                )
            else:
                conclusion = f"No — association not statistically significant (p = {pval:.3g})."

    # Pack the numeric results as the "object" so they can be consumed programmatically
    result_object = {
        "variable": var,
        "coef": coef,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
        "ci_95": (ci_lower, ci_upper),
        "nobs": nobs,
        "r_squared": r_squared,
        "conclusion": conclusion
    }

    description = (
        "Extracted coefficient and inference for StudentTeacherRatio from the fitted model. "
        "Coefficient is the estimated change in AvgScore for a one-unit increase in StudentTeacherRatio "
        "(one more student per teacher). A negative, statistically significant coefficient indicates that "
        "lower student–teacher ratios (fewer students per teacher) are associated with higher AvgScore."
    )

    return {
        "object": result_object,
        "description": description
    }