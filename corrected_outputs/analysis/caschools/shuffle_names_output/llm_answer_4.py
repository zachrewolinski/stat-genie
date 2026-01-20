def extract_final_answer(model_output):
    """
    Extract statistics for the 'StudentTeacherRatio' coefficient from a fitted
    statsmodels OLS RegressionResultsWrapper (fit with robust cov_type if desired).
    Returns a dict with keys:
      - "object": dict of numeric results (coef, se, t, p, 95% CI, n_obs, R2, conclusion)
      - "description": short human-readable interpretation in context.
    """
    res = model_output
    result = {"object": None, "description": ""}

    try:
        param_index = res.params.index
    except Exception:
        result["description"] = "Provided model_output does not appear to be a statsmodels results object with params."
        return result

    var = 'StudentTeacherRatio'
    if var not in param_index:
        result["description"] = f"Variable '{var}' not found in the fitted model. Available parameters: {list(param_index)}"
        return result

    # Extract numeric statistics
    coef = float(res.params[var])
    # bse, tvalues, pvalues should reflect the fitted model's covariance (including robust cov if fit used it)
    se = float(res.bse[var]) if var in res.bse.index else float(res.bse[param_index.get_loc(var)])
    tval = float(res.tvalues[var])
    pval = float(res.pvalues[var])
    # 95% confidence interval
    try:
        ci = res.conf_int().loc[var].tolist()
    except Exception:
        # fallback if conf_int returns ndarray
        ci_arr = res.conf_int()
        # try to find row corresponding to var
        try:
            ci = ci_arr[param_index.get_loc(var)].tolist()
        except Exception:
            ci = [None, None]

    # Additional model info
    nobs = int(res.nobs) if hasattr(res, 'nobs') else None
    rsq = float(res.rsquared) if hasattr(res, 'rsquared') else None

    # Interpretation: because StudentTeacherRatio is defined as students/teachers (higher = more students per teacher),
    # a negative coefficient means that higher ratio (more students per teacher) is associated with lower AvgScore,
    # which is equivalent to saying lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore.
    alpha = 0.05
    if pval < alpha:
        if coef < 0:
            conclusion = ("Evidence of a statistically significant negative association: "
                          "lower student-teacher ratios (fewer students per teacher) are associated with higher average academic performance "
                          f"(coef = {coef:.4f}, p = {pval:.3g}).")
        else:
            conclusion = ("Evidence of a statistically significant positive association: "
                          "higher student-teacher ratios (more students per teacher) are associated with higher average academic performance "
                          f"(coef = {coef:.4f}, p = {pval:.3g}).")
    else:
        conclusion = ("No statistically significant association detected between student-teacher ratio and average academic performance "
                      f"(coef = {coef:.4f}, p = {pval:.3g}).")

    obj = {
        "variable": var,
        "coef": coef,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
        "95%_CI": [float(ci[0]) if ci[0] is not None else None, float(ci[1]) if ci[1] is not None else None],
        "n_obs": nobs,
        "r_squared": rsq,
        "conclusion": conclusion
    }

    result["object"] = obj
    result["description"] = (
        "Extracted coefficient, uncertainty, and statistical test for 'StudentTeacherRatio'. "
        "Interpretation: negative coefficient implies that lower student-teacher ratios (fewer students per teacher) "
        "are associated with higher AvgScore. Statistical significance is judged at alpha=0.05; see 'conclusion' in the object."
    )
    return result