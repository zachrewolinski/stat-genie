def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, 95% CI, and sample size
    for the StudentTeacherRatio variable from a fitted statsmodels OLS result
    (RegressionResultsWrapper). Returns a dictionary with keys:
      - "object": a dict of numeric results
      - "description": a short interpretation in the context of the task

    Interpretation logic:
      - If the coefficient is negative and statistically significant (p < 0.05),
        this is interpreted as evidence that a lower student-teacher ratio
        (fewer students per teacher) is associated with higher AvgScore.
      - If the coefficient is positive and significant, it indicates the
        opposite.
      - If not significant, we state there is no strong evidence of an association.
    """
    import numpy as np

    res = model_output

    var = 'StudentTeacherRatio'
    # Safety checks
    try:
        params = res.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Provided model_output does not appear to be a fitted statsmodels results object: {e}"
        }

    if var not in params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in the fitted model parameters."
        }

    # Extract numeric statistics
    try:
        coef = float(res.params[var])
    except Exception:
        coef = float(np.asarray(res.params)[list(res.params.index).index(var)])

    try:
        se = float(res.bse[var])
    except Exception:
        se = float(np.asarray(res.bse)[list(res.bse.index).index(var)])

    try:
        pval = float(res.pvalues[var])
    except Exception:
        pval = float(np.asarray(res.pvalues)[list(res.pvalues.index).index(var)])

    # Confidence interval (default 95%)
    try:
        ci = res.conf_int().loc[var].tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        # fallback if conf_int returns array-like without index
        try:
            ci_arr = res.conf_int()
            idx = list(res.params.index).index(var)
            ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
        except Exception:
            ci_lower, ci_upper = (np.nan, np.nan)

    # Sample size if available
    nobs = getattr(res, "nobs", None)
    try:
        if nobs is not None:
            nobs = int(nobs)
    except Exception:
        nobs = None

    # Formulate conclusion
    alpha = 0.05
    if pval < alpha:
        if coef < 0:
            conclusion = (
                "Statistically significant (p < 0.05). The negative coefficient "
                "indicates that higher StudentTeacherRatio (more students per teacher) "
                "is associated with lower AvgScore; equivalently, a lower "
                "student-teacher ratio (fewer students per teacher) is associated "
                "with higher academic performance."
            )
        else:
            conclusion = (
                "Statistically significant (p < 0.05). The positive coefficient "
                "indicates that higher StudentTeacherRatio (more students per teacher) "
                "is associated with higher AvgScore (opposite to the commonly expected direction)."
            )
    else:
        if coef < 0:
            conclusion = (
                "Coefficient is negative but not statistically significant (p >= 0.05). "
                "There is no strong evidence that a lower student-teacher ratio is associated "
                "with higher academic performance in this model."
            )
        else:
            conclusion = (
                "Coefficient is positive but not statistically significant (p >= 0.05). "
                "There is no strong evidence of an association between student-teacher ratio "
                "and academic performance in this model."
            )

    result_object = {
        "variable": var,
        "coef": coef,
        "std_error": se,
        "p_value": pval,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "nobs": nobs
    }

    description = (
        f"Extracted results for '{var}': coefficient = {coef:.4f}, SE = {se:.4f}, "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}], p-value = {pval:.4g}. "
        + conclusion
    )

    return {"object": result_object, "description": description}