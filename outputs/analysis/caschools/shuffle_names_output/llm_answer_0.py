def extract_final_answer(model_output):
    """
    Extract statistics for the independent variable 'StudentTeacherRatio' from a
    statsmodels RegressionResultsWrapper and produce a brief interpretation.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, pvalue, conf_int, nobs, rsquared, statistically_significant)
      - "description": human-readable interpretation of the sign and significance in the context:
                       whether a lower student-teacher ratio (fewer students per teacher)
                       is associated with higher AvgScore.
    """
    # Defensive checks
    try:
        params = model_output.params
    except Exception:
        return {
            "object": None,
            "description": "Provided model_output does not appear to be a statsmodels RegressionResultsWrapper (no .params)."
        }

    var = "StudentTeacherRatio"
    if var not in params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in the fitted model. Available parameters: {list(params.index)}"
        }

    # Extract core statistics
    try:
        coef = float(model_output.params[var])
    except Exception:
        coef = None
    try:
        se = float(model_output.bse[var])
    except Exception:
        se = None
    try:
        pvalue = float(model_output.pvalues[var])
    except Exception:
        pvalue = None
    try:
        ci = model_output.conf_int().loc[var]
        ci_low = float(ci[0])
        ci_high = float(ci[1])
    except Exception:
        ci_low = ci_high = None
    # Additional context
    try:
        nobs = int(getattr(model_output, "nobs"))
    except Exception:
        nobs = None
    try:
        rsq = float(getattr(model_output, "rsquared", float("nan")))
    except Exception:
        rsq = None

    statistically_significant = (pvalue is not None) and (pvalue < 0.05)

    # Interpret direction in terms of "lower student-teacher ratio (fewer students per teacher)"
    if coef is None:
        direction_text = "Could not determine coefficient sign."
    else:
        # coef is change in AvgScore associated with a one-unit increase in StudentTeacherRatio.
        if coef < 0:
            # An increase in ratio (more students per teacher) reduces AvgScore,
            # so a lower ratio (fewer students per teacher) is associated with higher AvgScore.
            direction_text = (
                "Coefficient is negative: a lower student-teacher ratio (fewer students per teacher) "
                "is associated with higher district average academic performance (AvgScore)."
            )
        elif coef > 0:
            direction_text = (
                "Coefficient is positive: a lower student-teacher ratio (fewer students per teacher) "
                "is associated with lower district average academic performance (AvgScore)."
            )
        else:
            direction_text = "Coefficient is essentially zero: no directional association detected."

    # Compose a concise description incorporating significance
    sig_text = "statistically significant (p < 0.05)" if statistically_significant else "not statistically significant (p >= 0.05)"
    description = (
        f"StudentTeacherRatio coefficient = {coef:.4f}, SE = {se:.4f}, p-value = {pvalue:.4g}, "
        f"95% CI = [{ci_low:.4f}, {ci_high:.4f}] (n = {nobs}, R^2 = {rsq:.4f}). "
        f"{direction_text} The estimate is {sig_text}."
    )

    # Build the returned object with numeric fields for programmatic use
    result_object = {
        "variable": var,
        "coef": coef,
        "std_error": se,
        "p_value": pvalue,
        "conf_int_low": ci_low,
        "conf_int_high": ci_high,
        "nobs": nobs,
        "rsquared": rsq,
        "statistically_significant": statistically_significant
    }

    return {"object": result_object, "description": description}