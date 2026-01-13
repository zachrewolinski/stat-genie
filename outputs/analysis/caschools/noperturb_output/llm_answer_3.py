def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, p-value, and 95% CI for the StudentTeacherRatio variable
    from a statsmodels RegressionResultsWrapper and returns an interpretation.

    Returns a dict with keys:
      - "object": dict with numeric results:
           { "coef", "std_err", "p_value", "ci_lower", "ci_upper", "n_obs",
             "df_resid", "significant_at_0.05" }
      - "description": short textual interpretation of what the numbers imply
                       about whether a lower student-teacher ratio is associated
                       with higher academic performance.
    """
    res = model_output

    # Name of the variable we care about
    var = 'StudentTeacherRatio'

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    if var not in res.params.index:
        raise ValueError(f"Variable '{var}' not found in model results. Available params: {list(res.params.index)}")

    # Extract point estimate, standard error, p-value
    coef = float(res.params[var])
    # Use robust SE/pvalues that are part of the fitted results object
    try:
        std_err = float(res.bse[var])
    except Exception:
        # fallback if bse not available as Series-like
        std_err = float(res.bse[list(res.params.index).index(var)])

    try:
        p_value = float(res.pvalues[var])
    except Exception:
        p_value = float(res.pvalues[list(res.params.index).index(var)])

    # Confidence interval (95% by default)
    try:
        ci = res.conf_int().loc[var]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # conf_int may return ndarray in some versions; handle that
        ci_array = res.conf_int()
        idx = list(res.params.index).index(var)
        ci_lower = float(ci_array[idx, 0])
        ci_upper = float(ci_array[idx, 1])

    # Sample size and residual df if available
    try:
        n_obs = int(res.nobs)
    except Exception:
        n_obs = None
    try:
        df_resid = float(res.df_resid)
    except Exception:
        df_resid = None

    # Interpretation:
    # StudentTeacherRatio is "students per teacher". A negative coef means that
    # higher ratio (more students per teacher) is associated with lower AvgScore,
    # i.e., a lower ratio (fewer students per teacher / smaller classes) is
    # associated with higher AvgScore.
    coef_sign = "negative" if coef < 0 else ("positive" if coef > 0 else "zero")
    significant = (p_value < 0.05) if (p_value is not None) else False

    if coef < 0 and significant:
        conclusion = (
            "Coefficient is negative and statistically significant at the 5% level: "
            "this provides evidence that lower student-teacher ratios (smaller class sizes) "
            "are associated with higher district average academic performance."
        )
    elif coef < 0 and not significant:
        conclusion = (
            "Coefficient is negative but not statistically significant at the 5% level: "
            "point estimate suggests lower student-teacher ratios are associated with higher performance, "
            "but the evidence is weak (not statistically significant)."
        )
    elif coef > 0 and significant:
        conclusion = (
            "Coefficient is positive and statistically significant at the 5% level: "
            "this indicates higher student-teacher ratios (more students per teacher) are associated "
            "with higher district average academic performance (contrary to the hypothesis)."
        )
    elif coef > 0 and not significant:
        conclusion = (
            "Coefficient is positive but not statistically significant at the 5% level: "
            "no strong evidence that student-teacher ratio is associated with performance in either direction."
        )
    else:
        conclusion = "Coefficient is essentially zero; no association detected."

    result_object = {
        "variable": var,
        "coef": coef,
        "std_err": std_err,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "df_resid": df_resid,
        "significant_at_0.05": significant,
        "coef_sign": coef_sign,
    }

    description = (
        f"Estimate for '{var}': coef = {coef:.4f}, SE = {std_err:.4f}, p = {p_value:.4g}, "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}].\n"
        f"Interpretation: {conclusion} "
        f"(Recall: '{var}' is students per teacher, so a negative coefficient means smaller class sizes -> higher scores.)"
    )

    return {"object": result_object, "description": description}