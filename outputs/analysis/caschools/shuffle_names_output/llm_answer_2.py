def extract_final_answer(model_output):
    """
    Extracts the coefficient, robust SE, t-value, p-value, 95% CI, sample size,
    and a concise interpretation for the StudentTeacherRatio variable from a
    statsmodels RegressionResultsWrapper (or compatible) object.

    Returns a dictionary with keys:
      - "object": dict of numeric results and a boolean 'significant'
      - "description": human-readable interpretation in context
    """
    res = model_output

    var = 'StudentTeacherRatio'
    # Basic checks
    try:
        params = res.params
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a fitted statsmodels results object.") from e

    if var not in params.index:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    # Extract statistics (convert to native Python types)
    coef = float(res.params[var])
    # bse, tvalues, pvalues should exist for fitted results with cov_type specified
    se = float(res.bse[var]) if hasattr(res, 'bse') else None
    tvalue = float(res.tvalues[var]) if hasattr(res, 'tvalues') else None
    pvalue = float(res.pvalues[var]) if hasattr(res, 'pvalues') else None

    # 95% CI (if available)
    try:
        ci_df = res.conf_int(alpha=0.05)
        ci_lower = float(ci_df.loc[var, 0])
        ci_upper = float(ci_df.loc[var, 1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Sample size if available
    nobs = int(res.nobs) if hasattr(res, 'nobs') else None

    # Significance at alpha = 0.05
    significant = (pvalue is not None) and (pvalue < 0.05)

    # Interpretation in context:
    # Note: StudentTeacherRatio is "students per teacher".
    if coef < 0:
        direction_text = (
            "negative: higher student-teacher ratio (more students per teacher) is associated "
            "with LOWER average 5th-grade scores; equivalently, LOWER student-teacher ratios "
            "(fewer students per teacher) are associated with HIGHER scores."
        )
    elif coef > 0:
        direction_text = (
            "positive: higher student-teacher ratio (more students per teacher) is associated "
            "with HIGHER average 5th-grade scores; equivalently, LOWER student-teacher ratios "
            "are associated with LOWER scores."
        )
    else:
        direction_text = "coefficient is exactly zero (no association detected)."

    significance_text = "statistically significant at alpha=0.05" if significant else "NOT statistically significant at alpha=0.05"

    desc = (
        f"StudentTeacherRatio coefficient = {coef:.4f} (SE = {se:.4f}, t = {tvalue:.3f}, p = {pvalue:.3g}). "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. Sample size n = {nobs}. "
        f"Interpretation: {direction_text} The effect is {significance_text}."
    )

    result_object = {
        "variable": var,
        "coef": coef,
        "se": se,
        "tvalue": tvalue,
        "pvalue": pvalue,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "nobs": nobs,
        "significant_at_0.05": significant
    }

    return {"object": result_object, "description": desc}