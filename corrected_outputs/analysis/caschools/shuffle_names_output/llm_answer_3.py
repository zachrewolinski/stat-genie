def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels
    RegressionResultsWrapper (or similar) and returns a brief interpretation.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, t, p, ci_lower, ci_upper)
      - "description": textual interpretation of the sign, magnitude, and significance
                       in the context of whether a lower student-teacher ratio is
                       associated with higher academic performance.
    """
    res = model_output

    var = 'StudentTeacherRatio'
    # Check presence
    if var not in res.params.index:
        raise ValueError(f"Variable '{var}' not found in model results. Available params: {list(res.params.index)}")

    # Extract numeric estimates (convert to Python floats)
    coef = float(res.params[var])
    try:
        se = float(res.bse[var])
    except Exception:
        # fallback: compute se from cov_params diagonal if available
        cov = res.cov_params()
        se = float((cov.loc[var, var] ** 0.5) if hasattr(cov, 'loc') else (cov[res.params.index.get_loc(var), res.params.index.get_loc(var)] ** 0.5))

    tval = float(res.tvalues[var]) if hasattr(res, 'tvalues') else (coef / se if se != 0 else float('nan'))
    pval = float(res.pvalues[var]) if hasattr(res, 'pvalues') else float('nan')

    # Confidence interval (95%)
    try:
        ci = res.conf_int(alpha=0.05)
        # ci may be a DataFrame or ndarray
        if hasattr(ci, 'loc'):
            ci_lower = float(ci.loc[var].iloc[0])
            ci_upper = float(ci.loc[var].iloc[1])
        else:
            # ndarray: find index position
            idx = list(res.params.index).index(var)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
    except Exception:
        # fallback using coef +/- 1.96*se
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

    # Build numeric object
    numeric_result = {
        "variable": var,
        "coef": coef,
        "se": se,
        "t": tval,
        "p": pval,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "interpretation_note": "Coefficient units: change in AvgTestScore per one additional student per teacher (district-level)."
    }

    # Interpret direction and significance
    alpha = 0.05
    if pval < alpha:
        signif = "statistically significant (p < 0.05)"
    else:
        signif = "not statistically significant (p >= 0.05)"

    if coef < 0:
        direction = ("Higher student-teacher ratio (more students per teacher) is associated with LOWER "
                     "average test scores. Equivalently, a LOWER student-teacher ratio (fewer students per teacher) "
                     "is associated with HIGHER academic performance.")
    elif coef > 0:
        direction = ("Higher student-teacher ratio (more students per teacher) is associated with HIGHER "
                     "average test scores. Equivalently, a LOWER student-teacher ratio is associated with LOWER academic performance.")
    else:
        direction = "No association detected (coefficient is zero)."

    description = (
        f"Estimated effect of StudentTeacherRatio on AvgTestScore: coef = {coef:.4f}, SE = {se:.4f}, "
        f"t = {tval:.3f}, p = {pval:.3g}. 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"This effect is {signif}. {direction} "
        "Model was weighted by StudentsTotal and controls included ExpenditurePerStudent, "
        "PctReducedLunch, PctEnglishLearners, NumComputers, and AvgIncome."
    )

    return {"object": numeric_result, "description": description}