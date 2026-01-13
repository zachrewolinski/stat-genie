def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted statsmodels OLS result.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results:
           {
             "variable": variable name,
             "coef": float,
             "std_err": float,
             "t_value": float,
             "p_value": float,
             "ci_lower": float,
             "ci_upper": float,
             "significant": bool,
             "alpha": 0.05,
             "interpretation": str  # short conclusion about association
           }
      - "description": a short human-readable explanation of what the numbers mean
    """
    res = model_output

    var_name = 'StudentTeacherRatio'
    # Defensive checks
    try:
        params = res.params
    except Exception as e:
        raise ValueError("Provided model_output does not look like a statsmodels results object.") from e

    if var_name not in params.index:
        # try common alternative spellings (if model used different naming)
        alt_matches = [v for v in params.index if 'Student' in v and 'Teacher' in v]
        if len(alt_matches) == 1:
            var_name = alt_matches[0]
        else:
            raise KeyError(f"Could not find variable 'StudentTeacherRatio' in model parameters. Available params: {list(params.index)}")

    coef = float(res.params[var_name])
    std_err = float(res.bse[var_name]) if hasattr(res, 'bse') else None
    t_value = float(res.tvalues[var_name]) if hasattr(res, 'tvalues') else None
    p_value = float(res.pvalues[var_name]) if hasattr(res, 'pvalues') else None

    # 95% CI (if available)
    try:
        ci = res.conf_int(alpha=0.05).loc[var_name].tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None

    alpha = 0.05
    significant = (p_value is not None) and (p_value < alpha)

    # Interpretation logic:
    # Note: StudentTeacherRatio is measured as students per teacher (higher = larger classes).
    # A negative coefficient => higher ratio -> lower AvgScore, so lower ratio -> higher AvgScore.
    if significant:
        if coef < 0:
            interpretation = (
                "Statistically significant negative association: higher student-teacher ratio "
                "(more students per teacher) is associated with LOWER average test scores. "
                "Equivalently, a LOWER student-teacher ratio is associated with HIGHER academic performance."
            )
        else:
            interpretation = (
                "Statistically significant positive association: higher student-teacher ratio "
                "(more students per teacher) is associated with HIGHER average test scores. "
                "This is contrary to the expectation that smaller classes help performance."
            )
    else:
        if coef < 0:
            interpretation = (
                "Coefficient is negative (consistent with the hypothesis that smaller student-teacher "
                "ratios are associated with higher scores), but the effect is NOT statistically significant "
                f"at alpha={alpha}. Evidence is inconclusive."
            )
        elif coef > 0:
            interpretation = (
                "Coefficient is positive (suggesting larger ratios associate with higher scores), but the effect "
                f"is NOT statistically significant at alpha={alpha}. Evidence is inconclusive."
            )
        else:
            interpretation = "Estimated effect is essentially zero and not statistically significant."

    result_object = {
        "variable": var_name,
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "alpha": alpha,
        "interpretation": interpretation
    }

    description = (
        "Extracted coefficient, standard error, t-value, p-value, and 95% confidence interval for the "
        f"student-teacher ratio (variable '{var_name}'). The coefficient indicates the expected change in "
        "district average test score associated with a one-unit increase in students per teacher. "
        "If the coefficient is negative and statistically significant, that supports the claim that a lower "
        "student-teacher ratio (fewer students per teacher) is associated with higher academic performance."
    )

    return {"object": result_object, "description": description}