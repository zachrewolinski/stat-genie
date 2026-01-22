def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, t-stat, p-value, and 95% CI for the
    StudentTeacherRatio coefficient from a fitted statsmodels RegressionResultsWrapper.
    Returns a dictionary with "object" containing numeric results and a short
    conclusion, and "description" explaining what the numbers mean.
    """
    # Ensure required attributes exist
    if not hasattr(model_output, "params") or not hasattr(model_output, "pvalues"):
        raise TypeError("model_output does not appear to be a statsmodels results object with .params and .pvalues")

    name = "StudentTeacherRatio"

    params = model_output.params
    pvalues = model_output.pvalues
    bse = getattr(model_output, "bse", None)

    if name not in params.index:
        raise KeyError(f"Variable '{name}' not found in model parameters. Available params: {list(params.index)}")

    coef = float(params[name])
    se = float(bse[name]) if bse is not None and name in bse.index else None
    pval = float(pvalues[name])

    # Confidence interval (95% by default)
    try:
        ci_df = model_output.conf_int()
        if name in ci_df.index:
            ci_lower, ci_upper = float(ci_df.loc[name, 0]), float(ci_df.loc[name, 1])
        else:
            # fallback by position
            idx = list(params.index).index(name)
            ci_arr = model_output.conf_int().values
            ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
    except Exception:
        ci_lower, ci_upper = None, None

    # t-stat if available
    tstat = float(getattr(model_output, "tvalues", pd.Series()).get(name, float('nan'))) if 'pd' in globals() else float(model_output.tvalues[name])

    # Interpret direction: StudentTeacherRatio is students per teacher.
    # A negative coefficient implies that higher StudentTeacherRatio -> lower AvgScore,
    # i.e., lower ratio (fewer students per teacher) is associated with higher AvgScore.
    direction = "negative" if coef < 0 else ("positive" if coef > 0 else "zero")
    significance = (pval < 0.05)

    # Effect sizes for 1-unit and 5-unit changes
    effect_per_1 = coef
    effect_per_5 = coef * 5

    # Build a concise conclusion
    if coef < 0 and significance:
        conclusion = (
            "Evidence (p < 0.05) that lower student-teacher ratio (fewer students per teacher) "
            "is associated with higher district average test scores."
        )
    elif coef < 0 and not significance:
        conclusion = (
            "Point estimate suggests lower student-teacher ratio is associated with higher scores, "
            "but the effect is not statistically significant (p >= 0.05)."
        )
    elif coef > 0 and significance:
        conclusion = (
            "Evidence (p < 0.05) that higher student-teacher ratio (more students per teacher) "
            "is associated with higher district average test scores (opposite of the hypothesized direction)."
        )
    elif coef > 0 and not significance:
        conclusion = (
            "Point estimate suggests higher student-teacher ratio is associated with higher scores, "
            "but the effect is not statistically significant (p >= 0.05)."
        )
    else:
        conclusion = "No detectable association (coefficient essentially zero)."

    result_object = {
        "variable": name,
        "coefficient": coef,
        "std_error": se,
        "t_stat": tstat,
        "p_value": pval,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "effect_per_1_unit": effect_per_1,
        "effect_per_5_units": effect_per_5,
        "direction": direction,
        "statistically_significant_05": significance,
        "conclusion": conclusion
    }

    description = (
        "Extracted OLS estimate for StudentTeacherRatio from the fitted model. "
        "Coefficient is the change in AvgScore associated with a one-unit increase in students-per-teacher. "
        "Negative coefficient => fewer students per teacher (lower ratio) is associated with higher AvgScore. "
        "The dictionary 'object' contains the numeric estimates, 95% CI, and a short conclusion about statistical significance."
    )

    return {"object": result_object, "description": description}