def extract_final_answer(model_output):
    """
    Extracts the coefficient, robust SE, t-stat, p-value, and 95% CI for StudentTeacherRatio
    from a fitted statsmodels RegressionResultsWrapper and returns a short conclusion.

    Returns a dictionary with keys:
      - "object": dict with extracted numeric results and a short boolean/text conclusion
      - "description": brief explanation of what the numbers mean in context
    """
    res = model_output

    # Basic validation
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    param = "StudentTeacherRatio"
    if param not in res.params.index:
        raise ValueError(f"Model does not contain parameter '{param}'.")

    # Extract statistics (fit was called with cov_type='HC3', so these reflect robust estimates)
    coef = float(res.params[param])
    std_err = float(res.bse[param])
    t_value = float(res.tvalues[param])
    p_value = float(res.pvalues[param])

    # 95% confidence interval (uses the fitted results' cov type)
    try:
        ci_series = res.conf_int().loc[param]
        ci_lower = float(ci_series[0])
        ci_upper = float(ci_series[1])
    except Exception:
        # Fallback if conf_int() is unavailable for some reason
        ci_lower = coef - 1.96 * std_err
        ci_upper = coef + 1.96 * std_err

    # Interpret the direction and statistical significance
    alpha = 0.05
    if p_value < alpha:
        if coef < 0:
            conclusion_text = (
                "Yes — statistically significant negative association: "
                "lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore "
                f"(coef = {coef:.4f}, p = {p_value:.3g})."
            )
        else:
            conclusion_text = (
                "No — statistically significant positive association: "
                "higher student-teacher ratio is associated with higher AvgScore (opposite of the hypothesized direction) "
                f"(coef = {coef:.4f}, p = {p_value:.3g})."
            )
    else:
        if coef < 0:
            conclusion_text = (
                "No strong evidence of an association at alpha = 0.05: "
                "the point estimate is negative (suggesting lower ratio -> higher AvgScore) but it is not statistically significant "
                f"(coef = {coef:.4f}, p = {p_value:.3g})."
            )
        else:
            conclusion_text = (
                "No strong evidence of an association at alpha = 0.05: "
                "the point estimate is positive (suggesting higher ratio -> higher AvgScore) and is not statistically significant "
                f"(coef = {coef:.4f}, p = {p_value:.3g})."
            )

    result_object = {
        "parameter": param,
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "95%_CI": [ci_lower, ci_upper],
        "significant_at_0.05": p_value < alpha,
        "conclusion_text": conclusion_text
    }

    description = (
        "Extracted the estimated coefficient and robust inference for StudentTeacherRatio from the OLS model "
        "that controls for expenditure, income, lunch percent, English learner percent, computers per student, and district grade span. "
        "A negative coefficient means that a lower student-teacher ratio (smaller classes) is associated with higher average test scores."
    )

    return {"object": result_object, "description": description}