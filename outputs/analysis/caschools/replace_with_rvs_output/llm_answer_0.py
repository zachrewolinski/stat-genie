def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels
    RegressionResultsWrapper and returns a concise conclusion about whether a lower
    student-teacher ratio is associated with higher average 5th-grade performance.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, p_value, 95% CI) and a boolean conclusion
      - "description": brief human-readable interpretation in context
    """
    import numpy as np

    # Helper to find the exact parameter name if it's not exactly 'StudentTeacherRatio'
    param_name = None
    try:
        params_index = list(model_output.params.index)
    except Exception:
        # If model_output doesn't provide params, raise informative error
        raise ValueError("Provided model_output does not look like a fitted statsmodels results object.")

    for name in params_index:
        if name == 'StudentTeacherRatio' or 'StudentTeacherRatio' in name:
            param_name = name
            break
    if param_name is None:
        raise KeyError(f"'StudentTeacherRatio' not found in model parameters. Available params: {params_index}")

    # Extract statistics
    coef = float(model_output.params[param_name])
    se = float(model_output.bse[param_name])
    p_value = float(model_output.pvalues[param_name])

    # Confidence interval: handle DataFrame or ndarray returned by conf_int()
    try:
        ci = model_output.conf_int().loc[param_name]
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except Exception:
        # fallback: conf_int may return numpy array in same order as params_index
        ci_array = model_output.conf_int()
        idx = params_index.index(param_name)
        ci_low, ci_high = float(ci_array[idx, 0]), float(ci_array[idx, 1])

    # Interpretation logic:
    # - A negative coefficient means that increasing StudentTeacherRatio (more students per teacher)
    #   is associated with lower AvgScore. Therefore, a lower ratio (fewer students per teacher)
    #   would be associated with higher AvgScore when coef < 0.
    # - We consider p < 0.05 as evidence of statistical significance.
    direction = "negative" if coef < 0 else ("positive" if coef > 0 else "zero")
    significant = p_value < 0.05

    if coef < 0 and significant:
        conclusion_bool = True
        conclusion_text = (
            "Yes. The estimated coefficient for StudentTeacherRatio is negative and statistically significant "
            f"(coef = {coef:.4f}, SE = {se:.4f}, p = {p_value:.3g}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]). "
            "This implies that a lower student-teacher ratio (fewer students per teacher) is associated with higher "
            "average 5th-grade performance (AvgScore), holding controls constant."
        )
    elif coef < 0 and not significant:
        conclusion_bool = False
        conclusion_text = (
            "No strong evidence. The estimated coefficient for StudentTeacherRatio is negative but not statistically "
            f"significant (coef = {coef:.4f}, SE = {se:.4f}, p = {p_value:.3g}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]). "
            "The point estimate suggests that lower ratios might be associated with higher scores, but this effect is "
            "not distinguishable from zero at conventional significance levels."
        )
    elif coef > 0 and significant:
        conclusion_bool = False
        conclusion_text = (
            "No (and opposite). The estimated coefficient for StudentTeacherRatio is positive and statistically significant "
            f"(coef = {coef:.4f}, SE = {se:.4f}, p = {p_value:.3g}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]). "
            "This implies that a lower student-teacher ratio (fewer students per teacher) would be associated with "
            "lower AvgScore (the effect is in the opposite direction of the hypothesis)."
        )
    else:  # coef == 0 (very unlikely) or inconclusive positive/zero not significant
        conclusion_bool = False
        conclusion_text = (
            "No evidence of an effect. The estimated coefficient is approximately zero or not statistically significant "
            f"(coef = {coef:.4f}, SE = {se:.4f}, p = {p_value:.3g}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}])."
        )

    result_object = {
        "parameter": param_name,
        "coef": coef,
        "std_error": se,
        "p_value": p_value,
        "95%_CI": (ci_low, ci_high),
        "direction": direction,
        "significant_at_0.05": bool(significant),
        # Final boolean answer to the question "Is a lower student-teacher ratio associated with higher academic performance?"
        "lower_ratio_assoc_with_higher_performance": bool(conclusion_bool),
        "conclusion_text": conclusion_text
    }

    description = (
        "Extracted coefficient, robust standard error, p-value, and 95% CI for the StudentTeacherRatio term, "
        "and interpreted whether a lower student-teacher ratio is associated with higher AvgScore based on sign "
        "and statistical significance (alpha = 0.05). See 'object' for numeric results and the conclusion text."
    )

    return {"object": result_object, "description": description}