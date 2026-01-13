def extract_final_answer(model_output):
    """
    Extracts key statistics for the 'StudentTeacherRatio' coefficient from a fitted
    statsmodels RegressionResultsWrapper and returns a concise interpretation.

    Returns a dict with:
      - "object": dict with numeric results (coef, se, t, p, 95% CI, effect per 10 students, conclusion_bool)
      - "description": human-readable explanation of what these numbers mean for the question
                       "Is a lower student-teacher ratio associated with higher academic performance?"
    """
    # Basic validation
    if model_output is None:
        raise ValueError("model_output is None")
    # Ensure it's a statsmodels results object with necessary attributes
    for attr in ("params", "bse", "tvalues", "pvalues", "conf_int"):
        if not hasattr(model_output, attr):
            raise ValueError(f"model_output missing required attribute: {attr}")

    param_name = 'StudentTeacherRatio'
    params = model_output.params
    if param_name not in params.index:
        raise ValueError(f"Parameter '{param_name}' not found in model parameters: {list(params.index)}")

    coef = float(model_output.params[param_name])
    se = float(model_output.bse[param_name]) if param_name in model_output.bse.index else None
    t_stat = float(model_output.tvalues[param_name]) if param_name in model_output.tvalues.index else None
    p_value = float(model_output.pvalues[param_name]) if param_name in model_output.pvalues.index else None

    # 95% CI
    try:
        ci_df = model_output.conf_int(alpha=0.05)
        ci_lower, ci_upper = (float(ci_df.loc[param_name, 0]), float(ci_df.loc[param_name, 1]))
    except Exception:
        ci_lower, ci_upper = (None, None)

    # Practical interpretation: effect per 10-student change
    effect_per_10 = coef * 10.0

    # Conclusion about whether lower ratio (fewer students per teacher) is associated with higher performance.
    # Note: StudentTeacherRatio is "students per teacher"; a negative coef implies that increasing ratio (more students)
    # decreases scores, so lower ratio is associated with higher scores.
    if p_value is None:
        conclusion_bool = None
        conclusion_text = "Could not determine statistical significance (p-value not available)."
    else:
        statistically_significant = (p_value < 0.05)
        if coef < 0:
            direction = "lower student-teacher ratio is associated with higher AvgScore"
        elif coef > 0:
            direction = "lower student-teacher ratio is associated with lower AvgScore"
        else:
            direction = "no association (coefficient is zero)"

        if statistically_significant:
            conclusion_bool = (coef < 0)  # True if lower ratio -> higher score (coef negative) and significant
            concl_sign = "statistically significant (p < 0.05)"
        else:
            conclusion_bool = False
            concl_sign = "not statistically significant (p >= 0.05)"

        conclusion_text = (
            f"The estimated coefficient on StudentTeacherRatio is {coef:.4f} (SE={se:.4f}, t={t_stat:.2f}, p={p_value:.3g}). "
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}].\n"
            f"Interpretation: a one-unit increase in StudentTeacherRatio (one more student per teacher) is associated "
            f"with a change of {coef:.4f} points in AvgScore. Equivalently, reducing the ratio by 10 students is "
            f"associated with a change of {effect_per_10:.4f} points.\n"
            f"Direction: {direction}. Statistical significance: {concl_sign}.\n"
            f"Answer to the question 'Is a lower student-teacher ratio associated with higher academic performance?': "
            f"{'Yes' if conclusion_bool else 'No (not supported)'}."
        )

    result_object = {
        "coef": coef,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_per_10_students": effect_per_10,
        # conclusion_bool is True only when coef < 0 and statistically significant at 5%
        "conclusion_bool": conclusion_bool,
    }

    return {
        "object": result_object,
        "description": conclusion_text
    }