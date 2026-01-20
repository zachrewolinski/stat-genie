def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, t-stat, p-value, 95% CI, and a brief
    interpretation for the StudentTeacherRatio coefficient from a fitted
    statsmodels RegressionResultsWrapper.

    Returns a dict with:
      - "object": dict with numeric results (coef, se, t, p, ci_lower, ci_upper, significant)
      - "description": text interpretation of what the numeric results imply
                       for whether a lower student-teacher ratio is associated
                       with higher academic performance.
    """
    # Ensure the expected variable exists in the model results
    var = 'StudentTeacherRatio'
    if var not in getattr(model_output, 'params').index:
        raise KeyError(f"Variable '{var}' not found in model_output.params")

    # Extract numeric statistics
    coef = float(model_output.params[var])
    se = float(model_output.bse[var]) if hasattr(model_output, 'bse') else None
    t_stat = float(model_output.tvalues[var]) if hasattr(model_output, 'tvalues') else None
    p_value = float(model_output.pvalues[var]) if hasattr(model_output, 'pvalues') else None

    # Confidence interval (95%)
    try:
        ci_row = model_output.conf_int().loc[var]
        ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
    except Exception:
        # fallback by position
        idx = list(model_output.params.index).index(var)
        ci_row = model_output.conf_int().iloc[idx]
        ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])

    # Determine significance at alpha = 0.05 (two-sided)
    significant = (p_value is not None) and (p_value < 0.05)

    # Interpretation: recall that StudentTeacherRatio is "students per teacher"
    # so a negative coefficient means that higher students-per-teacher -> lower scores,
    # i.e., lower student-teacher ratio (fewer students per teacher) -> higher scores.
    if p_value is None:
        interpretation = (
            "Estimated coefficient = {:.4f}. P-value unavailable, so statistical "
            "significance cannot be assessed.".format(coef)
        )
    else:
        if significant:
            if coef < 0:
                direction = (
                    "Statistically significant negative association: higher students-per-teacher "
                    "is associated with lower average test scores."
                )
                implication = (
                    "This implies that a lower student-teacher ratio (fewer students per teacher) "
                    "is associated with higher academic performance."
                )
            else:
                direction = (
                    "Statistically significant positive association: higher students-per-teacher "
                    "is associated with higher average test scores."
                )
                implication = (
                    "This implies that a lower student-teacher ratio (fewer students per teacher) "
                    "is associated with lower academic performance."
                )
            interpretation = (
                f"{direction} Coefficient = {coef:.4f} (SE = {se:.4f}, t = {t_stat:.3f}, p = {p_value:.3g}). "
                f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. {implication}"
            )
        else:
            interpretation = (
                f"Estimated coefficient = {coef:.4f} (SE = {se:.4f}, t = {t_stat:.3f}, p = {p_value:.3g}). "
                f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. The association is not statistically significant "
                "(alpha = 0.05), so the data do not provide strong evidence that student-teacher ratio is "
                "associated with average test scores after the included controls and county fixed effects."
            )

    result_object = {
        "variable": var,
        "coef": coef,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_at_0.05": bool(significant)
    }

    return {
        "object": result_object,
        "description": interpretation
    }