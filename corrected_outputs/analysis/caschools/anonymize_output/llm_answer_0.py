def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, t-value, p-value, and 95% CI for
    the StudentTeacherRatio_z predictor from a statsmodels RegressionResultsWrapper,
    interprets the direction and statistical significance, and reports the
    implied effect of a 1-standard-deviation decrease in student-teacher ratio.

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Concise interpretation and yes/no conclusion"
      }
    """
    var = 'StudentTeacherRatio_z'

    # Basic validation
    if not hasattr(model_output, 'params'):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params).")
    if var not in model_output.params.index:
        raise ValueError(f"The variable '{var}' is not present in the model results.")

    # Extract estimates
    coef = float(model_output.params[var])
    se = float(model_output.bse[var])
    tvalue = float(model_output.tvalues[var])
    pvalue = float(model_output.pvalues[var])

    # 95% CI (statsmodels: conf_int returns array-like with [lower, upper])
    ci = model_output.conf_int().loc[var]
    ci_lower = float(ci[0])
    ci_upper = float(ci[1])

    # Interpretation: the coefficient is the change in AvgTestScore for a 1 SD increase
    # in StudentTeacherRatio. For a 1 SD decrease, the change = -coef.
    effect_1sd_increase = coef
    effect_1sd_decrease = -coef
    ci_decrease_lower = -ci_upper  # flip signs for the decrease
    ci_decrease_upper = -ci_lower

    # Statistical significance at conventional alpha = 0.05
    significant = (pvalue < 0.05)

    # Short human-readable conclusion
    if coef < 0 and significant:
        conclusion = (
            "Yes. The estimated coefficient for StudentTeacherRatio_z is negative and "
            f"statistically significant (coef = {coef:.3f}, SE = {se:.3f}, "
            f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}], p = {pvalue:.3f}). "
            "Because the coefficient is the change in AvgTestScore for a 1-SD increase "
            "in student-teacher ratio, a 1-SD decrease (fewer students per teacher) is "
            f"associated with an increase in AvgTestScore of {effect_1sd_decrease:.3f} "
            f"(95% CI [{ci_decrease_lower:.3f}, {ci_decrease_upper:.3f}])."
        )
    elif coef < 0 and not significant:
        conclusion = (
            "No strong evidence. The estimated coefficient is negative (coef = "
            f"{coef:.3f}) suggesting that lower student-teacher ratios might be "
            "associated with higher AvgTestScore, but this effect is not statistically "
            f"significant (SE = {se:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}], p = {pvalue:.3f})."
        )
    elif coef > 0 and significant:
        conclusion = (
            "No. The estimated coefficient for StudentTeacherRatio_z is positive and "
            f"statistically significant (coef = {coef:.3f}, SE = {se:.3f}, "
            f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}], p = {pvalue:.3f}). "
            "This means higher student-teacher ratios (more students per teacher) are "
            "associated with higher AvgTestScore; equivalently, a 1-SD decrease in "
            "ratio is associated with a decrease in AvgTestScore of "
            f"{effect_1sd_decrease:.3f} (95% CI [{ci_decrease_lower:.3f}, {ci_decrease_upper:.3f}])."
        )
    else:  # coef == 0 (unlikely) or exactly non-significant and zero sign
        conclusion = (
            "No evidence of an association. The estimated coefficient is approximately zero "
            f"(coef = {coef:.3f}, SE = {se:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}], "
            f"p = {pvalue:.3f})."
        )

    result_object = {
        "variable": var,
        "coef": coef,
        "std_err": se,
        "t_value": tvalue,
        "p_value": pvalue,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "effect_1sd_increase": effect_1sd_increase,
        "effect_1sd_decrease": effect_1sd_decrease,
        "ci_1sd_decrease_lower": ci_decrease_lower,
        "ci_1sd_decrease_upper": ci_decrease_upper,
        "significant_at_0.05": bool(significant),
    }

    return {"object": result_object, "description": conclusion}