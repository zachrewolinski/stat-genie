def extract_final_answer(model_output):
    """
    Extracts statistics for the 'stu_teacher_ratio' coefficient from a fitted statsmodels OLS result.

    Returns a dictionary with:
      - "object": a dict containing coefficient, std err, t-stat, p-value, 95% CI and a short conclusion.
      - "description": a brief interpretation of these statistics in the context of whether a lower
                       student-teacher ratio is associated with higher academic performance.

    Expects model_output to be a statsmodels RegressionResultsWrapper (the object returned by .fit()).
    """
    result = {"object": None, "description": None}

    param_name = "stu_teacher_ratio"
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
        tvalues = model_output.tvalues
        # conf_int may be a DataFrame or ndarray; handle both
        conf_int = model_output.conf_int()
    except Exception as e:
        result["description"] = f"Error: provided model_output does not look like a fitted statsmodels results object: {e}"
        return result

    if param_name not in params.index:
        result["description"] = f"Parameter '{param_name}' not found in the model results. Available params: {list(params.index)}"
        return result

    coef = float(params[param_name])
    se = float(bse[param_name]) if param_name in bse.index else None
    tstat = float(tvalues[param_name]) if param_name in tvalues.index else None
    pval = float(pvalues[param_name]) if param_name in pvalues.index else None

    # Extract 95% CI robustly
    try:
        # If conf_int is a DataFrame with index
        if hasattr(conf_int, "loc"):
            ci_lower, ci_upper = float(conf_int.loc[param_name, 0]), float(conf_int.loc[param_name, 1])
        else:
            # conf_int as ndarray: find row corresponding to param (assume same order as params)
            idx = list(params.index).index(param_name)
            ci_lower, ci_upper = float(conf_int[idx, 0]), float(conf_int[idx, 1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Interpret direction and significance.
    alpha = 0.05
    if pval is None:
        conclusion = "Could not determine statistical significance (p-value not available)."
    else:
        if pval < alpha:
            # Significant effect
            if coef < 0:
                conclusion = (
                    "Statistically significant negative association: higher student-teacher ratios (more students per teacher) "
                    "are associated with lower AvgScore. Equivalently, a lower student-teacher ratio (fewer students per teacher) "
                    "is associated with higher academic performance."
                )
            else:
                conclusion = (
                    "Statistically significant positive association: higher student-teacher ratios (more students per teacher) "
                    "are associated with higher AvgScore. This implies that a lower student-teacher ratio is associated with lower academic performance."
                )
        else:
            # Not significant
            if coef < 0:
                conclusion = (
                    "Negative point estimate (suggesting lower student-teacher ratio => higher AvgScore) but not statistically significant "
                    f"(p = {pval:.3g}). No strong evidence of an association."
                )
            elif coef > 0:
                conclusion = (
                    "Positive point estimate (suggesting lower student-teacher ratio => lower AvgScore) but not statistically significant "
                    f"(p = {pval:.3g}). No strong evidence of an association."
                )
            else:
                conclusion = f"Estimate is exactly zero (p = {pval:.3g}); no evidence of an association."

    obj = {
        "parameter": param_name,
        "coef": coef,
        "std_err": se,
        "t_stat": tstat,
        "p_value": pval,
        "95%_CI_lower": ci_lower,
        "95%_CI_upper": ci_upper,
        "conclusion": conclusion,
        "interpretation_short": (
            "Coefficient = change in AvgScore associated with a one-unit increase in students per teacher. "
            "Negative coef => fewer students per teacher (lower ratio) associated with higher AvgScore."
        )
    }

    result["object"] = obj
    result["description"] = (
        "Extracted coefficient, standard error, t-statistic, p-value, and 95% confidence interval for 'stu_teacher_ratio'. "
        "The 'conclusion' field states whether the association is statistically significant at alpha=0.05 and explains the direction "
        "in terms of whether a lower student-teacher ratio is associated with higher academic performance."
    )

    return result