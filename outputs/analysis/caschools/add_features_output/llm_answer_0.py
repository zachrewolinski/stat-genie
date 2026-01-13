def extract_final_answer(model_output):
    """
    Extracts the coefficient, robust p-value, 95% CI and a short interpretation
    for the 'student_teacher_ratio' coefficient from a fitted statsmodels OLS
    RegressionResultsWrapper (assumed to have been fit with robust (HC3)
    covariance).

    Returns:
      {
        "object": {
          "coef": float,                # estimated coefficient (change in AvgScore per 1 unit increase in ratio)
          "p_value": float,             # robust p-value
          "ci_lower": float,            # 95% CI lower bound
          "ci_upper": float,            # 95% CI upper bound
          "significant": bool,          # whether p_value < 0.05
          "direction": "negative"|"positive"  # sign of coef
        },
        "description": str              # brief interpretation in context
      }
    """
    import pandas as pd
    import numpy as np

    # Ensure the object looks like a statsmodels fitted result
    if model_output is None:
        raise ValueError("model_output is None")

    # The coefficient name we are interested in
    var_name = 'student_teacher_ratio'

    # Try to extract params, pvalues, and conf_int robustly
    try:
        params = model_output.params
        pvalues = model_output.pvalues
    except Exception as e:
        raise ValueError(f"Provided model_output doesn't have expected attributes: {e}")

    if var_name not in params.index:
        raise KeyError(f"Variable '{var_name}' not found in model parameters. Available params: {list(params.index)}")

    coef = float(params[var_name])
    pval = float(pvalues[var_name])

    # Confidence interval extraction (works whether conf_int returns DataFrame or ndarray)
    try:
        ci = model_output.conf_int()
        if isinstance(ci, pd.DataFrame):
            ci_lower, ci_upper = map(float, ci.loc[var_name])
        else:
            # ci as ndarray; find position of var_name in params index
            pos = list(params.index).index(var_name)
            ci_lower, ci_upper = map(float, ci[pos])
    except Exception:
        # Fallback: use coef +/- 1.96 * bse if conf_int fails
        try:
            bse = float(model_output.bse[var_name])
            ci_lower = coef - 1.96 * bse
            ci_upper = coef + 1.96 * bse
        except Exception as e:
            raise ValueError(f"Could not compute confidence interval: {e}")

    significant = (pval < 0.05)
    direction = "negative" if coef < 0 else ("positive" if coef > 0 else "zero")

    # Build a concise interpretation
    # Note: coef is change in AvgScore per one-unit increase in student_teacher_ratio.
    if direction == "negative" and significant:
        interp = (
            f"Yes. The estimated coefficient on student_teacher_ratio is {coef:.3f} "
            f"(p = {pval:.3g}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
            "Because the coefficient is negative and statistically significant, "
            "a lower student-teacher ratio (fewer students per teacher) is associated "
            "with higher average academic performance. "
            f"Interpretation: decreasing the student-teacher ratio by 1 student is associated "
            f"with an estimated increase of {abs(coef):.3f} points in AvgScore."
        )
    elif direction == "negative" and not significant:
        interp = (
            f"The estimated coefficient on student_teacher_ratio is {coef:.3f} "
            f"(p = {pval:.3g}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
            "The coefficient is negative (suggesting lower ratios link to higher scores) "
            "but it is not statistically significant at the 5% level, so the evidence is "
            "insufficient to conclude a reliable association."
        )
    elif direction == "positive" and significant:
        interp = (
            f"No. The estimated coefficient on student_teacher_ratio is {coef:.3f} "
            f"(p = {pval:.3g}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
            "Because the coefficient is positive and statistically significant, "
            "a higher student-teacher ratio (more students per teacher) is associated "
            "with higher average academic performance — the opposite of the hypothesis."
        )
    else:  # positive but not significant, or exactly zero
        interp = (
            f"The estimated coefficient on student_teacher_ratio is {coef:.3f} "
            f"(p = {pval:.3g}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
            "The coefficient is not statistically significant at the 5% level, so there is "
            "no reliable evidence of an association between student-teacher ratio and average performance."
        )

    result_object = {
        "coef": coef,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": bool(significant),
        "direction": direction
    }

    return {"object": result_object, "description": interp}