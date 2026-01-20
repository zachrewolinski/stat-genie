def extract_final_answer(model_output):
    """
    Extracts the coefficient, p-value, 95% CI, sample size, and R-squared for the
    'student_teacher_ratio' variable from a statsmodels OLS results object, and
    returns those values plus a short interpretation.

    Returns a dictionary with keys:
      - "object": dict with numeric outputs (coef, p_value, ci, n, r_squared, significant)
      - "description": brief plain-language interpretation of what the coefficient implies
    """
    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    params = res.params
    if 'student_teacher_ratio' not in params.index:
        raise KeyError("The model does not contain a parameter named 'student_teacher_ratio'.")

    # Extract coefficient and p-value (these reflect the HC3 robust cov if model was fit with cov_type='HC3')
    coef = float(params['student_teacher_ratio'])
    pval = float(res.pvalues['student_teacher_ratio'])

    # Extract 95% confidence interval for the parameter, handling both ndarray and DataFrame returns
    ci = res.conf_int(alpha=0.05)
    try:
        # If conf_int returns a DataFrame or has .loc
        if hasattr(ci, 'loc'):
            ci_row = ci.loc['student_teacher_ratio']
            ci_low, ci_high = float(ci_row[0]), float(ci_row[1])
        else:
            # Otherwise treat as ndarray with same ordering as params.index
            names = list(params.index)
            idx = names.index('student_teacher_ratio')
            ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        # Fallback: try indexing by parameter position
        names = list(params.index)
        idx = names.index('student_teacher_ratio')
        ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])

    # Additional model info
    nobs = int(getattr(res, "nobs", getattr(res, "df_resid", None) and int(res.df_resid + len(params)) or None))
    try:
        r_squared = float(getattr(res, "rsquared", float("nan")))
    except Exception:
        r_squared = float("nan")

    # Statistical significance at alpha = 0.05
    significant = (pval < 0.05)

    # Interpret direction: a negative coef means higher ratio (more students per teacher)
    # is associated with lower scores; equivalently a lower ratio (fewer students per teacher)
    # is associated with higher scores.
    if coef < 0:
        direction_text = ("A lower student-teacher ratio (fewer students per teacher) is associated "
                          "with higher average test scores.")
    elif coef > 0:
        direction_text = ("A lower student-teacher ratio (fewer students per teacher) is associated "
                          "with lower average test scores (the coefficient is positive).")
    else:
        direction_text = "No association (coefficient is exactly zero)."

    significance_text = ("This association is statistically significant at the 5% level."
                         if significant else
                         "This association is not statistically significant at the 5% level.")

    description = (
        f"Estimated effect of student_teacher_ratio on avg_score: coefficient = {coef:.4f}, "
        f"95% CI = [{ci_low:.4f}, {ci_high:.4f}], p = {pval:.4g}. {direction_text} "
        f"{significance_text} Interpretation: a one-unit change in student_teacher_ratio is associated "
        f"with a {abs(coef):.4f}-point {'increase' if coef < 0 else 'decrease' if coef > 0 else 'no change'} "
        f"in avg_score, holding the controls constant. (N = {nobs}, R^2 = {r_squared:.3f})"
    )

    result_object = {
        "coef": coef,
        "p_value": pval,
        "ci_95": [ci_low, ci_high],
        "n": nobs,
        "r_squared": r_squared,
        "significant_at_0.05": significant
    }

    return {"object": result_object, "description": description}