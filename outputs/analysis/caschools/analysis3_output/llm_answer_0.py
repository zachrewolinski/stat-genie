def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted statsmodels OLS result.
    Returns a dictionary with:
      - "object": a dict with numeric results (coefficient, p-value, 95% CI, standardized coefficient, nobs)
      - "description": a short plain-English interpretation answering whether a lower student-teacher
                       ratio is associated with higher academic performance (statistical sign & significance).
    """
    import numpy as np

    res = model_output

    param_name = 'StudentTeacherRatio'
    # Check that parameter exists
    if param_name not in list(res.params.index):
        return {
            "object": None,
            "description": f"Parameter '{param_name}' not found in the model results."
        }

    # Point estimate
    coef = float(res.params[param_name])

    # p-value (uses the model's cov_type, here HC3)
    pval = float(res.pvalues[param_name]) if param_name in res.pvalues.index else float('nan')

    # 95% confidence interval
    ci = res.conf_int(alpha=0.05)
    try:
        ci_lower, ci_upper = float(ci.loc[param_name, 0]), float(ci.loc[param_name, 1])
    except Exception:
        # conf_int may be an ndarray; find by index
        idx = list(res.params.index).index(param_name)
        ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])

    # Standardized coefficient (beta): coef * (std(x) / std(y))
    try:
        exog_names = list(res.model.exog_names)
        idx_col = exog_names.index(param_name)
        x = res.model.exog[:, idx_col]
        y = res.model.endog
        std_beta = float(coef * (np.std(x, ddof=1) / np.std(y, ddof=1)))
    except Exception:
        std_beta = None

    # Sample size
    try:
        nobs = int(res.nobs)
    except Exception:
        nobs = None

    # Interpret sign relative to the question:
    # Lower StudentTeacherRatio = fewer students per teacher.
    # If coef < 0: higher ratio -> lower scores, so lower ratio -> higher scores (supports the hypothesis).
    # If coef > 0: higher ratio -> higher scores, so lower ratio -> lower scores (opposes the hypothesis).
    significance = (pval < 0.05) if (pval == pval) else False  # guard against nan
    if coef < 0:
        association = True
        sign_statement = "A negative coefficient implies that a lower student-teacher ratio (fewer students per teacher) is associated with higher average test scores."
    elif coef > 0:
        association = False
        sign_statement = "A positive coefficient implies that a lower student-teacher ratio (fewer students per teacher) is associated with lower average test scores."
    else:
        association = False
        sign_statement = "The coefficient is exactly zero (no association)."

    sig_statement = ("This association is statistically significant at the 5% level."
                     if significance else
                     "This association is NOT statistically significant at the 5% level.")

    description = (
        f"Estimate for '{param_name}': coefficient = {coef:.4f}, p-value = {pval:.4g}, "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}], standardized coef ≈ "
        + (f"{std_beta:.4f}" if std_beta is not None else "NA")
        + (f". Sample size = {nobs}." if nobs is not None else ".")
        + " " + sign_statement + " " + sig_statement
    )

    result_object = {
        "coefficient": coef,
        "p_value": pval,
        "conf_int_95": [ci_lower, ci_upper],
        "standardized_coefficient": std_beta,
        "nobs": nobs,
        # a simple boolean answer to the question (True means: lower ratio associated with higher performance)
        "lower_ratio_associated_with_higher_performance": bool(association and significance),
        # also include whether the sign would support the hypothesis regardless of significance
        "sign_supports_hypothesis": bool(association)
    }

    return {
        "object": result_object,
        "description": description
    }