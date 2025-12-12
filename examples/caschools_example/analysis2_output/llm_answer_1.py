def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, t-statistic, p-value, 95% CI, and a standardized effect
    for the StudentTeacherRatio variable from a fitted statsmodels regression results object.

    Returns a dictionary:
      - "object": dict with numeric results for StudentTeacherRatio
      - "description": plain-English interpretation of what those numbers imply about whether
                       a lower student-teacher ratio is associated with higher academic performance.
    """
    import numpy as np

    # Ensure model_output looks like a statsmodels RegressionResults
    if not hasattr(model_output, "params") or not hasattr(model_output, "bse"):
        raise ValueError("model_output does not appear to be a statsmodels regression results object.")

    # Parameter name we care about
    param_name = 'StudentTeacherRatio'

    params = model_output.params
    if param_name not in params.index:
        raise ValueError(f"Parameter '{param_name}' not found in model_output.params")

    # Extract point estimate, SE, t, p
    coef = float(params[param_name])
    se = float(model_output.bse[param_name]) if param_name in model_output.bse.index else float(np.nan)
    tval = float(model_output.tvalues[param_name]) if param_name in model_output.tvalues.index else float(np.nan)
    pval = float(model_output.pvalues[param_name]) if param_name in model_output.pvalues.index else float(np.nan)

    # 95% confidence interval (robust handling if conf_int returns ndarray or DataFrame)
    try:
        ci_all = model_output.conf_int(alpha=0.05)
        # Try DataFrame-style access
        try:
            ci_lower, ci_upper = ci_all.loc[param_name].tolist()
        except Exception:
            # ndarray-style: find index position of param
            param_list = list(params.index)
            idx = param_list.index(param_name)
            ci_lower, ci_upper = float(ci_all[idx, 0]), float(ci_all[idx, 1])
    except Exception:
        ci_lower, ci_upper = float(np.nan), float(np.nan)

    # Compute a standardized (beta) coefficient if model stores the original exog & endog
    std_beta = None
    try:
        exog_names = list(model_output.model.exog_names)
        if param_name in exog_names:
            idx = exog_names.index(param_name)
            X = np.asarray(model_output.model.exog)[:, idx]
            Y = np.asarray(model_output.model.endog)
            # use sample standard deviation (ddof=1) to be conventional
            sx = X.std(ddof=1)
            sy = Y.std(ddof=1)
            if sy != 0:
                std_beta = float(coef * (sx / sy))
            else:
                std_beta = None
    except Exception:
        std_beta = None

    # Interpretation about direction and statistical significance
    # Note: StudentTeacherRatio is students per teacher. A negative coef means fewer students per teacher
    # (i.e., lower ratio) is associated with higher AvgScore.
    significance = "not statistically significant"
    if not np.isnan(pval):
        if pval < 0.01:
            significance = "statistically significant at p < 0.01"
        elif pval < 0.05:
            significance = "statistically significant at p < 0.05"
        elif pval < 0.1:
            significance = "marginally significant (p < 0.1)"
        else:
            significance = "not statistically significant (p >= 0.1)"

    if coef < 0:
        direction_statement = ("Negative coefficient: higher student-teacher ratio (more students per teacher) "
                               "is associated with LOWER AvgScore; equivalently, a lower student-teacher ratio "
                               "(fewer students per teacher) is associated with HIGHER AvgScore.")
    elif coef > 0:
        direction_statement = ("Positive coefficient: higher student-teacher ratio (more students per teacher) "
                               "is associated with HIGHER AvgScore; equivalently, a lower student-teacher ratio "
                               "(fewer students per teacher) is associated with LOWER AvgScore.")
    else:
        direction_statement = "Coefficient is zero (no estimated association)."

    # Build the object to return
    result_object = {
        "parameter": param_name,
        "coefficient": coef,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "standardized_beta": std_beta  # may be None if cannot be computed
    }

    description_lines = [
        f"Estimate for '{param_name}': coefficient = {coef:.4g}, SE = {se:.4g}, t = {tval:.4g}, p = {pval:.4g}.",
        f"95% CI = [{ci_lower:.4g}, {ci_upper:.4g}]." if not (np.isnan(ci_lower) or np.isnan(ci_upper)) else "95% CI unavailable.",
        f"Standardized (beta) coefficient = {std_beta:.4g}." if std_beta is not None else "Standardized coefficient unavailable.",
        direction_statement,
        f"Statistical evidence: {significance}.",
        "Interpretation: If the coefficient is negative and statistically significant, this provides evidence that a LOWER student-teacher ratio (fewer students per teacher) is associated with HIGHER 5th-grade average test scores, controlling for the included covariates. If the coefficient is not statistically significant, there is no strong evidence of an association after controlling for those covariates."
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}