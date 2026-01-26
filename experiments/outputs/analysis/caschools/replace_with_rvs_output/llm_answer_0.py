def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, t-stat, p-value, 95% CI, sample size,
    and a short interpretation for the 'StudentTeacherRatio' coefficient from a
    fitted statsmodels RegressionResultsWrapper.

    Returns:
      {
        "object": {
          "coef": float or None,
          "std_err": float or None,
          "t_value": float or None,
          "p_value": float or None,
          "ci_lower": float or None,
          "ci_upper": float or None,
          "n_obs": int or None,
          "significant_0.05": bool or None
        },
        "description": str
      }
    """
    # Ensure the object looks like a statsmodels result with params, bse, pvalues, conf_int
    required_attrs = ['params', 'bse', 'pvalues', 'conf_int']
    for attr in required_attrs:
        if not hasattr(model_output, attr):
            return {
                "object": None,
                "description": f"model_output is missing required attribute '{attr}'."
            }

    var = 'StudentTeacherRatio'
    if var not in model_output.params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in the fitted model."
        }

    # Extract statistics, converting to native Python types
    try:
        coef = float(model_output.params[var])
    except Exception:
        coef = None
    try:
        std_err = float(model_output.bse[var])
    except Exception:
        std_err = None
    try:
        t_value = float(model_output.tvalues[var])
    except Exception:
        t_value = None
    try:
        p_value = float(model_output.pvalues[var])
    except Exception:
        p_value = None
    try:
        ci = model_output.conf_int().loc[var]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        ci_lower = ci_upper = None
    try:
        n_obs = int(getattr(model_output, 'nobs', None)) if getattr(model_output, 'nobs', None) is not None else None
    except Exception:
        n_obs = None

    significant_0_05 = None
    if p_value is not None:
        significant_0_05 = (p_value < 0.05)

    # Interpretation in context:
    # StudentTeacherRatio is "students per teacher" so a lower ratio = fewer students per teacher.
    if coef is None:
        interpretation = "Could not extract coefficient for StudentTeacherRatio."
    else:
        # Directional interpretation
        if coef < 0:
            direction = "A negative coefficient indicates that increasing the student-teacher ratio (more students per teacher) is associated with lower AvgScore; therefore, a lower ratio (fewer students per teacher) is associated with higher academic performance."
        elif coef > 0:
            direction = "A positive coefficient indicates that increasing the student-teacher ratio (more students per teacher) is associated with higher AvgScore; therefore, a lower ratio (fewer students per teacher) would be associated with lower academic performance."
        else:
            direction = "The coefficient is approximately zero, indicating little estimated association between student-teacher ratio and AvgScore."

        sig_text = "This effect is statistically significant at the 5% level." if significant_0_05 else "This effect is not statistically significant at the 5% level."
        interpretation = (
            f"Estimated coefficient on StudentTeacherRatio = {coef:.4f} (SE = {std_err:.4f}, t = {t_value:.2f}, p = {p_value:.3g}). "
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}] (n = {n_obs}). {direction} {sig_text}"
        )

    return {
        "object": {
            "coef": coef,
            "std_err": std_err,
            "t_value": t_value,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_obs": n_obs,
            "significant_0.05": significant_0_05
        },
        "description": interpretation
    }