def extract_final_answer(model_output):
    """
    Extracts the coefficient and related statistics for 'StudentTeacherRatio'
    from a fitted statsmodels regression results object and returns a
    concise interpretation relevant to the question:
      "Is a lower student-teacher ratio associated with higher academic performance?"
    
    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, std_err, t, p_value, conf_int, n_obs, r_squared)
      - "description": short plain-language interpretation of the estimate and its significance
    
    Handles the case where 'StudentTeacherRatio' is not in the model.
    """
    import numpy as np

    result = {}
    description = ""
    try:
        params = model_output.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Provided model_output does not appear to be a fitted statsmodels results object: {e}"
        }

    var = 'StudentTeacherRatio'
    if var not in params.index:
        return {
            "object": None,
            "description": f"The model does not contain a coefficient for '{var}'."
        }

    try:
        coef = float(model_output.params[var])
        std_err = float(model_output.bse[var]) if hasattr(model_output, 'bse') else None
        t_stat = float(model_output.tvalues[var]) if hasattr(model_output, 'tvalues') else None
        p_value = float(model_output.pvalues[var]) if hasattr(model_output, 'pvalues') else None
        ci = model_output.conf_int().loc[var].tolist() if hasattr(model_output, 'conf_int') else [None, None]
        # Number of observations and R-squared if available
        n_obs = int(model_output.nobs) if hasattr(model_output, 'nobs') else None
        r_squared = float(model_output.rsquared) if hasattr(model_output, 'rsquared') else None

        result = {
            "variable": var,
            "coef": coef,
            "std_err": std_err,
            "t_stat": t_stat,
            "p_value": p_value,
            "conf_int_95": ci,
            "n_obs": n_obs,
            "r_squared": r_squared
        }

        # Interpretation in context: lower StudentTeacherRatio means fewer students per teacher.
        # If coef < 0 then lower ratio (fewer students per teacher) is associated with higher AvgScore.
        sign_desc = ""
        if coef < 0:
            sign_desc = ("The estimated coefficient is negative, meaning that a smaller "
                         "student-teacher ratio (fewer students per teacher) is associated "
                         "with higher district AvgScore.")
        elif coef > 0:
            sign_desc = ("The estimated coefficient is positive, meaning that a smaller "
                         "student-teacher ratio (fewer students per teacher) would be associated "
                         "with lower district AvgScore (contrary to the hypothesis).")
        else:
            sign_desc = "The estimated coefficient is exactly zero."

        # Statistical significance statement
        if p_value is None:
            sig_desc = "No p-value is available to assess statistical significance."
        else:
            if p_value < 0.01:
                sig_level = "p < 0.01"
            elif p_value < 0.05:
                sig_level = "p < 0.05"
            elif p_value < 0.1:
                sig_level = "p < 0.10"
            else:
                sig_level = f"p = {p_value:.3f}"
            sig_desc = f"The association has {sig_level} (based on the model's robust standard errors)."

        # Magnitude statement: interpret per one-unit change in ratio
        # Note: a one-unit change in StudentTeacherRatio is one additional student per teacher.
        magnitude_desc = (f"Point estimate: {coef:.3f}. "
                          f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}] (if available). "
                          f"This implies that a one-unit decrease in the student-teacher ratio "
                          f"is associated with a {(-coef if coef < 0 else -coef):.3f}-point change in AvgScore "
                          f"(direction depends on the sign).")

        description = " ".join([sign_desc, sig_desc, magnitude_desc])

        return {
            "object": result,
            "description": description
        }

    except Exception as e:
        return {
            "object": None,
            "description": f"Error extracting statistics for '{var}': {e}"
        }