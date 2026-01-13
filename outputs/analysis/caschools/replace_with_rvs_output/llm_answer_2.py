def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels
    RegressionResultsWrapper and returns a concise, programmatic conclusion.

    Returns:
      dict with keys:
        - "object": dict containing numeric results (coef, std_err, t, p, ci_lower, ci_upper,
                    standardized_coef, significant)
        - "description": short human-readable interpretation of the coefficient in context
    """
    # Attempt to find the parameter name in the model params index (handle slight name differences)
    params = getattr(model_output, "params", None)
    if params is None:
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    # Determine the exact parameter name to use
    target_names = ["StudentTeacherRatio", "studentteacherratio", "Student_Teacher_Ratio"]
    param_name = None
    for name in target_names:
        if name in params.index:
            param_name = name
            break
    # If not found, try to find any parameter that contains 'student' and 'teacher' as a fallback
    if param_name is None:
        for idx_name in params.index:
            low = str(idx_name).lower()
            if "student" in low and "teacher" in low:
                param_name = idx_name
                break

    if param_name is None:
        raise KeyError("Could not find a parameter name for the student-teacher ratio in model params.")

    # Extract core statistics
    coef = float(params[param_name])
    bse = float(model_output.bse[param_name]) if hasattr(model_output, "bse") else None
    tvalue = float(model_output.tvalues[param_name]) if hasattr(model_output, "tvalues") else None
    pvalue = float(model_output.pvalues[param_name]) if hasattr(model_output, "pvalues") else None

    # Confidence interval (robust CI since model was fit with cov_type='HC1')
    try:
        ci = model_output.conf_int(alpha=0.05)
        # ci may be a numpy array or a DataFrame-like object
        if hasattr(ci, "loc"):
            ci_lower = float(ci.loc[param_name, 0])
            ci_upper = float(ci.loc[param_name, 1])
        else:
            # find positional index of param_name in params.index
            pos = list(params.index).index(param_name)
            ci_lower = float(ci[pos, 0])
            ci_upper = float(ci[pos, 1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Compute a standardized (beta) coefficient if possible:
    std_beta = None
    try:
        # model.endog and model.exog are the original arrays used
        endog = model_output.model.endog
        exog = model_output.model.exog
        exog_names = list(model_output.model.exog_names)
        if param_name in exog_names:
            col_idx = exog_names.index(param_name)
        else:
            # try matching lower-case name
            lowered = [str(n).lower() for n in exog_names]
            col_idx = lowered.index(str(param_name).lower())
        x_col = exog[:, col_idx]
        # compute standard deviations (ddof=1)
        import numpy as _np
        sd_x = float(_np.std(x_col, ddof=1))
        sd_y = float(_np.std(endog, ddof=1))
        if sd_x > 0 and sd_y > 0:
            std_beta = float(coef * (sd_x / sd_y))
    except Exception:
        std_beta = None

    # Determine significance at conventional alpha = 0.05 (if pvalue available)
    significant = None
    if pvalue is not None:
        significant = (pvalue < 0.05)

    # Interpret direction: negative coef means lower ratio (fewer students per teacher) -> higher AvgScore
    if coef < 0:
        direction = "lower StudentTeacherRatio (fewer students per teacher) is associated with higher AvgScore"
    elif coef > 0:
        direction = "lower StudentTeacherRatio (fewer students per teacher) is associated with lower AvgScore (unexpected direction)"
    else:
        direction = "no linear association (coef = 0)"

    # Build a concise description
    if significant is True:
        conclusion = (
            f"The StudentTeacherRatio coefficient = {coef:.4f} (SE={bse:.4f}, p={pvalue:.3g}). "
            f"This is statistically significant (p < 0.05) and indicates that {direction}."
        )
    elif significant is False:
        conclusion = (
            f"The StudentTeacherRatio coefficient = {coef:.4f} (SE={bse:.4f}, p={pvalue:.3g}). "
            "This is not statistically significant at the 0.05 level, so there is insufficient evidence "
            f"to conclude a relationship; point estimate suggests that {direction}."
        )
    else:
        # p-value not available
        conclusion = (
            f"The StudentTeacherRatio coefficient = {coef:.4f} (SE={bse}), p-value not available. "
            f"Point estimate suggests that {direction}."
        )

    result_object = {
        "param_name": str(param_name),
        "coef": coef,
        "std_err": float(bse) if bse is not None else None,
        "t_value": float(tvalue) if tvalue is not None else None,
        "p_value": float(pvalue) if pvalue is not None else None,
        "ci_lower_95": float(ci_lower) if ci_lower is not None else None,
        "ci_upper_95": float(ci_upper) if ci_upper is not None else None,
        "standardized_coef": float(std_beta) if std_beta is not None else None,
        "significant_at_0.05": bool(significant) if significant is not None else None,
    }

    return {
        "object": result_object,
        "description": conclusion
    }