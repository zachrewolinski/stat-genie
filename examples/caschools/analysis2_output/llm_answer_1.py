def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, 95% CI, and a brief interpretation
    for the 'StudentTeacherRatio' variable from a fitted statsmodels RegressionResultsWrapper.

    Returns a dict with keys:
      - "object": dict of numeric results for StudentTeacherRatio
      - "description": brief text interpreting the sign and statistical significance
    """
    # Prepare return structure
    result_obj = {}
    description = ""
    
    # Basic checks
    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided."
        }
    
    try:
        params = model_output.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Provided object does not appear to be a statsmodels results object: {e}"
        }
    
    var_name = 'StudentTeacherRatio'
    if var_name not in params.index:
        return {
            "object": None,
            "description": f"Variable '{var_name}' not found in the model parameters."
        }
    
    # Extract statistics
    coef = float(params[var_name])
    try:
        std_err = float(model_output.bse[var_name])
    except Exception:
        # fallback: compute from covariance if available
        try:
            std_err = float(model_output.normalized_cov_params[var_name].get(var_name, float('nan')))
        except Exception:
            std_err = float('nan')
    try:
        t_stat = float(model_output.tvalues[var_name])
    except Exception:
        t_stat = float('nan')
    try:
        p_value = float(model_output.pvalues[var_name])
    except Exception:
        p_value = float('nan')
    try:
        ci = model_output.conf_int(alpha=0.05).loc[var_name].values
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = float('nan'), float('nan')
    
    # Significance at conventional 5% level
    significant_05 = False
    if (not (p_value != p_value)) and p_value < 0.05:  # check for NaN then threshold
        significant_05 = True
    
    # Directional interpretation:
    # StudentTeacherRatio is defined as number of students per teacher (higher = worse/larger ratio).
    # A negative coefficient implies that higher student-teacher ratio is associated with LOWER AvgScore,
    # which equivalently means a lower student-teacher ratio (fewer students per teacher) is associated
    # with HIGHER academic performance.
    if coef < 0:
        direction_text = (
            "Coefficient is negative: higher student-teacher ratio (more students per teacher) "
            "is associated with lower AvgScore. Thus, lower student-teacher ratios are associated "
            "with higher academic performance."
        )
    elif coef > 0:
        direction_text = (
            "Coefficient is positive: higher student-teacher ratio (more students per teacher) "
            "is associated with higher AvgScore. Thus, lower student-teacher ratios would be "
            "associated with lower academic performance."
        )
    else:
        direction_text = "Coefficient is exactly zero (no estimated association)."
    
    sig_text = (
        "The association is statistically significant at the 5% level."
        if significant_05 else
        "The association is NOT statistically significant at the 5% level."
    )
    
    # Build object to return
    result_obj = {
        "variable": var_name,
        "coef": coef,
        "std_err": std_err,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "significant_at_0.05": significant_05
    }
    
    description = (
        f"Estimated effect of '{var_name}' on AvgScore: coefficient = {coef:.4f}, "
        f"SE = {std_err:.4f}, t = {t_stat:.3f}, p = {p_value:.3g}, "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. {direction_text} {sig_text}"
    )
    
    return {"object": result_obj, "description": description}