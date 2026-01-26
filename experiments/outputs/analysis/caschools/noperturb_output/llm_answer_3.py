def extract_final_answer(model_output):
    """
    Extract statistics for the StudentTeacherRatio coefficient from a statsmodels
    RegressionResultsWrapper and summarize whether a lower student-teacher ratio
    is associated with higher academic performance.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, t, p, 95% CI, nobs, significant, conclusion)
      - "description": brief plain-language interpretation of those results

    The function is robust to small variations in the parameter name (looks for
    'StudentTeacherRatio' exactly, otherwise any parameter name containing 'student'
    and 'teacher' substrings).
    """
    # Defensive checks
    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided."
        }

    # Try to get parameter name
    param_name = None
    target = "StudentTeacherRatio"
    try:
        params = model_output.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Provided object does not appear to be a fitted statsmodels result: {e}"
        }

    if target in params.index:
        param_name = target
    else:
        # fallback: look for a param name containing both 'student' and 'teacher'
        for name in params.index:
            low = str(name).lower()
            if "student" in low and "teacher" in low:
                param_name = name
                break
        # fallback: any param containing 'student'
        if param_name is None:
            for name in params.index:
                if "student" in str(name).lower():
                    param_name = name
                    break

    if param_name is None:
        return {
            "object": None,
            "description": "Could not find a parameter name corresponding to student-teacher ratio in the model results."
        }

    # Extract statistics
    coef = float(model_output.params[param_name])
    # Some result objects provide bse, tvalues, pvalues, conf_int, nobs
    try:
        se = float(model_output.bse[param_name])
    except Exception:
        se = None
    try:
        tstat = float(model_output.tvalues[param_name])
    except Exception:
        tstat = None
    try:
        pval = float(model_output.pvalues[param_name])
    except Exception:
        pval = None
    try:
        ci = model_output.conf_int().loc[param_name].tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None
    try:
        nobs = int(getattr(model_output, "nobs", None))
    except Exception:
        nobs = None

    # Determine significance at conventional alpha=0.05 if p-value available
    significant = None
    if pval is not None:
        significant = pval < 0.05

    # Interpret direction: negative coef => higher ratio (more students per teacher) -> lower scores,
    # so lower ratio (fewer students per teacher) -> higher scores.
    if coef < 0:
        direction = "negative"
        conclusion_simple = ("A lower student-teacher ratio (fewer students per teacher) is associated "
                             "with higher AvgTestScore (negative coefficient).")
    elif coef > 0:
        direction = "positive"
        conclusion_simple = ("A lower student-teacher ratio would be associated with lower AvgTestScore "
                             "(positive coefficient indicates more students per teacher is associated with higher scores), "
                             "which is counterintuitive.")
    else:
        direction = "zero"
        conclusion_simple = "No linear association (coefficient is zero)."

    # Combine result object
    result_obj = {
        "param_name": param_name,
        "coef": coef,
        "se": se,
        "t": tstat,
        "p_value": pval,
        "95%_CI": [ci_lower, ci_upper],
        "nobs": nobs,
        "significant_at_0.05": significant,
        "direction": direction,
        "conclusion": conclusion_simple
    }

    # Short description suitable for the user question
    if pval is None:
        desc = (f"Estimated coefficient for {param_name} = {coef:.4g}. "
                "p-value not available; cannot determine statistical significance.")
    else:
        sig_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"
        desc = (f"The coefficient on {param_name} = {coef:.4g} (SE = {se:.4g}, t = {tstat:.4g}, "
                f"p = {pval:.4g}, 95% CI = [{ci_lower:.4g}, {ci_upper:.4g}]). "
                f"This indicates a {direction} association; {sig_text}. "
                "A negative and significant coefficient would mean that lower student-teacher ratios "
                "(fewer students per teacher) are associated with higher average test scores.")

    return {"object": result_obj, "description": desc}