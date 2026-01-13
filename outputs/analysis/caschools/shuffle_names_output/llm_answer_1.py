def extract_final_answer(model_output):
    """
    Extracts statistics about the StudentTeacherRatio coefficient from a fitted statsmodels
    RegressionResultsWrapper and returns a dictionary with the numeric results ("object")
    and a short interpretation ("description").

    Returned dictionary format:
      {
        "object": {
          "coef": float,
          "std_err": float,
          "p_value": float,
          "conf_int_low": float,
          "conf_int_high": float,
          "nobs": int,
          "r_squared": float or None,
          "effect_per_10_decrease": float,
          "effect_per_10_decrease_conf_int": (float, float)
        },
        "description": str
      }
    """
    # Basic validation
    if model_output is None:
        raise ValueError("model_output is None")

    # Ensure params attribute exists
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object (no .params).")

    params = model_output.params
    # Try to find the StudentTeacherRatio parameter name robustly
    target_name = None
    if 'StudentTeacherRatio' in params.index:
        target_name = 'StudentTeacherRatio'
    else:
        # fallback: look for a parameter name that contains both 'student' and 'teacher'
        for name in params.index:
            lname = str(name).lower()
            if ('student' in lname and 'teacher' in lname) or ('studentteacher' in lname.replace('_','')):
                target_name = name
                break

    if target_name is None:
        raise ValueError("Could not find a parameter matching 'StudentTeacherRatio' in the model parameters: "
                         f"found parameters {list(params.index)}")

    # Extract primary statistics
    coef = float(model_output.params[target_name])
    # robust retrievals for standard error, pvalue, conf_int
    try:
        std_err = float(model_output.bse[target_name])
    except Exception:
        std_err = None
    try:
        p_value = float(model_output.pvalues[target_name])
    except Exception:
        p_value = None
    try:
        ci = model_output.conf_int(alpha=0.05).loc[target_name].values
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except Exception:
        # try array-style access
        try:
            ci_arr = model_output.conf_int(alpha=0.05)
            # find row corresponding to target_name
            if hasattr(ci_arr, 'loc'):
                ci_low, ci_high = float(ci_arr.loc[target_name].values[0]), float(ci_arr.loc[target_name].values[1])
            else:
                # assume order matches params.index
                idx = list(params.index).index(target_name)
                ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
        except Exception:
            ci_low = ci_high = None

    # Additional model info
    nobs = int(getattr(model_output, 'nobs', getattr(model_output, 'model', None) and getattr(model_output.model, 'nobs', None) or None) or 0)
    r_squared = getattr(model_output, 'rsquared', None)

    # Interpretation-specific calculations:
    # coef is change in AvgScore per one-unit increase in StudentTeacherRatio (students per teacher).
    # A negative coef means that increasing the student-teacher ratio (more students per teacher) lowers AvgScore,
    # i.e., a lower ratio (fewer students per teacher) is associated with higher AvgScore.
    effect_per_10_decrease = -10.0 * coef  # estimated change in AvgScore when StudentTeacherRatio is reduced by 10
    if (ci_low is not None) and (ci_high is not None):
        # CI for -10 * coef is [-10*ci_high, -10*ci_low]
        effect_per_10_ci = (-10.0 * ci_high, -10.0 * ci_low)
    else:
        effect_per_10_ci = (None, None)

    # Significance judgement at alpha = 0.05
    if p_value is None:
        significance_text = "p-value unavailable; significance cannot be determined."
        significant = None
    else:
        significant = p_value < 0.05
        significance_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"

    # Direction interpretation
    if coef < 0:
        direction_text = ("Negative coefficient: higher student-teacher ratio (more students per teacher) is "
                          "associated with lower average scores; therefore LOWER student-teacher ratio "
                          "(fewer students per teacher) is associated with HIGHER academic performance.")
    elif coef > 0:
        direction_text = ("Positive coefficient: higher student-teacher ratio (more students per teacher) is "
                          "associated with HIGHER average scores; therefore LOWER student-teacher ratio "
                          "(fewer students per teacher) would be associated with LOWER academic performance.")
    else:
        direction_text = "Coefficient is exactly zero; no association detected."

    # Build result object (numbers rounded for readability)
    def _r(x):
        return None if x is None else round(x, 4)

    result_object = {
        "parameter_name": str(target_name),
        "coef": _r(coef),
        "std_err": _r(std_err),
        "p_value": _r(p_value),
        "conf_int_low": _r(ci_low),
        "conf_int_high": _r(ci_high),
        "nobs": int(nobs) if nobs is not None else None,
        "r_squared": _r(r_squared),
        "effect_per_10_decrease": _r(effect_per_10_decrease),
        "effect_per_10_decrease_conf_int": (_r(effect_per_10_ci[0]), _r(effect_per_10_ci[1])),
        "significant_at_0_05": significant
    }

    description = (
        f"Estimated effect of StudentTeacherRatio on AvgScore: coef = {result_object['coef']} "
        f"(SE = {result_object['std_err']}), 95% CI = ({result_object['conf_int_low']}, {result_object['conf_int_high']}), "
        f"p = {result_object['p_value']}, n = {result_object['nobs']}. "
        f"{direction_text} This effect is {significance_text}. "
        f"Estimated change in AvgScore for a reduction of 10 students per teacher = {result_object['effect_per_10_decrease']} "
        f"(95% CI = {result_object['effect_per_10_decrease_conf_int']})."
    )

    return {"object": result_object, "description": description}