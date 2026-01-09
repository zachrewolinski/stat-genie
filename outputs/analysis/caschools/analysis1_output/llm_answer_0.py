def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, and 95% CI for the
    StudentTeacherRatio coefficient from a statsmodels OLS results object
    (optionally from a robust-covariance wrapped results).

    Returns a dictionary with keys:
      - "object": dict with numeric results and an interpretation/conclusion
      - "description": text explaining what the returned numbers mean

    The interpretation evaluates direction (positive/negative) and
    statistical significance at alpha=0.05, and translates the sign into
    whether a LOWER student-teacher ratio (fewer students per teacher)
    is associated with HIGHER AvgScore.
    """
    import numpy as np

    res = model_output

    # Ensure we have params available
    if not hasattr(res, "params"):
        raise ValueError("The provided model_output does not have a .params attribute.")

    params = res.params

    # Build a list of parameter names in a robust way:
    if hasattr(params, "index"):
        names = list(params.index)
    elif hasattr(res, "model") and hasattr(res.model, "exog_names"):
        names = list(res.model.exog_names)
    elif hasattr(res, "param_names"):
        names = list(res.param_names)
    else:
        # Fallback: use positional names for the length of params
        try:
            length = len(params)
        except Exception:
            length = 0
        names = [f"param_{i}" for i in range(length)]

    # Try to find the parameter name corresponding to StudentTeacherRatio.
    # We accept names that include both 'student' and 'teacher' (case-insensitive).
    param_name = None
    for i, name in enumerate(names):
        lname = str(name).lower()
        if 'student' in lname and 'teacher' in lname:
            param_name = name
            break
        if 'studentteacher' in lname.replace('_', ''):
            param_name = name
            break

    if param_name is None:
        available = ", ".join(names[:20])
        raise KeyError(
            "Could not find a parameter for 'StudentTeacherRatio' in model params. "
            f"Available parameter names (first 20): {available}"
        )

    # Determine the position/index of the parameter
    try:
        pos = names.index(param_name)
    except ValueError:
        pos = None

    # Helper to retrieve a value either by name (if possible) or by position
    def _get_value(obj, name, pos):
        if obj is None:
            return np.nan
        # If obj supports mapping by name
        try:
            return float(obj[name])
        except Exception:
            pass
        # If obj supports positional indexing
        try:
            if pos is not None:
                return float(obj[pos])
        except Exception:
            pass
        # If obj has .get (like a Series) but above failed, try get with fallback
        try:
            val = getattr(obj, "get", lambda k, default=None: default)(name, None)
            if val is not None:
                return float(val)
        except Exception:
            pass
        return np.nan

    # Extract coefficient
    try:
        coef = _get_value(params, param_name, pos)
    except Exception:
        coef = np.nan

    # Standard error: robust wrapper should store bse reflecting the robust cov.
    bse_obj = getattr(res, "bse", None)
    std_err = _get_value(bse_obj, param_name, pos)

    # t-value and p-value (may be from robust results if wrapper used)
    t_obj = getattr(res, "tvalues", None)
    p_obj = getattr(res, "pvalues", None)

    t_value = _get_value(t_obj, param_name, pos)
    # if t_value is nan, set to None for clarity
    if np.isnan(t_value):
        t_value = None

    p_value = _get_value(p_obj, param_name, pos)
    if np.isnan(p_value):
        p_value = None

    # 95% confidence interval
    try:
        ci_obj = res.conf_int()
        if hasattr(ci_obj, "loc") and param_name in getattr(ci_obj, "index", []):
            row = ci_obj.loc[param_name]
            ci_low, ci_high = float(row[0]), float(row[1])
        elif hasattr(ci_obj, "iloc") and pos is not None:
            row = ci_obj.iloc[pos]
            ci_low, ci_high = float(row[0]), float(row[1])
        else:
            # assume numpy array-like
            ci_low, ci_high = float(ci_obj[pos, 0]), float(ci_obj[pos, 1])
    except Exception:
        ci_low, ci_high = (np.nan, np.nan)

    # Interpretation: direction and significance
    alpha = 0.05
    significant = False
    if p_value is not None:
        try:
            significant = (p_value < alpha)
        except Exception:
            significant = False

    if coef < 0:
        direction_text = ("negative: higher StudentTeacherRatio (more students per teacher, i.e., larger classes) "
                          "is associated with LOWER AvgScore. Equivalently, a LOWER student-teacher ratio "
                          "(smaller classes) is associated with HIGHER AvgScore.")
    elif coef > 0:
        direction_text = ("positive: higher StudentTeacherRatio (more students per teacher) is associated with HIGHER AvgScore. "
                          "Equivalently, a LOWER student-teacher ratio would be associated with LOWER AvgScore.")
    else:
        direction_text = "no association (coefficient is exactly zero)."

    significance_text = ("The association is statistically significant (p < 0.05)."
                         if significant else
                         "The association is NOT statistically significant at the 0.05 level.")

    conclusion = f"{direction_text} {significance_text}"

    result_object = {
        "parameter_name": param_name,
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "significant_at_0.05": bool(significant),
        "conclusion": conclusion
    }

    description = (
        "Extracted the estimated effect of StudentTeacherRatio on AvgScore from the fitted OLS model "
        "(with robust standard errors if the results object contains them). "
        "coef: estimated change in AvgScore associated with a one-unit increase in StudentTeacherRatio "
        "(one more student per teacher). A negative coef means smaller class sizes (lower ratio) are "
        "associated with higher AvgScore. p_value and confidence interval provide statistical inference."
    )

    return {"object": result_object, "description": description}