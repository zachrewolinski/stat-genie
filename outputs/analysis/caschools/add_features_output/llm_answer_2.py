def extract_final_answer(model_output):
    """
    Extract statistics about the StudentTeacherRatio coefficient from a statsmodels
    RegressionResultsWrapper (or similar object) and return a short interpretation.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, t, p, 95% CI, nobs, significant_0.05)
      - "description": short human-readable interpretation answering whether a LOWER
                       student-teacher ratio is associated with HIGHER AvgScore.
    """
    # Basic checks
    if model_output is None:
        raise ValueError("model_output is None")

    # Ensure the object has params, bse, tvalues, pvalues, conf_int methods/attributes
    for attr in ("params", "bse", "tvalues", "pvalues", "conf_int"):
        if not hasattr(model_output, attr):
            raise ValueError(f"model_output is missing required attribute/method: {attr}")

    var = "StudentTeacherRatio"
    params = model_output.params

    # Ensure we can get a list of parameter names
    try:
        param_index = list(params.index)
    except Exception:
        param_index = None

    if param_index is None or var not in param_index:
        # Try the Series lookup directly (will raise if not present)
        try:
            _ = params.loc[var]
        except Exception:
            available = getattr(params, "index", None)
            raise ValueError(f"Variable '{var}' not found in model output parameters. Available params: {list(available) if available is not None else available}")

    # Helper to safely extract a numeric value for var from a Series-like or array-like attribute
    def _get_value(attr):
        val = getattr(model_output, attr)
        # If it's a pandas Series/DataFrame-like with .loc
        try:
            if hasattr(val, "loc"):
                return float(val.loc[var])
        except Exception:
            pass
        # Fallback: if we have param_index, find positional index
        try:
            idx = param_index.index(var)
            return float(val[idx])
        except Exception:
            # As last resort, try direct indexing by var (may raise)
            try:
                return float(val[var])
            except Exception as e:
                raise ValueError(f"Could not extract '{var}' from model_output.{attr}: {e}")

    coef = float(params.loc[var])
    se = _get_value("bse")
    t = _get_value("tvalues")
    p = _get_value("pvalues")

    # Confidence interval handling
    try:
        ci = model_output.conf_int(alpha=0.05)
    except TypeError:
        # some implementations accept no args
        ci = model_output.conf_int()
    ci_lower = ci_upper = None
    try:
        if hasattr(ci, "loc") and var in ci.index:
            # ci.loc[var] might return a Series with two elements
            row = ci.loc[var]
            # row may be like [lower, upper] accessible by position 0/1
            ci_lower = float(row.iloc[0]) if hasattr(row, "iloc") else float(row[0])
            ci_upper = float(row.iloc[1]) if hasattr(row, "iloc") else float(row[1])
        else:
            # fallback to positional lookup
            if param_index is not None:
                idx = param_index.index(var)
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Number of observations if available
    nobs = None
    if hasattr(model_output, "nobs"):
        try:
            nobs = int(model_output.nobs)
        except Exception:
            try:
                nobs = int(getattr(model_output, "nobs"))
            except Exception:
                nobs = None

    # Determine significance at conventional levels
    significant_0_05 = (p < 0.05)
    significant_0_01 = (p < 0.01)

    # Interpret direction
    if coef < 0:
        direction = ("Negative: higher student-teacher ratio (more students per teacher) is "
                     "associated with LOWER AvgScore. Therefore, a LOWER ratio is associated with HIGHER AvgScore.")
    elif coef > 0:
        direction = ("Positive: higher student-teacher ratio (more students per teacher) is "
                     "associated with HIGHER AvgScore. Therefore, a LOWER ratio is associated with LOWER AvgScore.")
    else:
        direction = "Coefficient is 0 (no association detected)."

    # Formatting helper to guard against None
    def fmt(value, fmt_spec):
        if value is None:
            return "NA"
        try:
            return format(value, fmt_spec)
        except Exception:
            return str(value)

    # Build the returned object
    result_object = {
        "variable": var,
        "coef": coef,
        "se": se,
        "t": t,
        "p_value": p,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "nobs": nobs,
        "significant_0.05": significant_0_05,
        "significant_0.01": significant_0_01,
    }

    # Human-readable description answering the yes/no question
    if coef < 0 and significant_0_05:
        yes_no = "Yes"
        conclusion = (
            f"{yes_no}: There is a statistically significant association (p = {p:.3g}) such that "
            f"lower student-teacher ratio is associated with higher average test scores. "
            f"Estimated effect: a one-unit decrease in StudentTeacherRatio is associated with an "
            f"increase of {abs(coef):.3f} points in AvgScore (95% CI: {fmt(ci_lower, '.3f')} to {fmt(ci_upper, '.3f')})."
        )
    elif coef < 0 and not significant_0_05:
        yes_no = "Inconclusive"
        conclusion = (
            f"{yes_no}: The estimated association is in the expected direction (lower ratio -> higher AvgScore) "
            f"but it is not statistically significant at the 5% level (p = {p:.3g}). "
            f"Estimated effect: {coef:.3f} (SE = {fmt(se, '.3f')}; 95% CI: {fmt(ci_lower, '.3f')} to {fmt(ci_upper, '.3f')})."
        )
    elif coef > 0 and significant_0_05:
        yes_no = "No"
        conclusion = (
            f"{yes_no}: The association is statistically significant (p = {p:.3g}) but in the opposite direction: "
            f"higher student-teacher ratio (more students per teacher) is associated with higher AvgScore. "
            f"Estimated effect: a one-unit increase in StudentTeacherRatio -> {coef:.3f} points in AvgScore "
            f"(95% CI: {fmt(ci_lower, '.3f')} to {fmt(ci_upper, '.3f')})."
        )
    elif coef > 0 and not significant_0_05:
        yes_no = "No / Inconclusive"
        conclusion = (
            f"{yes_no}: The estimated association is positive (higher ratio -> higher AvgScore) but not "
            f"statistically significant at the 5% level (p = {p:.3g}). "
            f"Estimated effect: {coef:.3f} (SE = {fmt(se, '.3f')}; 95% CI: {fmt(ci_lower, '.3f')} to {fmt(ci_upper, '.3f')})."
        )
    else:
        # coef == 0
        yes_no = "No"
        conclusion = "No association estimated (coefficient = 0)."

    # Compose final description including the numeric summary
    description = (
        f"Summary for variable '{var}': coef = {fmt(coef, '.4f')}, SE = {fmt(se, '.4f')}, t = {fmt(t, '.3f')}, "
        f"p = {fmt(p, '.4g')}, 95% CI = [{fmt(ci_lower, '.4f')}, {fmt(ci_upper, '.4f')}], n = {nobs}. "
        f"{direction} {conclusion}"
    )

    return {"object": result_object, "description": description}