def extract_final_answer(model_output):
    """
    Extracts the estimated effect of StudentTeacherRatio from a fitted statsmodels
    RegressionResultsWrapper (with robust SEs or not) and returns a concise
    summary and numeric results.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coef, se, t, p, 95% CI, nobs,
                  sign, significant boolean)
      - "description": a short interpreted statement about whether a lower
                       student-teacher ratio is associated with higher academic performance.
    """
    import numpy as np
    # Name of the variable of interest
    var = "StudentTeacherRatio"

    # Basic validation / access
    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided."
        }

    # Ensure model_output exposes params
    try:
        params = model_output.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Provided model_output does not appear to be a statsmodels results object: {e}"
        }

    if var not in params.index:
        return {
            "object": None,
            "description": f"The model does not contain the variable '{var}'."
        }

    # Extract statistics (handle absence of some attributes defensively)
    coef = float(params[var])
    se = float(model_output.bse[var]) if hasattr(model_output, "bse") and var in model_output.bse.index else None
    tvalue = float(model_output.tvalues[var]) if hasattr(model_output, "tvalues") and var in model_output.tvalues.index else (coef / se if se not in (None, 0) else None)
    pvalue = float(model_output.pvalues[var]) if hasattr(model_output, "pvalues") and var in model_output.pvalues.index else None

    # Confidence interval
    try:
        ci = model_output.conf_int().loc[var]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        ci_lower = ci_upper = None

    # Number of observations if available
    nobs = int(getattr(model_output, "nobs", np.nan))

    # Interpret sign and significance
    sign = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
    significant = None
    if pvalue is not None:
        significant = bool(pvalue < 0.05)

    # Plain-language interpretation of effect:
    # Note: variable is defined so that higher = more students per teacher; lower = smaller class size.
    if coef < 0:
        effect_text = (
            f"The estimated coefficient on {var} is {coef:.3f} (SE={se:.3f}, p={pvalue:.3g}). "
            f"Because the coefficient is negative, it implies that lower student-to-teacher ratios "
            f"(smaller class sizes) are associated with higher AvgScore. "
            f"Specifically, a one-unit decrease in the StudentTeacherRatio is associated with an "
            f"average increase of {abs(coef):.3f} points in AvgScore."
        )
    elif coef > 0:
        effect_text = (
            f"The estimated coefficient on {var} is {coef:.3f} (SE={se:.3f}, p={pvalue:.3g}). "
            f"Because the coefficient is positive, it implies that lower student-to-teacher ratios "
            f"(smaller class sizes) are associated with lower AvgScore (i.e., the opposite of the hypothesis). "
            f"Specifically, a one-unit increase in the StudentTeacherRatio is associated with an average increase of {coef:.3f} points in AvgScore."
        )
    else:
        effect_text = f"The estimated coefficient on {var} is exactly zero."

    sig_text = ""
    if significant is True:
        sig_text = " The effect is statistically significant at the 5% level."
    elif significant is False:
        sig_text = " The effect is not statistically significant at the 5% level."
    else:
        sig_text = ""

    # Construct returned object (numeric results plus interpretation flags)
    result_object = {
        "variable": var,
        "coef": coef,
        "se": se,
        "t": tvalue,
        "p": pvalue,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "nobs": nobs,
        "sign": sign,
        "significant_at_0.05": significant
    }

    description = (
        f"{effect_text}{sig_text} "
        f"95% CI = [{ci_lower:.3f}, {ci_upper:.3f}] (if available). "
        f"Interpretation: since higher StudentTeacherRatio means more students per teacher, "
        f"a negative coefficient supports the hypothesis that smaller class sizes (lower ratio) are associated with higher academic performance."
    )

    return {"object": result_object, "description": description}