def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, and 95% CI for the
    'StudentTeacherRatio' predictor from a fitted statsmodels OLS result (or
    similar RegressionResultsWrapper). Interprets whether a lower
    student-teacher ratio (smaller number = fewer students per teacher)
    is associated with higher AvgTestScore.
    
    Returns a dict with keys:
      - "object": dict of numeric results (coef, se, t, p, ci_lower, ci_upper,
                  significant (bool), direction)
      - "description": brief human-readable interpretation in context
    """
    var = 'StudentTeacherRatio'
    result = {
        "object": None,
        "description": None
    }

    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
        tvalues = model_output.tvalues
        ci = model_output.conf_int(alpha=0.05)  # default 95% CI
    except Exception as e:
        result["description"] = f"Could not extract statistics from model_output: {e}"
        return result

    if var not in params.index:
        result["description"] = f"Variable '{var}' not found in model output."
        return result

    coef = float(params[var])
    se = float(bse[var]) if var in bse.index else None
    tstat = float(tvalues[var]) if var in tvalues.index else None
    pval = float(pvalues[var]) if var in pvalues.index else None
    ci_lower, ci_upper = (float(ci.loc[var, 0]), float(ci.loc[var, 1])) if var in ci.index else (None, None)

    # Interpretation logic:
    # Lower StudentTeacherRatio means fewer students per teacher.
    # If coef < 0 then lower ratio -> higher AvgTestScore (negative association).
    direction = "negative" if coef < 0 else ("positive" if coef > 0 else "none")
    # Statistical significance at conventional alpha = 0.05
    significant = (pval is not None) and (pval < 0.05)

    # Build the numeric object to return
    numeric_object = {
        "variable": var,
        "coef": coef,
        "std_err": se,
        "t_stat": tstat,
        "p_value": pval,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "direction": direction,
        "significant_at_0.05": bool(significant)
    }

    # Human-readable description
    if pval is None:
        desc = f"Extracted coefficient for '{var}' = {coef:.4g}. Could not extract p-value."
    else:
        sign_text = ("A negative coefficient" if coef < 0 else
                     "A positive coefficient" if coef > 0 else
                     "No association (coefficient = 0)")
        sig_text = ("statistically significant" if significant else "not statistically significant")
        # Map direction to answer to the yes/no question:
        if coef < 0 and significant:
            yesno = "Yes — lower student-teacher ratio is associated with higher academic performance (statistically significant)."
        elif coef < 0 and not significant:
            yesno = "Suggestive (not statistically significant) evidence that lower student-teacher ratio is associated with higher academic performance."
        elif coef > 0 and significant:
            yesno = "No — higher student-teacher ratio is associated with higher academic performance (statistically significant), i.e., the opposite of the hypothesis."
        elif coef > 0 and not significant:
            yesno = "No clear evidence that lower student-teacher ratio is associated with higher academic performance (coefficient positive but not statistically significant)."
        else:
            yesno = "No association detected (coefficient is essentially zero)."

        desc = (
            f"{sign_text} for '{var}' = {coef:.4g} (SE = {se:.4g}, t = {tstat:.4g}, p = {pval:.4g}). "
            f"95% CI = [{ci_lower:.4g}, {ci_upper:.4g}]. This estimate is {sig_text}. {yesno}"
        )

    result["object"] = numeric_object
    result["description"] = desc
    return result