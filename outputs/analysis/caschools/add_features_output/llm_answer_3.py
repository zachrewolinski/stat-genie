def extract_final_answer(model_output):
    """
    Extract key statistics for the StudentTeacherRatio coefficient from a fitted statsmodels OLS result.

    Returns a dictionary with:
      - "object": a dict containing numeric results (coefficient, se, t, p, 95% CI, sample size, adj. R^2)
      - "description": a brief plain-language interpretation answering whether a lower student-teacher
                       ratio is associated with higher academic performance (including significance and caveats).

    The function is defensive: if the coefficient is not present it returns a clear message.
    """
    result = {"object": None, "description": ""}

    varname = "StudentTeacherRatio"
    try:
        params = model_output.params
    except Exception as e:
        result["description"] = f"Provided model_output does not look like a fitted statsmodels result: {e}"
        return result

    if varname not in params.index:
        result["description"] = f"Variable '{varname}' not found in the model parameters. Available parameters: {list(params.index)}"
        return result

    # Extract statistics
    coef = float(model_output.params[varname])
    # bse, tvalue, pvalue, conf_int may be available
    try:
        se = float(model_output.bse[varname])
    except Exception:
        se = None
    try:
        tval = float(model_output.tvalues[varname])
    except Exception:
        tval = None
    try:
        pval = float(model_output.pvalues[varname])
    except Exception:
        pval = None
    try:
        ci = model_output.conf_int().loc[varname].tolist()  # [lower, upper]
        ci = [float(ci[0]), float(ci[1])]
    except Exception:
        ci = None

    # Sample size and adj. R-squared for context
    try:
        n_obs = int(model_output.nobs)
    except Exception:
        n_obs = None
    try:
        r2_adj = float(model_output.rsquared_adj)
    except Exception:
        # fallback to rsquared if adj not present
        try:
            r2_adj = float(model_output.rsquared)
        except Exception:
            r2_adj = None

    # Interpret direction and significance
    significance = None
    if pval is not None:
        significance = (pval < 0.05)
    # Because StudentTeacherRatio is defined as students per teacher (higher = more students per teacher),
    # a negative coefficient implies that higher ratio -> lower AcademicScore, i.e. lower ratio -> higher performance.
    if significance is True:
        if coef < 0:
            short_conclusion = ("Yes: statistically significant evidence (p={:.3g}) that a lower student-teacher "
                                "ratio is associated with higher academic performance.").format(pval)
        else:
            short_conclusion = ("No: statistically significant evidence (p={:.3g}) that a higher student-teacher "
                                "ratio is associated with higher academic performance (i.e., lower ratio -> lower performance).").format(pval)
    elif significance is False:
        # not significant
        if coef < 0:
            short_conclusion = ("Inconclusive: coefficient is negative (coef = {:.4g}) suggesting lower ratio might "
                                "be associated with higher performance, but this effect is not statistically "
                                "significant (p = {:.3g}).").format(coef, pval if pval is not None else float("nan"))
        else:
            short_conclusion = ("Inconclusive: coefficient is positive (coef = {:.4g}) suggesting lower ratio might "
                                "not be associated with higher performance, but this effect is not statistically "
                                "significant (p = {:.3g}).").format(coef, pval if pval is not None else float("nan"))
    else:
        short_conclusion = "Could not determine statistical significance (p-value unavailable)."

    # Build the object to return (numeric summary)
    numeric_summary = {
        "variable": varname,
        "coef": coef,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
        "95%_conf_int": ci,
        "n_obs": n_obs,
        "adj_R2": r2_adj,
        # Interpretative fields
        "significant_at_0.05": significance,
        "direction": "negative (higher ratio -> lower score)" if coef < 0 else "positive (higher ratio -> higher score)" if coef > 0 else "zero"
    }

    # Construct description that includes interpretation and caveats
    desc_lines = []
    desc_lines.append(short_conclusion)
    desc_lines.append("Estimated effect: a one-unit increase in StudentTeacherRatio (one more student per teacher) is associated with a change of {:.4g} points in AcademicScore (95% CI: {} to {}, p = {})."
                      .format(coef,
                              "{:.4g}".format(ci[0]) if ci is not None else "NA",
                              "{:.4g}".format(ci[1]) if ci is not None else "NA",
                              "{:.3g}".format(pval) if pval is not None else "NA"))
    desc_lines.append("Model controls included expenditure, income, calworks, lunch, english, log(students), county fixed effects, and grade-span fixed effects.")
    desc_lines.append("Caveat: this is an observational association, not necessarily causal. Results depend on model specification and data quality.")
    # Join into one paragraph
    description = " ".join(desc_lines)

    result["object"] = numeric_summary
    result["description"] = description

    return result