def extract_final_answer(model_output):
    """
    Extracts coefficient, p-value, confidence interval, t-stat, standardized beta (if possible),
    and a short interpretation for the StudentTeacherRatio coefficient from a statsmodels
    RegressionResultsWrapper.
    Returns a dict with keys:
      - "object": dict of numeric outputs (coef, pvalue, tstat, conf_int, std_beta, nobs, significant)
      - "description": text interpretation in context
    """
    res = model_output

    # Basic validation
    if not hasattr(res, "params"):
        return {
            "object": None,
            "description": "Provided model_output does not appear to be a statsmodels RegressionResults object (missing .params)."
        }

    var = "StudentTeacherRatio"
    if var not in res.params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in model parameters. Available params: {list(res.params.index)}"
        }

    # Extract statistics
    coef = float(res.params[var])
    pval = float(res.pvalues[var])
    tstat = float(res.tvalues[var])
    ci = res.conf_int().loc[var].tolist()  # [lower, upper]
    conf_lower, conf_upper = float(ci[0]), float(ci[1])
    nobs = int(res.nobs) if hasattr(res, "nobs") else None

    # Attempt to compute a standardized (beta) coefficient using model endog/exog if available
    std_beta = None
    try:
        exog_names = list(res.model.exog_names)
        if var in exog_names:
            idx = exog_names.index(var)
            x = res.model.exog[:, idx]
            y = res.model.endog
            sd_x = x.std(ddof=0)
            sd_y = y.std(ddof=0)
            if sd_y != 0:
                std_beta = float(coef * (sd_x / sd_y))
    except Exception:
        std_beta = None  # leave as None if any problem

    significant = pval < 0.05

    # Interpretation logic
    if coef < 0:
        direction = "negative"
        meaning = "lower student-teacher ratio (fewer students per teacher) is associated with higher AvgTestScore"
    elif coef > 0:
        direction = "positive"
        meaning = "higher student-teacher ratio (more students per teacher) is associated with higher AvgTestScore"
    else:
        direction = "no effect"
        meaning = "no association detected"

    significance_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"

    description = (
        f"StudentTeacherRatio coefficient = {coef:.4f} (t = {tstat:.2f}, p = {pval:.3g}); "
        f"95% CI = [{conf_lower:.4f}, {conf_upper:.4f}]. "
    )
    if std_beta is not None:
        description += f"Standardized beta ≈ {std_beta:.4f}. "
    description += (
        f"The association is {direction} and {significance_text}. "
        f"In context: {meaning}, controlling for other covariates and county fixed effects. "
        f"Sample size used in regression: {nobs}."
    )

    out_obj = {
        "variable": var,
        "coef": coef,
        "pvalue": pval,
        "tstat": tstat,
        "conf_int": [conf_lower, conf_upper],
        "standardized_beta": std_beta,
        "nobs": nobs,
        "significant_at_0_05": significant
    }

    return {"object": out_obj, "description": description}