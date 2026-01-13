def extract_final_answer(model_output):
    """
    Extracts statistics for the name-femininity variable from available fitted models.

    Returns a dict with keys:
      - "object": a dict containing extracted numeric results (model used, term name,
                  coefficient, standard error, p-value, 95% CI, nobs, R-squared when available)
                  or None if extraction was not possible.
      - "description": a short textual interpretation of the effect of name femininity
                       on the outcome and whether the result supports the hypothesis
                       that more feminine names lead to fewer deaths (i.e., negative effect).

    The function prefers the primary 'deaths_model' if present, otherwise falls back to
    'bivariate_deaths_model' and then 'damage_model'. It looks for the parameter name
    'masfem_z' (used in the modeling code) and then 'masfem' as a fallback.
    """
    # Helper to format a failure response
    def _no_result(msg):
        return {
            "object": None,
            "description": msg
        }

    if not isinstance(model_output, dict):
        return _no_result("model_output is not a dict as expected.")

    # Preferential order of models to inspect
    candidate_keys = ['deaths_model', 'bivariate_deaths_model', 'damage_model']

    res = None
    res_key = None
    for k in candidate_keys:
        if k in model_output and model_output[k] is not None:
            res = model_output[k]
            res_key = k
            break

    if res is None:
        return _no_result("No fitted models found in model_output (all are None).")

    # Basic validation that this looks like a statsmodels RegressionResults-like object
    if not hasattr(res, 'params'):
        return _no_result(f"Selected model '{res_key}' does not expose params; cannot extract estimates.")

    params = res.params
    # candidate parameter names used in the modeling code
    term_candidates = ['masfem_z', 'masfem']
    term = next((t for t in term_candidates if t in params.index), None)

    if term is None:
        return _no_result(
            f"Selected model '{res_key}' does not contain the expected femininity term (searched for {term_candidates})."
        )

    # Extract statistics robustly
    try:
        coef = float(params[term])
    except Exception:
        return _no_result("Could not read coefficient value for term '{}'.".format(term))

    # Standard error
    se = None
    if hasattr(res, 'bse') and term in res.bse.index:
        try:
            se = float(res.bse[term])
        except Exception:
            se = None

    # p-value
    pval = None
    if hasattr(res, 'pvalues') and term in res.pvalues.index:
        try:
            pval = float(res.pvalues[term])
        except Exception:
            pval = None

    # 95% confidence interval: handle DataFrame or numpy array returns from conf_int()
    ci_low = ci_high = None
    try:
        ci = res.conf_int()
        # If conf_int() returned a DataFrame-like object with index
        if hasattr(ci, 'loc') and term in ci.index:
            ci_low, ci_high = float(ci.loc[term, 0]), float(ci.loc[term, 1])
        else:
            # assume numpy array; find index of term
            idx = list(params.index).index(term)
            ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        ci_low = ci_high = None

    # nobs
    nobs = None
    try:
        # statsmodels sometimes exposes nobs as attribute
        if hasattr(res, 'nobs'):
            nobs = int(res.nobs)
        elif hasattr(res, 'model') and hasattr(res.model, 'nobs'):
            nobs = int(res.model.nobs)
    except Exception:
        nobs = None

    # R-squared when available (may not be meaningful for robust fits but is useful)
    rsq = None
    try:
        if hasattr(res, 'rsquared'):
            rsq = float(res.rsquared)
    except Exception:
        rsq = None

    # Build the numeric result object
    result_object = {
        "model_used": res_key,
        "term": term,
        "coefficient": coef,
        "std_error": se,
        "p_value": pval,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_obs": nobs,
        "r_squared": rsq
    }

    # Interpret the coefficient with respect to the hypothesis:
    # Hypothesis: higher femininity -> fewer deaths (negative coef expected).
    direction = "negative" if coef < 0 else ("zero" if coef == 0 else "positive")
    significance = None
    if pval is None:
        significance = "p-value not available"
    else:
        if pval < 0.01:
            significance = "highly statistically significant (p < 0.01)"
        elif pval < 0.05:
            significance = "statistically significant (p < 0.05)"
        elif pval < 0.10:
            significance = "marginally significant (p < 0.10)"
        else:
            significance = "not statistically significant (p >= 0.10)"

    # Conclusion re: hypothesis
    if pval is not None and coef < 0 and pval < 0.05:
        conclusion = (
            "The estimated effect of name femininity on the log number of deaths is negative "
            "and statistically significant, which is consistent with the hypothesis that "
            "hurricanes with more feminine names are associated with fewer fatalities (all else equal)."
        )
    elif pval is not None and coef < 0 and pval >= 0.05:
        conclusion = (
            "The estimated effect is negative (higher femininity -> fewer deaths) but not "
            "statistically significant; there is not strong evidence to conclude a real effect."
        )
    elif pval is not None and coef >= 0:
        conclusion = (
            "The estimated effect is {} (higher femininity -> {} deaths) and {}. "
            "This does not provide evidence supporting the hypothesis that more feminine names lead "
            "to fewer fatalities.".format(direction, "fewer" if coef < 0 else "more", significance)
        )
    else:
        # pval is None
        conclusion = (
            "Coefficient is {} but p-value is not available; cannot make a statistical conclusion. "
            "See numeric results returned in 'object' for details.".format(direction)
        )

    # Compose description
    description_lines = [
        f"Model used: {res_key}. Term examined: '{term}'.",
        f"Estimated coefficient = {coef:.4g}",
    ]
    if se is not None:
        description_lines.append(f"SE = {se:.4g}")
    if pval is not None:
        description_lines.append(f"p-value = {pval:.4g} ({significance})")
    if ci_low is not None and ci_high is not None:
        description_lines.append(f"95% CI = [{ci_low:.4g}, {ci_high:.4g}]")
    if nobs is not None:
        description_lines.append(f"Number of observations used in this fit = {nobs}")
    if rsq is not None:
        description_lines.append(f"R-squared = {rsq:.4g}")
    description_lines.append(conclusion)

    description = " ".join(description_lines)

    return {
        "object": result_object,
        "description": description
    }