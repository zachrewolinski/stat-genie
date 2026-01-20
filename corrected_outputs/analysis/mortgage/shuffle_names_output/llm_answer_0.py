def extract_final_answer(model_output):
    """
    Extract the effect of 'is_female' from a fitted statsmodels GLM/GLMResultsWrapper (robust or not).
    Returns a dictionary with:
      - "object": dict with numeric results (coefficient, p-value, odds ratio, CIs, nobs, significance)
      - "description": short human-readable interpretation of the effect in context
    
    This function attempts to handle:
      - statsmodels result wrappers returned by GLM.fit() or .get_robustcov_results()
      - cases where p-values or conf_int() are not present by computing them from covariance
    """
    import numpy as np
    from math import exp
    try:
        from scipy import stats
    except Exception:
        # If scipy is not available, use normal approximation via math and numpy for cdf
        stats = None

    # Unwrap if a list/tuple was passed
    res = model_output
    if isinstance(res, (list, tuple)) and len(res) > 0:
        res = res[0]

    # Ensure we have params
    if not hasattr(res, "params"):
        raise ValueError("Provided model_output does not have 'params' attribute. Expected a statsmodels results object.")

    params = res.params
    if 'is_female' not in params.index:
        raise ValueError("The model does not contain a parameter named 'is_female'.")

    # Extract coefficient (log-odds)
    coef = float(params.loc['is_female'])

    # Try to obtain p-value directly; otherwise compute from covariance
    pvalue = None
    try:
        pvalue = float(res.pvalues.loc['is_female'])
    except Exception:
        try:
            cov = res.cov_params()
            se = float(np.sqrt(cov.loc['is_female', 'is_female']))
            z = coef / se
            if stats is not None:
                pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
            else:
                # approximate normal cdf via numpy
                pvalue = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(z) / np.sqrt(2.0))))
        except Exception:
            pvalue = None

    # Confidence interval on log-odds: try res.conf_int(); otherwise compute using cov
    ci_log = None
    try:
        ci_df = res.conf_int()
        # conf_int may return array or DataFrame
        try:
            lower, upper = ci_df.loc['is_female'].tolist()
        except Exception:
            # conf_int returned array with rows in same order as params
            # find position
            idx = list(params.index).index('is_female')
            lower, upper = ci_df[idx].tolist() if isinstance(ci_df, (list, tuple)) else (float(ci_df[idx, 0]), float(ci_df[idx, 1]))
        ci_log = [float(lower), float(upper)]
    except Exception:
        try:
            cov = res.cov_params()
            se = float(np.sqrt(cov.loc['is_female', 'is_female']))
            # 95% CI using normal approximation
            z_crit = 1.96
            ci_log = [coef - z_crit * se, coef + z_crit * se]
        except Exception:
            ci_log = [None, None]

    # Odds ratio and its CI
    try:
        odds_ratio = float(exp(coef))
    except Exception:
        odds_ratio = None
    if ci_log[0] is not None and ci_log[1] is not None:
        try:
            ci_odds = [float(exp(ci_log[0])), float(exp(ci_log[1]))]
        except Exception:
            ci_odds = [None, None]
    else:
        ci_odds = [None, None]

    # Sample size
    nobs = None
    try:
        # statsmodels stores number of observations in several places
        if hasattr(res, 'nobs') and res.nobs is not None:
            nobs = int(res.nobs)
        elif hasattr(res, 'model') and hasattr(res.model, 'nobs') and res.model.nobs is not None:
            nobs = int(res.model.nobs)
        else:
            # fallback: length of endogenous if available
            if hasattr(res, 'endog') and res.endog is not None:
                nobs = int(len(res.endog))
    except Exception:
        nobs = None

    # Significance at alpha = 0.05 (if p-value available)
    significant = None
    if pvalue is not None:
        significant = (pvalue < 0.05)

    # Percent change in odds (interpretation)
    pct_change_odds = None
    if odds_ratio is not None:
        pct_change_odds = (odds_ratio - 1.0) * 100.0

    # Build the object to return
    result_object = {
        "term": "is_female",
        "coef_log_odds": coef,
        "p_value": pvalue,
        "odds_ratio": odds_ratio,
        "95%_CI_log_odds": ci_log,
        "95%_CI_odds_ratio": ci_odds,
        "percent_change_in_odds": pct_change_odds,
        "nobs": nobs,
        "significant_at_0.05": significant
    }

    # Human-readable description
    if pvalue is None:
        ptext = "p-value could not be determined"
    else:
        ptext = f"p = {pvalue:.4g}"
    if odds_ratio is None:
        ortext = "odds ratio could not be computed"
    else:
        ortext = f"OR = {odds_ratio:.3f} (95% CI: {ci_odds[0]:.3f} to {ci_odds[1]:.3f})" if (ci_odds[0] is not None and ci_odds[1] is not None) else f"OR = {odds_ratio:.3f}"

    if significant is True:
        sign_text = "This effect is statistically significant at alpha=0.05."
    elif significant is False:
        sign_text = "This effect is not statistically significant at alpha=0.05."
    else:
        sign_text = ""

    description = (
        f"'is_female' coefficient (log-odds) = {coef:.4f}; {ptext}. "
        f"{ortext}. {sign_text} "
        + (f"Based on n = {nobs} observations." if nobs is not None else "")
    ).strip()

    return {"object": result_object, "description": description}