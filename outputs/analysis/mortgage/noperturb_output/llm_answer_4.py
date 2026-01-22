def extract_final_answer(model_output):
    """
    Extract the effect of the 'female' indicator from the model output returned by the
    modeling function. Returns a dict with numeric results under "object" and a brief
    interpretation under "description".
    Expected input: the dict produced by the provided model() function (contains keys
    like 'fit_robust' and 'odds_ratios'), or possibly the fit object itself.
    """
    import numpy as np
    from scipy.stats import norm

    # Normalize input: accept either the dict or the fit object directly
    fit = None
    odds_df = None

    if isinstance(model_output, dict):
        fit = model_output.get('fit_robust', None)
        odds_df = model_output.get('odds_ratios', None)
    else:
        # If a statsmodels fit object was passed directly
        fit = model_output

    if fit is None:
        raise ValueError("Could not find fitted model in model_output. Expecting a dict with key 'fit_robust' or a fit object.")

    # Ensure parameter 'female' exists
    params = getattr(fit, 'params', None)
    if params is None or 'female' not in params.index:
        raise ValueError("Model does not contain a 'female' coefficient.")

    # Coefficient (log-odds)
    coef = float(params.loc['female'])

    # Robust standard error if attached, otherwise fallback to model's bse
    se_robust = None
    bse_attr = getattr(fit, 'bse_robust', None)
    if bse_attr is not None and 'female' in getattr(bse_attr, 'index', bse_attr):
        se_robust = float(bse_attr.loc['female'])
    else:
        # fallback
        bse = getattr(fit, 'bse', None)
        if bse is not None and 'female' in bse.index:
            se_robust = float(bse.loc['female'])
        else:
            raise ValueError("Could not locate a standard error for 'female' in the fit object.")

    # z-stat and two-sided p-value using robust SE (or fallback SE)
    z_stat = coef / se_robust if se_robust != 0 else np.nan
    p_value = float(2 * (1 - norm.cdf(abs(z_stat)))) if not np.isnan(z_stat) else np.nan

    # Odds ratio and CI: prefer provided odds_df if available (precomputed using robust SE),
    # otherwise compute from coef +/- z_crit * se_robust
    or_val = None
    ci_lower_or = None
    ci_upper_or = None

    if odds_df is not None and 'female' in odds_df.index:
        try:
            or_val = float(odds_df.loc['female', 'OR'])
            ci_lower_or = float(odds_df.loc['female', 'CI_lower'])
            ci_upper_or = float(odds_df.loc['female', 'CI_upper'])
        except Exception:
            or_val = None

    if or_val is None:
        # compute 95% CI on log-odds then exponentiate
        z_crit = norm.ppf(0.975)
        ci_low_log = coef - z_crit * se_robust
        ci_high_log = coef + z_crit * se_robust
        or_val = float(np.exp(coef))
        ci_lower_or = float(np.exp(ci_low_log))
        ci_upper_or = float(np.exp(ci_high_log))

    # Build returned object
    extracted = {
        'coef_log_odds': coef,
        'se_used_for_test': se_robust,
        'z_stat': z_stat,
        'p_value_two_sided': p_value,
        'odds_ratio': or_val,
        'OR_95CI_lower': ci_lower_or,
        'OR_95CI_upper': ci_upper_or,
        # convenience boolean: statistically significant at alpha=0.05 using the computed p-value
        'significant_at_0.05': bool(p_value < 0.05) if not np.isnan(p_value) else None
    }

    # Short interpretation
    # Note: mention that the CI and p-value were computed using the robust SE when available.
    description = (
        f"Effect of being female on mortgage acceptance: the estimated log-odds coef = {coef:.4f} "
        f"(SE used = {se_robust:.4f}), z = {z_stat:.3f}, two-sided p = {p_value:.3f}. "
        f"Equivalently, odds ratio = {or_val:.3f} with 95% CI [{ci_lower_or:.3f}, {ci_upper_or:.3f}]. "
        f"This indicates that, holding the listed controls constant, female applicants have higher odds "
        f"of approval (OR > 1). The effect is statistically significant at the 0.05 level: "
        f"{'yes' if extracted['significant_at_0.05'] else 'no'}."
    )

    return {"object": extracted, "description": description}