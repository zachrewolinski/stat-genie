def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator on mortgage denial from the model output.

    Returns a dict with:
      - "object": a dict containing numeric results (coefficient, p-value, odds ratio, 95% CI, significance)
      - "description": a short plain-language explanation of what those numbers mean

    Expects model_output to contain at least either:
      - 'model_result' : a statsmodels results wrapper (preferred), or
      - 'odds_ratios', 'odds_ratio_CI_lower', 'odds_ratio_CI_upper' series/dicts.
    """
    # Defensive checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Try to get stats from statsmodels result if available
    coef = None
    p_value = None
    ci_lower = None
    ci_upper = None
    odds_ratio = None

    result = model_output.get('model_result', None)
    try:
        if result is not None:
            # params, pvalues, conf_int are pandas Series/DataFrame with index = covariate names
            coef = float(result.params['female'])
            p_value = float(result.pvalues['female'])
            ci = result.conf_int().loc['female']  # two-element array-like [lower, upper] on log-odds scale
            ci_lower_log, ci_upper_log = float(ci[0]), float(ci[1])
            # convert to odds ratio scale
            odds_ratio = float(np.exp(coef))
            ci_lower = float(np.exp(ci_lower_log))
            ci_upper = float(np.exp(ci_upper_log))
    except Exception:
        # If any of the above fails, we'll try to read from precomputed odds ratios in model_output
        coef = coef  # keep what we might have
        p_value = p_value

    # Fallback to odds_ratios keys if odds ratio or CI not captured above
    if odds_ratio is None and 'odds_ratios' in model_output:
        ors = model_output['odds_ratios']
        try:
            odds_ratio = float(ors['female'])
        except Exception:
            try:
                # if it's a numpy array or list in same order as covariates_used
                covs = model_output.get('covariates_used', [])
                if 'female' in covs:
                    idx = covs.index('female')
                    odds_ratio = float(ors.iloc[idx]) if hasattr(ors, "iloc") else float(ors[idx])
            except Exception:
                odds_ratio = None

    if (ci_lower is None or ci_upper is None) and 'odds_ratio_CI_lower' in model_output and 'odds_ratio_CI_upper' in model_output:
        try:
            ci_lower = float(model_output['odds_ratio_CI_lower']['female'])
            ci_upper = float(model_output['odds_ratio_CI_upper']['female'])
        except Exception:
            try:
                covs = model_output.get('covariates_used', [])
                if 'female' in covs:
                    idx = covs.index('female')
                    lower_series = model_output['odds_ratio_CI_lower']
                    upper_series = model_output['odds_ratio_CI_upper']
                    ci_lower = float(lower_series.iloc[idx]) if hasattr(lower_series, "iloc") else float(lower_series[idx])
                    ci_upper = float(upper_series.iloc[idx]) if hasattr(upper_series, "iloc") else float(upper_series[idx])
            except Exception:
                ci_lower = ci_lower
                ci_upper = ci_upper

    # If coefficient or p-value missing but model_result exists, try to extract just those individually
    if (coef is None or p_value is None) and result is not None:
        try:
            if coef is None:
                coef = float(result.params['female'])
            if p_value is None:
                p_value = float(result.pvalues['female'])
        except Exception:
            pass

    # Build final object with what we have
    obj = {
        'coefficient_log_odds': coef,
        'p_value': p_value,
        'odds_ratio': odds_ratio,
        'odds_ratio_CI': (ci_lower, ci_upper) if (ci_lower is not None and ci_upper is not None) else None,
        'significant_at_0.05': (p_value is not None and p_value < 0.05)
    }

    # Plain-language description
    if odds_ratio is not None and obj['odds_ratio_CI'] is not None and p_value is not None:
        direction = "lower" if odds_ratio < 1 else "higher" if odds_ratio > 1 else "no change"
        desc = (
            f"The model coefficient for 'female' (female=1 vs male=0) is {coef:.4f} on the log-odds scale "
            f"(p = {p_value:.3g}). The estimated odds ratio is {odds_ratio:.3f} "
            f"with 95% CI [{obj['odds_ratio_CI'][0]:.3f}, {obj['odds_ratio_CI'][1]:.3f}]. "
            f"This indicates that, controlling for the listed covariates, female applicants have {direction} odds "
            f"of mortgage denial compared to male applicants. "
            f"{'The effect is statistically significant at the 5% level.' if obj['significant_at_0.05'] else 'The effect is not statistically significant at the 5% level.'}"
        )
    else:
        desc = "Could not extract all statistics for 'female' from model_output. Returned whatever values were found in the 'object' field."

    return {"object": obj, "description": desc}