def extract_final_answer(model_output):
    """
    Extract statistics relevant to the effect of applicant gender (Female) on mortgage approval
    from the provided model_output (expected to be the dict returned by the model function).

    Returns a dict with keys:
      - "object": a dict with numeric results for:
            * Female (main effect: women vs men among non-Black applicants)
            * Female_Black (interaction term)
            * Female_effect_black_applicants (sum of Female + Female_Black: effect of being female among Black applicants)
        Each entry contains: coef (log-odds), se, pvalue, odds_ratio, ci_lower, ci_upper (95% CI on OR scale).
      - "description": short interpretation of the results in plain language.
    """
    import numpy as np
    from scipy import stats

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")
    if 'model' not in model_output:
        raise ValueError("model_output missing 'model' key.")

    res = model_output['model']  # statsmodels GLMResultsWrapper expected

    # Ensure required parameters exist
    required_params = ['Female', 'Female_Black']
    for p in required_params:
        if p not in res.params.index:
            raise KeyError(f"Expected parameter '{p}' not found in model parameters: {list(res.params.index)}")

    # Extract parameter-level statistics for Female and Female_Black
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    conf = res.conf_int()  # log-odds scale
    conf.columns = ['2.5%', '97.5%']

    def summarize_term(name):
        coef = float(params[name])
        se = float(bse[name])
        pval = float(pvalues[name])
        or_ratio = float(np.exp(coef))
        ci_low = float(np.exp(conf.loc[name, '2.5%']))
        ci_high = float(np.exp(conf.loc[name, '97.5%']))
        return {
            'coef_log_odds': coef,
            'se': se,
            'pvalue': pval,
            'odds_ratio': or_ratio,
            'ci_95_odds_ratio': (ci_low, ci_high)
        }

    female_stats = summarize_term('Female')
    female_black_stats = summarize_term('Female_Black')

    # Compute combined effect for Black applicants: Female + Female_Black
    cov = res.cov_params()  # covariance matrix of coefficients
    # variance of sum = var(Female) + var(Female_Black) + 2*cov(Female, Female_Black)
    var_sum = (
        cov.loc['Female', 'Female']
        + cov.loc['Female_Black', 'Female_Black']
        + 2.0 * cov.loc['Female', 'Female_Black']
    )
    se_sum = float(np.sqrt(var_sum))
    coef_sum = float(params['Female'] + params['Female_Black'])
    z_sum = coef_sum / se_sum if se_sum > 0 else np.nan
    p_sum = float(2.0 * stats.norm.sf(abs(z_sum))) if se_sum > 0 else np.nan
    or_sum = float(np.exp(coef_sum))
    ci_low_sum = float(np.exp(coef_sum - 1.96 * se_sum))
    ci_high_sum = float(np.exp(coef_sum + 1.96 * se_sum))

    female_effect_black = {
        'coef_log_odds': coef_sum,
        'se': se_sum,
        'pvalue': p_sum,
        'odds_ratio': or_sum,
        'ci_95_odds_ratio': (ci_low_sum, ci_high_sum)
    }

    # Build a short interpretation
    # Determine statistical significance (two-sided alpha=0.05)
    def signif(p):
        return (p is not None) and (not np.isnan(p)) and (p < 0.05)

    parts = []
    # Main effect among non-Black applicants
    parts.append(
        f"Main effect (Female vs Male) among non-Black applicants: OR={female_stats['odds_ratio']:.3f}, "
        f"95% CI=({female_stats['ci_95_odds_ratio'][0]:.3f}, {female_stats['ci_95_odds_ratio'][1]:.3f}), "
        f"p={female_stats['pvalue']:.3f}."
    )
    if signif(female_stats['pvalue']):
        parts.append("This is a statistically significant difference.")
    else:
        parts.append("This is NOT statistically significant (CI includes 1 / p >= 0.05).")

    # Interaction term
    parts.append(
        f"Interaction (Female x Black): OR={female_black_stats['odds_ratio']:.3f}, "
        f"95% CI=({female_black_stats['ci_95_odds_ratio'][0]:.3f}, {female_black_stats['ci_95_odds_ratio'][1]:.3f}), "
        f"p={female_black_stats['pvalue']:.3f}."
    )
    if signif(female_black_stats['pvalue']):
        parts.append("Interaction is statistically significant (gender effect differs by Black status).")
    else:
        parts.append("Interaction is NOT statistically significant.")

    # Combined effect among Black applicants
    parts.append(
        f"Combined effect for Black applicants (Female + Female_Black): OR={female_effect_black['odds_ratio']:.3f}, "
        f"95% CI=({female_effect_black['ci_95_odds_ratio'][0]:.3f}, {female_effect_black['ci_95_odds_ratio'][1]:.3f}), "
        f"p={female_effect_black['pvalue']:.3f}."
    )
    if signif(female_effect_black['pvalue']):
        parts.append("This is statistically significant among Black applicants.")
    else:
        parts.append("This is NOT statistically significant among Black applicants.")

    description = " ".join(parts)

    result_object = {
        'Female': female_stats,
        'Female_Black': female_black_stats,
        'Female_effect_black_applicants': female_effect_black
    }

    return {
        'object': result_object,
        'description': description
    }