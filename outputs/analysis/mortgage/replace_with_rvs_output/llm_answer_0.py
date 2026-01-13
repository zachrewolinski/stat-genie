def extract_final_answer(model_output):
    """
    Extract and interpret the gender effect from the fitted logistic regression model output.
    Expects model_output to be a dict containing at least the key 'results' with a
    statsmodels BinaryResultsWrapper object (as produced by the provided modeling code).

    Returns a dict with:
      - "object": a dict of extracted numeric results (coefficients, SEs, p-values,
                  odds ratios, 95% CIs) for:
            * Female (effect among non-Black applicants)
            * Female_Black (interaction term)
            * Female_effect_for_Black (combined Female + Female_Black)
      - "description": a brief textual interpretation of those results in context.
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output.get('results', None)
    if res is None:
        raise ValueError("model_output must contain 'results' (statsmodels fitted result).")

    # Extract parameter estimates, standard errors, p-values, and covariance matrix
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    cov = res.cov_params()

    # Helper to build odds ratio and CI from log-odds coef and se (Wald CI)
    def summarize_term(name):
        coef = float(params[name])
        se = float(bse[name])
        p = float(pvals[name])
        or_ = float(np.exp(coef))
        # Wald 95% CI on log-odds scale
        z = norm.ppf(0.975)
        lo_log, hi_log = coef - z * se, coef + z * se
        ci_or = (float(np.exp(lo_log)), float(np.exp(hi_log)))
        return {
            'coef_logit': coef,
            'se': se,
            'p_value': p,
            'odds_ratio': or_,
            'odds_ratio_CI_95': ci_or
        }

    # Summaries for Female and interaction term
    female_summary = summarize_term('Female')
    interaction_summary = summarize_term('Female_Black')

    # Combined effect for Black applicants: Female + Female_Black
    coef_f = params['Female']
    coef_fb = params['Female_Black']
    coef_sum = float(coef_f + coef_fb)

    # variance of sum = var(Female) + var(Female_Black) + 2*cov(Female, Female_Black)
    var_f = cov.loc['Female', 'Female']
    var_fb = cov.loc['Female_Black', 'Female_Black']
    cov_f_fb = cov.loc['Female', 'Female_Black']
    se_sum = float(np.sqrt(var_f + var_fb + 2.0 * cov_f_fb))

    z_stat = coef_sum / se_sum if se_sum > 0 else np.nan
    p_sum = float(2.0 * (1.0 - norm.cdf(abs(z_stat)))) if not np.isnan(z_stat) else np.nan
    or_sum = float(np.exp(coef_sum))
    z = norm.ppf(0.975)
    lo_log_sum, hi_log_sum = coef_sum - z * se_sum, coef_sum + z * se_sum
    ci_or_sum = (float(np.exp(lo_log_sum)), float(np.exp(hi_log_sum)))

    female_effect_on_black = {
        'coef_logit': coef_sum,
        'se': se_sum,
        'p_value': p_sum,
        'odds_ratio': or_sum,
        'odds_ratio_CI_95': ci_or_sum
    }

    # Build returned object
    out_object = {
        'Female_nonBlack': female_summary,
        'Female_Black_interaction': interaction_summary,
        'Female_effect_for_Black': female_effect_on_black
    }

    # Short interpretation
    # Determine significance at alpha=0.05
    sig_f = female_summary['p_value'] < 0.05
    sig_inter = interaction_summary['p_value'] < 0.05
    sig_f_black = female_effect_on_black['p_value'] < 0.05

    desc_lines = []
    desc_lines.append(
        "Interpretation (logistic regression results):"
    )
    desc_lines.append(
        f"- Among non-Black applicants, being female has odds ratio = {female_summary['odds_ratio']:.3f} "
        f"(95% CI {female_summary['odds_ratio_CI_95'][0]:.3f} to {female_summary['odds_ratio_CI_95'][1]:.3f}), "
        f"p = {female_summary['p_value']:.3g}. "
        + ("Statistically significant (p<0.05)." if sig_f else "Not statistically significant (p>=0.05).")
    )
    desc_lines.append(
        f"- The Female x Black interaction term has odds ratio = {interaction_summary['odds_ratio']:.3f} "
        f"(95% CI {interaction_summary['odds_ratio_CI_95'][0]:.3f} to {interaction_summary['odds_ratio_CI_95'][1]:.3f}), "
        f"p = {interaction_summary['p_value']:.3g}. "
        + ("Statistically significant (p<0.05)." if sig_inter else "Not statistically significant (p>=0.05).")
    )
    desc_lines.append(
        f"- For Black applicants, the combined female effect (Female + Female x Black) has odds ratio = {female_effect_on_black['odds_ratio']:.3f} "
        f"(95% CI {female_effect_on_black['odds_ratio_CI_95'][0]:.3f} to {female_effect_on_black['odds_ratio_CI_95'][1]:.3f}), "
        f"p = {female_effect_on_black['p_value']:.3g}. "
        + ("Statistically significant (p<0.05)." if sig_f_black else "Not statistically significant (p>=0.05).")
    )
    desc_lines.append(
        "In plain terms: an odds ratio < 1 indicates lower odds of loan acceptance for females; > 1 indicates higher odds."
    )

    description = " ".join(desc_lines)

    return {
        'object': out_object,
        'description': description
    }