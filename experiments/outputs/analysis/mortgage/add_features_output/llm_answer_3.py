def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from the provided model_output.
    Returns a dictionary with keys:
      - "object": a dict of numeric results (coef, se, p-value, odds ratio, 95% CI, n, significance flag)
      - "description": a short plain-language interpretation of the effect of gender on approval

    Expects model_output to be a dict like the one produced by the provided modeling function,
    containing at least 'result' (a statsmodels BinaryResultsWrapper). If available, it will use
    'odds_ratios' and 'conf_int_odds' from model_output; otherwise those will be computed.
    """
    import numpy as np
    import pandas as pd

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing the fitted model/result.")

    result = model_output.get('result')
    if result is None:
        raise ValueError("model_output does not contain a 'result' object.")

    # Safely extract coefficient, se, p-value for 'female'
    try:
        coef = float(result.params['female'])
        se = float(result.bse['female'])
        pval = float(result.pvalues['female'])
    except Exception as e:
        # Try using .loc in case of different index types
        coef = float(result.params.loc['female'])
        se = float(result.bse.loc['female'])
        pval = float(result.pvalues.loc['female'])

    # Odds ratio (use precomputed if present)
    if 'odds_ratios' in model_output and model_output['odds_ratios'] is not None:
        try:
            odds_ratio = float(model_output['odds_ratios'].loc['female'])
        except Exception:
            odds_ratio = float(model_output['odds_ratios']['female'])
    else:
        odds_ratio = float(np.exp(coef))

    # 95% CI for odds ratio
    if 'conf_int_odds' in model_output and model_output['conf_int_odds'] is not None:
        conf_odds = model_output['conf_int_odds']
        try:
            ci_lower = float(conf_odds.loc['female', '2.5%'])
            ci_upper = float(conf_odds.loc['female', '97.5%'])
        except Exception:
            # fallback for different indexing
            row = conf_odds.loc['female']
            ci_lower = float(row.iloc[0])
            ci_upper = float(row.iloc[1])
    else:
        conf = result.conf_int().loc['female']
        ci_lower = float(np.exp(conf.iloc[0]))
        ci_upper = float(np.exp(conf.iloc[1]))

    # Sample size if available
    n_obs = int(result.nobs) if hasattr(result, 'nobs') else None

    significant_at_0_05 = (pval < 0.05)

    obj = {
        'coefficient_logit': coef,
        'std_err': se,
        'p_value': pval,
        'odds_ratio': odds_ratio,
        'odds_ratio_95ci': [ci_lower, ci_upper],
        'n_obs': n_obs,
        'significant_at_0.05': bool(significant_at_0_05)
    }

    # Plain-language description
    percent_change = (odds_ratio - 1.0) * 100.0
    direction = "higher" if odds_ratio > 1 else ("lower" if odds_ratio < 1 else "no difference")
    signif_text = "statistically significant (p < 0.05)" if significant_at_0_05 else "not statistically significant (p ≥ 0.05)"
    description = (
        f"Controlling for the listed covariates, the female indicator has log-odds coefficient {coef:.3f} "
        f"(SE={se:.3f}, p={pval:.3f}); estimated odds ratio = {odds_ratio:.3f} "
        f"with 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]. "
        f"This implies female applicants have {abs(percent_change):.1f}% {direction} odds of mortgage approval "
        f"compared to male applicants, and the effect is {signif_text}."
    )

    return {"object": obj, "description": description}