def extract_final_answer(model_output):
    """
    Extract statistics relevant to the effect of IsHomo on AMTL from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with keys:
      - "object": dict of numeric results (coefficient, se, z, p, 95% CI, odds ratio and its 95% CI,
                                  average marginal effect if available)
      - "description": human-readable interpretation of those results in the context of the task.
    """
    import numpy as np

    res = model_output

    # Name of coefficient of interest
    coef_name = 'IsHomo'

    # Basic existence checks
    try:
        params = res.params
        pvalues = res.pvalues
        bse = res.bse
        conf = res.conf_int()
    except Exception as e:
        raise ValueError(f"Provided model_output does not look like a statsmodels results object: {e}")

    if coef_name not in params.index:
        raise KeyError(f"Coefficient '{coef_name}' not found in model. Available coefficients: {list(params.index)}")

    # Extract primary statistics and cast to native Python types
    coef = float(params[coef_name])
    se = float(bse[coef_name])
    # compute z (Wald) statistic
    z_stat = float(coef / se) if se != 0 else None
    pval = float(pvalues[coef_name])
    ci_lower = float(conf.loc[coef_name, 0])
    ci_upper = float(conf.loc[coef_name, 1])

    # Odds ratio and CI
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Try to obtain average marginal effect (AME) for IsHomo (change in predicted probability)
    ame_result = None
    try:
        # get_margeff may raise if not available; use overall average ('at'='overall') for dy/dx
        me = res.get_margeff(at='overall', method='dydx')
        me_df = me.summary_frame()
        # The summary frame uses the same index names as params; ensure presence
        if coef_name in me_df.index:
            ame = float(me_df.loc[coef_name, 'dy/dx'])
            ame_se = float(me_df.loc[coef_name, 'Std. Err.'])
            ame_p = float(me_df.loc[coef_name, 'P>|z|'])
            # approximate 95% CI for AME
            ame_ci_lower = ame - 1.96 * ame_se
            ame_ci_upper = ame + 1.96 * ame_se
            ame_result = {
                'ame': ame,
                'ame_se': ame_se,
                'ame_pvalue': ame_p,
                'ame_ci_95': [ame_ci_lower, ame_ci_upper]
            }
    except Exception:
        # If margin effects cannot be computed, leave as None (this is non-fatal)
        ame_result = None

    # Build object to return
    output_object = {
        'coefficient_name': coef_name,
        'coef_log_odds': coef,
        'std_error': se,
        'z_stat': z_stat,
        'p_value': pval,
        'coef_95ci_log_odds': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'odds_ratio_95ci': [or_ci_lower, or_ci_upper],
        'average_marginal_effect': ame_result  # may be None if not available
    }

    # Interpret results in context
    if pval < 0.05:
        direction = 'higher' if coef > 0 else 'lower'
        significance_statement = (
            f"The coefficient for {coef_name} is statistically significant (p = {pval:.3g}). "
            f"Modern humans (IsHomo = 1) have {direction} odds of AMTL compared to non-human primates "
            f"(log-odds = {coef:.3f}, odds ratio = {odds_ratio:.3f}, 95% CI for OR = [{or_ci_lower:.3f}, {or_ci_upper:.3f}])."
        )
    else:
        direction = 'higher' if coef > 0 else 'lower'
        significance_statement = (
            f"The coefficient for {coef_name} is NOT statistically significant (p = {pval:.3g}). "
            f"There is no strong evidence that modern humans differ from non-human primates in AMTL after "
            f"accounting for age, sex (prob_male), and tooth class. The point estimate indicates {direction} odds "
            f"(log-odds = {coef:.3f}, odds ratio = {odds_ratio:.3f}, 95% CI for OR = [{or_ci_lower:.3f}, {or_ci_upper:.3f}])."
        )

    ame_text = ""
    if ame_result is not None:
        ame_text = (
            f" The average marginal effect of IsHomo on the probability of AMTL is approximately {ame_result['ame']:.4f} "
            f"(SE = {ame_result['ame_se']:.4f}, p = {ame_result['ame_pvalue']:.3g}, 95% CI ≈ [{ame_result['ame_ci_95'][0]:.4f}, {ame_result['ame_ci_95'][1]:.4f}]). "
            "This is the estimated change in probability of a tooth being missing associated with being a modern human, "
            "averaged over the sample covariate distribution."
        )

    description = (
        "Extracted statistics for the IsHomo predictor from the fitted binomial GLM.\n"
        + significance_statement
        + ame_text
        + " Interpretation: a positive log-odds (and OR > 1) indicates higher AMTL in modern humans, "
        "a negative log-odds (OR < 1) indicates lower AMTL. Statistical significance is judged at alpha = 0.05."
    )

    return {'object': output_object, 'description': description}