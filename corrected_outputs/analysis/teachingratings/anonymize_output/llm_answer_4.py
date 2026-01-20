def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of instructor beauty on evaluation scores
    from a fitted statsmodels OLS RegressionResultsWrapper (with clustered SEs).

    Returns a dictionary with:
      - "object": dict of numeric results (coefficients, SEs, p-values, CIs,
                  marginal effects for female and male instructors)
      - "description": brief interpretation of the results in plain language.
    """
    import numpy as np
    try:
        from scipy import stats as _scipy_stats
        _has_scipy = True
    except Exception:
        # Fall back to normal approx if scipy not available
        _has_scipy = False
        from math import erf as _erf
        def _norm_sf(x):
            # survival function of standard normal
            return 0.5 * (1.0 - (2.0/_erf(1)))  # dummy - will not be used realistically
    res = model_output

    # Basic checks
    required_names = ['Beauty_centered', 'Beauty_x_Gender']
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("model_output does not have .params attribute. Expecting a statsmodels results object.")
    for name in required_names:
        if name not in params.index:
            raise ValueError(f"Required parameter '{name}' not found in model_output.params. Available params: {list(params.index)}")

    # Extract point estimates
    beta_beauty = float(params['Beauty_centered'])
    beta_inter = float(params['Beauty_x_Gender'])

    # Extract covariance matrix, p-values, and confidence intervals
    cov = getattr(res, "cov_params", None)
    if cov is None:
        raise ValueError("model_output does not have .cov_params method/attribute.")
    covmat = res.cov_params()  # covariance matrix consistent with cluster cov_type used in fitting
    pvals = getattr(res, "pvalues", None)
    if pvals is None:
        raise ValueError("model_output does not have .pvalues attribute.")
    p_beauty = float(pvals['Beauty_centered'])
    p_inter = float(pvals['Beauty_x_Gender'])

    # Standard errors (from covariance matrix)
    se_beauty = float(np.sqrt(covmat.loc['Beauty_centered', 'Beauty_centered']))
    se_inter = float(np.sqrt(covmat.loc['Beauty_x_Gender', 'Beauty_x_Gender']))

    # Confidence intervals (use model's conf_int if available)
    try:
        conf_df = res.conf_int()
        ci_beauty = [float(conf_df.loc['Beauty_centered', 0]), float(conf_df.loc['Beauty_centered', 1])]
        ci_inter = [float(conf_df.loc['Beauty_x_Gender', 0]), float(conf_df.loc['Beauty_x_Gender', 1])]
    except Exception:
        # fallback to normal-approx 95% CI using se
        z = 1.96
        ci_beauty = [beta_beauty - z * se_beauty, beta_beauty + z * se_beauty]
        ci_inter = [beta_inter - z * se_inter, beta_inter + z * se_inter]

    # Marginal effects by gender:
    # Female (Gender_male = 0): effect = beta_beauty
    est_female = beta_beauty
    var_female = float(covmat.loc['Beauty_centered', 'Beauty_centered'])
    se_female = float(np.sqrt(var_female))
    # Male (Gender_male = 1): effect = beta_beauty + beta_inter
    est_male = beta_beauty + beta_inter
    var_male = float(
        covmat.loc['Beauty_centered', 'Beauty_centered']
        + covmat.loc['Beauty_x_Gender', 'Beauty_x_Gender']
        + 2.0 * covmat.loc['Beauty_centered', 'Beauty_x_Gender']
    )
    se_male = float(np.sqrt(max(var_male, 0.0)))

    # Compute p-values for linear combinations (use t-distribution with residual df if scipy available; otherwise normal approx)
    df_resid = getattr(res, "df_resid", None)
    def two_sided_p_from_t(stat, df):
        if _has_scipy and df is not None:
            return float(2.0 * _scipy_stats.t.sf(abs(stat), df))
        else:
            # normal approximation
            return float(2.0 * (1.0 - _scipy_stats.norm.cdf(abs(stat))))

    t_female = est_female / se_female if se_female > 0 else np.nan
    p_female = two_sided_p_from_t(t_female, df_resid)

    t_male = est_male / se_male if se_male > 0 else np.nan
    p_male = two_sided_p_from_t(t_male, df_resid)

    # 95% CIs for marginal effects (use normal approx; this is conventional)
    ci_female = [est_female - 1.96 * se_female, est_female + 1.96 * se_female]
    ci_male = [est_male - 1.96 * se_male, est_male + 1.96 * se_male]

    # Decide significance at alpha=0.05
    alpha = 0.05
    beauty_significant = p_beauty < alpha
    interaction_significant = p_inter < alpha
    female_effect_significant = p_female < alpha
    male_effect_significant = p_male < alpha

    # Package numeric results (convert numpy types to native Python floats)
    result_object = {
        'coef_Beauty_centered': float(beta_beauty),
        'se_Beauty_centered': float(se_beauty),
        'p_Beauty_centered': float(p_beauty),
        'ci95_Beauty_centered': [float(ci_beauty[0]), float(ci_beauty[1])],

        'coef_Beauty_x_Gender': float(beta_inter),
        'se_Beauty_x_Gender': float(se_inter),
        'p_Beauty_x_Gender': float(p_inter),
        'ci95_Beauty_x_Gender': [float(ci_inter[0]), float(ci_inter[1])],

        'marginal_effect_female': float(est_female),
        'se_marginal_effect_female': float(se_female),
        'p_marginal_effect_female': float(p_female),
        'ci95_marginal_effect_female': [float(ci_female[0]), float(ci_female[1])],
        'significant_marginal_effect_female': bool(female_effect_significant),

        'marginal_effect_male': float(est_male),
        'se_marginal_effect_male': float(se_male),
        'p_marginal_effect_male': float(p_male),
        'ci95_marginal_effect_male': [float(ci_male[0]), float(ci_male[1])],
        'significant_marginal_effect_male': bool(male_effect_significant),

        'interaction_significant': bool(interaction_significant),
        'beauty_main_effect_significant': bool(beauty_significant),
        # metadata
        'df_resid': float(df_resid) if df_resid is not None else None,
        'params_index': list(params.index),
    }

    # Brief interpretation text
    # We convey: effect per one unit increase in centered beauty on EvalScore (scale 1-5).
    if interaction_significant:
        desc = (
            "The interaction between beauty and gender is statistically significant (p = "
            f"{p_inter:.3g}), indicating the effect of beauty differs by instructor gender. "
            "Estimated marginal effects:\n"
            f"- Female instructors (Gender_male=0): a one-unit increase in centered beauty is associated with a change in evaluation score of {est_female:.3f} "
            f"(SE = {se_female:.3f}, 95% CI [{ci_female[0]:.3f}, {ci_female[1]:.3f}], p = {p_female:.3g}).\n"
            f"- Male instructors (Gender_male=1): a one-unit increase in centered beauty is associated with a change in evaluation score of {est_male:.3f} "
            f"(SE = {se_male:.3f}, 95% CI [{ci_male[0]:.3f}, {ci_male[1]:.3f}], p = {p_male:.3g})."
        )
    else:
        desc = (
            "The interaction between beauty and gender is NOT statistically significant (p = "
            f"{p_inter:.3g}). The main (average) effect of beauty is:\n"
            f"- Average effect: a one-unit increase in centered beauty is associated with a change in evaluation score of {beta_beauty:.3f} "
            f"(SE = {se_beauty:.3f}, 95% CI [{ci_beauty[0]:.3f}, {ci_beauty[1]:.3f}], p = {p_beauty:.3g}).\n"
            "Because the interaction is not significant, the gender-specific marginal effects are similar to this average effect."
        )

    return {
        "object": result_object,
        "description": desc
    }