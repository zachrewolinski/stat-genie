def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of having children on number of affairs
    from the provided model_output (expects 'nb_model' and/or 'poisson_model').

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "concise interpretation in plain language"
      }
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy.stats import norm
    except Exception:
        # Fallback: approximate normal p-values using exp approximation (should not be necessary in usual envs)
        def _norm_sf(x):
            # very crude fallback (not recommended); try to avoid using this path
            return 0.5 * (1.0 - (2.0 / np.pi) ** 0.5 * (x / (1.0 + 0.2316419 * abs(x)))) 
        class _FakeNorm:
            @staticmethod
            def sf(x):
                return _norm_sf(x)
        norm = _FakeNorm()

    # Prefer Negative Binomial model (handles overdispersion); if absent, fall back to Poisson
    model = None
    model_name = None
    if model_output.get('nb_model') is not None:
        model = model_output['nb_model']
        model_name = 'Negative Binomial'
    elif model_output.get('poisson_model') is not None:
        model = model_output['poisson_model']
        model_name = 'Poisson'
    else:
        raise KeyError("model_output must contain 'nb_model' or 'poisson_model'")

    params = model.params
    bse = model.bse
    pvalues = model.pvalues
    conf_int = model.conf_int()  # DataFrame with columns [0,1]
    cov = model.cov_params()

    # Identify term names robustly
    # Main children effect should be exactly 'Children'
    if 'Children' not in params.index:
        raise KeyError("The model does not contain a 'Children' coefficient in params.index")

    # Interaction term name might be 'Children:Gender_Male' (statsmodels uses ':' for interactions)
    interaction_name_candidates = [n for n in params.index if ('Children' in n) and ('Gender_Male' in n)]
    interaction_name = interaction_name_candidates[0] if interaction_name_candidates else None

    # Extract female (reference, Gender_Male=0) effect: coefficient of 'Children'
    coef_children = float(params['Children'])
    se_children = float(bse['Children'])
    p_children = float(pvalues['Children'])
    ci_children = (float(conf_int.loc['Children', 0]), float(conf_int.loc['Children', 1]))
    irr_children = float(np.exp(coef_children))
    irr_children_ci = (float(np.exp(ci_children[0])), float(np.exp(ci_children[1])))

    # Prepare male effect (Children effect when Gender_Male=1): Children + interaction (if present)
    if interaction_name is not None:
        coef_inter = float(params[interaction_name])
        se_inter = float(bse[interaction_name])
        p_inter = float(pvalues[interaction_name])
        ci_inter = (float(conf_int.loc[interaction_name, 0]), float(conf_int.loc[interaction_name, 1]))

        coef_children_male = coef_children + coef_inter
        # var(Children + interaction) = var(Children) + var(interaction) + 2*cov(Children, interaction)
        cov_ch_inter = float(cov.loc['Children', interaction_name])
        var_children_male = float(cov.loc['Children', 'Children']) + float(cov.loc[interaction_name, interaction_name]) + 2.0 * cov_ch_inter
        se_children_male = sqrt(max(var_children_male, 0.0))
        z_male = coef_children_male / (se_children_male + 1e-20)
        p_children_male = float(2.0 * norm.sf(abs(z_male)))
        ci_children_male = (coef_children_male - 1.96 * se_children_male, coef_children_male + 1.96 * se_children_male)
        irr_children_male = float(np.exp(coef_children_male))
        irr_children_male_ci = (float(np.exp(ci_children_male[0])), float(np.exp(ci_children_male[1])))
    else:
        # No interaction: effect is same for males and females
        coef_inter = None
        se_inter = None
        p_inter = None
        ci_inter = (None, None)

        coef_children_male = coef_children
        se_children_male = se_children
        p_children_male = p_children
        ci_children_male = ci_children
        irr_children_male = irr_children
        irr_children_male_ci = irr_children_ci

    # Interaction term p-value (if present) indicates whether the effect differs by gender
    interaction_pvalue = float(p_inter) if p_inter is not None else None

    # Build a short conclusion: interpret sign and significance (alpha=0.05)
    def interpret_effect(coef, p):
        if p < 0.05:
            if coef < 0:
                return "statistically significant decrease"
            elif coef > 0:
                return "statistically significant increase"
            else:
                return "no change (coef ~ 0, but p < 0.05 — unusual)"
        else:
            if coef < 0:
                return "non-significant decrease (evidence weak)"
            elif coef > 0:
                return "non-significant increase (evidence weak)"
            else:
                return "no evidence of effect"
    interp_female = interpret_effect(coef_children, p_children)
    interp_male = interpret_effect(coef_children_male, p_children_male)

    # Overall concise conclusion string
    if (p_children < 0.05) and (p_children_male < 0.05):
        overall = "Having children is associated with a reduction in reported extramarital sexual encounters for both women (reference) and men (when male effect is computed), according to the {} model.".format(model_name)
    elif (p_children < 0.05) and (p_children_male >= 0.05):
        overall = "Having children is associated with a statistically significant reduction for women (reference group), but the effect for men is not statistically significant ({} model).".format(model_name)
    elif (p_children >= 0.05) and (p_children_male < 0.05):
        overall = "No significant effect for women (reference), but having children is associated with a statistically significant change for men ({} model).".format(model_name)
    else:
        overall = "No strong evidence that having children decreases extramarital affairs in this sample ({} model); estimated effects are small or not statistically significant.".format(model_name)

    # Compose results object
    results_obj = {
        'model_used': model_name,
        'n_obs': int(model_output.get('n_obs', getattr(model, 'nobs', np.nan))),
        'poisson_dispersion_pearson_chi2_div_df': model_output.get('poisson_dispersion_pearson_chi2_div_df', None),
        'children_female': {
            'coef_log_count': coef_children,
            'se': se_children,
            'p_value': p_children,
            '95ci_log_count': ci_children,
            'irr (exp(coef))': irr_children,
            '95ci_irr': irr_children_ci,
            'interpretation': interp_female
        },
        'children_male': {
            'coef_log_count': coef_children_male,
            'se': se_children_male,
            'p_value': p_children_male,
            '95ci_log_count': ci_children_male,
            'irr (exp(coef))': irr_children_male,
            '95ci_irr': irr_children_male_ci,
            'interpretation': interp_male
        },
        'interaction_term': {
            'name': interaction_name,
            'coef_log_count': coef_inter,
            'se': se_inter,
            'p_value': interaction_pvalue,
            '95ci_log_count': ci_inter
        },
        'conclusion': overall
    }

    description = (
        "This output reports the estimated effect of having children on the expected count of extramarital "
        "sexual encounters from the fitted count model (negative binomial preferred if present). "
        "Coefficients are on the log-count scale; exponentiated coefficients (IRRs) give multiplicative "
        "changes in expected counts. Results are shown separately for the female reference group "
        "(Gender_Male=0) and for men (Gender_Male=1) using the interaction term. The 'conclusion' field "
        "gives a concise interpretation about whether having children appears to decrease extramarital affairs."
    )

    return {"object": results_obj, "description": description}