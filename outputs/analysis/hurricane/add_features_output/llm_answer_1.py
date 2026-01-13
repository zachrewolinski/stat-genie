def extract_final_answer(model_output):
    """
    Extracts statistics related to the masfem_c predictor from the provided model_output dict.
    Returns a dict with keys:
      - "object": a dict with numeric summaries for the Negative Binomial (primary) and OLS (robustness) models,
                  plus a boolean 'supports_hypothesis' based on the primary model.
      - "description": a short plain-language interpretation of the results in the context of the hypothesis.
    """
    import numpy as np

    # Basic checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")
    if 'nb_model' not in model_output or 'ols_model' not in model_output:
        raise ValueError("model_output must contain 'nb_model' and 'ols_model' keys.")

    nb = model_output['nb_model']
    ols = model_output['ols_model']

    # Ensure masfem_c is present in model parameters
    param_name = 'masfem_c'
    if param_name not in nb.params.index:
        raise KeyError(f"Parameter '{param_name}' not found in nb_model.params")
    if param_name not in ols.params.index:
        raise KeyError(f"Parameter '{param_name}' not found in ols_model.params")

    # Extract stats from Negative Binomial (primary)
    nb_coef = float(nb.params[param_name])
    nb_se = float(nb.bse[param_name])
    nb_p = float(nb.pvalues[param_name])
    nb_ci = nb.conf_int().loc[param_name].astype(float).tolist()  # [lower, upper]
    # Incidence Rate Ratio (IRR) and CI
    irr = float(np.exp(nb_coef))
    irr_ci = [float(np.exp(nb_ci[0])), float(np.exp(nb_ci[1]))]

    # Extract stats from OLS on log(1+deaths) (robustness)
    ols_coef = float(ols.params[param_name])
    ols_se = float(ols.bse[param_name])
    ols_p = float(ols.pvalues[param_name])
    ols_ci = ols.conf_int().loc[param_name].astype(float).tolist()
    # Approximate percent change interpretation for log outcome: 100*(exp(beta)-1)
    ols_pct = float((np.exp(ols_coef) - 1.0) * 100.0)
    ols_pct_ci = [float((np.exp(ols_ci[0]) - 1.0) * 100.0), float((np.exp(ols_ci[1]) - 1.0) * 100.0)]

    # Overdispersion and sample size if available
    overdispersion = model_output.get('overdispersion_ratio_var_over_mean', None)
    n_obs = model_output.get('n_observations', None)

    # Decide whether the primary model supports the hypothesis:
    # Hypothesis: higher masfem -> more deaths (positive association).
    supports_hypothesis = (nb_coef > 0) and (nb_p < 0.05)

    result_object = {
        'n_observations': n_obs,
        'overdispersion_var_over_mean': overdispersion,
        'negative_binomial': {
            'coef': nb_coef,
            'std_err': nb_se,
            'p_value': nb_p,
            'conf_int_95': nb_ci,
            'incidence_rate_ratio (IRR)': irr,
            'IRR_conf_int_95': irr_ci
        },
        'ols_log1p': {
            'coef_on_log1p': ols_coef,
            'std_err': ols_se,
            'p_value': ols_p,
            'conf_int_95': ols_ci,
            'approx_pct_change_per_sd_in_masfem': ols_pct,
            'pct_change_conf_int_95': ols_pct_ci
        },
        'supports_hypothesis_primary_model': bool(supports_hypothesis)
    }

    # Short description / interpretation
    if supports_hypothesis:
        conclusion_text = (
            "Primary (Negative Binomial) model: masfem_c has a positive coefficient "
            f"({nb_coef:.4g}), IRR={irr:.4g}, p={nb_p:.3g}, 95% CI for IRR [{irr_ci[0]:.4g}, {irr_ci[1]:.4g}]. "
            "This is consistent with the hypothesis that more-feminine names are associated with more deaths "
            "(interpreted as fewer precautions), and the effect is statistically significant at p<0.05. "
        )
    else:
        conclusion_text = (
            "Primary (Negative Binomial) model: masfem_c coefficient is "
            f"{'positive' if nb_coef>0 else 'negative'} ({nb_coef:.4g}), p={nb_p:.3g}. "
            "This does not provide statistically significant support (p<0.05) for the hypothesis "
            "that more-feminine names lead to more deaths." if nb_p >= 0.05 else ""
        )

    # Add notes about robustness and dispersion
    conclusion_text += (
        f" Robustness (OLS on log(1+deaths)) shows coef={ols_coef:.4g}, p={ols_p:.3g}, "
        f"approx. % change per unit masfem_c: {ols_pct:.2f}% (95% CI [{ols_pct_ci[0]:.2f}%, {ols_pct_ci[1]:.2f}%])."
    )
    if overdispersion is not None:
        conclusion_text += f" Overdispersion (var/mean) = {overdispersion:.3g} (>>1), which supports using a NB model."

    return {
        "object": result_object,
        "description": conclusion_text
    }