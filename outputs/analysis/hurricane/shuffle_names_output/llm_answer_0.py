def extract_final_answer(model_output):
    """
    Extract summary statistics for the effect of 'fem_z' (perceived femininity)
    from the provided model_output dict.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results for the NB and OLS models
      - "description": a short plain-language interpretation of those results

    Expects model_output to contain:
      - 'nb_model': statsmodels GLMResultsWrapper (NegativeBinomial or Poisson fallback)
      - 'ols_log_model': statsmodels RegressionResultsWrapper (OLS on log(1+deaths))
      - optionally 'model_dataframe' for sample size/context
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    if 'nb_model' not in model_output or 'ols_log_model' not in model_output:
        raise ValueError("model_output must contain 'nb_model' and 'ols_log_model' keys.")

    nb = model_output['nb_model']
    ols = model_output['ols_log_model']

    # Ensure the parameter of interest exists
    for res, name in [(nb, 'nb_model'), (ols, 'ols_log_model')]:
        if 'fem_z' not in res.params.index:
            raise KeyError(f"'fem_z' not found in params of {name}.")

    # Negative Binomial (GLM)
    nb_coef = float(nb.params['fem_z'])
    nb_se = float(nb.bse['fem_z'])
    nb_p = float(nb.pvalues['fem_z'])
    nb_ci = nb.conf_int().loc['fem_z'].astype(float)
    nb_ci_lower = float(nb_ci[0])
    nb_ci_upper = float(nb_ci[1])

    # Exponentiate to get incidence rate ratio (IRR) and its CI
    nb_irr = float(np.exp(nb_coef))
    nb_irr_ci_lower = float(np.exp(nb_ci_lower))
    nb_irr_ci_upper = float(np.exp(nb_ci_upper))
    nb_percent_change = (nb_irr - 1.0) * 100.0  # percent change in expected deaths per 1-unit increase in fem_z

    # OLS on log(1 + deaths)
    ols_coef = float(ols.params['fem_z'])
    ols_se = float(ols.bse['fem_z'])
    # p-values for OLS when using cov_type HC3 are in .pvalues
    ols_p = float(ols.pvalues['fem_z'])
    ols_ci = ols.conf_int().loc['fem_z'].astype(float)
    ols_ci_lower = float(ols_ci[0])
    ols_ci_upper = float(ols_ci[1])

    # Approximate percent change in (1 + deaths) using exp(coef) - 1
    ols_approx_pct_change = (np.exp(ols_coef) - 1.0) * 100.0
    ols_approx_pct_ci_lower = (np.exp(ols_ci_lower) - 1.0) * 100.0
    ols_approx_pct_ci_upper = (np.exp(ols_ci_upper) - 1.0) * 100.0

    # Sample size if available
    n_obs = None
    if hasattr(nb, 'nobs'):
        try:
            n_obs = int(nb.nobs)
        except Exception:
            n_obs = None
    elif 'model_dataframe' in model_output:
        try:
            n_obs = int(model_output['model_dataframe'].shape[0])
        except Exception:
            n_obs = None

    result_object = {
        "n_obs": n_obs,
        "nb_model": {
            "coef_fem_z": round(nb_coef, 4),
            "se": round(nb_se, 4),
            "p_value": round(nb_p, 4),
            "ci_95": [round(nb_ci_lower, 4), round(nb_ci_upper, 4)],
            "irr": round(nb_irr, 4),
            "irr_95_ci": [round(nb_irr_ci_lower, 4), round(nb_irr_ci_upper, 4)],
            "percent_change_in_expected_deaths": round(nb_percent_change, 2),
            "significant_at_0.05": bool(nb_p < 0.05)
        },
        "ols_log_model": {
            "coef_fem_z": round(ols_coef, 4),
            "se": round(ols_se, 4),
            "p_value": round(ols_p, 4),
            "ci_95": [round(ols_ci_lower, 4), round(ols_ci_upper, 4)],
            "approx_percent_change_in_1_plus_deaths": round(float(ols_approx_pct_change), 2),
            "approx_percent_change_95_ci": [round(float(ols_approx_pct_ci_lower), 2), round(float(ols_approx_pct_ci_upper), 2)],
            "significant_at_0.05": bool(ols_p < 0.05)
        }
    }

    # Human-readable description
    desc_lines = []
    desc_lines.append(f"Sample size (observations): {n_obs if n_obs is not None else 'unknown'}")
    desc_lines.append("Negative Binomial GLM on death counts:")
    desc_lines.append(
        f" - fem_z coef = {result_object['nb_model']['coef_fem_z']}, SE = {result_object['nb_model']['se']}, "
        f"p = {result_object['nb_model']['p_value']}."
    )
    desc_lines.append(
        f" - Incidence rate ratio (IRR) = {result_object['nb_model']['irr']}, 95% CI = {result_object['nb_model']['irr_95_ci']}."
    )
    desc_lines.append(
        f" - Interpretation: a one-unit (one SD, since fem_z is standardized) increase in perceived femininity is associated with a "
        f"{result_object['nb_model']['percent_change_in_expected_deaths']}% change in expected deaths. "
        f"{'This effect is statistically significant (p < 0.05).' if result_object['nb_model']['significant_at_0.05'] else 'This effect is not statistically significant (p >= 0.05).'}"
    )

    desc_lines.append("Robustness: OLS on log(1 + deaths):")
    desc_lines.append(
        f" - fem_z coef (log scale) = {result_object['ols_log_model']['coef_fem_z']}, SE = {result_object['ols_log_model']['se']}, "
        f"p = {result_object['ols_log_model']['p_value']}."
    )
    desc_lines.append(
        f" - Approx. percent change in (1 + deaths) per 1 SD increase in fem_z = {result_object['ols_log_model']['approx_percent_change_in_1_plus_deaths']}%, "
        f"95% CI = {result_object['ols_log_model']['approx_percent_change_95_ci']}."
    )
    desc_lines.append(
        f" - {'Statistically significant (p < 0.05).' if result_object['ols_log_model']['significant_at_0.05'] else 'Not statistically significant (p >= 0.05).'}"
    )

    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}