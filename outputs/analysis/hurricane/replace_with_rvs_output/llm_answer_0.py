def extract_final_answer(model_output):
    """
    Extracts the effect of 'masfem_c' from the provided model_output dictionary
    (expects keys 'nb_model' and 'ols_model' containing fitted statsmodels results).
    Returns a dictionary with keys:
      - "object": a dict containing coefficients, standard errors, p-values, 95% CIs,
                  exponentiated effect (for NB), percent-change interpretation, and
                  a boolean 'supports_hypothesis' flag for the primary NB model.
      - "description": human-readable explanation of what the numbers mean.
    """
    import numpy as np
    import pandas as pd

    # Helper to safe-get attribute and raise informative error
    def _get_result(model_output, key):
        if key not in model_output:
            raise KeyError(f"Expected key '{key}' in model_output but not found.")
        return model_output[key]

    nb_res = _get_result(model_output, 'nb_model')
    ols_res = _get_result(model_output, 'ols_model')

    varname = 'masfem_c'
    # Ensure variable exists in model results
    if varname not in nb_res.params.index:
        raise KeyError(f"Variable '{varname}' not found in nb_model params: {list(nb_res.params.index)}")
    if varname not in ols_res.params.index:
        raise KeyError(f"Variable '{varname}' not found in ols_model params: {list(ols_res.params.index)}")

    # Negative-binomial (primary) extraction
    coef_nb = float(nb_res.params[varname])
    se_nb = float(nb_res.bse[varname]) if hasattr(nb_res, 'bse') else None
    p_nb = float(nb_res.pvalues[varname]) if hasattr(nb_res, 'pvalues') else None
    # Confidence interval: statsmodels returns array-like; align indices if needed
    try:
        ci_nb = nb_res.conf_int().loc[varname].astype(float)
        ci_lower_nb = float(ci_nb[0])
        ci_upper_nb = float(ci_nb[1])
    except Exception:
        # fallback if conf_int returns ndarray without index
        ci_array = np.asarray(nb_res.conf_int())
        param_idx = list(nb_res.params.index).index(varname)
        ci_lower_nb = float(ci_array[param_idx, 0])
        ci_upper_nb = float(ci_array[param_idx, 1])

    # Interpret NB coefficient: multiplicative effect on expected count
    rate_ratio = float(np.exp(coef_nb))
    rr_ci_lower = float(np.exp(ci_lower_nb))
    rr_ci_upper = float(np.exp(ci_upper_nb))
    pct_change = (rate_ratio - 1.0) * 100.0
    pct_ci_lower = (rr_ci_lower - 1.0) * 100.0
    pct_ci_upper = (rr_ci_upper - 1.0) * 100.0

    # Determine whether the NB result supports the hypothesis:
    # Hypothesis expects more feminine names -> more deaths (positive coef).
    supports_hypothesis = (coef_nb > 0) and (p_nb is not None and p_nb < 0.05)

    # OLS robustness extraction (log(1+deaths))
    coef_ols = float(ols_res.params[varname])
    se_ols = float(ols_res.bse[varname]) if hasattr(ols_res, 'bse') else None
    p_ols = float(ols_res.pvalues[varname]) if hasattr(ols_res, 'pvalues') else None
    try:
        ci_ols = ols_res.conf_int().loc[varname].astype(float)
        ci_lower_ols = float(ci_ols[0])
        ci_upper_ols = float(ci_ols[1])
    except Exception:
        ci_array_ols = np.asarray(ols_res.conf_int())
        param_idx_ols = list(ols_res.params.index).index(varname)
        ci_lower_ols = float(ci_array_ols[param_idx_ols, 0])
        ci_upper_ols = float(ci_array_ols[param_idx_ols, 1])

    # For log outcome, approximate percent change ~ coef_ols * 100 (small changes)
    approx_pct_change_ols = coef_ols * 100.0
    approx_pct_ci_lower_ols = ci_lower_ols * 100.0
    approx_pct_ci_upper_ols = ci_upper_ols * 100.0

    result_object = {
        'nb_model': {
            'variable': varname,
            'coef': coef_nb,
            'se': se_nb,
            'p_value': p_nb,
            'ci_95': [ci_lower_nb, ci_upper_nb],
            'rate_ratio': rate_ratio,
            'rate_ratio_95_ci': [rr_ci_lower, rr_ci_upper],
            'percent_change_in_deaths': pct_change,
            'percent_change_95_ci': [pct_ci_lower, pct_ci_upper],
            'supports_hypothesis_at_p<.05': bool(supports_hypothesis)
        },
        'ols_model_log_outcome': {
            'variable': varname,
            'coef': coef_ols,
            'se': se_ols,
            'p_value': p_ols,
            'ci_95': [ci_lower_ols, ci_upper_ols],
            'approx_percent_change_in_1_plus_deaths': approx_pct_change_ols,
            'approx_percent_change_95_ci': [approx_pct_ci_lower_ols, approx_pct_ci_upper_ols]
        }
    }

    # Human-readable description
    description_lines = []
    description_lines.append(
        "Primary result (Negative-Binomial GLM on raw death counts): "
        f"Coefficient for '{varname}' = {coef_nb:.4f} (SE={se_nb:.4f}, p={p_nb:.4f}). "
        f"This corresponds to a rate ratio = {rate_ratio:.3f} (95% CI [{rr_ci_lower:.3f}, {rr_ci_upper:.3f}]), "
        f"i.e. an expected change of {pct_change:.1f}% in deaths per one-unit increase in '{varname}' "
        f"(95% CI: [{pct_ci_lower:.1f}%, {pct_ci_upper:.1f}%])."
    )
    if supports_hypothesis:
        description_lines.append("The effect is positive and statistically significant at p<0.05, which is consistent with the hypothesis "
                                 "that more-feminine hurricane names are associated with higher deaths (interpreted as fewer precautions).")
    else:
        description_lines.append("The effect is NOT statistically significant at p<0.05 and therefore does not provide reliable support for the hypothesis.")
    description_lines.append(
        "Robustness (OLS on log(1 + deaths)): "
        f"Coefficient = {coef_ols:.4f} (SE={se_ols:.4f}, p={p_ols:.4f}), approximately equal to {approx_pct_change_ols:.2f}% change in (1+deaths) per unit."
    )
    description_lines.append(
        "Note: 'masfem_c' is mean-centered; interpretation is per one-unit increase in that centered femininity rating. "
        "NB model is primary because the outcome is an overdispersed count; OLS on log-transformed deaths is presented as a robustness check."
    )

    description = " ".join(description_lines)

    return {
        "object": result_object,
        "description": description
    }