def extract_final_answer(model_output):
    """
    Extracts relevant statistics from the fitted model objects returned by the modeling function.
    Expects model_output to be a dict with at least the key 'nb_deaths' pointing to a statsmodels GLMResultsWrapper
    (negative binomial or Poisson fallback). Also extracts secondary info for 'gender_female' if present.
    
    Returns a dictionary with:
      - "object": dict with numeric results (coef, se, p-value, 95% CI, exponentiated coef (IRR), IRR CI,
                          and percent change in expected fatalities per 1-unit (1 SD) increase)
                   for 'masfem_scaled' (primary) and for 'gender_female' (secondary).
      - "description": brief interpretation of the primary result in context.
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict as returned by the modeling function.")
    if 'nb_deaths' not in model_output:
        raise KeyError("model_output must contain the key 'nb_deaths' with the fitted GLM results.")

    nb_res = model_output['nb_deaths']

    # Helper to safely extract statistics for a variable name
    def _get_stats(res, varname):
        # Ensure variable exists in the result parameters
        params = getattr(res, 'params', None)
        if params is None or varname not in params.index:
            return None

        coef = float(params[varname])
        # standard error
        bse = float(res.bse[varname])
        # p-value (if available)
        pval = float(res.pvalues[varname]) if (hasattr(res, 'pvalues') and varname in res.pvalues.index) else None
        # 95% conf interval
        try:
            ci = res.conf_int().loc[varname].values.astype(float)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower, ci_upper = None, None

        # Exponentiated coefficient: incidence rate ratio (IRR) for count models
        try:
            irr = float(np.exp(coef))
            irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
            irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
        except Exception:
            irr, irr_ci_lower, irr_ci_upper = None, None, None

        # Percent change in expected fatalities for a one-unit increase in predictor:
        # (exp(coef) - 1) * 100
        pct_change = (irr - 1.0) * 100.0 if irr is not None else None

        return {
            'variable': varname,
            'coef': coef,
            'std_error': bse,
            'p_value': pval,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'irr': irr,
            'irr_95_ci_lower': irr_ci_lower,
            'irr_95_ci_upper': irr_ci_upper,
            'percent_change_expected_deaths_per_unit': pct_change
        }

    # Extract stats for primary variable (masfem_scaled) and secondary (gender_female)
    masfem_stats = _get_stats(nb_res, 'masfem_scaled')
    gender_stats = _get_stats(nb_res, 'gender_female')

    # Prepare object to return
    result_object = {
        'model_family': str(getattr(nb_res, 'family', 'Unknown')),
        'primary': masfem_stats,
        'secondary': gender_stats
    }

    # Construct a concise description interpreting the primary result
    if masfem_stats is None:
        description = "The fitted model does not contain a parameter named 'masfem_scaled'. Cannot interpret effect."
    else:
        coef = masfem_stats['coef']
        p = masfem_stats['p_value']
        pct = masfem_stats['percent_change_expected_deaths_per_unit']
        direction = "increase" if coef > 0 else "decrease" if coef < 0 else "no change"
        sig_text = ""
        if p is None:
            sig_text = " (p-value not available)"
        else:
            sig_text = " (statistically significant at p < 0.05)" if p < 0.05 else " (not statistically significant at p < 0.05)"
        description = (
            f"Primary result for 'masfem_scaled': coefficient = {coef:.4f}, which implies an IRR = {masfem_stats['irr']:.4f}. "
            f"This corresponds to a {pct:.2f}% expected {direction} in fatalities per 1-SD increase in name femininity. "
            f"p-value = {p:.4g}{sig_text}. "
            "Positive coefficient means more feminine names are associated with higher fatalities (consistent with the hypothesis that feminine names elicit fewer precautions); "
            "negative coefficient would imply the opposite."
        )

    return {
        "object": result_object,
        "description": description
    }