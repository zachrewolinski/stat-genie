def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of the 'Female' indicator from the model_output produced
    by the provided modeling function.

    Returns a dictionary with:
      - "object": a dict of numeric results (coefficient, SE, p-value, odds ratio, OR 95% CI, n, aic, significance)
      - "description": a short plain-language interpretation of the gender effect on mortgage approval
    """
    import numpy as np

    # Get the fitted statsmodels result object if present
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('model_result', None)
    else:
        # If the function was passed the raw statsmodels result itself
        res = model_output

    if res is None:
        raise ValueError("model_output does not contain a 'model_result' statsmodels object.")

    # Preferred name for the gender variable in the model
    var_name = 'Female'
    if var_name not in res.params.index:
        # try common alternatives
        for alt in ['female', 'FEMALE', 'sex_female', 'is_female']:
            if alt in res.params.index:
                var_name = alt
                break

    # Extract estimates
    coef = float(res.params[var_name])
    se = float(res.bse[var_name]) if hasattr(res, 'bse') else None
    p_value = float(res.pvalues[var_name]) if hasattr(res, 'pvalues') else None

    # Confidence interval for coefficient and convert to odds ratio scale
    conf = res.conf_int()
    if var_name in conf.index:
        ci_low_coef = float(conf.loc[var_name, 0])
        ci_high_coef = float(conf.loc[var_name, 1])
    else:
        # fallback: try to index by position (not recommended but safe)
        idx = list(res.params.index).index(var_name)
        ci_low_coef = float(conf.iloc[idx, 0])
        ci_high_coef = float(conf.iloc[idx, 1])

    or_point = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low_coef))
    or_ci_high = float(np.exp(ci_high_coef))

    # Additional available summary info
    n_obs = int(getattr(res, 'nobs', model_output.get('n_obs', np.nan)))
    aic = float(getattr(res, 'aic', model_output.get('aic', np.nan)))

    # Determine statistical significance at alpha = 0.05 (two-sided)
    significant = (p_value is not None) and (p_value < 0.05)

    # Build object to return
    result_object = {
        'variable': var_name,
        'coef_log_odds': coef,
        'std_error': se,
        'p_value': p_value,
        'odds_ratio': or_point,
        'or_ci_lower': or_ci_low,
        'or_ci_upper': or_ci_high,
        'n_obs': n_obs,
        'aic': aic,
        'significant_0.05': bool(significant)
    }

    # Plain-language description
    # Direction and magnitude: compare female=1 to male=0 controlling for covariates
    direction = "higher" if or_point > 1 else ("lower" if or_point < 1 else "no difference")
    descr = (
        f"Controlling for PI_ratio, loan_to_value, housing_expense_ratio, self_employed, married, "
        f"and bad_history, the estimated odds ratio for Female vs Male is {or_point:.3f} "
        f"(95% CI {or_ci_low:.3f}–{or_ci_high:.3f}), log-odds coef = {coef:.3f}, p = {p_value:.3g}. "
        f"This indicates that female applicants have {direction} odds of mortgage approval compared "
        f"with otherwise similar male applicants. The effect is "
        f"{'statistically significant at the 5% level' if significant else 'not statistically significant at the 5% level'}."
    )

    return {
        "object": result_object,
        "description": descr
    }