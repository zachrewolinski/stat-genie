def extract_final_answer(model_output):
    """
    Extract the effect of the IsHuman indicator from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with:
      - "object": dict of extracted statistics (coef, se, z, p, 95% CI, odds ratio, OR 95% CI, n_obs, significant)
      - "description": brief plain-language interpretation in the context of the task

    Expects model_output to be a statsmodels.results.GLMResultsWrapper (or similar) from the provided model().
    """
    import numpy as np

    # Ensure params exist
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError("model_output does not appear to be a fitted statsmodels results object.") from e

    # Find parameter name that corresponds to the IsHuman indicator.
    # This is defensive in case the parameter is named e.g. 'IsHuman' or 'IsHuman[T.True]' etc.
    param_name = None
    for name in params.index:
        if 'IsHuman' in name:
            param_name = name
            break
    if param_name is None:
        raise KeyError("Could not find a parameter containing 'IsHuman' in model_output.params. "
                       "Available parameter names: {}".format(list(params.index)))

    # Extract statistics
    coef = float(params[param_name])
    try:
        se = float(model_output.bse[param_name])
    except Exception:
        # fallback: compute from cov_params if available
        se = float(np.sqrt(model_output.cov_params().loc[param_name, param_name]))
    # z value (Wald statistic)
    z = coef / se if se != 0 else float('nan')
    # p-value
    p_value = float(model_output.pvalues[param_name]) if param_name in model_output.pvalues.index else float('nan')
    # 95% confidence interval on the log-odds (coefficient)
    try:
        ci = model_output.conf_int().loc[param_name].astype(float).tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = float('nan'), float('nan')

    # Odds ratio and its CI
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else float('nan')
    or_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else float('nan')

    # Sample size if available
    try:
        n_obs = int(model_output.nobs)
    except Exception:
        n_obs = None

    # Significance at alpha = 0.05
    significant = (p_value < 0.05)

    # Build return object
    result_object = {
        'param_name': param_name,
        'coef_log_odds': coef,
        'std_error': se,
        'z_value': z,
        'p_value': p_value,
        'ci_95_log_odds': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'odds_ratio_95_CI': [or_ci_lower, or_ci_upper],
        'n_obs': n_obs,
        'significant_at_0.05': bool(significant)
    }

    # Build human-readable description / conclusion
    # Interpret coefficient: positive coef => higher odds of AMTL in humans
    direction = "higher" if coef > 0 else "lower" if coef < 0 else "no difference"
    significance_text = ("statistically significant (p = {:.3g})".format(p_value)
                         if significant else "not statistically significant (p = {:.3g})".format(p_value))
    description = (
        "Parameter '{}': coefficient = {:+.3f} (SE = {:.3f}, z = {:.2f}, p = {:.3g}), 95% CI for log-odds = [{:.3f}, {:.3f}]. "
        "This corresponds to an odds ratio = {:.3f} (95% CI = [{:.3f}, {:.3f}]). "
        "Interpretation: after controlling for tooth class, centered age and age^2, and sex, modern humans (IsHuman=1) "
        "have {} odds of antemortem tooth loss compared to non-human primates. The effect is {}. "
        "Model n_obs = {}."
    ).format(
        param_name, coef, se, z, p_value, ci_lower, ci_upper,
        odds_ratio, or_ci_lower, or_ci_upper,
        direction, significance_text, n_obs if n_obs is not None else "unknown"
    )

    return {"object": result_object, "description": description}