def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator on mortgage acceptance from the provided
    model_output (the dict returned by the modeling function). Returns a dictionary with:
      - "object": dict of extracted numeric results (coef, SE, z, p, OR, 95% CI)
      - "description": plain-language interpretation of the effect
    
    Compatible with both the robust-fallback object used in the model code and with
    regular statsmodels result objects.
    """
    import numpy as np
    from scipy import stats

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Prefer robust results if available
    robust = model_output.get('robust_results', None)
    model_obj = model_output.get('model_object', None)

    if robust is None and model_obj is None:
        raise ValueError("model_output must contain at least 'robust_results' or 'model_object'.")

    # Get parameter estimates
    params = None
    try:
        params = robust.params
    except Exception:
        try:
            params = model_obj.params
        except Exception:
            raise RuntimeError("Could not retrieve parameter estimates from model_output.")

    # Ensure 'female' is present
    if 'female' not in params.index:
        raise KeyError("'female' not found in model parameters.")

    coef = float(params.loc['female'])

    # Obtain covariance / standard error
    se_female = None
    cov_matrix = None
    # Try robust.cov (fallback object stores .cov), or cov_params() if available, or bse
    try:
        if hasattr(robust, 'cov') and getattr(robust, 'cov') is not None:
            cov_matrix = np.asarray(robust.cov)
        elif hasattr(robust, 'cov_params'):
            cov_matrix = np.asarray(robust.cov_params())
    except Exception:
        cov_matrix = None

    # If still None, try model's cov_params() or bse
    if cov_matrix is None:
        try:
            if hasattr(model_obj, 'cov_params'):
                cov_matrix = np.asarray(model_obj.cov_params())
        except Exception:
            cov_matrix = None

    if cov_matrix is not None:
        # Find index of 'female' in params to get SE
        try:
            idx = list(params.index).index('female')
            se_female = float(np.sqrt(cov_matrix[idx, idx]))
        except Exception:
            se_female = None

    # Fall back to bse if cov not available
    if se_female is None:
        try:
            if hasattr(robust, 'bse'):
                se_female = float(robust.bse.loc['female'])
            elif hasattr(model_obj, 'bse'):
                se_female = float(model_obj.bse.loc['female'])
        except Exception:
            se_female = None

    # If still missing, cannot compute z/p; raise
    if se_female is None or se_female == 0:
        raise RuntimeError("Could not compute standard error for 'female' coefficient.")

    # z-statistic and two-sided p-value (normal approximation)
    z = coef / se_female
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))

    # Confidence interval for coefficient: try robust.conf_int(); else use coef +/- 1.96*se
    ci_lower_coef = None
    ci_upper_coef = None
    try:
        conf = robust.conf_int()
        # conf may be a DataFrame with index including 'female'
        if 'female' in conf.index:
            # conf columns might be labeled differently; take first two columns
            ci_vals = conf.loc['female'].values
            ci_lower_coef = float(ci_vals[0])
            ci_upper_coef = float(ci_vals[1])
    except Exception:
        conf = None

    if ci_lower_coef is None or ci_upper_coef is None:
        zcrit = 1.96
        ci_lower_coef = coef - zcrit * se_female
        ci_upper_coef = coef + zcrit * se_female

    # Convert to odds ratio and CI on OR scale
    OR = float(np.exp(coef))
    OR_ci_lower = float(np.exp(ci_lower_coef))
    OR_ci_upper = float(np.exp(ci_upper_coef))

    # Build numeric result object
    result_obj = {
        'coef_female': coef,
        'se_female': se_female,
        'z_female': z,
        'p_value_female': p_value,
        'OR_female': OR,
        'OR_CI_lower': OR_ci_lower,
        'OR_CI_upper': OR_ci_upper
    }

    # Interpret result in plain language
    # Determine significance at alpha=0.05 and note borderline at 0.10
    sig_level = 0.05
    borderline_level = 0.10
    if p_value < sig_level:
        significance = f"statistically significant (p = {p_value:.3f} < {sig_level})"
    elif p_value < borderline_level:
        significance = f"marginally significant (p = {p_value:.3f} < {borderline_level})"
    else:
        significance = f"not statistically significant (p = {p_value:.3f} ≥ {sig_level})"

    description = (
        f"Effect of being female on mortgage acceptance:\n"
        f"- Logistic coefficient = {coef:.4f}, SE = {se_female:.4f}, z = {z:.3f}, p = {p_value:.3f}.\n"
        f"- Odds ratio = {OR:.3f} with 95% CI [{OR_ci_lower:.3f}, {OR_ci_upper:.3f}].\n"
        f"- Interpretation: Being female is associated with a multiplicative change in the odds "
        f"of loan acceptance of {OR:.3f} (95% CI [{OR_ci_lower:.3f}, {OR_ci_upper:.3f}]). "
        f"This effect is {significance}.\n"
        f"- Note: the model controls for race, employment type, marital status, credit history, debt ratios, "
        f"loan-to-value, and credit scores as specified in the model."
    )

    return {"object": result_obj, "description": description}