def extract_final_answer(model_output):
    """
    Extracts the IsDark coefficient and robust (clustered if available) inference
    from the provided model_output dict.

    Returns a dictionary with:
      - "object": a dict containing coefficient, se, z, p-value, 95% CI on log scale,
                  incidence rate ratio (IRR = exp(coef)) and 95% CI for IRR,
                  inference method used, and nobs if available.
      - "description": a short plain-language interpretation answering whether
                       dark-skinned players are more likely to receive red cards.
    """
    import numpy as np
    from scipy import stats

    # Prefer the clustered-se fit if present, otherwise fall back to the raw fit
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('nb_model_clustered_se') or model_output.get('nb_model_raw')
    else:
        res = model_output

    if res is None:
        return {
            "object": None,
            "description": "No model result found in model_output."
        }

    # Try to get parameter vector
    try:
        params = res.params  # pandas Series
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not read params from model result: {e}"
        }

    # Determine index for IsDark
    if 'IsDark' in params.index:
        idx_name = 'IsDark'
        idx = params.index.get_loc('IsDark')
    else:
        # fallback: try to assume the second parameter is IsDark (after const)
        if len(params) >= 2:
            idx = 1
            idx_name = params.index[idx]
        else:
            return {
                "object": None,
                "description": "Model parameters do not contain 'IsDark' and no suitable fallback."
            }

    coef = float(params.iloc[idx])

    # Try to get covariance matrix (will reflect cov_type used when fit was called)
    se = None
    method = "unknown"
    try:
        cov = res.cov_params()
        # cov may be numpy matrix/array or DataFrame
        if hasattr(cov, "values"):
            cov_vals = np.asarray(cov.values)
        else:
            cov_vals = np.asarray(cov)
        se = float(np.sqrt(np.diag(cov_vals)[idx]))
        # infer method from result object attrs if possible
        method = getattr(res, "cov_type", "clustered_or_raw")
    except Exception:
        # fallback to model-provided bse (may be the default non-clustered SE)
        try:
            se = float(res.bse.iloc[idx])
            method = "bse_fallback"
        except Exception as e:
            return {
                "object": None,
                "description": f"Could not obtain standard errors from model result: {e}"
            }

    # Compute z, p-value, CI, and IRR
    if se == 0 or np.isnan(se):
        return {
            "object": None,
            "description": "Standard error for IsDark is zero or NaN; cannot compute inference."
        }

    z_stat = coef / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    ci_lower = float(coef - 1.96 * se)
    ci_upper = float(coef + 1.96 * se)
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Number of observations if available
    nobs = None
    try:
        nobs = int(getattr(res, "nobs"))
    except Exception:
        # some results expose .model.endog or len(.fittedvalues)
        try:
            nobs = int(len(res.model.endog))
        except Exception:
            nobs = None

    # Plain-language interpretation: since model is a log-link NB with offset=log(games),
    # coef is log rate ratio of red cards per game for dark vs light players.
    if np.isnan(p_value):
        verdict = "Cannot determine statistical significance (p-value is NaN)."
    else:
        alpha = 0.05
        if (p_value < alpha) and (coef > 0):
            verdict = ("Yes — the coefficient for IsDark is positive and statistically significant "
                       f"(p = {p_value:.3g}). Dark-skinned players have a higher rate of red cards. ")
        elif (p_value < alpha) and (coef < 0):
            verdict = ("No — the coefficient for IsDark is negative and statistically significant "
                       f"(p = {p_value:.3g}). Dark-skinned players have a lower rate of red cards. ")
        else:
            verdict = ("No strong evidence of a difference — the IsDark coefficient is not "
                       f"statistically significant at α = {alpha} (p = {p_value:.3g}).")

    # Compose the object to return
    result_object = {
        "parameter_name": idx_name,
        "coef_log_rate": coef,
        "se": se,
        "z": z_stat,
        "p_value": p_value,
        "ci_log_lower": ci_lower,
        "ci_log_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "method_used_for_se": str(method),
        "nobs": nobs
    }

    description = (
        f"Coefficient for '{idx_name}' = {coef:.4f} (SE = {se:.4f}, z = {z_stat:.3f}, p = {p_value:.3g}). "
        f"This is a log rate-ratio from a Negative Binomial model with offset = log(games). "
        f"Exp(coef) = IRR = {irr:.3f} (95% CI: {irr_ci_lower:.3f} to {irr_ci_upper:.3f}). "
        f"{verdict}"
    )

    return {"object": result_object, "description": description}