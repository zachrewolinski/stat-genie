def extract_final_answer(model_output):
    """
    Extracts statistics for the 'DarkSkin' coefficient from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with keys:
      - "object": a dict with numeric results (coefficient, SE, p-value, CI, IRR, IRR CI, significance, n_obs)
      - "description": a short interpretation of what these numbers mean for the research question.
    """
    import numpy as np

    res = model_output

    # Attempt to find the parameter name corresponding to the DarkSkin variable
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("The provided model_output does not have a 'params' attribute.")

    # Common expected names; fall back to substring search
    candidate_names = ["DarkSkin", "DarkSkin[T.1]", "DarkSkin_1", "darkSkin", "dark_skin"]
    param_name = None
    for nm in candidate_names:
        if nm in params.index:
            param_name = nm
            break
    if param_name is None:
        # fallback: any param name containing 'Dark' (case-insensitive)
        for nm in params.index:
            if "dark" in str(nm).lower():
                param_name = nm
                break

    if param_name is None:
        raise KeyError("Could not find a parameter name for 'DarkSkin' in model params. "
                       "Available param names: {}".format(list(params.index)))

    # Extract values (use try/except to handle different result object behaviors)
    try:
        coef = float(res.params[param_name])
    except Exception:
        coef = float(params[param_name])

    # Standard error
    try:
        se = float(res.bse[param_name])
    except Exception:
        # fallback: compute from covariance matrix diagonal if available
        cov = getattr(res, "cov_params", None)
        if callable(cov):
            covmat = cov()
        else:
            covmat = getattr(res, "cov_params()", None)
        if covmat is not None and param_name in covmat.index:
            se = float(np.sqrt(covmat.loc[param_name, param_name]))
        else:
            se = None

    # p-value
    try:
        pvalue = float(res.pvalues[param_name])
    except Exception:
        pvalue = None

    # Confidence interval (try res.conf_int(), else use coef +/- 1.96*se)
    try:
        ci_df = res.conf_int()
        if param_name in ci_df.index:
            ci_lower, ci_upper = float(ci_df.loc[param_name, 0]), float(ci_df.loc[param_name, 1])
        else:
            raise KeyError
    except Exception:
        if se is not None:
            z = 1.96
            ci_lower = coef - z * se
            ci_upper = coef + z * se
        else:
            ci_lower = ci_upper = None

    # Exponentiate to get incidence rate ratio (IRR) and its CI
    try:
        irr = float(np.exp(coef))
    except Exception:
        irr = None
    try:
        irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        irr_ci_lower = irr_ci_upper = None

    # Significance flag (two-sided alpha=0.05) if p-value available
    significant = None
    if pvalue is not None:
        significant = bool(pvalue < 0.05)

    # Number of observations if available
    n_obs = getattr(res, "nobs", None)
    try:
        if n_obs is not None:
            n_obs = int(n_obs)
    except Exception:
        pass

    # Build the object to return
    result_object = {
        "param_name": param_name,
        "coef_log_rate": coef,                     # log rate ratio (model scale)
        "se": se,
        "p_value": pvalue,
        "ci_95_log_rate": (ci_lower, ci_upper),
        "IRR": irr,                                # incidence rate ratio = exp(coef)
        "IRR_95_CI": (irr_ci_lower, irr_ci_upper),
        "significant_at_0.05": significant,
        "n_obs": n_obs
    }

    # Short human-readable interpretation
    if (irr is not None) and (significant is not None):
        if significant:
            if irr > 1:
                meaning = ("The model estimates that players coded as DarkSkin receive red cards at a higher rate "
                           "than LightSkin players. Estimated IRR = {:.3f} (95% CI: {:.3f} to {:.3f}), p = {:.3g}."
                           .format(irr, irr_ci_lower, irr_ci_upper, pvalue))
            else:
                meaning = ("The model estimates that players coded as DarkSkin receive red cards at a lower rate "
                           "than LightSkin players. Estimated IRR = {:.3f} (95% CI: {:.3f} to {:.3f}), p = {:.3g}."
                           .format(irr, irr_ci_lower, irr_ci_upper, pvalue))
        else:
            meaning = ("No statistically significant difference in red-card rates between DarkSkin and LightSkin players "
                       "was detected at alpha=0.05. Estimated IRR = {:.3f} (95% CI: {:.3f} to {:.3f}), p = {:.3g}."
                       .format(irr, irr_ci_lower, irr_ci_upper, pvalue))
    else:
        meaning = ("Extracted statistics for parameter '{}'. IRR and significance could not be fully determined "
                   "because some quantities (SE or p-value) were unavailable.".format(param_name))

    description = (
        "Extracted coefficient for '{}'. Coefficient is on the log-rate scale for redCards per game "
        "(offset used). Exponentiating the coefficient gives the incidence rate ratio (IRR): "
        "IRR > 1 means higher red-card rate for DarkSkin players; IRR < 1 means lower rate. "
        "{}"
    ).format(param_name, meaning)

    return {"object": result_object, "description": description}