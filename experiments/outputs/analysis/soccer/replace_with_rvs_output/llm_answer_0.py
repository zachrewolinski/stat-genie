def extract_final_answer(model_output):
    """
    Extracts the estimated effect of 'dark_binary' from a fitted statsmodels result object
    (including robust/clustered-covariance wrappers returned by get_robustcov_results).
    Returns a dictionary with keys:
      - "object": dictionary of numeric results (coef, se, pvalue, conf_int, IRR, IRR_CI, nobs, decision)
      - "description": brief plain-language interpretation of the result in context.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to safely get attributes that may exist on different result wrappers
    def _get_attr(attr_name):
        return getattr(res, attr_name, None)

    # Try to obtain parameter series, bse, pvalues, conf_int
    params = _get_attr('params')
    bse = _get_attr('bse')
    pvalues = _get_attr('pvalues')
    try:
        conf = res.conf_int()
    except Exception:
        conf = None

    if params is None:
        raise ValueError("The provided model_output does not expose .params. "
                         "Make sure this is a statsmodels result object or a robust-covariance wrapper.")

    # Locate the parameter corresponding to the dark vs light contrast.
    # Prefer exact name 'dark_binary', otherwise look for any parameter containing that substring.
    param_index = None
    if 'dark_binary' in params.index:
        param_index = 'dark_binary'
    else:
        matches = [n for n in params.index if 'dark_binary' in str(n)]
        if len(matches) == 1:
            param_index = matches[0]
        elif len(matches) > 1:
            # Pick the first match but warn via ValueError would be raised; here choose first deterministically
            param_index = matches[0]

    if param_index is None:
        raise KeyError("Could not find a parameter named 'dark_binary' in the model's parameters. "
                       "Available parameters: {}".format(list(params.index)))

    # Extract numeric values
    coef = float(params[param_index])
    se = float(bse[param_index]) if (bse is not None and param_index in bse.index) else None
    pval = float(pvalues[param_index]) if (pvalues is not None and param_index in pvalues.index) else None

    # Confidence interval for coefficient (log rate ratio)
    if conf is not None:
        # conf could be a DataFrame with index same as params.index
        try:
            ci_low, ci_high = conf.loc[param_index].astype(float).tolist()
        except Exception:
            # conf might be an ndarray in which case try to find by position
            try:
                idx = list(params.index).index(param_index)
                ci_low, ci_high = float(conf[idx, 0]), float(conf[idx, 1])
            except Exception:
                ci_low, ci_high = None, None
    else:
        ci_low, ci_high = None, None

    # Exponentiate to get incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
    irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

    # Sample size / number of observations if available
    nobs = _get_attr('nobs')
    if nobs is None:
        # try model endog
        try:
            nobs = int(res.model.endog.shape[0])
        except Exception:
            nobs = None

    # Decision rule: statistically significant at alpha = 0.05
    decision = {}
    alpha = 0.05
    if pval is None:
        decision_text = ("Unable to determine statistical significance because the p-value "
                         "for 'dark_binary' is not available.")
    else:
        if pval < alpha:
            if coef > 0:
                decision_text = (
                    "Yes — statistically significant evidence (p = {:.3g}) that dark-skinned players "
                    "receive more red cards per game than light-skinned players. "
                    "Estimated IRR = {:.3f} (95% CI [{:.3f}, {:.3f}])."
                ).format(pval, irr, irr_ci_low if irr_ci_low is not None else float('nan'),
                         irr_ci_high if irr_ci_high is not None else float('nan'))
            else:
                decision_text = (
                    "No (significant difference in the opposite direction) — statistically significant evidence (p = {:.3g}) that "
                    "dark-skinned players receive fewer red cards per game than light-skinned players. "
                    "Estimated IRR = {:.3f} (95% CI [{:.3f}, {:.3f}])."
                ).format(pval, irr, irr_ci_low if irr_ci_low is not None else float('nan'),
                         irr_ci_high if irr_ci_high is not None else float('nan'))
        else:
            decision_text = (
                "No strong evidence of a difference in red card rates between dark- and light-skinned players "
                "(p = {:.3g}). Estimated IRR = {:.3f} (95% CI [{:.3f}, {:.3f}])."
            ).format(pval, irr, irr_ci_low if irr_ci_low is not None else float('nan'),
                     irr_ci_high if irr_ci_high is not None else float('nan'))

    # Build returned object
    result_object = {
        "parameter_name": param_index,
        "coef_log_rate_ratio": coef,
        "std_error": se,
        "p_value": pval,
        "conf_int_log": [ci_low, ci_high],
        "IRR": irr,
        "conf_int_IRR": [irr_ci_low, irr_ci_high],
        "nobs": int(nobs) if (nobs is not None and float(nobs).is_integer()) else nobs,
        "alpha": alpha,
    }

    description = (
        "The model coefficient for '{}' is {:.6f} (SE = {}). This coefficient is on the log scale "
        "for the red-card rate per game (offset by log(games)). Exponentiating gives an incidence rate ratio (IRR) = {:.3f}, "
        "which indicates the multiplicative change in red-card rate for dark-skinned players relative to light-skinned players. "
        "{}"
    ).format(param_index,
             coef,
             ("{:.6f}".format(se) if se is not None else "NA"),
             irr,
             decision_text)

    return {"object": result_object, "description": description}