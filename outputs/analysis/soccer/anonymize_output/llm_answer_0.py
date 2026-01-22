def extract_final_answer(model_output):
    """
    Extracts the estimated effect of the dark vs light skin indicator from a fitted
    statsmodels GLM (negative binomial) results object (optionally robust/clustered).
    Returns a dict with the numeric results (coefficient, SE, z, p, 95% CI, IRR and IRR CI)
    and a short interpretation about whether players with dark skin are more likely to
    receive red cards than those with light skin.

    Returned dictionary:
      {
        "object": { ... numeric extraction ... },
        "description": "Plain-language interpretation"
      }
    """
    import numpy as np

    res = model_output

    # Helper: locate the parameter name for the skin indicator.
    param_name = None
    # prefer exact 'dark_binary'
    try:
        idx = list(res.params.index)
    except Exception:
        # If params not available, raise a helpful error
        raise ValueError("The model_output does not expose .params; please pass the fitted statsmodels results object.")

    if 'dark_binary' in idx:
        param_name = 'dark_binary'
    else:
        # fallback: find any parameter name containing 'dark' (case-insensitive)
        for name in idx:
            if 'dark' in str(name).lower():
                param_name = name
                break

    if param_name is None:
        raise KeyError("Could not find a parameter for the skin-tone indicator (searched for 'dark_binary' or names containing 'dark'). "
                       "Available parameter names: {}".format(idx))

    # Extract coefficient, standard error, p-value
    coef = float(res.params[param_name])
    # bse and pvalues should exist on the results object
    se = float(res.bse[param_name]) if hasattr(res, 'bse') else None
    pval = float(res.pvalues[param_name]) if hasattr(res, 'pvalues') else None

    # z / t stat (depending on results); compute if se available
    z_stat = float(coef / se) if se not in (None, 0) else None

    # 95% CI for coefficient
    try:
        ci_df = res.conf_int()  # typically a DataFrame indexed by param names
        # conf_int may return numpy array; handle both
        if hasattr(ci_df, 'loc'):
            ci_lower = float(ci_df.loc[param_name, 0])
            ci_upper = float(ci_df.loc[param_name, 1])
        else:
            # array-like: find the row corresponding to param index
            param_idx = list(res.params.index).index(param_name)
            ci_lower = float(ci_df[param_idx, 0])
            ci_upper = float(ci_df[param_idx, 1])
    except Exception:
        # If conf_int not available, set to None
        ci_lower = ci_upper = None

    # Exponentiate coefficient to get incidence rate ratio (IRR) and IRR CI
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
    irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

    # Determine a simple statistical decision (alpha = 0.05)
    if pval is None:
        decision = "Unable to determine statistical significance (p-value not available)."
        significant = None
    else:
        significant = bool(pval < 0.05)
        if significant:
            if irr > 1:
                decision = ("Statistically significant: players rated as dark-skinned have a higher rate of "
                            "red cards vs light-skinned players (IRR = {:.3f}, 95% CI [{:.3f}, {:.3f}], p = {:.3g})."
                            .format(irr, irr_ci_lower if irr_ci_lower is not None else np.nan,
                                    irr_ci_upper if irr_ci_upper is not None else np.nan, pval))
            else:
                decision = ("Statistically significant: players rated as dark-skinned have a lower rate of "
                            "red cards vs light-skinned players (IRR = {:.3f}, 95% CI [{:.3f}, {:.3f}], p = {:.3g})."
                            .format(irr, irr_ci_lower if irr_ci_lower is not None else np.nan,
                                    irr_ci_upper if irr_ci_upper is not None else np.nan, pval))
        else:
            decision = ("No statistically significant difference at alpha=0.05 between dark- and light-skinned players "
                        "in rates of red cards (estimated IRR = {:.3f}, 95% CI [{:.3f}, {:.3f}], p = {:.3g})."
                        .format(irr, irr_ci_lower if irr_ci_lower is not None else np.nan,
                                irr_ci_upper if irr_ci_upper is not None else np.nan, pval))

    # Build the numeric object to return
    numeric_result = {
        "param_name": param_name,
        "coef": coef,
        "std_error": se,
        "z_stat": z_stat,
        "p_value": pval,
        "coef_ci_lower": ci_lower,
        "coef_ci_upper": ci_upper,
        "IRR": irr,
        "IRR_ci_lower": irr_ci_lower,
        "IRR_ci_upper": irr_ci_upper,
        "significant_at_0.05": significant
    }

    description = (
        "Extracted effect of '{}' from the fitted model. Coefficient is on the log count scale; "
        "IRR = exp(coef) is the multiplicative change in red-card rate for dark vs light players, "
        "adjusting for covariates and using matches as exposure; standard errors were clustered by referee "
        "if the results object included clustered/robust covariances. Decision: {}"
    ).format(param_name, decision)

    return {"object": numeric_result, "description": description}