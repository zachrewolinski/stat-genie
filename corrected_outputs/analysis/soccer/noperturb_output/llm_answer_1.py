def extract_final_answer(model_output):
    """
    Extract coefficient, standard error, p-value, and 95% CI for the IsDark effect
    from a fitted statsmodels results object (possibly a clustered-robust results wrapper).
    Also compute the incidence rate ratio (IRR = exp(coef)) and its 95% CI,
    and give a short conclusion about whether dark-skinned players are more likely
    to receive red cards than light-skinned players (based on sign and p-value).

    Returns:
      {
        "object": { ... numeric results and conclusion ... },
        "description": "Text explanation of what these numbers mean"
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to get parameter name matching IsDark (robust to factor coding)
    param_index = None
    try:
        idx = getattr(res, 'params').index
    except Exception:
        # If params not available
        return {
            "object": None,
            "description": "The provided model object does not expose .params; cannot extract IsDark effect."
        }

    # Find a parameter name that contains 'IsDark'
    candidates = [name for name in idx if 'IsDark' in str(name)]
    if len(candidates) == 0:
        return {
            "object": None,
            "description": ("No parameter with name containing 'IsDark' was found in model params. "
                            "Available parameter names: " + ", ".join(map(str, idx)))
        }

    # If multiple matches, prefer exact 'IsDark', then the first match
    if 'IsDark' in candidates:
        param = 'IsDark'
    else:
        param = candidates[0]

    # Safely extract statistics; handle cases where attributes may be missing
    try:
        coef = float(res.params[param])
    except Exception:
        coef = None

    try:
        se = float(res.bse[param])
    except Exception:
        se = None

    try:
        pval = float(res.pvalues[param])
    except Exception:
        pval = None

    try:
        ci_df = res.conf_int()
        # conf_int may return ndarray or DataFrame; handle both
        if isinstance(ci_df, (pd.DataFrame, pd.Series)):
            ci_low = float(ci_df.loc[param][0]) if param in ci_df.index else float(ci_df.loc[param].iloc[0])
            ci_high = float(ci_df.loc[param][1]) if param in ci_df.index else float(ci_df.loc[param].iloc[1])
        else:
            # If it's a numpy array, find index of param in params index
            pos = list(res.params.index).index(param)
            ci_low = float(ci_df[pos, 0])
            ci_high = float(ci_df[pos, 1])
    except Exception:
        ci_low = None
        ci_high = None

    # Compute incidence rate ratio (IRR) and CI on exponentiated scale (since offset used, coef is log rate ratio)
    irr = np.exp(coef) if coef is not None else None
    irr_ci_low = np.exp(ci_low) if ci_low is not None else None
    irr_ci_high = np.exp(ci_high) if ci_high is not None else None

    # Number of observations if available
    n_obs = getattr(res, 'nobs', None)
    try:
        n_obs = int(n_obs) if n_obs is not None else None
    except Exception:
        pass

    # Simple conclusion based on sign and p-value (alpha = 0.05)
    conclusion = None
    if (pval is not None) and (coef is not None):
        if pval < 0.05:
            if coef > 0:
                conclusion = ("Statistically significant evidence (p < 0.05) that dark-skinned players "
                              "receive red cards at a higher rate than light-skinned players (controlling for "
                              "the listed covariates).")
            else:
                conclusion = ("Statistically significant evidence (p < 0.05) that dark-skinned players "
                              "receive red cards at a lower rate than light-skinned players (controlling for "
                              "the listed covariates).")
        else:
            conclusion = ("No statistically significant difference at the 0.05 level in red-card rates "
                          "between dark- and light-skinned players (the IsDark coefficient is not statistically significant).")
    else:
        conclusion = "Insufficient information to form a statistical conclusion (missing coef or p-value)."

    # Build the object to return (numbers + concise conclusion)
    obj = {
        "parameter_name": param,
        "coef_log_rate_ratio": coef,
        "std_error": se,
        "p_value": pval,
        "ci_95_log_scale": (ci_low, ci_high),
        "incidence_rate_ratio": irr,
        "irr_95_ci": (irr_ci_low, irr_ci_high),
        "n_obs": n_obs,
        "conclusion": conclusion
    }

    # Human-readable description summarizing what the extracted numbers mean
    description = (
        "The model coefficient for '{}' is the log difference in red-card rate per game for dark-skinned vs light-skinned players, "
        "controlling for games (offset), age, height, weight, goals, yellowCards, country-level bias measures, and position. "
        "Exponentiating the coefficient gives the incidence rate ratio (IRR): the multiplicative change in expected red-card rate "
        "per game for dark-skinned players compared to light-skinned players. ".format(param) +
        "The returned fields include the coefficient, its standard error, p-value, 95% CI on the log scale, "
        "and the corresponding IRR and its 95% CI. The 'conclusion' field gives a simple interpretation at alpha=0.05."
    )

    return {"object": obj, "description": description}