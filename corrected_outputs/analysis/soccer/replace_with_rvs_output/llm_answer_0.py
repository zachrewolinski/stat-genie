def extract_final_answer(model_output):
    """
    Extract the coefficient, standard error, p-value, 95% CI, and incidence-rate ratio (IRR)
    for the 'dark_vs_light' predictor from a fitted statsmodels GLM/RegressionResults object,
    and provide a short interpretation.

    Returns:
      dict with keys:
        - "object": dict with numeric results (coef, se, p_value, conf_int, irr, irr_conf_int, percent_change, nobs, param_name)
        - "description": human-readable interpretation of the effect of dark_vs_light on redCards
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper: find parameter name that corresponds to the dark_vs_light predictor
    param_candidates = []
    try:
        idx = res.params.index
    except Exception:
        # If model_output does not have params, return informative message
        return {
            "object": None,
            "description": "Provided model_output does not have 'params' attribute; cannot extract statistics."
        }

    for name in idx:
        if "dark_vs_light" in str(name):
            param_candidates.append(name)

    if len(param_candidates) == 0:
        # try exact match if variable was numeric and named exactly
        if "dark_vs_light" in idx:
            param_candidates = ["dark_vs_light"]
        else:
            return {
                "object": None,
                "description": "No parameter matching 'dark_vs_light' found in model parameters. Available params: "
                               + ", ".join([str(x) for x in idx])
            }

    # If multiple matches (unlikely), pick the first
    param_name = param_candidates[0]

    # Extract coefficient, SE, p-value
    try:
        coef = float(res.params[param_name])
    except Exception:
        return {
            "object": None,
            "description": f"Could not extract coefficient for parameter '{param_name}'."
        }

    # Standard error: try robust bse if available on the object (res.bse), otherwise fallback
    try:
        se = float(res.bse[param_name])
    except Exception:
        # Some robust result wrappers store bse in .bse or .cov_params; try diag of cov
        try:
            cov = res.cov_params()
            se = float(np.sqrt(np.diag(cov))[list(res.params.index).index(param_name)])
        except Exception:
            se = None

    # p-value
    try:
        p_value = float(res.pvalues[param_name])
    except Exception:
        p_value = None

    # Confidence intervals
    try:
        ci_df = res.conf_int()
        # conf_int returns a DataFrame; columns could be [0,1] or named
        lower = float(ci_df.loc[param_name].iloc[0])
        upper = float(ci_df.loc[param_name].iloc[1])
        conf_int = [lower, upper]
    except Exception:
        conf_int = None
        lower = None
        upper = None

    # Incidence Rate Ratio (IRR) and CI
    try:
        irr = float(np.exp(coef))
        irr_conf_int = [float(np.exp(lower)), float(np.exp(upper))] if (lower is not None and upper is not None) else None
    except Exception:
        irr = None
        irr_conf_int = None

    # Percent change in rate
    percent_change = (irr - 1) * 100 if irr is not None else None

    # Number of observations, if available
    try:
        nobs = int(getattr(res, "nobs"))
    except Exception:
        try:
            nobs = int(res.model.endog.shape[0])
        except Exception:
            nobs = None

    # Statistical significance at alpha=0.05
    sig_text = None
    if p_value is not None:
        sig_text = "statistically significant (p < 0.05)" if p_value < 0.05 else "not statistically significant (p >= 0.05)"

    # Build the object to return
    result_object = {
        "param_name": str(param_name),
        "coef": coef,
        "std_error": se,
        "p_value": p_value,
        "conf_int_95": conf_int,
        "irr": irr,
        "irr_conf_int_95": irr_conf_int,
        "percent_change_in_rate": percent_change,
        "nobs": nobs
    }

    # Build a readable description
    if irr is not None and percent_change is not None:
        effect_descr = f"The estimated coefficient for '{param_name}' is {coef:.4f}"
        if se is not None:
            effect_descr += f" (SE = {se:.4f})"
        if p_value is not None:
            effect_descr += f", p = {p_value:.3g}"
        if conf_int is not None:
            effect_descr += f", 95% CI for coef = [{conf_int[0]:.4f}, {conf_int[1]:.4f}]"
        effect_descr += "."
        effect_descr += f" Exponentiating gives an IRR = {irr:.3f}"
        if irr_conf_int is not None:
            effect_descr += f" (95% CI = [{irr_conf_int[0]:.3f}, {irr_conf_int[1]:.3f}])."
        else:
            effect_descr += "."
        effect_descr += f" This implies a {percent_change:.1f}% {'increase' if percent_change > 0 else 'decrease' if percent_change < 0 else 'no change'} in the red-card rate for the group coded as '{param_name}' relative to the reference group, holding controls constant."
        if sig_text is not None:
            effect_descr += " The effect is " + sig_text + "."
    else:
        effect_descr = f"Could not compute IRR/interpretation for parameter '{param_name}'. Raw coef = {coef}."

    # Note about exposure offset and clustering (contextual)
    effect_descr += " The model used a log(offset = number of games) to model redCards per game; the estimate therefore reflects the relative rate of red cards per game. If the provided model output included cluster-robust SEs (by referee), the SE/p-value/confidence intervals reflect that."

    return {
        "object": result_object,
        "description": effect_descr
    }