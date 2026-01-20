def extract_final_answer(model_output):
    """
    Extracts the effect of the skin-tone binary variable (dark vs light) on red card counts
    from a fitted statsmodels GLM/GLMResultsWrapper.

    Returns a dict with:
      - "object": a dict with numeric results (coef, se, pval, conf_int, IRR, IRR_CI, nobs)
      - "description": short plain-language interpretation answering whether darker players
                       are more likely to receive red cards than lighter players.

    The function attempts to locate a parameter whose name contains 'SkinToneBin' (the
    binary indicator). If not found, it will fall back to 'SkinToneMean' (continuous).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic extractions (handle different result types robustly)
    try:
        params = res.params
    except Exception:
        raise ValueError("model_output does not have .params")

    try:
        conf = res.conf_int()
    except Exception:
        # If conf_int fails, construct NaNs
        conf = pd.DataFrame(np.nan, index=params.index, columns=[0, 1])

    try:
        pvalues = res.pvalues
    except Exception:
        pvalues = pd.Series(np.nan, index=params.index)

    try:
        nobs = int(getattr(res, 'nobs', getattr(res.model, 'nobs', len(res.model.endog))))
    except Exception:
        nobs = None

    # Find the parameter name for the binary dark-vs-light variable
    param_name = None
    # 1) prefer any parameter containing 'SkinToneBin' (this should capture factor-coded dummies)
    for idx in params.index:
        if 'SkinToneBin' in str(idx):
            param_name = idx
            break
    # 2) fall back to any param name that explicitly mentions 'Dark' and 'SkinTone' (safety)
    if param_name is None:
        for idx in params.index:
            if ('SkinTone' in str(idx)) and ('Dark' in str(idx)):
                param_name = idx
                break
    # 3) fall back to continuous mean if binary not present
    used_continuous = False
    if param_name is None:
        for idx in params.index:
            if 'SkinToneMean' in str(idx):
                param_name = idx
                used_continuous = True
                break

    if param_name is None:
        # nothing found: return available SkinTone-related stats if any, otherwise fail gracefully
        skin_params = [idx for idx in params.index if 'SkinTone' in str(idx)]
        if len(skin_params) == 0:
            return {
                "object": None,
                "description": "No parameter matching 'SkinToneBin' or 'SkinToneMean' was found in the model results. Cannot extract an answer."
            }
        else:
            # pick first related param
            param_name = skin_params[0]

    # Extract stats
    coef = float(params[param_name])
    se = float(getattr(res, 'bse', pd.Series(params.index, index=params.index)).get(param_name, np.nan))
    pval = float(pvalues.get(param_name, np.nan))
    try:
        ci_low = float(conf.loc[param_name][0])
        ci_high = float(conf.loc[param_name][1])
    except Exception:
        # conf could be ndarray or other; try alternatives
        try:
            ci = res.conf_int().iloc[list(params.index).index(param_name)]
            ci_low, ci_high = float(ci[0]), float(ci[1])
        except Exception:
            ci_low, ci_high = np.nan, np.nan

    # Incidence Rate Ratio and CI (exp of coef/CI)
    irr = float(np.exp(coef))
    irr_low = float(np.exp(ci_low)) if (not np.isnan(ci_low)) else np.nan
    irr_high = float(np.exp(ci_high)) if (not np.isnan(ci_high)) else np.nan

    # Interpret sign and significance
    alpha = 0.05
    significant = (not np.isnan(pval)) and (pval < alpha)
    direction = None
    if coef > 0:
        direction = "higher"
    elif coef < 0:
        direction = "lower"
    else:
        direction = "no difference"

    # Formulate a concise description
    if used_continuous:
        var_label = f"{param_name} (continuous SkinToneMean)"
        hypothesis_text = ("(continuous SkinToneMean used as robustness check)")
    else:
        var_label = f"{param_name} (Dark vs Light)"
        hypothesis_text = "(Dark relative to Light)"

    desc = (
        f"Parameter extracted: {var_label}. Sample size (nobs) = {nobs}.\n"
        f"Coefficient (log IRR) = {coef:.4f}, SE = {se:.4f}, p = {pval:.4f}.\n"
        f"Incidence Rate Ratio (IRR) = {irr:.3f}; 95% CI for IRR = [{irr_low:.3f}, {irr_high:.3f}].\n"
    )

    if significant:
        desc += (
            f"Interpretation: The {('binary dark vs light' if not used_continuous else 'continuous skin tone')} "
            f"effect is statistically significant (p < {alpha}). Players in the 'Dark' group have a {direction} "
            f"rate of receiving red cards compared to the 'Light' group (IRR = {irr:.3f}). {hypothesis_text}"
        )
    else:
        desc += (
            f"Interpretation: The effect is not statistically significant at alpha={alpha} (p = {pval:.4f}). "
            f"There is no strong evidence that darker-skinned players receive more red cards than lighter-skinned players "
            f"based on this model. {hypothesis_text}"
        )

    # Prepare object with numeric outputs for downstream use
    result_object = {
        "parameter_name": str(param_name),
        "coef_log_IRR": coef,
        "se": se,
        "p_value": pval,
        "coef_CI_lower": ci_low,
        "coef_CI_upper": ci_high,
        "IRR": irr,
        "IRR_CI_lower": irr_low,
        "IRR_CI_upper": irr_high,
        "nobs": nobs,
        "significant_at_0.05": bool(significant),
        "direction": direction,
        "used_continuous_as_fallback": bool(used_continuous)
    }

    return {"object": result_object, "description": desc}