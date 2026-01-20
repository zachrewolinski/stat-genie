def extract_final_answer(model_output):
    """
    Extracts the coefficient, uncertainty, p-value, and incidence-rate interpretation
    for the 'SkinDark' variable from a fitted statsmodels GLM/robust-results object.

    Returns a dict with:
      - "object": dict with numeric results (coef, se, z, p, 95% CI, IRR and IRR CI, significant)
      - "description": short plain-language interpretation of the effect in context
    """
    import math

    res = model_output

    # Check that params exist
    try:
        params = res.params
    except Exception:
        return {
            "object": None,
            "description": "The provided model object has no 'params' attribute; cannot extract results."
        }

    # Find the parameter name for SkinDark (be flexible in case of slight naming differences)
    param_names = [str(n) for n in params.index]
    if "SkinDark" in param_names:
        key = "SkinDark"
    else:
        matches = [n for n in param_names if "SkinDark" in n]
        if matches:
            key = matches[0]
        else:
            return {
                "object": None,
                "description": "No parameter matching 'SkinDark' found in the model parameters."
            }

    # Extract coefficient
    try:
        coef = float(params[key])
    except Exception:
        return {
            "object": None,
            "description": f"Unable to read coefficient for parameter '{key}'."
        }

    # Standard error (robust or model-based)
    se = None
    try:
        se = float(res.bse[key])
    except Exception:
        # if unavailable, leave as None
        se = None

    # p-value (prefer model-provided). If missing, compute from z using normal approximation.
    p_value = None
    try:
        p_value = float(res.pvalues[key])
    except Exception:
        if se is not None and se > 0:
            z_approx = coef / se
            # normal cdf via error function from stdlib
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_approx) / math.sqrt(2))))
        else:
            p_value = None

    # z/t statistic if possible
    z_value = None
    if se is not None and se > 0:
        z_value = coef / se

    # 95% confidence interval for coefficient
    ci = [None, None]
    try:
        ci_obj = res.conf_int()
        # ci_obj might be a DataFrame-like with .loc or a numpy array
        if hasattr(ci_obj, "loc"):
            row = ci_obj.loc[key]
            ci = [float(row.iloc[0]), float(row.iloc[1])]
        else:
            # assume numpy array aligned with params order
            idx = list(params.index).index(key)
            ci = [float(ci_obj[idx, 0]), float(ci_obj[idx, 1])]
    except Exception:
        ci = [None, None]

    # Incidence Rate Ratio (IRR) and its CI (exp of coef and CI)
    try:
        irr = math.exp(coef)
    except Exception:
        irr = None
    irr_ci = [None, None]
    try:
        if ci[0] is not None and ci[1] is not None:
            irr_ci = [math.exp(ci[0]), math.exp(ci[1])]
    except Exception:
        irr_ci = [None, None]

    # Statistical significance at alpha=0.05 (if p-value available)
    significant = None
    if p_value is not None:
        significant = (p_value < 0.05)

    # Plain-language interpretation
    if p_value is None or irr is None:
        interpretation = (
            "Extracted coefficient and/or p-value not available; cannot give a definitive inference "
            "about whether dark-skinned players receive red cards at a different rate."
        )
    else:
        # Determine direction and significance
        if significant:
            if irr > 1:
                interpretation = (
                    f"Estimated effect: dark-skinned players receive red cards at a higher rate. "
                    f"IRR = {irr:.3f} (95% CI {irr_ci[0]:.3f}–{irr_ci[1]:.3f}), p = {p_value:.3g}. "
                    "This is statistically significant at the 0.05 level."
                )
            elif irr < 1:
                interpretation = (
                    f"Estimated effect: dark-skinned players receive red cards at a lower rate. "
                    f"IRR = {irr:.3f} (95% CI {irr_ci[0]:.3f}–{irr_ci[1]:.3f}), p = {p_value:.3g}. "
                    "This is statistically significant at the 0.05 level."
                )
            else:
                interpretation = (
                    f"Estimated IRR is 1.00 (no effect). IRR = {irr:.3f}, p = {p_value:.3g}."
                )
        else:
            interpretation = (
                f"No statistically significant difference in red-card rates by skin tone at alpha=0.05. "
                f"Estimated IRR = {irr:.3f} (95% CI {irr_ci[0]:.3f}–{irr_ci[1]:.3f}), p = {p_value:.3g}."
            )

    # Build the returned object with numeric results and interpretation
    result_object = {
        "parameter_name": key,
        "coef": coef,
        "std_err": se,
        "z_value": z_value,
        "p_value": p_value,
        "coef_95CI": ci,
        "IRR": irr,
        "IRR_95CI": irr_ci,
        "significant_at_0.05": significant
    }

    return {
        "object": result_object,
        "description": interpretation
    }