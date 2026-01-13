def extract_final_answer(model_output):
    """
    Extract key statistics for the effect of IsHuman from a fitted statsmodels GLMResultsWrapper
    and return an interpretable summary.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, z, p, conf_int, OR, OR_conf_int, significant)
      - "description": a short plain-language interpretation answering whether modern humans
                       have higher AMTL after accounting for controls (based on sign and p-value).

    The function is robust to variants of the parameter name, e.g. "IsHuman", "IsHuman[T.True]",
    or any parameter name containing the substring "IsHuman".
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Find the parameter name for IsHuman (allow variants)
    params_index = list(res.params.index)
    candidates = [name for name in params_index if 'IsHuman' in name]
    if not candidates:
        raise KeyError("Could not find a parameter matching 'IsHuman' in model parameters. "
                       f"Available params: {params_index}")

    # Prefer exact match if present
    param_name = 'IsHuman' if 'IsHuman' in params_index else candidates[0]

    coef = float(res.params[param_name])
    se = float(res.bse[param_name]) if hasattr(res, 'bse') else None
    # z/t value: statsmodels for GLM uses z-statistic (or t for some models); compute if possible
    test_stat = None
    pvalue = None
    if hasattr(res, 'pvalues') and param_name in res.pvalues:
        pvalue = float(res.pvalues[param_name])
    if se is not None and se != 0:
        test_stat = coef / se

    # Confidence interval (default 95%)
    try:
        ci = res.conf_int().loc[param_name].values.astype(float)
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        # fallback: try as ndarray
        try:
            ci_arr = res.conf_int()
            # if conf_int is ndarray with matching order, find index
            idx = params_index.index(param_name)
            ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
        except Exception:
            ci_lower, ci_upper = None, None

    # Odds ratio and CI (since binomial GLM uses logit link)
    try:
        or_est = float(np.exp(coef))
        or_ci = (float(np.exp(ci_lower)) if ci_lower is not None else None,
                 float(np.exp(ci_upper)) if ci_upper is not None else None)
    except Exception:
        or_est = None
        or_ci = (None, None)

    significant = (pvalue is not None) and (pvalue < 0.05)

    # Prepare numeric object to return
    object_dict = {
        "parameter_name": param_name,
        "coef_log_odds": coef,
        "std_error": se,
        "test_statistic": test_stat,
        "p_value": pvalue,
        "conf_int_95": (ci_lower, ci_upper),
        "odds_ratio": or_est,
        "odds_ratio_conf_int_95": or_ci,
        "significant_at_0.05": significant
    }

    # Short interpretation answering the task question
    if pvalue is None:
        interpretation = (
            f"Extracted parameter '{param_name}' (coef = {coef:.4g}). "
            "P-value could not be determined from the model output, so statistical significance is unknown. "
            "Positive coefficient indicates higher log-odds of AMTL in modern humans, negative indicates lower."
        )
    else:
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        sig_text = "statistically significant" if significant else "not statistically significant"
        interpretation = (
            f"Parameter '{param_name}': log-odds coef = {coef:.4g}, SE = {se:.4g}, "
            f"z/stat = {test_stat:.4g} , p = {pvalue:.4g}. 95% CI (log-odds) = ({ci_lower:.4g}, {ci_upper:.4g}). "
            f"Odds ratio = {or_est:.4g} with 95% CI = ({or_ci[0]:.4g}, {or_ci[1]:.4g}). "
            f"This indicates that modern humans have {direction} AMTL compared to the non-human primates "
            f"after adjusting for age, sex estimate, and tooth class; the effect is {sig_text} at alpha=0.05."
        )

    return {"object": object_dict, "description": interpretation}