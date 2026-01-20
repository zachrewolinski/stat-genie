def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted statsmodels binary
    regression result (e.g., Logit or GLM Binomial). Returns a dictionary with:
      - "object": a dict of extracted numeric statistics (coef, se, p, CI, odds ratio, odds ratio CI)
      - "description": plain-language interpretation of the effect of being female on mortgage approval

    The function is defensive: if the 'female' variable is not present or some stats are
    unavailable it returns an explanatory message.
    """
    import numpy as np

    result = model_output

    param = "female"
    out = {"object": None, "description": None}

    # Basic checks
    try:
        params = result.params
    except Exception:
        out["description"] = "The provided model output does not have .params. Not a supported statsmodels results object."
        return out

    if param not in params.index:
        out["description"] = f"The fitted model does not include a parameter named '{param}'."
        return out

    # Extract coefficient
    coef = float(params[param])

    # Standard error
    try:
        se = float(result.bse[param])
    except Exception:
        # try sqrt of diagonal of covariance matrix
        try:
            cov = result.cov_params()
            idx = list(params.index).index(param)
            se = float(np.sqrt(np.abs(cov.iloc[idx, idx])))
        except Exception:
            se = None

    # p-value
    try:
        pval = float(result.pvalues[param])
    except Exception:
        pval = None

    # Confidence interval (95%)
    try:
        ci = result.conf_int(alpha=0.05)
        # conf_int may be a DataFrame or ndarray
        if hasattr(ci, "loc"):
            ci_lower = float(ci.loc[param, 0])
            ci_upper = float(ci.loc[param, 1])
        else:
            # fallback by index
            idx = list(params.index).index(param)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
    except Exception:
        ci_lower = None
        ci_upper = None

    # Odds ratio and its CI (exp of coef and CI)
    try:
        odds_ratio = float(np.exp(coef))
    except Exception:
        odds_ratio = None
    try:
        or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        or_ci_lower = or_ci_upper = None

    # Number of observations if available
    try:
        n_obs = int(getattr(result, "nobs", None))
    except Exception:
        n_obs = None

    # Compose object
    obj = {
        "parameter": param,
        "coefficient_log_odds": coef,
        "std_error": se,
        "p_value": pval,
        "conf_int_95": [ci_lower, ci_upper],
        "odds_ratio": odds_ratio,
        "odds_ratio_conf_int_95": [or_ci_lower, or_ci_upper],
        "n_obs": n_obs,
        "notes": "Coefficient is on the log-odds scale from a logistic model (Logit or GLM Binomial)."
    }

    # Build description interpreting the coefficient
    if pval is not None:
        sig_text = ("statistically significant" if pval < 0.05 else "not statistically significant")
    else:
        sig_text = "statistical significance could not be determined (p-value unavailable)"

    if odds_ratio is not None:
        if odds_ratio > 1:
            direction = "higher"
            magnitude_text = f"Odds ratio = {odds_ratio:.3f} (95% CI: {or_ci_lower:.3f} to {or_ci_upper:.3f})" if or_ci_lower is not None else f"Odds ratio = {odds_ratio:.3f}"
            interpret = f"Being female is associated with {direction} odds of mortgage approval compared with being male — {magnitude_text}."
        elif odds_ratio < 1:
            direction = "lower"
            magnitude_text = f"Odds ratio = {odds_ratio:.3f} (95% CI: {or_ci_lower:.3f} to {or_ci_upper:.3f})" if or_ci_lower is not None else f"Odds ratio = {odds_ratio:.3f}"
            interpret = f"Being female is associated with {direction} odds of mortgage approval compared with being male — {magnitude_text}."
        else:
            interpret = "Estimated odds ratio is 1.0 (no difference in odds between female and male)."
    else:
        interpret = "Could not compute odds ratio from the available coefficient."

    desc_lines = [
        f"Parameter examined: '{param}'.",
        f"Estimated log-odds coefficient = {coef:.4f}" + (f" (SE = {se:.4f})" if se is not None else ""),
        f"P-value = {pval:.4g}." if pval is not None else "P-value unavailable.",
        f"This effect is {sig_text}.",
        interpret,
        "Model adjusts for credit and demographic controls specified in the original analysis."
    ]
    description = " ".join(desc_lines)

    out["object"] = obj
    out["description"] = description
    return out