def extract_final_answer(model_output):
    """
    Extracts statistics for the IsHuman coefficient from a fitted statsmodels GLMResultsWrapper
    (binomial GLM) and provides a concise interpretation about whether modern humans (Homo sapiens)
    have higher antemortem tooth loss (AMTL) than non-human primates after adjusting for controls.

    Returns:
      dict with keys:
        - "object": dict of extracted numeric results:
            {
              "param_name": str,
              "coef": float,            # log-odds coefficient
              "se": float,              # standard error
              "z": float,               # z-statistic (coef / se)
              "p_value": float,
              "ci_low": float,          # 95% conf int lower (log-odds)
              "ci_high": float,         # 95% conf int upper (log-odds)
              "odds_ratio": float,      # exp(coef)
              "or_ci_low": float,       # exp(ci_low)
              "or_ci_high": float,      # exp(ci_high)
              "significant": bool       # p_value < 0.05
            }
        - "description": str with brief interpretation in plain language.
    """
    import numpy as np

    # Basic validation
    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided."
        }

    # Obtain parameter names from the fitted model
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        ci = model_output.conf_int()  # DataFrame or ndarray with rows aligned to params
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract necessary attributes from model_output: {e}"
        }

    # Decide which parameter corresponds to IsHuman.
    # We attempt to find an exact match 'IsHuman' first, otherwise any parameter containing 'IsHuman'.
    param_index = None
    for name in params.index:
        if name == 'IsHuman':
            param_index = name
            break
    if param_index is None:
        # fallback: any parameter name containing 'IsHuman'
        matches = [n for n in params.index if 'IsHuman' in n]
        if len(matches) >= 1:
            # pick the first match (commonly there's only one)
            param_index = matches[0]

    if param_index is None:
        return {
            "object": None,
            "description": "Model does not contain a parameter named 'IsHuman' (or similar)."
        }

    # Extract numeric values
    coef = float(params[param_index])
    se = float(bse[param_index]) if (param_index in bse.index) else float(np.nan)
    # z-statistic (for GLM binomial, params / se is appropriate)
    z = float(coef / se) if se != 0 and not np.isnan(se) else float(np.nan)
    p_value = float(pvalues[param_index]) if (param_index in pvalues.index) else float(np.nan)

    # Confidence interval
    try:
        # conf_int may be a DataFrame with index matching params.index
        if hasattr(ci, "loc"):
            ci_low = float(ci.loc[param_index][0])
            ci_high = float(ci.loc[param_index][1])
        else:
            # ndarray: find row by position
            idx_pos = list(params.index).index(param_index)
            ci_low = float(ci[idx_pos, 0])
            ci_high = float(ci[idx_pos, 1])
    except Exception:
        ci_low, ci_high = float(np.nan), float(np.nan)

    # Odds ratio and CI on odds ratio scale
    try:
        odds_ratio = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low)) if not np.isnan(ci_low) else float(np.nan)
        or_ci_high = float(np.exp(ci_high)) if not np.isnan(ci_high) else float(np.nan)
    except Exception:
        odds_ratio = or_ci_low = or_ci_high = float(np.nan)

    significant = (not np.isnan(p_value)) and (p_value < 0.05)

    result_object = {
        "param_name": param_index,
        "coef": coef,
        "se": se,
        "z": z,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "odds_ratio": odds_ratio,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "significant": bool(significant)
    }

    # Prepare description interpreting direction and significance
    if np.isnan(coef) or np.isnan(p_value):
        description = ("Could not compute coefficient or p-value for parameter "
                       f"'{param_index}'.")
    else:
        direction = "higher" if coef > 0 else "lower" if coef < 0 else "no difference"
        sig_phrase = "statistically significant" if significant else "not statistically significant"
        description = (
            f"The model coefficient for '{param_index}' is {coef:.4f} (SE = {se:.4f}), "
            f"z = {z:.2f}, p = {p_value:.3g}. The 95% CI for the log-odds is "
            f"[{ci_low:.4f}, {ci_high:.4f}], which corresponds to an odds ratio of "
            f"{odds_ratio:.3f} (95% CI: [{or_ci_low:.3f}, {or_ci_high:.3f}]).\n\n"
            f"Interpretation: specimens coded as modern humans (IsHuman) have {direction} "
            f"AMTL compared to non-human primates after adjusting for age, sex probability, "
            f"and tooth class; this effect is {sig_phrase} (alpha = 0.05)."
        )

    return {
        "object": result_object,
        "description": description
    }