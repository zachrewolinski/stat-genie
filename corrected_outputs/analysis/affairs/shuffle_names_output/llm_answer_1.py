def extract_final_answer(model_output):
    """
    Extract statistics for the effect of 'HasChildren' from a fitted statsmodels binary
    regression results object (Logit/GLM/etc.) and produce a concise interpretation.

    Returns:
      {
        "object": {  # numeric summary values (or None if not found)
          "param_name": str,
          "coef": float or None,
          "se": float or None,
          "p_value": float or None,
          "odds_ratio": float or None,
          "ci_lower": float or None,
          "ci_upper": float or None,
          "nobs": int or None
        },
        "description": str  # brief plain-language interpretation of the effect
      }
    """
    import numpy as np

    res = model_output

    # Try to access parameter vector
    try:
        params = res.params
    except Exception:
        return {
            "object": None,
            "description": "The provided model_output does not expose a .params attribute; cannot extract results."
        }

    # Identify parameter name for the 'HasChildren' variable in a robust way
    param_candidates = [name for name in params.index if 'haschild' in str(name).lower() or 'child' in str(name).lower() or 'children' in str(name).lower()]
    # Prefer exact 'HasChildren' if present
    param_name = None
    if 'HasChildren' in params.index:
        param_name = 'HasChildren'
    elif len(param_candidates) > 0:
        # pick the first reasonable candidate
        param_name = param_candidates[0]
    else:
        return {
            "object": None,
            "description": "No parameter matching 'HasChildren' (or 'child/children') found in model parameters."
        }

    # Safely extract statistics, allowing for different result object shapes
    def safe_get(series_like, key):
        try:
            return series_like[key]
        except Exception:
            try:
                # maybe numeric-indexable
                idx = list(params.index).index(key)
                return series_like[idx]
            except Exception:
                return None

    coef = safe_get(params, param_name)
    # standard error
    se = None
    try:
        se = safe_get(res.bse, param_name)
    except Exception:
        se = None
    # p-value
    p_value = None
    try:
        p_value = safe_get(res.pvalues, param_name)
    except Exception:
        p_value = None

    # confidence interval (2.5%, 97.5%)
    ci_lower = ci_upper = None
    try:
        conf = res.conf_int()
        # conf might be a DataFrame-like with index matching params
        try:
            row = conf.loc[param_name]
            ci_lower, ci_upper = float(row[0]), float(row[1])
        except Exception:
            # conf might be an ndarray with rows aligned to params
            try:
                idx = list(params.index).index(param_name)
                ci_lower, ci_upper = float(conf[idx, 0]), float(conf[idx, 1])
            except Exception:
                ci_lower = ci_upper = None
    except Exception:
        ci_lower = ci_upper = None

    # Odds ratio and CI on OR scale
    odds_ratio = None
    or_ci_lower = None
    or_ci_upper = None
    try:
        if coef is not None:
            odds_ratio = float(np.exp(coef))
        if ci_lower is not None:
            or_ci_lower = float(np.exp(ci_lower))
        if ci_upper is not None:
            or_ci_upper = float(np.exp(ci_upper))
    except Exception:
        odds_ratio = or_ci_lower = or_ci_upper = None

    # sample size
    nobs = None
    try:
        nobs = int(res.nobs)
    except Exception:
        try:
            # fallback to model endog length
            nobs = int(getattr(res.model, 'nobs', len(getattr(res.model, 'endog', []))))
        except Exception:
            nobs = None

    # Build summary object
    result_obj = {
        "param_name": param_name,
        "coef": float(coef) if coef is not None else None,
        "se": float(se) if se is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "odds_ratio": float(odds_ratio) if odds_ratio is not None else None,
        "ci_lower": float(or_ci_lower) if or_ci_lower is not None else None,
        "ci_upper": float(or_ci_upper) if or_ci_upper is not None else None,
        "nobs": int(nobs) if nobs is not None else None
    }

    # Interpretation text
    if result_obj["odds_ratio"] is None:
        description = f"Extracted parameter '{param_name}', but could not compute odds ratio/CI. Raw coefficient = {result_obj['coef']}, p = {result_obj['p_value']}."
    else:
        direction = "decrease" if result_obj["odds_ratio"] < 1 else "increase"
        ptext = ("statistically significant" if (result_obj["p_value"] is not None and result_obj["p_value"] < 0.05)
                 else "not statistically significant")
        description = (
            f"Effect of '{param_name}' on odds of having an extramarital affair: coefficient = {result_obj['coef']:.4f}, "
            f"OR = {result_obj['odds_ratio']:.3f} (95% CI: {result_obj['ci_lower']:.3f} to {result_obj['ci_upper']:.3f}), "
            f"p = {result_obj['p_value']:.4g} (n = {result_obj['nobs']}).\n"
            f"This implies that having children is associated with a {abs(1 - result_obj['odds_ratio'])*100:.1f}% {direction} in the odds of reporting an affair; the effect is {ptext}."
        )

    return {"object": result_obj, "description": description}