def extract_final_answer(model_output):
    """
    Extract coefficients, standard errors, p-values, 95% CIs, and multiplicative (percent)
    effects for the predictors age_c, sex_male, and help_yes from a fitted statsmodels
    result object (MixedLMResultsWrapper or RegressionResultsWrapper).
    
    Returns:
      dict with keys:
        - "object": dict mapping each predictor -> extracted statistics
        - "description": human-readable summary of effects (direction, significance,
                         and percent-change interpretation on the original scale)
    """
    import numpy as np

    predictors = ['age_c', 'sex_male', 'help_yes']
    out = {}
    # Helper to safely get an attribute or None
    def safe_get(obj, name):
        return getattr(obj, name) if hasattr(obj, name) else None

    # Try to get params, bse, pvalues, conf_int
    params = safe_get(model_output, 'params')
    bse = safe_get(model_output, 'bse')
    pvalues = safe_get(model_output, 'pvalues')
    try:
        ci = model_output.conf_int()
    except Exception:
        ci = None

    # If any are returned as numpy arrays without index, convert to dictionaries if possible
    # But most statsmodels result objects return pandas Series/DataFrame for these.
    # We'll attempt index-based lookup, and fallback to KeyError handling.
    for var in predictors:
        res = {
            'coef': None,
            'std_err': None,
            'p_value': None,
            'ci_lower': None,
            'ci_upper': None,
            'percent_change': None,   # (exp(coef)-1)*100: percent change in nuts/min
            'significant_p_lt_0_05': None,
            'present_in_model': False
        }
        try:
            # params may be a Series (preferred)
            if params is not None and var in params.index:
                coef = float(params.loc[var])
                res['coef'] = coef
                res['present_in_model'] = True
            else:
                # try positional access if params is ndarray or no index
                if params is not None and hasattr(params, '__len__'):
                    # No reliable name lookup; skip in that case
                    coef = None
                else:
                    coef = None
        except Exception:
            coef = None

        # std err
        try:
            if bse is not None and var in bse.index:
                res['std_err'] = float(bse.loc[var])
            else:
                # fallback: try to infer from console if available
                res['std_err'] = None
        except Exception:
            res['std_err'] = None

        # p-value
        try:
            if pvalues is not None and var in pvalues.index:
                res['p_value'] = float(pvalues.loc[var])
            else:
                res['p_value'] = None
        except Exception:
            res['p_value'] = None

        # confidence interval
        try:
            if ci is not None:
                # conf_int usually returns DataFrame with columns [0,1] or named columns
                if hasattr(ci, 'loc') and var in ci.index:
                    lower = ci.loc[var].iat[0]
                    upper = ci.loc[var].iat[1]
                    res['ci_lower'] = float(lower)
                    res['ci_upper'] = float(upper)
                else:
                    res['ci_lower'] = None
                    res['ci_upper'] = None
        except Exception:
            res['ci_lower'] = None
            res['ci_upper'] = None

        # percent change interpretation (dependent variable was log-transformed)
        try:
            if res['coef'] is not None:
                res['percent_change'] = float((np.expm1(res['coef'])) * 100.0)
        except Exception:
            res['percent_change'] = None

        # significance flag
        try:
            if res['p_value'] is not None:
                res['significant_p_lt_0_05'] = bool(res['p_value'] < 0.05)
            else:
                res['significant_p_lt_0_05'] = None
        except Exception:
            res['significant_p_lt_0_05'] = None

        out[var] = res

    # Build a concise human-readable description
    desc_lines = []
    model_name = type(model_output).__name__
    desc_lines.append(f"Extracted results from model object of type: {model_name}.")

    for var in predictors:
        r = out[var]
        if not r['present_in_model']:
            desc_lines.append(f"- {var}: NOT found in model coefficients.")
            continue
        coef = r['coef']
        p = r['p_value']
        pct = r['percent_change']
        ci_l = r['ci_lower']; ci_u = r['ci_upper']
        sig = r['significant_p_lt_0_05']
        # Direction
        if coef is None:
            direction = "no estimate"
        elif coef > 0:
            direction = "positive"
        elif coef < 0:
            direction = "negative"
        else:
            direction = "no change"

        # Compose line
        line = f"- {var}: coef={coef:.4f}"
        if r['std_err'] is not None:
            line += f" (SE={r['std_err']:.4f})"
        if p is not None:
            line += f", p={p:.3f}"
        if ci_l is not None and ci_u is not None:
            line += f", 95% CI=[{ci_l:.4f}, {ci_u:.4f}]"
        if pct is not None:
            line += f" → multiplicative effect: {pct:.1f}% change in nuts/min"
        line += f"; direction: {direction}"
        if sig is True:
            line += " (statistically significant at α=0.05)"
        elif sig is False:
            line += " (not significant at α=0.05)"
        else:
            line += " (significance unknown)"
        desc_lines.append(line)

    description = " ".join(desc_lines)

    return {"object": out, "description": description}