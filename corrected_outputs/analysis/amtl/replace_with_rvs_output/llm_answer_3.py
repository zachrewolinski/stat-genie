def extract_final_answer(model_output):
    """
    Extracts the effect of 'is_human' from a fitted statsmodels GLM (binomial) result.

    Returns a dictionary with keys:
      - "object": a dict of numeric results (coefficient, SE, test stat, p-value,
                  95% CI on log-odds scale, odds-ratio and its 95% CI, verdict)
      - "description": a human-readable interpretation answering whether modern
                       humans have higher AMTL after controlling for covariates.
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output must be a dict with keys 'model' and optionally 'clustered'."
        }

    # Prefer cluster-robust results if present, otherwise use the raw model
    res = None
    if model_output.get('clustered') is not None:
        res = model_output['clustered']
    elif model_output.get('model') is not None:
        res = model_output['model']
    else:
        return {
            "object": None,
            "description": "No fit object found in model_output (expected key 'model' or 'clustered')."
        }

    # Ensure we have parameter info
    try:
        params = res.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not read parameters from result object: {e}"
        }

    # Find the parameter name corresponding to the is_human predictor
    # This is robust to small naming differences (e.g., if patsy produced a decorated name).
    param_names = list(params.index)
    matches = [n for n in param_names if 'is_human' in n]
    if len(matches) == 0:
        # Try exact 'is_human' as fallback
        if 'is_human' in param_names:
            matches = ['is_human']
    if len(matches) == 0:
        return {
            "object": None,
            "description": "Could not find a parameter name containing 'is_human' in the model parameters."
        }

    param = matches[0]

    # Extract statistics, using best available attributes
    try:
        coef = float(params[param])
    except Exception:
        return {
            "object": None,
            "description": f"Parameter '{param}' found but could not extract coefficient."
        }

    # standard error
    se = None
    if hasattr(res, 'bse') and param in res.bse.index:
        se = float(res.bse[param])

    # test statistic: prefer tvalues, else zvalues, else None
    test_stat = None
    if hasattr(res, 'tvalues') and param in res.tvalues.index:
        test_stat = float(res.tvalues[param])
        test_name = 'z'  # statsmodels uses 'tvalues' name for GLM but it's a z-stat in GLM context
    elif hasattr(res, 'zvalues') and param in res.zvalues.index:
        test_stat = float(res.zvalues[param])
        test_name = 'z'
    else:
        test_name = None

    # p-value
    pvalue = None
    if hasattr(res, 'pvalues') and param in res.pvalues.index:
        pvalue = float(res.pvalues[param])

    # confidence interval on coefficient (log-odds) scale
    ci_low = ci_high = None
    try:
        ci = res.conf_int()
        # conf_int may be a DataFrame or ndarray
        if hasattr(ci, 'loc') and param in ci.index:
            ci_low = float(ci.loc[param, 0])
            ci_high = float(ci.loc[param, 1])
        else:
            # assume same ordering as params
            idx = param_names.index(param)
            ci_low = float(ci[idx, 0]) if hasattr(ci, '__getitem__') else None
            ci_high = float(ci[idx, 1]) if hasattr(ci, '__getitem__') else None
    except Exception:
        ci_low = ci_high = None

    # Convert to odds ratio scale
    or_est = None
    or_ci = (None, None)
    try:
        or_est = float(np.exp(coef))
        if (ci_low is not None) and (ci_high is not None):
            or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
    except Exception:
        or_est = None

    # Make a simple verdict at alpha = 0.05
    verdict = "insufficient evidence"
    if pvalue is not None:
        if pvalue < 0.05:
            if coef > 0:
                verdict = "modern humans have a significantly higher AMTL frequency"
            elif coef < 0:
                verdict = "modern humans have a significantly lower AMTL frequency"
            else:
                verdict = "no difference in AMTL frequency"
        else:
            verdict = "no statistically significant difference in AMTL frequency"
    else:
        verdict = "could not determine statistical significance (p-value missing)"

    # Prepare numeric object to return
    numeric_result = {
        "param_name": param,
        "coef_log_odds": coef,
        "se": se,
        f"{test_name}_stat" if test_name is not None else "test_stat": test_stat,
        "p_value": pvalue,
        "ci_log_odds": (ci_low, ci_high),
        "odds_ratio": or_est,
        "odds_ratio_ci": or_ci,
        "verdict": verdict
    }

    # Build human-readable description
    # Round numbers for display if available
    def r(x, d=3):
        return None if x is None else round(x, d)

    desc_lines = []
    desc_lines.append("Model type: Binomial GLM (logit link assumed).")
    desc_lines.append(f"Parameter examined: '{param}' (effect of being modern human).")
    desc_lines.append(
        "Estimated coefficient (log-odds): "
        f"{r(coef)}" + (f" (SE = {r(se)})" if se is not None else "")
    )
    if test_stat is not None and pvalue is not None:
        desc_lines.append(f"{test_name}-stat = {r(test_stat)}, p = {r(pvalue,4)}.")
    elif pvalue is not None:
        desc_lines.append(f"p = {r(pvalue,4)}.")
    if (ci_low is not None) and (ci_high is not None):
        desc_lines.append(f"95% CI (log-odds): [{r(ci_low)}, {r(ci_high)}].")
    if or_est is not None:
        desc_lines.append(f"Odds ratio = {r(or_est)}; 95% CI = [{r(or_ci[0])}, {r(or_ci[1])}]")
    desc_lines.append(f"Interpretation/verdict: {verdict} (alpha = 0.05).")
    description = " ".join(desc_lines)

    return {
        "object": numeric_result,
        "description": description
    }