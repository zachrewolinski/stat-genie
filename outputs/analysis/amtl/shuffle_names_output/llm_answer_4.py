def extract_final_answer(model_output):
    """
    Extract statistics for the primary comparison (Homo sapiens vs non-human primates)
    from the model output returned by the provided `model()` function.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, z, p, 95% CI, odds ratio + CI, conclusion, model_used)
      - "description": a brief human-readable interpretation of the results

    The function will:
      - Prefer clustered-robust results if available (primary_result_clustered).
      - Fall back to primary_result if necessary.
      - Attempt to find the 'is_human' parameter; if not found, try to locate a parameter
        that references Homo (e.g., a genus categorical parameter).
    """
    import numpy as np

    # Helper to safely get attributes from results
    def safe_attr(res, name):
        return getattr(res, name, None)

    # Choose the result object: prefer clustered if present
    res = None
    if isinstance(model_output, dict):
        # try clustered first
        res = model_output.get('primary_result_clustered') or model_output.get('primary_result')
        # fallback to any object that looks like a result
        if res is None:
            # try other keys
            for k in ('primary_result', 'genus_result_clustered', 'genus_result'):
                if k in model_output and model_output[k] is not None:
                    res = model_output[k]
                    break
    else:
        res = model_output

    if res is None:
        return {
            "object": None,
            "description": "No model result object found in model_output."
        }

    # Get parameter names and values
    params = safe_attr(res, 'params')
    pvalues = safe_attr(res, 'pvalues')
    bse = safe_attr(res, 'bse')
    conf_int = None
    try:
        conf_int = res.conf_int()
    except Exception:
        # some objects might not implement conf_int; leave as None
        conf_int = None

    # Ensure params is a pandas Series or similar mapping; otherwise try to get names from model
    param_name = None
    if params is None:
        return {
            "object": None,
            "description": "The result object has no 'params' attribute."
        }

    # Find the parameter corresponding to is_human.
    # Common name in the primary model: 'is_human'
    possible_names = []
    try:
        # try index if pandas Series
        idx = list(params.index)
        possible_names = idx
    except Exception:
        # if params is numpy array, get names from model.exog_names
        exog_names = safe_attr(getattr(res, 'model', None), 'exog_names')
        if exog_names:
            possible_names = list(exog_names)

    # Preferred exact match
    if 'is_human' in possible_names:
        param_name = 'is_human'
    else:
        # fallback: look for parameter name that contains 'Homo' or 'homo' or 'human'
        for nm in possible_names:
            if isinstance(nm, str) and ('Homo' in nm or 'homo' in nm or 'human' in nm):
                param_name = nm
                break
        # final fallback: try any name that looks like a binary indicator (exact match of common alternatives)
        if param_name is None:
            for alt in ['C(genus)[T.Homo sapiens]', 'C(genus)[T.Homo sapiens]']:
                if alt in possible_names:
                    param_name = alt
                    break

    if param_name is None:
        return {
            "object": None,
            "description": "Could not find a parameter corresponding to the human indicator ('is_human' or similar) in the model parameters."
        }

    # Extract numeric values
    coef = float(params[param_name])
    se = float(bse[param_name]) if (bse is not None and param_name in bse.index) else (float(params[param_name]) * np.nan)
    pval = float(pvalues[param_name]) if (pvalues is not None and param_name in pvalues.index) else None

    # z / t value
    z_val = coef / se if (se is not None and not np.isnan(se) and se != 0.0) else None

    # Confidence interval on log-odds scale
    ci_low, ci_high = (None, None)
    if conf_int is not None:
        try:
            # conf_int may be a DataFrame or ndarray; handle both
            if hasattr(conf_int, 'loc') and param_name in conf_int.index:
                ci_low = float(conf_int.loc[param_name, 0])
                ci_high = float(conf_int.loc[param_name, 1])
            else:
                # try to locate by position
                # assume same ordering as params
                pos = possible_names.index(param_name) if param_name in possible_names else None
                if pos is not None:
                    ci_low = float(conf_int[pos, 0])
                    ci_high = float(conf_int[pos, 1])
        except Exception:
            ci_low, ci_high = (None, None)

    # Convert to odds ratio and CI
    try:
        odds_ratio = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
    except Exception:
        odds_ratio = None
        or_ci_low = None
        or_ci_high = None

    # Conclusion: do humans have higher AMTL?
    # A positive coef => higher log-odds (hence higher probability) in humans vs baseline.
    significance = None
    if pval is not None:
        significance = (pval < 0.05)
    else:
        significance = None

    if coef > 0:
        direction = "higher"
    elif coef < 0:
        direction = "lower"
    else:
        direction = "no difference (coef = 0)"

    if significance is True:
        conclusion_text = f"Statistically significant ({pval:.3g}) evidence that modern humans have {direction} AMTL compared to non-human primates, controlling for age, sex, tooth class, and age uncertainty."
    elif significance is False:
        conclusion_text = f"No statistically significant evidence (p = {pval:.3g}) that modern humans differ from non-human primates in AMTL after controlling for covariates; the point estimate indicates {direction} AMTL in humans but it is not statistically significant."
    else:
        conclusion_text = f"Unable to determine statistical significance (p-value unavailable). Point estimate indicates {direction} AMTL in humans."

    # Build output object
    stats = {
        "parameter_name": param_name,
        "coef_log_odds": coef,
        "se": se,
        "z_or_t": z_val,
        "p_value": pval,
        "conf_int_log_odds": (ci_low, ci_high),
        "odds_ratio": odds_ratio,
        "odds_ratio_conf_int": (or_ci_low, or_ci_high),
        "humans_higher": True if (coef > 0 and significance is True) else (False if (coef <= 0 and significance is True) else None),
        "significant": significance,
        "model_used": "primary_result_clustered" if ('primary_result_clustered' in model_output and model_output.get('primary_result_clustered') is not None) else "primary_result"
    }

    # Create a short description
    description = (
        f"Parameter '{param_name}' from the primary binomial GLM: coef (log-odds) = {coef:.4f}, SE = {se:.4f}, "
        f"z = {z_val:.3f} , p = {pval:.3g}. 95% CI (log-odds) = [{ci_low:.4f}, {ci_high:.4f}]. "
        f"Odds ratio = {odds_ratio:.3f} (95% CI = [{or_ci_low:.3f}, {or_ci_high:.3f}]). "
        + conclusion_text
    )

    return {
        "object": stats,
        "description": description
    }