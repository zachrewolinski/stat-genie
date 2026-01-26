def extract_final_answer(model_output):
    """
    Extracts the estimated effect of IsHomo from a fitted binomial GLM model output.

    Expects model_output to be the dict returned by the modeling function:
      {'glm_result': res, 'glm_clustered_result': clustered_res, ...}

    Returns a dictionary with:
      - "object": dict of numeric results for the IsHomo coefficient (coef, SE, p, CI, odds ratio, OR CI)
      - "description": short plain-English interpretation of whether modern humans (IsHomo=1)
                       have higher AMTL after controlling for covariates.

    The function prefers cluster-robust results if present (glm_clustered_result),
    otherwise uses the naive glm_result.
    """
    import math

    # Preferred use of clustered result if available
    res = None
    used_clustered = False
    if isinstance(model_output, dict) and model_output.get('glm_clustered_result') is not None:
        res = model_output.get('glm_clustered_result')
        used_clustered = True
    elif isinstance(model_output, dict) and model_output.get('glm_result') is not None:
        res = model_output.get('glm_result')
        used_clustered = False
    else:
        raise ValueError("model_output must be a dict containing 'glm_result' or 'glm_clustered_result'.")

    # Parameter name expected in the model
    param = 'IsHomo'

    # Ensure parameter exists
    try:
        params = res.params
    except Exception:
        raise ValueError("The provided result object does not expose .params. Provide a statsmodels results object.")

    if param not in params.index:
        raise KeyError(f"Parameter '{param}' not found in the model results. Available parameters: {list(params.index)}")

    # Extract statistics, converting to native Python floats
    coef = float(res.params[param])
    # bse and pvalues should exist for statsmodels results (including robust results)
    try:
        se = float(res.bse[param])
    except Exception:
        # fallback: compute se from covariance if available
        try:
            cov = res.cov_params()
            se = float((cov.loc[param, param]) ** 0.5)
        except Exception:
            se = None

    try:
        pvalue = float(res.pvalues[param])
    except Exception:
        pvalue = None

    # Confidence interval for the coefficient (log-odds scale)
    try:
        ci = res.conf_int().loc[param]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # fallback if conf_int() returned an array without index
        try:
            ci_matrix = res.conf_int()
            # find row by position
            idx = list(res.params.index).index(param)
            ci_lower = float(ci_matrix[idx, 0])
            ci_upper = float(ci_matrix[idx, 1])
        except Exception:
            ci_lower = None
            ci_upper = None

    # Convert to odds ratio scale
    try:
        odds_ratio = float(math.exp(coef))
        or_ci_lower = float(math.exp(ci_lower)) if ci_lower is not None else None
        or_ci_upper = float(math.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        odds_ratio = None
        or_ci_lower = None
        or_ci_upper = None

    # Determine significance at alpha = 0.05 if p-value available
    significant = None
    if pvalue is not None:
        significant = (pvalue < 0.05)

    # Build the object to return
    result_obj = {
        'parameter': param,
        'used_clustered_SEs': bool(used_clustered),
        'coef_log_odds': coef,
        'se': se,
        'p_value': pvalue,
        'coef_ci_95': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'odds_ratio_ci_95': [or_ci_lower, or_ci_upper],
        'significant_at_0.05': significant
    }

    # Human-readable interpretation
    if pvalue is None:
        interp = (
            "Could not retrieve a p-value for IsHomo; statistics extracted but significance unknown. "
            "Coefficient is presented on the log-odds scale; exponentiate to interpret as an odds ratio."
        )
    else:
        if significant:
            if coef > 0:
                interp = (
                    f"The coefficient for IsHomo is positive (log-odds = {coef:.3f}, SE = {se:.3f}, p = {pvalue:.3g}), "
                    f"corresponding to an odds ratio of {odds_ratio:.3f} (95% CI: {or_ci_lower:.3f}–{or_ci_upper:.3f}). "
                    "This indicates that, after adjusting for age, sex, and tooth class, modern humans (Homo sapiens) have "
                    "statistically significantly higher odds of antemortem tooth loss compared to the non-human primates included."
                )
            else:
                interp = (
                    f"The coefficient for IsHomo is negative (log-odds = {coef:.3f}, SE = {se:.3f}, p = {pvalue:.3g}), "
                    f"corresponding to an odds ratio of {odds_ratio:.3f} (95% CI: {or_ci_lower:.3f}–{or_ci_upper:.3f}). "
                    "This indicates that, after adjusting for age, sex, and tooth class, modern humans (Homo sapiens) have "
                    "statistically significantly lower odds of antemortem tooth loss compared to the non-human primates included."
                )
        else:
            interp = (
                f"The coefficient for IsHomo is {coef:.3f} (SE = {se:.3f}, p = {pvalue:.3g}), corresponding to an odds ratio of "
                f"{odds_ratio:.3f} (95% CI: {or_ci_lower:.3f}–{or_ci_upper:.3f}). The effect is not statistically significant at alpha=0.05, "
                "so there is no strong evidence that modern humans differ from the sampled non-human primates in odds of AMTL after adjustment."
            )

    return {
        "object": result_obj,
        "description": interp
    }