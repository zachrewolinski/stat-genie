def extract_final_answer(model_output):
    """
    Extracts the IsHuman effect from the model_output and returns a human-readable
    conclusion about whether modern humans have higher AMTL frequencies than
    non-human primates (controlling for covariates).

    Returns a dict with:
      - "object": dict of numeric results (coef, se, pvalue, odds_ratio, CI95, significant, direction)
      - "description": short plain-language interpretation answering the yes/no question.
    """
    import math
    import numpy as np

    # Helper to safely pull from model_output dict or from an embedded statsmodels result
    def safe_get(key):
        if key in model_output:
            return model_output[key]
        # try to extract from glm_clustered/results object if present
        for candidate in ('glm_clustered', 'glm_raw', 'glm'):
            res = model_output.get(candidate)
            if res is None:
                continue
            try:
                if key == 'IsHuman_coef':
                    return float(res.params.get('IsHuman'))
                if key == 'IsHuman_se':
                    # clustered bse may be in bse attribute
                    return float(res.bse.get('IsHuman'))
                if key == 'IsHuman_pvalue':
                    return float(res.pvalues.get('IsHuman'))
                if key == 'IsHuman_odds_ratio_CI95':
                    ci = res.conf_int().loc['IsHuman']
                    return (float(np.exp(ci[0])), float(np.exp(ci[1])))
                if key == 'IsHuman_odds_ratio':
                    coef = float(res.params.get('IsHuman'))
                    return float(math.exp(coef))
            except Exception:
                continue
        return None

    # Extract values with fallbacks
    coef = safe_get('IsHuman_coef')
    se = safe_get('IsHuman_se')
    pval = safe_get('IsHuman_pvalue')
    oratio = safe_get('IsHuman_odds_ratio')
    ci95 = safe_get('IsHuman_odds_ratio_CI95')

    # Ensure numeric types where possible
    try:
        coef_f = float(coef) if coef is not None else None
    except Exception:
        coef_f = None
    try:
        se_f = float(se) if se is not None else None
    except Exception:
        se_f = None
    try:
        pval_f = float(pval) if pval is not None else None
    except Exception:
        pval_f = None
    try:
        or_f = float(oratio) if oratio is not None else (math.exp(coef_f) if coef_f is not None else None)
    except Exception:
        or_f = None
    try:
        if ci95 is not None:
            ci_low, ci_high = float(ci95[0]), float(ci95[1])
        else:
            ci_low = ci_high = None
    except Exception:
        ci_low = ci_high = None

    # Determine significance and direction
    alpha = 0.05
    if pval_f is None:
        significant = None
    else:
        significant = (pval_f < alpha)

    if coef_f is None:
        direction = "unknown"
    else:
        if significant is True:
            direction = "higher" if coef_f > 0 else "lower"
        elif significant is False:
            direction = "no_evidence_of_difference"
        else:
            direction = "unknown"

    # Build concise interpretation string answering the yes/no question
    if pval_f is None or coef_f is None:
        description = ("Could not reliably extract the IsHuman effect from the model output. "
                       "Required statistics (coefficient/p-value) are missing.")
    else:
        # coefficient is on log-odds scale: positive -> higher odds for humans
        if significant:
            if coef_f > 0:
                description = (f"Yes — the IsHuman coefficient is positive and statistically significant "
                               f"(coef={coef_f:.4f}, p={pval_f:.3g}). This indicates modern humans have "
                               f"higher odds of AMTL than non-human primates (OR={or_f:.3g}, 95% CI=[{ci_low:.3g}, {ci_high:.3g}]).")
            else:
                description = (f"No — the IsHuman coefficient is negative and statistically significant "
                               f"(coef={coef_f:.4f}, p={pval_f:.3g}), indicating modern humans have lower odds of AMTL "
                               f"than non-human primates (OR={or_f:.3g}, 95% CI=[{ci_low:.3g}, {ci_high:.3g}]).")
        else:
            # not significant
            description = (f"No evidence that modern humans have higher AMTL after accounting for covariates: "
                           f"IsHuman coef = {coef_f:.4f} (log-odds), p = {pval_f:.3g}. "
                           f"Estimated OR = {or_f:.3g} with 95% CI = [{ci_low:.3g}, {ci_high:.3g}]. "
                           "The effect is not statistically significant and the confidence interval includes values below and above 1, "
                           "so the direction and magnitude are uncertain.")

    # Compose object to return (numeric results + interpretation flags)
    result_object = {
        'coef_log_odds': coef_f,
        'se': se_f,
        'p_value': pval_f,
        'odds_ratio': or_f,
        'odds_ratio_CI95': (ci_low, ci_high),
        'significant_at_0.05': significant,
        'direction_interpretation': direction
    }

    return {
        'object': result_object,
        'description': description
    }