def extract_final_answer(model_output):
    """
    Extracts the effect of 'is_human' from a fitted statsmodels GLMResults-like object
    (e.g., statsmodels.genmod.generalized_linear_model.GLMResultsWrapper).
    
    Returns a dictionary with:
      - "object": a dict containing coefficient (log-odds), standard error,
                  p-value, 95% CI for coefficient, odds ratio, and 95% CI for OR.
      - "description": a short interpretation of whether modern humans (is_human=1)
                       have higher AMTL than non-human primates after controlling
                       for age, sex, and tooth class.
    """
    import numpy as _np

    res = model_output

    # Try to get parameter name for is_human (cover small naming variations)
    try:
        params_index = list(res.params.index)
    except Exception:
        # If the results object doesn't behave like a statsmodels results object
        raise ValueError("model_output does not appear to be a statsmodels results object with fitted parameters.")

    # Find the parameter name that corresponds to is_human
    target_names = [name for name in params_index if name == 'is_human' or 'is_human' in name]
    if not target_names:
        raise ValueError("Could not find a parameter named 'is_human' in the model results. Available params: {}".format(params_index))
    # If multiple matches, prefer exact match
    if 'is_human' in target_names:
        target = 'is_human'
    else:
        target = target_names[0]

    # Extract statistics
    coef = float(res.params[target])
    # Standard error: try res.bse, fallback to cov_params
    try:
        se = float(res.bse[target])
    except Exception:
        try:
            cov = res.cov_params()
            se = float(_np.sqrt(cov.loc[target, target]))
        except Exception:
            se = None

    # p-value
    try:
        pvalue = float(res.pvalues[target])
    except Exception:
        pvalue = None

    # 95% CI for coefficient
    try:
        ci = res.conf_int().loc[target].tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Odds ratio and CI on odds ratio scale
    try:
        or_coef = float(_np.exp(coef))
        or_ci_lower = float(_np.exp(ci_lower)) if ci_lower is not None else None
        or_ci_upper = float(_np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        or_coef = or_ci_lower = or_ci_upper = None

    # Simple interpretation using alpha = 0.05
    if pvalue is not None:
        if pvalue < 0.05:
            if coef > 0:
                conclusion = ("Statistically significant positive association: specimens coded as Homo sapiens "
                              "have higher odds of AMTL than non-human primates after adjusting for age, sex, "
                              "and tooth class (p = {:.3g}).").format(pvalue)
            else:
                conclusion = ("Statistically significant negative association: specimens coded as Homo sapiens "
                              "have lower odds of AMTL than non-human primates after adjusting for controls (p = {:.3g}).").format(pvalue)
        else:
            conclusion = ("No statistically significant difference in AMTL between Homo sapiens and non-human primates "
                          "after adjusting for age, sex, and tooth class (p = {:.3g}).").format(pvalue)
    else:
        conclusion = "Could not determine statistical significance (p-value unavailable)."

    # Build the object to return
    result_object = {
        "parameter_name": target,
        "coef_log_odds": coef,
        "std_error": se,
        "p_value": pvalue,
        "coef_95CI": [ci_lower, ci_upper],
        "odds_ratio": or_coef,
        "odds_ratio_95CI": [or_ci_lower, or_ci_upper],
    }

    description = (
        "Effect of 'is_human' on probability of AMTL (binomial GLM, logit link), adjusted for age_at_death, "
        "sex_male_prob, and tooth_class. Interpretation: {} Effect size: OR = {:.3g} (95% CI: {:.3g} to {:.3g})."
        .format(conclusion,
                or_coef if or_coef is not None else float('nan'),
                or_ci_lower if or_ci_lower is not None else float('nan'),
                or_ci_upper if or_ci_upper is not None else float('nan'))
    )

    return {"object": result_object, "description": description}