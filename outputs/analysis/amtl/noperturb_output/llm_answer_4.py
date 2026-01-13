def extract_final_answer(model_output):
    """
    Extracts the coefficient, uncertainty, p-value, confidence interval, and
    odds-ratio for the 'is_homo' predictor from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with keys:
      - "object": dict with numeric results
      - "description": textual interpretation of whether Homo sapiens show higher AMTL
    """
    import numpy as np

    res = model_output

    # Basic checks and attribute extraction
    try:
        params = res.params
        pvalues = res.pvalues
        bse = res.bse
        conf = res.conf_int()
    except Exception as e:
        raise ValueError(f"model_output does not appear to be a statsmodels results object or is missing attributes: {e}")

    if 'is_homo' not in params.index:
        raise KeyError("Coefficient 'is_homo' not found in model results. Check model specification / variable names.")

    # Extract coefficient, SE, p-value
    coef = float(params['is_homo'])
    se = float(bse['is_homo']) if 'is_homo' in bse.index else None
    pval = float(pvalues['is_homo'])

    # Extract confidence interval robustly (conf may be DataFrame or ndarray)
    try:
        # DataFrame-like: conf.loc['is_homo'] -> array-like [lower, upper]
        ci_row = conf.loc['is_homo'].values
    except Exception:
        # conf might be an ndarray in which case find index of 'is_homo' in params
        try:
            idx = list(params.index).index('is_homo')
            ci_row = np.asarray(conf)[idx]
        except Exception as e:
            raise ValueError(f"Unable to extract confidence interval for 'is_homo': {e}")

    ci_lower = float(ci_row[0])
    ci_upper = float(ci_row[1])

    # Odds ratio and its CI (since GLM link is logit)
    odds_ratio = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower))
    or_ci_upper = float(np.exp(ci_upper))

    # Try to extract z/t value if available
    stat_name = None
    stat_value = None
    if hasattr(res, 'tvalues') and 'is_homo' in getattr(res, 'tvalues').index:
        stat_name = 'tvalue'
        stat_value = float(res.tvalues['is_homo'])
    elif hasattr(res, 'zvalues') and 'is_homo' in getattr(res, 'zvalues').index:
        stat_name = 'zvalue'
        stat_value = float(res.zvalues['is_homo'])

    # Sample size metadata if attached by the modeling function
    n_obs = None
    n_specimens = None
    try:
        md = getattr(res, 'model_data', None)
        if isinstance(md, dict):
            n_obs = md.get('n_obs', None)
            n_specimens = md.get('n_specimens', None)
    except Exception:
        pass

    # Formulate a concise conclusion based on coefficient sign and p-value
    alpha = 0.05
    if pval < alpha:
        if coef > 0:
            conclusion = (
                "Yes: The coefficient for 'is_homo' is positive and statistically significant "
                f"(coef = {coef:.4f}, p = {pval:.3g}), indicating higher odds of antemortem "
                "tooth loss in Homo sapiens compared to the reference non-human primates, "
                "after adjusting for age, sex probability, and tooth class."
            )
        else:
            conclusion = (
                "No: The coefficient for 'is_homo' is negative and statistically significant "
                f"(coef = {coef:.4f}, p = {pval:.3g}), indicating lower odds of antemortem "
                "tooth loss in Homo sapiens compared to the reference non-human primates."
            )
    else:
        if coef > 0:
            conclusion = (
                "No strong evidence: The coefficient for 'is_homo' is positive but not statistically significant "
                f"(coef = {coef:.4f}, p = {pval:.3g}). We cannot conclude that Homo sapiens have higher AMTL "
                "after accounting for the covariates."
            )
        else:
            conclusion = (
                "No strong evidence: The coefficient for 'is_homo' is negative (or non-positive) and not statistically significant "
                f"(coef = {coef:.4f}, p = {pval:.3g}). We cannot conclude that Homo sapiens have higher AMTL."
            )

    result_object = {
        'predictor': 'is_homo',
        'coef': coef,
        'std_error': se,
        'statistic_name': stat_name,
        'statistic_value': stat_value,
        'p_value': pval,
        'conf_int': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'odds_ratio_conf_int': [or_ci_lower, or_ci_upper],
        'n_obs': n_obs,
        'n_specimens': n_specimens
    }

    description = (
        "Extracted statistics for the 'is_homo' coefficient from the fitted binomial GLM. "
        "coef is on the log-odds scale; odds_ratio is exp(coef). The confidence intervals are 95% by default. "
        + conclusion
    )

    return {"object": result_object, "description": description}