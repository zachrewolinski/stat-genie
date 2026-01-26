def extract_final_answer(model_output):
    """
    Extract statistics for the 'is_human' effect from a fitted statsmodels GLM/GLMResultsWrapper
    (optionally with clustered robust covariance). Returns a dict with:
      - "object": dict of extracted numeric results (coef, se, p, CI, OR, OR_CI, significance, conclusion)
      - "description": brief plain-language interpretation of the result in the context of the task.
    
    The function is defensive about the exact parameter name (e.g. 'is_human' vs. alternatives).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to get parameter name for is_human (handles possible naming variations)
    candidate_names = ['is_human', 'is_human[T.True]', 'C(is_human)[T.1]']
    params_index = None
    try:
        params_index = list(res.params.index)
    except Exception:
        # If params not a Series with index, try to coerce to pandas Index
        try:
            params_index = list(pd.Index(res.params).tolist())
        except Exception:
            params_index = None

    param_name = None
    if params_index is not None:
        for n in candidate_names:
            if n in params_index:
                param_name = n
                break
        # fallback: try to find any name that contains 'is_human'
        if param_name is None:
            for n in params_index:
                if 'is_human' in str(n):
                    param_name = n
                    break

    if param_name is None:
        raise KeyError("Could not find a parameter name for 'is_human' in the model results. "
                       "Available parameter names: %s" % (params_index,))

    # Extract coefficient, se, p-value
    try:
        coef = float(res.params[param_name])
    except Exception:
        coef = float(res.params.loc[param_name])

    # Use robust/clustered SEs if available (res.bse should reflect that if res is robustcov_results)
    try:
        se = float(res.bse[param_name])
    except Exception:
        se = float(res.bse.loc[param_name])

    try:
        pvalue = float(res.pvalues[param_name])
    except Exception:
        pvalue = float(res.pvalues.loc[param_name])

    # Confidence interval
    try:
        ci_df = res.conf_int()
        # conf_int returns a DataFrame-like or ndarray. Prefer indexing by name.
        try:
            ci_low, ci_high = float(ci_df.loc[param_name, 0]), float(ci_df.loc[param_name, 1])
        except Exception:
            # ci_df might be a numpy array; find parameter position
            if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                names = list(res.model.exog_names)
                pos = names.index(param_name)
                ci_low, ci_high = float(ci_df[pos, 0]), float(ci_df[pos, 1])
            else:
                # fallback: use param ordering in params
                pos = list(res.params.index).index(param_name)
                ci_low, ci_high = float(ci_df[pos, 0]), float(ci_df[pos, 1])
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    # Odds ratio and CI on odds ratio scale
    or_coef = float(np.exp(coef))
    or_ci_low = float(np.exp(ci_low)) if not np.isnan(ci_low) else np.nan
    or_ci_high = float(np.exp(ci_high)) if not np.isnan(ci_high) else np.nan

    # Significance at alpha = 0.05
    significant = (pvalue < 0.05)

    # Direction: positive coef => higher log-odds (and odds) of AMTL in humans
    if np.isnan(coef):
        conclusion = "Unable to determine: coefficient is NaN."
    else:
        if significant:
            if coef > 0:
                conclusion = ("Yes — after adjusting for age, sex, stdev_age, and tooth class, "
                              "modern humans have significantly higher odds of antemortem tooth loss (AMTL) "
                              "compared to non-human primates (coef={:+.3f}, p={:.3g}).").format(coef, pvalue)
            else:
                conclusion = ("No — after adjusting for covariates, modern humans have significantly lower odds of AMTL "
                              "compared to non-human primates (coef={:+.3f}, p={:.3g}).").format(coef, pvalue)
        else:
            # Not statistically significant
            if coef > 0:
                conclusion = ("No strong evidence — point estimate suggests higher AMTL in modern humans (coef={:+.3f}), "
                              "but this difference is not statistically significant (p={:.3g}).").format(coef, pvalue)
            elif coef < 0:
                conclusion = ("No strong evidence — point estimate suggests lower AMTL in modern humans (coef={:+.3f}), "
                              "but this difference is not statistically significant (p={:.3g}).").format(coef, pvalue)
            else:
                conclusion = ("No evidence of a difference in AMTL between modern humans and non-human primates "
                              "(coef=0.0, p={:.3g}).").format(pvalue)

    result_object = {
        'param_name': param_name,
        'coef_log_odds': coef,
        'std_error': se,
        'p_value': pvalue,
        'conf_int_log_odds': (ci_low, ci_high),
        'odds_ratio': or_coef,
        'odds_ratio_ci': (or_ci_low, or_ci_high),
        'significant_at_0.05': bool(significant),
        'conclusion': conclusion
    }

    description = (
        "Extracted the coefficient for the 'is_human' indicator from the fitted binomial (logit) GLM. "
        "coef_log_odds is the estimated difference in log-odds of AMTL for modern humans vs non-human primates, "
        "odds_ratio is exp(coef). conf_int_log_odds and odds_ratio_ci are 95% confidence intervals. "
        "The 'conclusion' field gives a plain-language yes/no/uncertain answer about whether modern humans "
        "have higher AMTL after adjusting for age, sex (prob_male), age uncertainty (stdev_age), and tooth class "
        "(clustered SEs by specimen assumed to be applied in the supplied model object)."
    )

    return {'object': result_object, 'description': description}