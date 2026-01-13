def extract_final_answer(model_output):
    """
    Extract the estimated effect of 'female' from a fitted statsmodels Logit result
    or from a dict returned by the modeling function (with keys 'fit', 'odds_ratios', 'conf_odds').

    Returns a dict with:
      - "object": dict with numeric results (coef, se, p-value, odds ratio, 95% CI, significance)
      - "description": brief interpretation of those statistics in context
    """
    import numpy as np

    # Unwrap if model_output is the dict fallback
    if isinstance(model_output, dict) and 'fit' in model_output:
        res = model_output['fit']
        odds_ratios_attr = model_output.get('odds_ratios', None)
        conf_odds_attr = model_output.get('conf_odds', None)
    else:
        res = model_output
        odds_ratios_attr = getattr(res, 'odds_ratios', None)
        conf_odds_attr = getattr(res, 'conf_odds', None)

    # Helper to error if female not present
    def _raise_no_female():
        raise KeyError("The fitted model does not appear to contain a parameter named 'female'.")

    # Try to extract coefficient, se, p-value
    try:
        params = res.params
    except Exception:
        raise ValueError("Cannot access params on the provided model object.")

    # Determine index/key for female
    female_key = None
    try:
        if hasattr(params, 'index') and 'female' in params.index:
            female_key = 'female'
        elif hasattr(res, 'model') and hasattr(res.model, 'exog_names') and 'female' in res.model.exog_names:
            # params may be a numpy array; find corresponding position
            idx = res.model.exog_names.index('female')
            female_key = idx
        else:
            _raise_no_female()
    except Exception:
        _raise_no_female()

    # Extract numeric statistics with fallbacks
    try:
        coef = float(params[female_key])
    except Exception:
        coef = float(params.iloc[female_key])

    # standard error
    try:
        se = float(res.bse[female_key])
    except Exception:
        # fallback: compute from Hessian-inverse if available
        try:
            se = float(res.bse.iloc[female_key])
        except Exception:
            se = None

    # p-value
    try:
        p_value = float(res.pvalues[female_key])
    except Exception:
        try:
            p_value = float(res.pvalues.iloc[female_key])
        except Exception:
            p_value = None

    # odds ratio and CI: prefer provided odds_ratios/conf_odds if present, else compute
    try:
        if odds_ratios_attr is not None:
            # odds_ratios_attr may be a Series/DataFrame/ndarray
            try:
                odds_ratio = float(odds_ratios_attr[female_key])
            except Exception:
                odds_ratio = float(odds_ratios_attr.iloc[female_key])
        else:
            odds_ratio = float(np.exp(coef))
    except Exception:
        odds_ratio = None

    # confidence interval for odds ratio
    ci_low = ci_high = None
    try:
        if conf_odds_attr is not None:
            try:
                ci_low = float(conf_odds_attr.loc['female'][0]) if hasattr(conf_odds_attr, 'loc') else float(conf_odds_attr[female_key, 0])
                ci_high = float(conf_odds_attr.loc['female'][1]) if hasattr(conf_odds_attr, 'loc') else float(conf_odds_attr[female_key, 1])
            except Exception:
                # try positional
                try:
                    ci_row = conf_odds_attr.iloc[female_key]
                    ci_low, ci_high = float(ci_row[0]), float(ci_row[1])
                except Exception:
                    ci_low = ci_high = None
        else:
            # compute from conf_int on log-odds, then exponentiate
            conf = res.conf_int()
            if hasattr(conf, 'loc') and 'female' in conf.index:
                lower_log, upper_log = float(conf.loc['female'][0]), float(conf.loc['female'][1])
            else:
                # positional
                lower_log, upper_log = float(conf.iloc[female_key, 0]), float(conf.iloc[female_key, 1])
            ci_low, ci_high = float(np.exp(lower_log)), float(np.exp(upper_log))
    except Exception:
        ci_low = ci_high = None

    significance = None
    if p_value is not None:
        significance = (p_value < 0.05)

    result_object = {
        'coef_log_odds': coef,
        'std_error': se,
        'p_value': p_value,
        'odds_ratio': odds_ratio,
        'odds_ratio_ci_lower': ci_low,
        'odds_ratio_ci_upper': ci_high,
        'significant_at_0.05': significance
    }

    # Build short description interpreting direction and statistical evidence
    if odds_ratio is None:
        descr_nums = "Could not compute odds ratio / CI from the model output."
    else:
        direction = "higher" if odds_ratio > 1 else ("lower" if odds_ratio < 1 else "no change")
        descr_nums = (f"Estimated odds ratio for female vs male = {odds_ratio:.3g}"
                      + (f" (95% CI: {ci_low:.3g} to {ci_high:.3g})" if (ci_low is not None and ci_high is not None) else "")
                      + f"; log-odds coef = {coef:.3g}; p = {p_value:.3g}" if p_value is not None else "")

    if significance is True:
        conclusion = f"The effect is statistically significant at the 5% level: female applicants have {direction} odds of mortgage approval compared to males."
    elif significance is False:
        conclusion = "The effect is not statistically significant at the 5% level: no strong evidence that female applicants are treated differently than male applicants in approval odds."
    else:
        conclusion = "Significance could not be determined from the available output."

    description = f"{descr_nums} {conclusion}"

    return {
        "object": result_object,
        "description": description
    }