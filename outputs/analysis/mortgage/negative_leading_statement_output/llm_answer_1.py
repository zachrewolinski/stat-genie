def extract_final_answer(model_output):
    """
    Extracts statistics for the primary independent variable 'female' from the model output.

    Returns a dictionary with keys:
      - "object": dict with numeric results (log-odds coef, odds ratio, 95% CI for OR,
                  p-value, percent change in odds, significant flag)
      - "description": brief interpretation in context

    The function handles either:
      - A dict containing keys like 'model', 'odds_ratios', 'conf_int_odds', 'pvalues'
      - Or a statsmodels result object (from which params, pvalues, conf_int can be computed)
    """
    import numpy as np
    import pandas as pd

    # Helper to get variable index/label for 'female'
    def _find_index(series_like, name='female'):
        # series_like may be pandas Series with index, or dict-like
        try:
            idx = list(series_like.index)
        except Exception:
            # try dict keys
            try:
                idx = list(series_like.keys())
            except Exception:
                idx = []
        # exact match
        if name in idx:
            return name
        # case-insensitive or contains match
        name_lower = name.lower()
        for key in idx:
            try:
                if str(key).lower() == name_lower or name_lower in str(key).lower():
                    return key
            except Exception:
                continue
        # not found
        return None

    # Normalize to get params, pvalues, odds_ratios, conf_int_odds
    params = None
    pvalues = None
    odds_ratios = None
    conf_int_odds = None

    # If model_output is a dict with precomputed items:
    if isinstance(model_output, dict):
        # Try direct extraction
        if 'odds_ratios' in model_output:
            odds_ratios = model_output['odds_ratios']
        if 'pvalues' in model_output:
            pvalues = model_output['pvalues']
        if 'conf_int_odds' in model_output:
            conf_int_odds = model_output['conf_int_odds']
        if 'model' in model_output and (params is None or pvalues is None):
            res = model_output['model']
            # attempt to extract params and pvalues from res if missing
            try:
                if params is None:
                    params = res.params
                if pvalues is None:
                    pvalues = res.pvalues
                if odds_ratios is None:
                    odds_ratios = np.exp(res.params)
                if conf_int_odds is None:
                    conf = res.conf_int()
                    conf_int_odds = np.exp(conf)
            except Exception:
                # fall through; we'll try other keys below
                pass

    # If model_output is a statsmodels result directly
    if odds_ratios is None or pvalues is None or conf_int_odds is None:
        # try treating model_output as statsmodels results object
        try:
            res = model_output
            params = getattr(res, 'params', params)
            pvalues = getattr(res, 'pvalues', pvalues)
            if odds_ratios is None and params is not None:
                odds_ratios = np.exp(params)
            if conf_int_odds is None and hasattr(res, 'conf_int'):
                conf = res.conf_int()
                conf_int_odds = np.exp(conf)
        except Exception:
            pass

    # Final fallback: try to compute odds_ratios/conf from params if available
    if odds_ratios is None and params is not None:
        odds_ratios = np.exp(params)
    if conf_int_odds is None and params is not None and hasattr(model_output, 'conf_int'):
        try:
            conf = model_output.conf_int()
            conf_int_odds = np.exp(conf)
        except Exception:
            conf_int_odds = None

    # At this point, require pvalues and odds_ratios at least
    if pvalues is None or odds_ratios is None:
        raise ValueError("Could not find necessary statistics (p-values or odds ratios) in model_output.")

    # Identify label/key for 'female'
    female_key = _find_index(pvalues, 'female')
    if female_key is None:
        # try in odds_ratios
        female_key = _find_index(odds_ratios, 'female')
    if female_key is None:
        raise KeyError("Could not find variable 'female' in model output indices/keys.")

    # Extract numeric values, converting to Python floats
    try:
        coef_logodds = float(params[female_key]) if params is not None else None
    except Exception:
        # If params missing, compute from odds ratio via log
        try:
            coef_logodds = float(np.log(odds_ratios[female_key]))
        except Exception:
            coef_logodds = None

    pval = float(pvalues[female_key])
    oratio = float(odds_ratios[female_key])

    # Confidence interval for odds ratio
    ci_low = ci_high = None
    if conf_int_odds is not None:
        try:
            # conf_int_odds may be a DataFrame with index
            if hasattr(conf_int_odds, 'loc') and female_key in conf_int_odds.index:
                row = conf_int_odds.loc[female_key]
                # Many shapes: two columns; take first two values
                ci_low = float(row.iloc[0])
                ci_high = float(row.iloc[1])
            else:
                # try positional match: find index position of female_key in odds_ratios index
                try:
                    idx_list = list(odds_ratios.index)
                    pos = idx_list.index(female_key)
                    row = conf_int_odds.iloc[pos]
                    ci_low = float(row.iloc[0])
                    ci_high = float(row.iloc[1])
                except Exception:
                    # If conf_int_odds is dict-like
                    try:
                        row = conf_int_odds[female_key]
                        ci_low = float(row[0])
                        ci_high = float(row[1])
                    except Exception:
                        ci_low = ci_high = None
        except Exception:
            ci_low = ci_high = None

    percent_change = (oratio - 1.0) * 100.0
    significant = (pval < 0.05)

    result_object = {
        'coefficient_log_odds': None if coef_logodds is None else round(coef_logodds, 6),
        'odds_ratio': round(oratio, 6),
        'odds_ratio_95ci': None if (ci_low is None or ci_high is None) else [round(ci_low, 6), round(ci_high, 6)],
        'p_value': round(pval, 6),
        'percent_change_in_odds': round(percent_change, 3),
        'significant_at_0.05': bool(significant)
    }

    # Build a concise interpretation description
    if result_object['odds_ratio_95ci'] is not None:
        ci_text = f"95% CI for OR = [{result_object['odds_ratio_95ci'][0]}, {result_object['odds_ratio_95ci'][1]}]"
    else:
        ci_text = "95% CI for OR not available"

    significance_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"
    desc = (
        f"Controlling for the listed covariates, the 'female' indicator has an estimated "
        f"odds ratio of {result_object['odds_ratio']} ({ci_text}), p = {result_object['p_value']}. "
        f"This implies females have about {abs(result_object['percent_change_in_odds']):.1f}% "
        f"{'higher' if result_object['percent_change_in_odds'] > 0 else 'lower'} odds of mortgage approval "
        f"compared with males, and the effect is {significance_text}. "
        f"Note: these are odds (multiplicative) effects from a logistic regression, not direct probability differences."
    )

    return {"object": result_object, "description": desc}