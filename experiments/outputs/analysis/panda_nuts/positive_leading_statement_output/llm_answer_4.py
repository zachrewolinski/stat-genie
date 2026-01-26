def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, 95% CIs, and multiplicative interpretation (percent change)
    for the predictors of interest from a statsmodels result object (MixedLMResultsWrapper or
    RegressionResults/RobustResults).

    Returns a dictionary with keys:
      - "object": dict mapping each predictor ('age','sex_male','help_yes') to its extracted stats
      - "description": concise interpretation of the extracted stats (including sign, significance,
                       and multiplicative effect on nuts-opened-per-minute because the outcome
                       is log-transformed)

    The function is defensive: it tries multiple attribute names used by different statsmodels result
    objects.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Try to get parameter estimates
    params = None
    pvalues = None
    conf = None

    # get params
    for attr in ("params", "fe_params"):
        if hasattr(res, attr):
            try:
                params = getattr(res, attr)
                break
            except Exception:
                params = None
    # get p-values
    for attr in ("pvalues", "pvalues_fe"):
        if hasattr(res, attr):
            try:
                pvalues = getattr(res, attr)
                break
            except Exception:
                pvalues = None
    # get conf int
    try:
        conf = res.conf_int()
    except Exception:
        # Some objects require alpha kw; try that
        try:
            conf = res.conf_int(alpha=0.05)
        except Exception:
            conf = None

    # Ensure params/pvalues are pandas Series if possible
    if params is not None and not isinstance(params, (pd.Series, pd.DataFrame)):
        try:
            params = pd.Series(params)
        except Exception:
            pass
    if pvalues is not None and not isinstance(pvalues, (pd.Series, pd.DataFrame)):
        try:
            pvalues = pd.Series(pvalues)
        except Exception:
            pass
    if conf is not None and isinstance(conf, np.ndarray):
        try:
            conf = pd.DataFrame(conf, index=getattr(params, "index", None))
        except Exception:
            pass

    predictors = ['age', 'sex_male', 'help_yes']

    results = {}
    significance_level = 0.05

    for pred in predictors:
        entry = {
            'coefficient': None,
            'std_error': None,
            'p_value': None,
            'ci_95': (None, None),
            'percent_change': None,   # (exp(coef)-1)*100
            'significant_at_0.05': None
        }

        if params is None or pred not in params.index:
            # parameter not found
            results[pred] = entry
            continue

        coef = float(params[pred])
        entry['coefficient'] = coef

        # try to get std error if available
        se = None
        if hasattr(res, 'bse') and isinstance(getattr(res, 'bse'), (pd.Series, np.ndarray, list, tuple)):
            try:
                se = float(res.bse[pred])
            except Exception:
                se = None
        # fallback to bse_fe
        if se is None and hasattr(res, 'bse_fe'):
            try:
                se = float(res.bse_fe[pred])
            except Exception:
                se = None
        entry['std_error'] = se

        # p-value
        if pvalues is not None and pred in pvalues.index:
            try:
                pv = float(pvalues[pred])
            except Exception:
                pv = None
            entry['p_value'] = pv

        # conf int
        if conf is not None and pred in conf.index:
            try:
                low = float(conf.loc[pred, 0])
                high = float(conf.loc[pred, 1])
                entry['ci_95'] = (low, high)
            except Exception:
                # some conf_int returns DataFrame with named columns
                try:
                    cols = list(conf.columns)
                    low = float(conf.loc[pred, cols[0]])
                    high = float(conf.loc[pred, cols[1]])
                    entry['ci_95'] = (low, high)
                except Exception:
                    entry['ci_95'] = (None, None)

        # percent change interpretation (outcome is log-scale)
        try:
            pct = (np.exp(coef) - 1.0) * 100.0
            entry['percent_change'] = float(pct)
        except Exception:
            entry['percent_change'] = None

        # significance
        if entry['p_value'] is not None:
            entry['significant_at_0.05'] = entry['p_value'] < significance_level

        results[pred] = entry

    # Build a short human-readable description summarizing findings
    lines = []
    for pred in predictors:
        e = results[pred]
        if e['coefficient'] is None:
            lines.append(f"{pred}: estimate not found in model output.")
            continue
        sig = ("significant" if e['significant_at_0.05'] else "not significant")
        pct_text = (f"{e['percent_change']:.1f}% change" if e['percent_change'] is not None else "N/A")
        ci = e['ci_95']
        ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None and ci[1] is not None else "N/A"
        lines.append(
            f"{pred}: coef={e['coefficient']:.4f}, p={e['p_value']:.3g}" 
            f" ({sig}); 95% CI={ci_text}; approx. {pct_text} in nuts/min."
        )

    description = (
        "Fixed-effect estimates extracted for predictors of interest. Coefficients are on the log scale "
        "of nuts_opened_per_min (log-transformed). 'percent_change' = (exp(coef)-1)*100 gives the approximate "
        "percent multiplicative change in nuts opened per minute associated with a one-unit increase in the predictor. "
        "Summary per predictor:\n" + "\n".join(lines)
    )

    return {"object": results, "description": description}