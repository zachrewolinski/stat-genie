def extract_final_answer(model_output):
    """
    Extracts the estimated effect of the 'female' indicator from a fitted statsmodels
    logistic regression result (BinaryResultsWrapper or Results object).
    
    Returns a dictionary:
      - "object": dict with numeric results (coef, se, z, p, OR, 95% CI for OR, significance boolean)
      - "description": plain-language interpretation of what the numbers say about the effect
    """
    import numpy as np
    import pandas as pd

    # Basic checks
    if model_output is None:
        raise ValueError("model_output is None")

    # Try to access params, bse, pvalues, conf_int
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        conf = model_output.conf_int()
    except Exception as e:
        raise ValueError(f"Provided model_output does not look like a statsmodels results object: {e}")

    # Ensure 'female' is present
    if 'female' not in params.index:
        raise KeyError("The model results do not contain a parameter named 'female'.")

    # Index of the parameter (for conf matrix that may be positional)
    param_index = list(params.index).index('female')

    # Extract numeric quantities
    coef = float(params['female'])
    se = float(bse['female']) if 'female' in bse.index else float(bse.iloc[param_index])
    # z/statistic might be available as tvalues or zvalues; compute if not present
    z = None
    if hasattr(model_output, 'tvalues') and 'female' in getattr(model_output, 'tvalues').index:
        z = float(model_output.tvalues['female'])
    elif hasattr(model_output, 'zvalues') and 'female' in getattr(model_output, 'zvalues').index:
        z = float(model_output.zvalues['female'])
    else:
        # compute z as coef / se
        z = coef / se if se != 0 else float('nan')

    p = float(pvalues['female'])

    # Confidence interval for coefficient (log-odds)
    # conf can be DataFrame or ndarray-like; handle both
    try:
        if isinstance(conf, pd.DataFrame):
            lower_log = float(conf.iloc[param_index, 0])
            upper_log = float(conf.iloc[param_index, 1])
        else:
            # assume numpy array-like with shape (k,2)
            lower_log = float(conf[param_index, 0])
            upper_log = float(conf[param_index, 1])
    except Exception as e:
        raise ValueError(f"Could not extract confidence interval from model_output.conf_int(): {e}")

    # Odds ratio and CI on OR scale
    or_est = float(np.exp(coef))
    or_lower = float(np.exp(lower_log))
    or_upper = float(np.exp(upper_log))

    # Significance at conventional 0.05 level
    significant = (p < 0.05)

    # Build the numeric object to return
    result_object = {
        'coef_log_odds': round(coef, 6),
        'se': round(se, 6),
        'z': round(z, 6),
        'p_value': round(p, 6),
        'odds_ratio': round(or_est, 6),
        'OR_CI_lower': round(or_lower, 6),
        'OR_CI_upper': round(or_upper, 6),
        'significant_at_0.05': bool(significant)
    }

    # Plain-language description
    direction = "higher" if or_est > 1 else ("no change" if np.isclose(or_est, 1.0) else "lower")
    pct_change = (or_est - 1.0) * 100.0
    pct_str = f"{pct_change:+.2f}%"  # signed percentage change in odds

    significance_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"

    description = (
        f"The estimated coefficient on 'female' is {coef:.4f} (log-odds), with SE={se:.4f}, z={z:.3f}, p={p:.4g}. "
        f"This corresponds to an odds ratio of {or_est:.4f} (95% CI: [{or_lower:.4f}, {or_upper:.4f}]). "
        f"Interpreted on the odds scale, being female is associated with {direction} odds of mortgage approval "
        f"({pct_str} change in odds) compared to being male, holding the listed controls constant. "
        f"The effect is {significance_text}."
    )

    return {"object": result_object, "description": description}