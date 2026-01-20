def extract_final_answer(model_output):
    """
    Extract relevant statistics for the IsHuman effect from a fitted statsmodels GLMResultsWrapper.

    Returns a dictionary with keys:
      - "object": a dict of extracted numeric results (coef, se, z, p, 95% CI on link scale,
                  odds ratio and its 95% CI)
      - "description": a brief plain-language interpretation about whether modern humans
                       have higher AMTL than non-human primates after controlling for
                       age, sex, and tooth class.
    """
    import numpy as np

    res = model_output

    # Ensure parameter table exists
    try:
        params = res.params
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a fitted statsmodels results object.") from e

    # Find the parameter name corresponding to IsHuman (be tolerant to encoding)
    param_names = list(params.index)
    ishuman_candidates = [n for n in param_names if 'IsHuman' in n]
    if not ishuman_candidates:
        raise ValueError("No parameter name containing 'IsHuman' found in model parameters. "
                         "Parameter names: {}".format(param_names))
    param_name = ishuman_candidates[0]

    # Extract statistics (use available attributes with fallbacks)
    coef = float(params[param_name])
    # Standard error
    try:
        se = float(res.bse[param_name])
    except Exception:
        # compute approximate se from conf_int if available
        try:
            ci = res.conf_int().loc[param_name].values
            se = float((ci[1] - ci[0]) / (2 * 1.96))
        except Exception:
            se = None

    # z- or t-statistic
    z_val = None
    if hasattr(res, 'tvalues') and param_name in getattr(res, 'tvalues', {}):
        z_val = float(res.tvalues[param_name])
    elif hasattr(res, 'zvalues') and param_name in getattr(res, 'zvalues', {}):
        z_val = float(res.zvalues[param_name])
    else:
        # compute z from coef/se if available
        if se not in (None, 0):
            z_val = float(coef / se)

    # p-value
    try:
        pval = float(res.pvalues[param_name])
    except Exception:
        pval = None

    # 95% confidence interval on link (log-odds) scale
    try:
        ci_low, ci_high = res.conf_int().loc[param_name].astype(float).tolist()
    except Exception:
        ci_low, ci_high = (None, None)

    # Odds ratio and its CI
    or_est = np.exp(coef) if coef is not None else None
    or_ci_low = np.exp(ci_low) if ci_low is not None else None
    or_ci_high = np.exp(ci_high) if ci_high is not None else None

    # Interpretation
    alpha = 0.05
    if pval is None:
        conclusion = ("Could not determine statistical significance for IsHuman (p-value unavailable). "
                      "Extracted coefficient = {:.4f} (log-odds).".format(coef))
    else:
        if pval < alpha:
            if coef > 0:
                conclusion = ("Modern humans (IsHuman=1) have statistically significantly higher odds of "
                              "antemortem tooth loss than non-human primates after controlling for age, sex, "
                              "and tooth class (coef = {:+.4f}, OR = {:.3f}, 95% CI [{:.3f}, {:.3f}], p = {:.3e})."
                              .format(coef, or_est, or_ci_low, or_ci_high, pval))
            else:
                conclusion = ("Modern humans have statistically significantly lower odds of antemortem tooth loss "
                              "than non-human primates after controlling for covariates "
                              "(coef = {:+.4f}, OR = {:.3f}, 95% CI [{:.3f}, {:.3f}], p = {:.3e})."
                              .format(coef, or_est, or_ci_low, or_ci_high, pval))
        else:
            conclusion = ("No statistically significant difference in AMTL odds between modern humans and "
                          "non-human primates after controlling for age, sex, and tooth class "
                          "(coef = {:+.4f}, OR = {:.3f}, 95% CI [{:.3f}, {:.3f}], p = {:.3e})."
                          .format(coef, or_est, or_ci_low, or_ci_high, pval))

    result_object = {
        "param_name": param_name,
        "coef_log_odds": coef,
        "std_error": se,
        "z_or_t_value": z_val,
        "p_value": pval,
        "conf_int_95_log_odds": (ci_low, ci_high),
        "odds_ratio": float(or_est) if or_est is not None else None,
        "odds_ratio_95_ci": (float(or_ci_low) if or_ci_low is not None else None,
                             float(or_ci_high) if or_ci_high is not None else None),
        "significance_alpha": alpha
    }

    return {
        "object": result_object,
        "description": conclusion
    }