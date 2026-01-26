def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator from a fitted statsmodels binary model (Logit or GLM).
    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, p-value, 95% CI, odds ratio and CI, marginal effect at means)
      - "description": plain-language interpretation (direction, significance, and magnitude)
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("The provided model_output does not appear to be a statsmodels results object with .params")

    params = res.params
    if 'female' not in params.index:
        raise ValueError("The fitted model does not contain a 'female' coefficient in params.")

    coef = float(params['female'])

    # Standard error and p-value (if present)
    se = None
    pval = None
    try:
        se = float(res.bse['female'])
    except Exception:
        # try alternative attribute
        se = None
    try:
        pval = float(res.pvalues['female'])
    except Exception:
        pval = None

    # 95% confidence interval for coefficient
    try:
        ci = res.conf_int().loc['female'].astype(float).values  # [lower, upper]
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Odds ratio and CI
    try:
        odds_ratio = float(np.exp(coef))
        odds_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        odds_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        odds_ratio = None
        odds_ci_lower = None
        odds_ci_upper = None

    # Marginal effect: difference in predicted probability when female=1 vs female=0,
    # holding other covariates at their means (i.e., average marginal effect at means).
    marginal_effect_at_means = None
    try:
        # obtain exogenous design matrix and names
        exog = res.model.exog
        exog_names = list(res.model.exog_names)
        if 'female' in exog_names:
            idx = exog_names.index('female')
            x_mean = np.nanmean(exog, axis=0)
            x0 = x_mean.copy()
            x1 = x_mean.copy()
            # set female to 0 and 1
            x0[idx] = 0.0
            x1[idx] = 1.0
            lin0 = float(np.dot(x0, params))
            lin1 = float(np.dot(x1, params))
            # logistic link
            logistic = lambda x: 1.0 / (1.0 + np.exp(-x))
            p0 = logistic(lin0)
            p1 = logistic(lin1)
            marginal_effect_at_means = float(p1 - p0)  # absolute change in probability
    except Exception:
        marginal_effect_at_means = None

    # Build the object to return
    result_object = {
        'coef_female_log_odds': coef,
        'std_error': se,
        'p_value': pval,
        'conf_int_95': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'odds_ratio_95_CI': [odds_ci_lower, odds_ci_upper],
        'marginal_effect_at_means': marginal_effect_at_means  # in probability points (0-1)
    }

    # Interpretation / description
    # Decide significance at conventional 5% if p-value available
    if pval is None:
        sig_statement = "p-value not available, so statistical significance cannot be determined."
    else:
        if pval < 0.05:
            sig_statement = f"The effect is statistically significant at the 5% level (p = {pval:.3g})."
        else:
            sig_statement = f"The effect is not statistically significant at the 5% level (p = {pval:.3g})."

    # Direction and magnitude statement
    if odds_ratio is not None:
        if odds_ratio > 1:
            direction = "Female applicants have higher odds of acceptance than male applicants."
        elif odds_ratio < 1:
            direction = "Female applicants have lower odds of acceptance than male applicants."
        else:
            direction = "No change in odds for female applicants relative to male applicants."
        or_ci_text = f"Odds ratio = {odds_ratio:.3f}"
        if odds_ci_lower is not None and odds_ci_upper is not None:
            or_ci_text += f" (95% CI: {odds_ci_lower:.3f} — {odds_ci_upper:.3f})"
    else:
        direction = "Could not compute odds ratio."
        or_ci_text = ""

    me_text = ""
    if marginal_effect_at_means is not None:
        # convert to percentage points for readability
        me_pct = marginal_effect_at_means * 100.0
        me_text = f"The estimated change in predicted probability (holding controls at their means) is {me_pct:.2f} percentage points."
    else:
        me_text = "Marginal effect at means could not be computed."

    description = (
        f"Coefficient on 'female' (log-odds): {coef:.4f}. {sig_statement} {direction} "
        f"{or_ci_text}. {me_text}"
    )

    return {"object": result_object, "description": description}