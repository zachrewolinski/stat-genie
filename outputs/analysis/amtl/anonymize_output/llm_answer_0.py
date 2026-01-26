def extract_final_answer(model_output):
    """
    Extracts the effect of the IsHuman indicator from a fitted statsmodels GLMResults (or wrapper).
    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coefficient, se, z, p, 95% CI, odds ratio, OR 95% CI, significance, and a conclusion)
      - "description": a human-readable explanation of what these numbers mean for the task question

    Expects model_output to be a statsmodels results object (e.g., GLMResultsWrapper) where the parameter
    corresponding to the human indicator contains the substring 'IsHuman' (the code is defensive and will
    find the first parameter name containing 'IsHuman').
    """
    import numpy as np

    res = model_output

    # Basic validation
    if not hasattr(res, "params"):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params).")

    param_names = list(res.params.index.astype(str))

    # Find parameter name for IsHuman (be flexible: exact match or name containing 'IsHuman')
    matching = [name for name in param_names if name == 'IsHuman' or 'IsHuman' in name]
    if len(matching) == 0:
        raise ValueError(f"Could not find a parameter corresponding to 'IsHuman'. Available params: {param_names}")

    param = matching[0]

    # Extract stats
    coef = float(res.params[param])
    # use bse (standard errors)
    try:
        se = float(res.bse[param])
    except Exception:
        # fallback if bse not available
        se = float(np.nan)

    # compute z (or wald stat) and p-value
    z = float(coef / se) if (se is not None and not np.isnan(se) and se != 0) else float(np.nan)
    try:
        pval = float(res.pvalues[param])
    except Exception:
        pval = float(np.nan)

    # confidence intervals (on log-odds scale)
    try:
        ci_df = res.conf_int()
        # conf_int() typically returns a DataFrame indexed by param names
        if hasattr(ci_df, "loc"):
            ci_row = ci_df.loc[param]
            lower, upper = float(ci_row.iloc[0]), float(ci_row.iloc[1])
        else:
            # array-like fallback: find index of param
            idx = param_names.index(param)
            lower, upper = float(ci_df[idx, 0]), float(ci_df[idx, 1])
    except Exception:
        lower, upper = float(np.nan), float(np.nan)

    # transform to odds ratio scale
    or_est = float(np.exp(coef)) if not np.isnan(coef) else float(np.nan)
    or_ci_lower = float(np.exp(lower)) if not np.isnan(lower) else float(np.nan)
    or_ci_upper = float(np.exp(upper)) if not np.isnan(upper) else float(np.nan)

    # significance decision (two-sided alpha = 0.05)
    significance = None
    if not np.isnan(pval):
        significance = (pval < 0.05)

    # Build a concise conclusion about whether modern humans have higher AMTL
    if np.isnan(coef) or np.isnan(pval):
        conclusion = "Could not determine effect: coefficient or p-value is NaN."
    else:
        direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
        if significance is True:
            conclusion = (
                f"Statistically significant effect: modern humans have {direction} odds of antemortem tooth loss "
                f"(IsHuman coefficient = {coef:.4f}, p = {pval:.4g}; OR = {or_est:.3f}, 95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}])."
            )
        elif significance is False:
            conclusion = (
                f"Not statistically significant: the estimated effect indicates {direction} odds for modern humans "
                f"(IsHuman coefficient = {coef:.4f}, p = {pval:.4g}; OR = {or_est:.3f}, 95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}]), "
                "so we do not have strong evidence to claim a difference after controlling for age, sex, and tooth class."
            )
        else:
            conclusion = (
                f"Unable to assess significance (p-value missing). Estimated effect: coef = {coef:.4f}, OR = {or_est:.3f}."
            )

    result_object = {
        "parameter_name": param,
        "coef_log_odds": coef,
        "std_error": se,
        "z_value": z,
        "p_value": pval,
        "ci_log_odds": [lower, upper],
        "odds_ratio": or_est,
        "odds_ratio_ci": [or_ci_lower, or_ci_upper],
        "significant_at_0.05": significance,
        "conclusion": conclusion
    }

    description = (
        "This output extracts the coefficient for the IsHuman indicator from the fitted binomial GLM. "
        "Coefficient is on the log-odds scale: a positive value means higher log-odds (and OR>1 means higher odds) "
        "of a tooth being missing for modern humans relative to the reference non-human primates, "
        "after controlling for age (Age_c), sex estimate (SexEstimate), and tooth class (ToothClass). "
        "The dictionary under 'object' contains the raw estimates, their standard error, z-statistic, p-value, "
        "95% confidence interval on the log-odds scale, the odds ratio and its 95% CI, a boolean for significance "
        "at alpha=0.05, and a brief textual conclusion interpreting whether modern humans have higher AMTL."
    )

    return {"object": result_object, "description": description}