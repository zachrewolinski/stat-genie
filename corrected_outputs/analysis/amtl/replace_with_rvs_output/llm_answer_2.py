def extract_final_answer(model_output):
    """
    Extracts statistics for the 'IsHuman' coefficient from a fitted statsmodels GLMResultsWrapper
    (binomial logit model). Returns a dictionary with:
      - "object": dict of extracted numeric results (coef, se, z, p, conf_int, odds_ratio, or_ci, significant)
      - "description": short interpretation of whether modern humans have higher AMTL after adjustment.
    """
    import numpy as np

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("Provided model_output does not appear to be a fitted statsmodels results object.")

    params = model_output.params
    if 'IsHuman' not in params.index:
        raise ValueError("Model output does not contain an 'IsHuman' parameter. Check model specification or parameter names.")

    # Extract coefficient and standard error
    coef = float(params['IsHuman'])
    # bse usually exists
    try:
        se = float(model_output.bse['IsHuman'])
    except Exception:
        # fallback: compute from covariance matrix if available
        cov = getattr(model_output, "cov_params", None)
        if cov is None:
            raise RuntimeError("Could not retrieve standard error for 'IsHuman'.")
        else:
            se = float(np.sqrt(np.diag(cov()))[list(params.index).index('IsHuman')])

    # z-statistic (coef / se)
    z_stat = coef / se if se != 0 else np.nan

    # p-value
    pval = float(model_output.pvalues['IsHuman']) if hasattr(model_output, "pvalues") else np.nan

    # 95% confidence interval for the log-odds parameter
    try:
        conf = model_output.conf_int()
        # conf can be DataFrame or ndarray; find row for IsHuman
        if hasattr(conf, "loc") and 'IsHuman' in getattr(conf, "index", []):
            conf_low, conf_high = map(float, conf.loc['IsHuman'].values)
        else:
            # fallback by position
            pos = list(params.index).index('IsHuman')
            conf_low, conf_high = float(conf[pos, 0]), float(conf[pos, 1])
    except Exception:
        conf_low, conf_high = (np.nan, np.nan)

    # Odds ratio and its CI (exp of coef and CI)
    try:
        or_point = float(np.exp(coef))
        or_ci_low, or_ci_high = float(np.exp(conf_low)), float(np.exp(conf_high))
    except Exception:
        or_point, or_ci_low, or_ci_high = (np.nan, np.nan, np.nan)

    # Significance at alpha = 0.05
    significant = False
    if not np.isnan(pval):
        significant = (pval < 0.05)

    # Direction (higher/lower)
    if np.isnan(coef):
        direction = "unknown"
    elif coef > 0:
        direction = "higher"
    elif coef < 0:
        direction = "lower"
    else:
        direction = "no difference"

    # Prepare object to return
    result_object = {
        "parameter": "IsHuman",
        "coef_log_odds": coef,
        "std_error": se,
        "z_stat": z_stat,
        "p_value": pval,
        "conf_int_log_odds": (conf_low, conf_high),
        "odds_ratio": or_point,
        "odds_ratio_conf_int": (or_ci_low, or_ci_high),
        "significant_at_0.05": bool(significant),
        "direction": direction
    }

    # Compose concise description / conclusion
    if np.isnan(pval):
        conclusion = (
            "Could not determine statistical significance for 'IsHuman' because p-value is not available."
        )
    else:
        if significant:
            if coef > 0:
                conclusion = (
                    f"Result: The IsHuman coefficient is positive and statistically significant (coef={coef:.3f}, "
                    f"p={pval:.3g}). After adjusting for age, sex probability, and tooth class, modern humans "
                    f"have higher odds of antemortem tooth loss compared to the non-human primates in the sample. "
                    f"Estimated odds ratio = {or_point:.3f} (95% CI: {or_ci_low:.3f}–{or_ci_high:.3f})."
                )
            else:
                conclusion = (
                    f"Result: The IsHuman coefficient is negative and statistically significant (coef={coef:.3f}, "
                    f"p={pval:.3g}). After adjusting for covariates, modern humans have lower odds of antemortem "
                    f"tooth loss compared to the non-human primates in the sample. "
                    f"Estimated odds ratio = {or_point:.3f} (95% CI: {or_ci_low:.3f}–{or_ci_high:.3f})."
                )
        else:
            conclusion = (
                f"Result: The IsHuman coefficient is not statistically significant (coef={coef:.3f}, "
                f"p={pval:.3g}). There is no strong evidence that modern humans differ from the non-human primates "
                f"in odds of antemortem tooth loss after adjusting for age, sex probability, and tooth class. "
                f"Estimated odds ratio = {or_point:.3f} (95% CI: {or_ci_low:.3f}–{or_ci_high:.3f})."
            )

    return {
        "object": result_object,
        "description": conclusion
    }