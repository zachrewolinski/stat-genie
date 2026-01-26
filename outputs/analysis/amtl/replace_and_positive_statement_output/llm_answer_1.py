def extract_final_answer(model_output):
    """
    Extracts the effect of 'is_human' from a statsmodels GLM results object (possibly
    adjusted with clustered robust SEs). Returns a dictionary with numeric results and
    a short interpretation.

    Returned dictionary keys:
      - "object": dict with numeric results:
          coef, se, pvalue, conf_int (logit scale), odds_ratio, odds_ratio_ci, significant (bool)
      - "description": short human-readable interpretation about whether modern humans
                       have higher AMTL after accounting for controls.
    """
    import numpy as np
    import pandas as pd

    res = model_output
    param = "is_human"

    # Ensure results has parameter information
    try:
        params = res.params
    except Exception as e:
        raise ValueError(f"Provided model_output does not appear to be a fitted results object: {e}")

    # Check parameter presence
    if hasattr(params, "index"):
        if param not in params.index:
            raise KeyError(f"Parameter '{param}' not found in model results. Available params: {list(params.index)}")
    else:
        # If params has no index (unlikely for statsmodels), try to proceed by position (not recommended)
        raise KeyError("Model results.params does not expose an index. Cannot locate 'is_human' parameter reliably.")

    # Extract statistics
    coef = float(res.params[param])
    # bse may be in res.bse or res.bse_robust depending on how results were produced; try common locations
    try:
        se = float(res.bse[param])
    except Exception:
        # fallback: use sqrt of diagonal of cov_params if available
        try:
            cov = res.cov_params()
            se = float(np.sqrt(np.asarray(cov.loc[param, param])))
        except Exception as e:
            raise RuntimeError(f"Could not extract standard error for '{param}': {e}")

    # p-value
    try:
        pval = float(res.pvalues[param])
    except Exception:
        # If pvalues unavailable, compute z-stat and p from normal
        z = coef / se
        from scipy import stats
        pval = float(2 * (1 - stats.norm.cdf(abs(z))))

    # Confidence interval (logit scale)
    try:
        ci_df = res.conf_int()
        if isinstance(ci_df, (pd.DataFrame, pd.Series)):
            lower_ci = float(ci_df.loc[param, 0]) if 0 in ci_df.columns else float(ci_df.loc[param][0])
            upper_ci = float(ci_df.loc[param, 1]) if 1 in ci_df.columns else float(ci_df.loc[param][1])
        else:
            # conf_int returned ndarray with index; find row by param name
            idx = list(res.params.index).index(param)
            lower_ci, upper_ci = float(ci_df[idx, 0]), float(ci_df[idx, 1])
    except Exception:
        # fallback using normal approximation
        lower_ci = coef - 1.96 * se
        upper_ci = coef + 1.96 * se

    # Odds ratio and CI on OR scale
    odds_ratio = float(np.exp(coef))
    odds_ratio_ci = [float(np.exp(lower_ci)), float(np.exp(upper_ci))]

    # Significance and short conclusion
    alpha = 0.05
    significant = bool(pval < alpha)
    if coef > 0 and significant:
        conclusion = ("Result: Modern humans (Homo sapiens) have a statistically significant "
                      "higher odds of antemortem tooth loss (AMTL) compared to the non-human "
                      "primates in the dataset, after adjusting for age, sex probability, and tooth class.")
    elif coef > 0 and not significant:
        conclusion = ("Result: The estimated effect points toward modern humans having higher AMTL "
                      "than non-human primates, but this difference is not statistically significant "
                      f"(p = {pval:.3g}).")
    elif coef < 0 and significant:
        conclusion = ("Result: Modern humans have a statistically significant lower odds of AMTL "
                      "compared to the non-human primates in the dataset, after adjusting for controls.")
    else:
        conclusion = ("Result: No statistically significant difference in AMTL between modern humans "
                      "and the non-human primates was detected after adjusting for age, sex probability, and tooth class.")

    # Package numeric object for downstream use
    result_obj = {
        "parameter": param,
        "coef_logit": coef,
        "se": se,
        "pvalue": pval,
        "conf_int_logit": [lower_ci, upper_ci],
        "odds_ratio": odds_ratio,
        "odds_ratio_conf_int": odds_ratio_ci,
        "significant_at_0.05": significant
    }

    return {
        "object": result_obj,
        "description": conclusion
    }