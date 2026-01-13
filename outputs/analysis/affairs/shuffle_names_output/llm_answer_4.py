def extract_final_answer(model_output):
    """
    Extract statistics for the 'HasChildren' coefficient from a fitted statsmodels OLS results
    (assumed to have been fit with cov_type='HC3' as in the modelling code).

    Returns a dictionary with:
      - "object": a dict of numeric statistics (coef, se, t, pvalue, 95% CI, nobs, rsquared)
      - "description": a short interpretation describing whether having children is associated
                       with a statistically significant increase or decrease in affair frequency.
    """
    import numpy as np

    res = model_output
    param = 'HasChildren'

    # Basic existence checks
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")
    if param not in res.params.index:
        raise ValueError(f"Parameter '{param}' not found in model_output.params index. Available params: {list(res.params.index)}")

    # Extract point estimate and robust (HC3) summary stats
    coef = float(res.params[param])
    se = float(res.bse[param]) if hasattr(res, 'bse') else float(np.nan)
    tstat = float(res.tvalues[param]) if hasattr(res, 'tvalues') else float(np.nan)
    pvalue = float(res.pvalues[param]) if hasattr(res, 'pvalues') else float(np.nan)

    # Confidence interval (uses the covariance stored in the fitted result)
    try:
        ci_all = res.conf_int()
        # conf_int may be a DataFrame or ndarray; handle both
        if hasattr(ci_all, 'loc'):
            ci_row = ci_all.loc[param]
            ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
        else:
            # ndarray: find index of parameter
            idx = list(res.params.index).index(param)
            ci_lower, ci_upper = float(ci_all[idx, 0]), float(ci_all[idx, 1])
    except Exception:
        ci_lower, ci_upper = float(np.nan), float(np.nan)

    # Other useful model stats if available
    nobs = int(res.nobs) if hasattr(res, 'nobs') else None
    rsq = float(res.rsquared) if hasattr(res, 'rsquared') else None

    stats = {
        "parameter": param,
        "coef": coef,
        "se": se,
        "t": tstat,
        "pvalue": pvalue,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "nobs": nobs,
        "rsquared": rsq
    }

    # Simple interpretation regarding direction and statistical significance
    alpha = 0.05
    if np.isfinite(pvalue) and pvalue < alpha:
        if coef < 0:
            direction = "a statistically significant decrease"
        else:
            direction = "a statistically significant increase"
        description = (
            f"Having children is associated with {direction} in reported extramarital "
            f"affair frequency (coef = {coef:.3f}, SE = {se:.3f}, t = {tstat:.3f}, p = {pvalue:.3g}; "
            f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}])."
        )
    else:
        description = (
            f"No statistically significant association between having children and reported extramarital "
            f"affair frequency was found (coef = {coef:.3f}, SE = {se:.3f}, t = {tstat:.3f}, p = {pvalue:.3g}; "
            f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}])."
        )

    return {"object": stats, "description": description}