def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, 95% CI, and incidence-rate-ratio (IRR)
    for the 'SkinDark' variable from a fitted statsmodels results object (GLM/GLMResultsWrapper
    or a robust-covariance wrapped results object).

    Returns a dict with keys:
      - "object": a dict with numeric fields (coef, se, pvalue, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper, nobs, significance)
      - "description": a human-readable interpretation of the result in context.

    Notes:
      - The model is assumed to be a log-link count model (NegativeBinomial/Poisson) with an exposure offset log(games),
        so the coefficient is on the log-rate scale and exp(coef) is the rate ratio (IRR) per game.
      - The function tries to be robust to whether conf_int() returns a DataFrame or numpy array and to whether
        the results object contains 'SkinDark' exactly or a variant containing that substring.
    """
    import numpy as np

    res = model_output

    # Try to extract required objects from the results
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        conf = res.conf_int()
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract parameter table from model_output: {e}"
        }

    # Find the parameter name corresponding to SkinDark
    target = 'SkinDark'
    if target not in params.index:
        # Try to find any parameter name that contains 'SkinDark'
        candidates = [name for name in params.index if 'SkinDark' in str(name)]
        if len(candidates) == 0:
            return {
                "object": None,
                "description": "The model does not contain a coefficient for 'SkinDark'. Available coefficients: "
                               + ", ".join(map(str, params.index.tolist()))
            }
        target = candidates[0]

    # Safely get numeric values (handle pandas Series or numpy arrays)
    try:
        coef = float(params[target])
        se = float(bse[target])
        pval = float(pvalues[target])
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to read numeric values for '{target}': {e}"
        }

    # Extract confidence interval for the coefficient
    try:
        # conf may be a DataFrame-like (with .loc) or a numpy array
        if hasattr(conf, 'loc'):
            ci_lower, ci_upper = map(float, conf.loc[target])
        else:
            # assume array in same order as params.index
            idx = list(params.index).index(target)
            ci_lower, ci_upper = float(conf[idx, 0]), float(conf[idx, 1])
    except Exception as e:
        # fallback: approximate CI using coef +/- 1.96*se
        ci_lower = coef - 1.96 * se
        ci_upper = coef + 1.96 * se

    # Convert to incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Number of observations if available
    nobs = None
    try:
        nobs = int(getattr(res, 'nobs'))
    except Exception:
        try:
            nobs = int(getattr(res.model, 'nobs'))
        except Exception:
            nobs = None

    significance = (pval < 0.05)

    # Build the object to return
    result_object = {
        "parameter": target,
        "coef_log_rate": coef,
        "se": se,
        "p_value": pval,
        "ci_lower_log_rate": ci_lower,
        "ci_upper_log_rate": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "nobs": nobs,
        "significant_at_0.05": bool(significance)
    }

    # Build a human-readable interpretation
    if significance:
        interpretation = (
            f"The model estimates a statistically significant association between skin tone and red-card rate "
            f"(parameter '{target}'): coef = {coef:.4f} (SE = {se:.4f}, p = {pval:.3g}). "
            f"On the rate scale, the incidence rate ratio (IRR = exp(coef)) = {irr:.3f} "
            f"(95% CI {irr_ci_lower:.3f} to {irr_ci_upper:.3f}). "
            f"Interpretation: players in the 'dark' group are estimated to receive {100*(irr-1):.1f}% "
            f"{'more' if irr>1 else 'fewer'} red cards per game than players in the 'light' group, "
            f"and this difference is statistically significant at alpha = 0.05. "
        )
    else:
        interpretation = (
            f"The model does not find a statistically significant association between skin tone and red-card rate "
            f"(parameter '{target}'): coef = {coef:.4f} (SE = {se:.4f}, p = {pval:.3g}). "
            f"IRR = {irr:.3f} (95% CI {irr_ci_lower:.3f} to {irr_ci_upper:.3f}), which includes 1, "
            f"so we do not have evidence to conclude that dark-skinned players receive red cards at a different "
            f"rate than light-skinned players. "
        )

    # Add a reminder about model scale and offset
    interpretation += (
        "Model notes: this is from a log-link count model (Negative Binomial or Poisson fallback) with offset = log(games), "
        "so coefficients are log-rate differences (IRR = exp(coef)). Standard errors were requested clustered by referee if available."
    )

    return {
        "object": result_object,
        "description": interpretation
    }