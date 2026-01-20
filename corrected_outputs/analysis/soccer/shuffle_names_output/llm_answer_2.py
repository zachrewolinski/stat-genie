def extract_final_answer(model_output):
    """
    Extracts the DarkSkin effect from a fitted statsmodels GLMResultsWrapper (Negative Binomial).
    Returns a dictionary with:
      - "object": dict of extracted statistics (coef, se, p-value, conf_int, rate_ratio, rate_ratio_conf_int)
      - "description": human-readable interpretation of the effect in context
    
    The function is defensive about the exact parameter name: it looks for a parameter
    whose name contains 'DarkSkin' if 'DarkSkin' is not an exact match.
    """
    import numpy as np
    import pandas as pd

    result = {}

    # Try to obtain parameter name for DarkSkin
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
        ci = model_output.conf_int()
    except Exception as e:
        raise ValueError(f"Provided model_output does not have the expected attributes: {e}")

    # Look for exact name first, otherwise any name containing 'DarkSkin'
    param_name = None
    if 'DarkSkin' in params.index:
        param_name = 'DarkSkin'
    else:
        matches = [name for name in params.index if 'DarkSkin' in name]
        if len(matches) >= 1:
            param_name = matches[0]

    if param_name is None:
        raise KeyError("Could not find a parameter for 'DarkSkin' in model_output.params index. "
                       f"Available parameters: {list(params.index)}")

    # Extract statistics
    coef = float(params[param_name])
    se = float(bse[param_name]) if param_name in bse.index else float(np.nan)
    pval = float(pvalues[param_name]) if param_name in pvalues.index else float(np.nan)
    ci_low, ci_high = tuple(ci.loc[param_name]) if param_name in ci.index else (np.nan, np.nan)

    # Exponentiate to get rate ratio per match (because model used log offset for exposure)
    rate_ratio = float(np.exp(coef))
    rr_ci_low, rr_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    # Simple significance flag and textual summary
    significance = (pval < 0.05) if (not np.isnan(pval)) else None
    if significance is True:
        sig_text = "statistically significant at p < 0.05"
    elif significance is False:
        sig_text = "not statistically significant at p < 0.05"
    else:
        sig_text = "p-value not available"

    # Build object to return (numeric results)
    result_obj = {
        "parameter_name": param_name,
        "coef_log_rate": coef,
        "se": se,
        "p_value": pval,
        "coef_95ci": [ci_low, ci_high],
        "rate_ratio_per_match": rate_ratio,
        "rate_ratio_95ci": [rr_ci_low, rr_ci_high],
        "significant_at_0.05": significance
    }

    # Build descriptive interpretation
    # Interpretation: coefficient is log(rate ratio) per match; exp(coef) is multiplicative factor in red card rate per match.
    interpretation = (
        f"Parameter '{param_name}': log-rate coef = {coef:.4f} (SE = {se:.4f}), p = {pval:.4g}; "
        f"95% CI for log-coef = [{ci_low:.4f}, {ci_high:.4f}].\n"
        f"Exponentiated -> rate ratio (per match) = {rate_ratio:.4f} "
        f"(95% CI = [{rr_ci_low:.4f}, {rr_ci_high:.4f}]).\n"
        f"Interpretation: Holding the included controls constant, a player coded as 'DarkSkin' "
        f"has an expected red-card rate per match that is {rate_ratio:.2f} times that of a 'Light' player. "
        f"This effect is {sig_text}.\n"
        f"Note: This is an association (controlling for the model covariates and exposure); "
        f"it should not be interpreted as definitive proof of causation."
    )

    return {"object": result_obj, "description": interpretation}