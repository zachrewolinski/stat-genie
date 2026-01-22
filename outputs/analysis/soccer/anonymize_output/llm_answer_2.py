def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, test statistic, p-value, 95% CI,
    and incidence-rate-ratio (IRR) for the SkinDark predictor from a fitted
    statsmodels results object (e.g., GLMResultsWrapper or a robustified result).
    Returns a dictionary with keys "object" (a dict with numeric results) and
    "description" (a brief interpretation answering whether dark-skinned players
    are more likely to receive red cards).
    """
    import math
    import numpy as np

    res = model_output

    # Ensure params exist
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    params = res.params

    # Find the parameter name for SkinDark (handles possible naming variations)
    skin_candidates = [name for name in params.index if 'SkinDark' in str(name)]
    if len(skin_candidates) == 0:
        raise ValueError("Could not find a coefficient with name containing 'SkinDark' in model params.")
    skin_name = skin_candidates[0]

    # Extract coefficient
    coef = float(params[skin_name])

    # Extract standard error (try direct attribute, else compute from covariance)
    se = None
    if hasattr(res, 'bse') and skin_name in getattr(res, 'bse').index:
        se = float(res.bse[skin_name])
    else:
        # Try cov_params
        if hasattr(res, 'cov_params'):
            cov = res.cov_params()
            try:
                se = float(np.sqrt(cov.loc[skin_name, skin_name]))
            except Exception:
                pass
    if se is None:
        raise ValueError("Could not obtain a standard error for the SkinDark coefficient.")

    # Test statistic (use z = coef / se; for clustered/robust results this is the usual approach)
    z_stat = coef / se

    # p-value: prefer res.pvalues if available; otherwise approximate from normal
    p_value = None
    if hasattr(res, 'pvalues') and skin_name in getattr(res, 'pvalues').index:
        p_value = float(res.pvalues[skin_name])
    else:
        # two-sided normal approximation
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_stat) / math.sqrt(2))))

    # Confidence interval: try res.conf_int(), else approximate using normal critical value
    try:
        ci_table = res.conf_int()
        # ci_table may have 2 columns; pick them
        if skin_name in ci_table.index:
            row = ci_table.loc[skin_name]
            # handle both possible column namings
            lower = float(row.iloc[0])
            upper = float(row.iloc[1])
        else:
            # fallback to normal approx
            crit = 1.96
            lower = coef - crit * se
            upper = coef + crit * se
    except Exception:
        crit = 1.96
        lower = coef - crit * se
        upper = coef + crit * se

    # Incidence rate ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_lower = float(np.exp(lower))
    irr_upper = float(np.exp(upper))

    # Determine whether effect supports "dark-skinned players more likely"
    # We require a positive coefficient (IRR>1) and statistical significance at alpha=0.05.
    supports_hypothesis = (coef > 0) and (p_value < 0.05)

    # Prepare returned numeric object (rounded for readability)
    numeric_result = {
        "parameter_name": skin_name,
        "coef_log_rate": round(coef, 6),
        "std_error": round(se, 6),
        "z_stat": round(z_stat, 4),
        "p_value": round(p_value, 6),
        "ci_lower_log_rate": round(lower, 6),
        "ci_upper_log_rate": round(upper, 6),
        "IRR": round(irr, 4),
        "IRR_ci_lower": round(irr_lower, 4),
        "IRR_ci_upper": round(irr_upper, 4),
        "supports_hypothesis_at_0.05": bool(supports_hypothesis)
    }

    # Short interpretation
    if supports_hypothesis:
        conclusion_text = (
            "Yes — the SkinDark coefficient is positive and statistically significant "
            f"(coef = {numeric_result['coef_log_rate']}, p = {numeric_result['p_value']}). "
            "Interpretation: controlling for matches (exposure), position, age, height, weight, "
            "and country-level bias measures, players classified as Dark have a higher rate of red "
            f"cards than Light players. IRR = {numeric_result['IRR']} "
            f"(95% CI: {numeric_result['IRR_ci_lower']} to {numeric_result['IRR_ci_upper']})."
        )
    else:
        # Distinguish between no effect and opposite direction
        if p_value < 0.05 and coef < 0:
            direction_text = "fewer"
        else:
            direction_text = "not statistically different"
        conclusion_text = (
            f"No — the analysis does not provide evidence that dark-skinned players receive more red cards. "
            f"The SkinDark coefficient = {numeric_result['coef_log_rate']} (p = {numeric_result['p_value']}). "
            "This indicates that, after adjusting for covariates and exposure, the rate of red cards for Dark "
            f"players is {direction_text} compared to Light players. IRR = {numeric_result['IRR']} "
            f"(95% CI: {numeric_result['IRR_ci_lower']} to {numeric_result['IRR_ci_upper']})."
        )

    description = (
        "Extracted statistics for the SkinDark predictor from the fitted Negative Binomial GLM "
        "(offset = log(Matches)). Coefficient is on the log rate scale; IRR = exp(coef). "
        "Standard errors and p-values reflect the fitted model (cluster-robust SEs if the returned "
        "object was produced by get_robustcov_results). " + conclusion_text
    )

    return {"object": numeric_result, "description": description}