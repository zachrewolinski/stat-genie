def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator on mortgage approval from a fitted statsmodels logit result.

    Returns a dict:
      - "object": dict containing numeric statistics (coef, se, z, p-value, odds ratio, 95% CI on log-odds and odds ratio)
      - "description": brief plain-language interpretation of whether gender (female) affects approval and whether the effect is statistically significant.

    The function prefers robust (HC3) covariance if it is attached to the results object (via cov_robust or a cov_params() method that returns robust cov).
    """
    import math
    import numpy as np

    res = model_output

    # Helper: get parameter series
    try:
        params = res.params
    except Exception:
        raise ValueError("The provided model_output does not expose .params")

    # Find the name for the female coefficient (exact match or contains 'female')
    female_name = None
    if 'female' in params.index:
        female_name = 'female'
    else:
        # fallback: search for any parameter name that contains 'female'
        for name in params.index:
            if 'female' in str(name).lower():
                female_name = name
                break
    if female_name is None:
        raise ValueError("No parameter named 'female' (or containing 'female') found in model parameters.")

    coef = float(params[female_name])

    # Attempt to obtain a covariance matrix (robust if available)
    cov = None
    try:
        # If the results object has an attribute cov_robust (set by the model function), use it
        if hasattr(res, 'cov_robust'):
            cov = getattr(res, 'cov_robust')
        else:
            # cov_params might be a callable (method) or an attribute (matrix)
            cov_params_attr = getattr(res, 'cov_params', None)
            if callable(cov_params_attr):
                try:
                    cov = cov_params_attr()
                except Exception:
                    # cov_params might require no args but still raise; ignore
                    cov = None
            else:
                cov = cov_params_attr
    except Exception:
        cov = None

    se = None
    if cov is not None:
        # cov should be an array-like with indices aligned to params
        try:
            # if cov is a DataFrame, get diag by index
            if hasattr(cov, 'loc'):
                se = float(np.sqrt(np.diag(cov.loc[params.index, params.index]))[list(params.index).index(female_name)])
            else:
                # assume numpy array aligned with params order
                idx = list(params.index).index(female_name)
                se = float(np.sqrt(np.diag(cov))[idx])
        except Exception:
            se = None

    # If covariance not available or se extraction failed, fall back to res.bse if present
    if se is None:
        try:
            bse = res.bse
            se = float(bse[female_name])
            cov_source = "model bse (non-robust or pre-computed)"
        except Exception:
            raise ValueError("Could not determine standard error for the 'female' coefficient from the model output.")
    else:
        cov_source = "covariance (robust if provided by model output)"

    # Compute z-stat and two-sided p-value using normal approximation
    z = coef / se if se != 0 else float('nan')
    # two-sided p-value: use erfc to avoid dependency on scipy: p = erfc(|z|/sqrt(2))
    p_value = float(math.erfc(abs(z) / math.sqrt(2)))

    # 95% CI on log-odds scale and transform to odds ratio
    z_crit = 1.96
    ci_low = coef - z_crit * se
    ci_high = coef + z_crit * se
    odds_ratio = float(math.exp(coef))
    ci_or_low = float(math.exp(ci_low))
    ci_or_high = float(math.exp(ci_high))

    # Prepare numeric object (JSON-serializable)
    stats = {
        "parameter_name": str(female_name),
        "coef_log_odds": coef,
        "se_used": se,
        "covariance_source": cov_source,
        "z_stat": z,
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "95ci_log_odds": [ci_low, ci_high],
        "95ci_odds_ratio": [ci_or_low, ci_or_high],
    }

    # Simple interpretation
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    direction = ("decrease" if coef < 0 else "increase" if coef > 0 else "no change")
    description = (
        f"The 'female' coefficient is {coef:.4f} (log-odds). "
        f"That corresponds to an odds ratio of {odds_ratio:.3f} (95% CI: [{ci_or_low:.3f}, {ci_or_high:.3f}]). "
        f"Using the standard error from the model ({cov_source}), the two-sided p-value is {p_value:.3g}, "
        f"so the effect is {significance} at the 5% level. "
        f"In plain terms: being female is associated with a {direction} in the odds of mortgage approval "
        f"(relative to being male); see the odds ratio and confidence interval above for magnitude and precision."
    )

    return {"object": stats, "description": description}