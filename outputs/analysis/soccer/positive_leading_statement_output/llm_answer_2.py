def extract_final_answer(model_output):
    """
    Extracts statistics for the 'SkinDark' coefficient from a fitted statsmodels GLMResultsWrapper
    (negative binomial with cluster-robust SEs as in the provided modeling function).
    
    Returns a dictionary with:
      - "object": a dict of numeric results (coefficient, SE, z, p-value, 95% CI, IRR and its CI, nobs, parameter name, significance boolean)
      - "description": a human-readable interpretation of the result in the context of whether dark-skinned players
                       are more likely than light-skinned players to receive red cards (per game).
    """
    import numpy as np
    import math

    res = model_output

    # Ensure params exist
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    # Find the parameter name that corresponds to SkinDark
    param_names = list(res.params.index)
    matches = [n for n in param_names if "SkinDark" in n]
    if len(matches) == 0:
        raise KeyError(f'No parameter name containing "SkinDark" found in model params: {param_names}')
    # If multiple matches, pick the first (typical case: exactly "SkinDark")
    param = matches[0]

    # Extract coefficient
    coef = float(res.params[param])

    # Extract standard error (prefer bse if available)
    try:
        se = float(res.bse[param])
    except Exception:
        # Fall back to covariance matrix diagonal
        cov = res.cov_params()
        se = float(np.sqrt(cov.loc[param, param]))

    # z (or Wald) statistic and p-value
    z_stat = coef / se if se != 0 else float("nan")
    try:
        p_value = float(res.pvalues[param])
    except Exception:
        # two-sided normal approximation
        from scipy import stats
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # 95% CI for coefficient
    try:
        ci = res.conf_int().loc[param].astype(float)
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # approximate using normal quantiles
        z97 = 1.96
        ci_lower = coef - z97 * se
        ci_upper = coef + z97 * se

    # Incidence Rate Ratio (IRR) and CI on multiplicative scale
    irr = float(math.exp(coef))
    irr_ci_lower = float(math.exp(ci_lower))
    irr_ci_upper = float(math.exp(ci_upper))

    # Number of observations (fallbacks if attribute missing)
    n_obs = None
    if hasattr(res, "nobs"):
        try:
            n_obs = int(res.nobs)
        except Exception:
            n_obs = None
    if n_obs is None:
        try:
            n_obs = int(getattr(res.model, "nobs"))
        except Exception:
            try:
                n_obs = int(len(res.model.endog))
            except Exception:
                n_obs = None

    significant = (p_value < 0.05)

    # Create interpretation
    if significant:
        if irr > 1:
            interpretation = (
                f"The SkinDark coefficient is positive and statistically significant "
                f"(coef={coef:.3f}, SE={se:.3f}, z={z_stat:.2f}, p={p_value:.3g}). "
                f"Estimated incidence rate ratio (IRR) = {irr:.2f} (95% CI [{irr_ci_lower:.2f}, {irr_ci_upper:.2f}]). "
                "This indicates that, controlling for model covariates and exposure (games), "
                "players with a dark skin tone receive red cards at a higher rate per game than light-skinned players."
            )
        else:
            interpretation = (
                f"The SkinDark coefficient is negative and statistically significant "
                f"(coef={coef:.3f}, SE={se:.3f}, z={z_stat:.2f}, p={p_value:.3g}). "
                f"Estimated incidence rate ratio (IRR) = {irr:.2f} (95% CI [{irr_ci_lower:.2f}, {irr_ci_upper:.2f}]). "
                "This indicates that, controlling for model covariates and exposure (games), "
                "players with a dark skin tone receive red cards at a lower rate per game than light-skinned players."
            )
    else:
        interpretation = (
            f"No statistically significant difference was found in red-card rates for dark- vs light-skinned players "
            f"(coef={coef:.3f}, SE={se:.3f}, z={z_stat:.2f}, p={p_value:.3g}). "
            f"The estimated IRR = {irr:.2f} (95% CI [{irr_ci_lower:.2f}, {irr_ci_upper:.2f}]), "
            "which should be interpreted as the estimated multiplicative change in red-card rate per game for dark-skinned "
            "players relative to light-skinned players, controlling for the listed covariates and clustering by referee."
        )

    result_object = {
        "param_name": param,
        "coef": coef,
        "se": se,
        "z": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "n_obs": n_obs,
        "significant": significant
    }

    return {"object": result_object, "description": interpretation}