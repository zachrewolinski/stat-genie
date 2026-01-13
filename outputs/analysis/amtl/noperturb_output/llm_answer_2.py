def extract_final_answer(model_output):
    """
    Extracts the coefficient, clustered SE, test statistic, p-value, 95% CI, and odds-ratio for the IsHuman predictor
    from a fitted statsmodels GLMResults-like object whose covariance and bse may have been overridden for clustering.
    
    Returns:
      dict with keys:
        - "object": dict with numeric values (coef, se, z, p_value, ci_95, odds_ratio, odds_ratio_95, conclusion)
        - "description": human-readable interpretation of the result in context
    """
    import numpy as np
    from scipy import stats

    # Get parameter names and locate the IsHuman parameter (robust to small naming variations)
    params = model_output.params
    param_names = list(params.index)
    # find parameter name containing 'IsHuman'
    param_name = next((n for n in param_names if 'IsHuman' in n), None)
    if param_name is None:
        raise ValueError("Could not find a parameter whose name contains 'IsHuman' in model_output.params")

    # Coefficient value
    coef = float(params[param_name])

    # Standard error: handle both Series-like bse or numpy array bse (the user's code overwrote res.bse)
    try:
        se = float(model_output.bse[param_name])
    except Exception:
        # fallback to positional indexing
        idx = param_names.index(param_name)
        se = float(np.asarray(model_output.bse)[idx])

    # z-statistic and two-sided p-value (normal approximation for GLM coefficient)
    z = coef / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% Wald confidence interval
    z975 = stats.norm.ppf(0.975)
    ci_lower = coef - z975 * se
    ci_upper = coef + z975 * se

    # Exponentiate to get odds ratio interpretation (since Binomial GLM uses logit link by default)
    odds_ratio = float(np.exp(coef))
    odds_ratio_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))

    # Simple conclusion at alpha = 0.05
    significant = (p_value < 0.05)
    direction = "higher" if coef > 0 else "lower"
    conclusion = (
        "Yes" if (coef > 0 and significant) else
        "No" if (not (coef > 0 and significant)) else "Inconclusive"
    )

    # Prepare the returned objects
    result_obj = {
        "param_name": param_name,
        "coef": coef,
        "se": se,
        "z": z,
        "p_value": p_value,
        "ci_95": (ci_lower, ci_upper),
        "odds_ratio": odds_ratio,
        "odds_ratio_95": odds_ratio_ci,
        "significant_at_0.05": significant,
        "conclusion_yes_modern_humans_higher_AMTL": (coef > 0 and significant)
    }

    description = (
        f"Parameter '{param_name}': coefficient = {coef:.4f} (SE = {se:.4f}), z = {z:.3f}, p = {p_value:.3e}. "
        f"95% CI for coefficient = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"Odds ratio = {odds_ratio:.3f} (95% CI = [{odds_ratio_ci[0]:.3f}, {odds_ratio_ci[1]:.3f}]). "
        f"Interpretation: {'Evidence supports' if (coef > 0 and significant) else 'No statistically significant evidence that'} "
        f"modern humans have {direction} AMTL compared to non-human primates after controlling for age, sex, and tooth class "
        f"(alpha = 0.05)."
    )

    return {"object": result_obj, "description": description}