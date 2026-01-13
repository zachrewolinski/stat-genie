def extract_final_answer(model_output):
    """
    Extracts the IsHuman effect from a fitted statsmodels GLM (or robust-covariance result).
    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, z, p, 95% CI, odds ratio and its CI, significant flag)
      - "description": human-readable interpretation of whether modern humans have higher AMTL
    """
    import numpy as np
    try:
        from scipy import stats
        _norm_cdf = stats.norm.cdf
        _norm_ppf = stats.norm.ppf
    except Exception:
        # Fallback implementations if scipy is not available
        import math
        def _norm_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        def _norm_ppf(q):
            # approximate inverse CDF using a simple approximation is complicated;
            # but we only need 97.5% critical value -> use known constant
            if abs(q - 0.975) < 1e-6:
                return 1.95996398454005
            raise RuntimeError("scipy not available for general ppf; only q=0.975 supported")

    # Ensure the model output has parameter estimates
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not expose .params; expected a statsmodels results object.")

    params = model_output.params
    # Check that IsHuman is present
    if "IsHuman" not in params.index:
        raise KeyError("The fitted model does not contain a parameter named 'IsHuman'.")

    coef = float(params["IsHuman"])

    # Attempt to get robust/clustered standard errors (model_output.bse should reflect robust SE if get_robustcov_results was used)
    if hasattr(model_output, "bse") and "IsHuman" in model_output.bse.index:
        se = float(model_output.bse["IsHuman"])
    else:
        # fallback: try model_output.std_errors or raise
        try:
            se = float(model_output.bse["IsHuman"])
        except Exception:
            raise ValueError("Could not extract standard error for 'IsHuman' from the model output.")

    # Compute z (Wald) statistic and two-sided p-value using normal approximation
    z = coef / se if se != 0 else np.nan
    pvalue = float(2.0 * (1.0 - _norm_cdf(abs(z)))) if not np.isnan(z) else np.nan

    # 95% CI using normal approximation and the obtained standard error
    crit = _norm_ppf(0.975)
    ci_lower = coef - crit * se
    ci_upper = coef + crit * se

    # Convert log-odds effect to odds ratio and CI
    try:
        or_coef = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower))
        or_ci_upper = float(np.exp(ci_upper))
    except Exception:
        or_coef = or_ci_lower = or_ci_upper = float("nan")

    # Determine significance and direction
    alpha = 0.05
    significant = (pvalue < alpha) if (not np.isnan(pvalue)) else False
    if significant:
        if coef > 0:
            conclusion = (
                f"Statistically significant positive effect: modern humans (IsHuman=1) have higher AMTL "
                f"than non-human primates after controlling for age, sex proxy, and tooth class "
                f"(coef = {coef:.4f}, SE = {se:.4f}, z = {z:.3f}, p = {pvalue:.3g}). "
                f"Odds ratio = {or_coef:.3f} (95% CI: {or_ci_lower:.3f}–{or_ci_upper:.3f})."
            )
        else:
            conclusion = (
                f"Statistically significant negative effect: modern humans (IsHuman=1) have lower AMTL "
                f"than non-human primates after controlling for covariates "
                f"(coef = {coef:.4f}, SE = {se:.4f}, z = {z:.3f}, p = {pvalue:.3g}). "
                f"Odds ratio = {or_coef:.3f} (95% CI: {or_ci_lower:.3f}–{or_ci_upper:.3f})."
            )
    else:
        # Not significant: state no evidence for difference (report direction and effect size)
        if coef > 0:
            direction = "higher"
        elif coef < 0:
            direction = "lower"
        else:
            direction = "no difference (coef = 0)"

        conclusion = (
            f"No statistically significant difference in AMTL between modern humans and non-human primates "
            f"after controlling for covariates (coef = {coef:.4f}, SE = {se:.4f}, z = {z:.3f}, p = {pvalue:.3g}). "
            f"Point estimate suggests {direction} AMTL in modern humans; odds ratio = {or_coef:.3f} "
            f"(95% CI: {or_ci_lower:.3f}–{or_ci_upper:.3f})."
        )

    result_object = {
        "parameter": "IsHuman",
        "coef_log_odds": coef,
        "se": se,
        "z": z,
        "p_value": pvalue,
        "ci_95_log_odds": [ci_lower, ci_upper],
        "odds_ratio": or_coef,
        "odds_ratio_95_ci": [or_ci_lower, or_ci_upper],
        "significant_at_0.05": bool(significant),
    }

    return {"object": result_object, "description": conclusion}