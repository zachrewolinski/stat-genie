def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, confidence interval, and
    exponentiated effect (rate ratio) for the SkinTone contrast (Dark vs Light)
    from a fitted statsmodels GLM/Results object (possibly with cluster-robust cov).
    
    Returns:
      dict with keys:
        - "object": dict containing numeric results for the SkinTone (Dark) term
        - "description": human-readable interpretation of the effect in context
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Get parameter names
    try:
        param_index = list(res.params.index)
    except Exception:
        raise ValueError("Provided model_output does not appear to have .params")

    # Find the SkinTone coefficient name.
    # Prefer a parameter name that contains both 'SkinTone' (case-insensitive) and 'Dark'.
    candidates = [n for n in param_index if ('skintone' in n.lower() or 'skintone' in n)]
    # If none found, broaden search to any name containing 'skin' or 'dark'
    if not candidates:
        candidates = [n for n in param_index if 'skin' in n.lower()]
    if not candidates:
        candidates = [n for n in param_index if 'dark' in n.lower()]

    if not candidates:
        raise ValueError("Could not locate a parameter name corresponding to SkinTone in model_output.params:\n"
                         f"available params: {param_index}")

    # If multiple candidates, prefer the one that explicitly references Dark
    chosen = None
    for n in candidates:
        if 'dark' in n.lower():
            chosen = n
            break
    if chosen is None:
        # fallback: take the first candidate
        chosen = candidates[0]

    term_name = chosen

    # Extract coefficient
    coef = float(res.params[term_name])

    # Standard error: prefer res.bse (works for robust results), else compute from cov_params
    try:
        se = float(res.bse[term_name])
    except Exception:
        # Compute from covariance matrix
        try:
            cov = res.cov_params()
            se = float(np.sqrt(np.abs(cov.loc[term_name, term_name])))
        except Exception:
            raise ValueError("Unable to obtain standard errors from the model output.")

    # z-stat and p-value
    # Prefer res.pvalues if available
    try:
        p_value = float(res.pvalues[term_name])
        # compute z from coef and se for reporting
        z_stat = float(coef / se) if se != 0 else np.nan
    except Exception:
        # compute p-value from normal approx
        z_stat = float(coef / se) if se != 0 else np.nan
        p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # 95% CI: prefer res.conf_int(), else normal approximation
    try:
        ci_df = res.conf_int()
        if term_name in ci_df.index:
            ci_lower = float(ci_df.loc[term_name, 0])
            ci_upper = float(ci_df.loc[term_name, 1])
        else:
            # If conf_int exists but term not in index, fallback
            zcrit = stats.norm.ppf(0.975)
            ci_lower = coef - zcrit * se
            ci_upper = coef + zcrit * se
    except Exception:
        zcrit = stats.norm.ppf(0.975)
        ci_lower = coef - zcrit * se
        ci_upper = coef + zcrit * se

    # Exponentiate to get rate ratio (incidence rate ratio for negative binomial)
    rate_ratio = float(np.exp(coef))
    rr_ci_lower = float(np.exp(ci_lower))
    rr_ci_upper = float(np.exp(ci_upper))

    # Significance flag (alpha = 0.05)
    significant = bool(p_value < 0.05)

    # Build the "object" to return (actual numeric results)
    result_object = {
        "term": term_name,
        "coef": coef,
        "se": se,
        "z": z_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rate_ratio": rate_ratio,
        "rr_ci_lower": rr_ci_lower,
        "rr_ci_upper": rr_ci_upper,
        "significant_at_0.05": significant,
        # interpretation helpers
        "effect_direction": "higher" if coef > 0 else ("lower" if coef < 0 else "no_difference"),
        "note": "Rate ratio >1 means Dark skin tone players receive red cards at a higher rate than Light (per game)."
    }

    # Human-readable description
    if significant:
        desc = (
            f"The model term '{term_name}' has coefficient={coef:.4f} (SE={se:.4f}, z={z_stat:.2f}, p={p_value:.3g}),\n"
            f"95% CI for coef = [{ci_lower:.4f}, {ci_upper:.4f}].\n"
            f"Exponentiated, the incidence rate ratio (IRR) = {rate_ratio:.3f} "
            f"(95% CI = [{rr_ci_lower:.3f}, {rr_ci_upper:.3f}]).\n"
            f"This indicates a statistically significant {result_object['effect_direction']} rate of red cards for Dark vs Light skinned players "
            f"(alpha=0.05)."
        )
    else:
        desc = (
            f"The model term '{term_name}' has coefficient={coef:.4f} (SE={se:.4f}, z={z_stat:.2f}, p={p_value:.3g}),\n"
            f"95% CI for coef = [{ci_lower:.4f}, {ci_upper:.4f}].\n"
            f"Exponentiated, the incidence rate ratio (IRR) = {rate_ratio:.3f} "
            f"(95% CI = [{rr_ci_lower:.3f}, {rr_ci_upper:.3f}]).\n"
            f"This does NOT provide statistically significant evidence that Dark vs Light skinned players differ in red card rates (alpha=0.05)."
        )

    return {"object": result_object, "description": desc}