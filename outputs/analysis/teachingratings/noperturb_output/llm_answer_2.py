def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of instructor beauty on evaluations
    from a fitted statsmodels RegressionResultsWrapper (with clustered SEs).

    Returns a dict with:
      - "object": a dict containing coefficients, SEs, t-stats, p-values, 95% CIs
                  for Beauty_z and Beauty_z_sq, plus the marginal effect at mean
                  beauty (0, because Beauty_z is standardized) with its SE and CI.
      - "description": a short interpretation of these statistics in context.
    """
    import numpy as np

    res = model_output

    # Prepare output structure
    result_obj = {}

    # Safely access param-related attributes
    params = getattr(res, "params", None)
    bse = getattr(res, "bse", None)
    tvals = getattr(res, "tvalues", None)
    pvals = getattr(res, "pvalues", None)
    try:
        conf = res.conf_int()  # DataFrame with two columns [lower, upper]
    except Exception:
        conf = None

    # If any of the required pieces are missing, return informative message
    if params is None or bse is None or pvals is None or tvals is None:
        return {
            "object": None,
            "description": "Model output does not have the expected attributes (params, bse, pvalues, tvalues)."
        }

    # Helper to extract stats for a given term
    def term_stats(term):
        if term not in params.index:
            return None
        lower_ci, upper_ci = (None, None)
        if conf is not None and term in conf.index:
            lower_ci = float(conf.loc[term, 0])
            upper_ci = float(conf.loc[term, 1])
        return {
            "coef": float(params[term]),
            "std_err": float(bse[term]),
            "t_stat": float(tvals[term]),
            "p_value": float(pvals[term]),
            "ci_95_lower": lower_ci,
            "ci_95_upper": upper_ci
        }

    # Extract for Beauty_z and Beauty_z_sq
    result_obj["Beauty_z"] = term_stats("Beauty_z")
    result_obj["Beauty_z_sq"] = term_stats("Beauty_z_sq")

    # Compute marginal effect at mean beauty (Beauty_z = 0).
    # Marginal effect = dEval/dBeauty = beta1 + 2 * beta2 * beauty
    beta1 = float(params.get("Beauty_z", 0.0)) if "Beauty_z" in params.index else None
    beta2 = float(params.get("Beauty_z_sq", 0.0)) if "Beauty_z_sq" in params.index else 0.0

    marginal_at_mean = None
    if beta1 is not None:
        me = beta1 + 2.0 * beta2 * 0.0  # at beauty = 0
        # Compute SE for the marginal effect using covariance matrix (delta method)
        try:
            cov = res.cov_params()
            # var(me) = var(beta1) + (2*beauty)^2 var(beta2) + 2*(2*beauty) cov(beta1,beta2)
            beauty_val = 0.0
            var_me = float(cov.loc["Beauty_z", "Beauty_z"])
            if "Beauty_z_sq" in cov.index:
                var_me += (2.0 * beauty_val) ** 2 * float(cov.loc["Beauty_z_sq", "Beauty_z_sq"])
                var_me += 2.0 * (2.0 * beauty_val) * float(cov.loc["Beauty_z", "Beauty_z_sq"])
            se_me = float(np.sqrt(var_me))
            ci_lower = me - 1.96 * se_me
            ci_upper = me + 1.96 * se_me
        except Exception:
            se_me = None
            ci_lower = None
            ci_upper = None

        marginal_at_mean = {
            "marginal_effect_at_mean_beauty": float(me),
            "std_err": se_me,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper
        }

    result_obj["marginal_at_mean"] = marginal_at_mean

    # Short interpretation
    desc_lines = []
    desc_lines.append("Extracted coefficients and cluster-robust inference for the beauty terms.")
    if result_obj["Beauty_z"] is not None:
        bz = result_obj["Beauty_z"]
        sig = ("statistically significant (p < 0.05)" if bz["p_value"] < 0.05 else "not statistically significant (p >= 0.05)")
        desc_lines.append(
            f"Beauty_z: coef = {bz['coef']:.4f}, SE = {bz['std_err']:.4f}, "
            f"95% CI = [{bz['ci_95_lower']:.4f}, {bz['ci_95_upper']:.4f}] (p = {bz['p_value']:.4g}) -> {sig}."
        )
        # Quadratic term
        if result_obj["Beauty_z_sq"] is not None:
            bzs = result_obj["Beauty_z_sq"]
            sigsq = ("statistically significant (p < 0.05)" if bzs["p_value"] < 0.05 else "not statistically significant (p >= 0.05)")
            desc_lines.append(
                f"Beauty_z_sq: coef = {bzs['coef']:.4f}, SE = {bzs['std_err']:.4f}, "
                f"95% CI = [{bzs['ci_95_lower']:.4f}, {bzs['ci_95_upper']:.4f}] (p = {bzs['p_value']:.4g}) -> {sigsq}."
            )
        # Marginal at mean
        if marginal_at_mean is not None:
            me = marginal_at_mean
            if me["std_err"] is not None:
                desc_lines.append(
                    f"Marginal effect at mean beauty (0): {me['marginal_effect_at_mean_beauty']:.4f} (SE = {me['std_err']:.4f}), "
                    f"95% CI = [{me['ci_95_lower']:.4f}, {me['ci_95_upper']:.4f}]. "
                    "Interpretation: a one-standard-deviation increase in instructor attractiveness is associated with this change in the evaluation score."
                )
            else:
                desc_lines.append(
                    f"Marginal effect at mean beauty (0): {me['marginal_effect_at_mean_beauty']:.4f}. SE/CI could not be computed."
                )
    else:
        desc_lines.append("Beauty_z term not found in the model results.")

    description = " ".join(desc_lines)

    return {"object": result_obj, "description": description}