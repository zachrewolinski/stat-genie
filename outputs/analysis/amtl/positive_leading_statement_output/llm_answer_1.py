import numpy as np

def extract_final_answer(model_output):
    """
    Extract the effect of 'is_human' from a fitted statsmodels GLM results object.

    Input:
      model_output: dict-like with keys:
        - 'glm_result': a fitted statsmodels GLMResultsWrapper (required)
        - 'glm_result_clustered_by_pop': optional clustered-robust result (use if not None)

    Returns: dict with keys:
      - "object": dict with numeric results (coef, se, p_value, 95% CI on log-odds and odds ratio, which covariance was used)
      - "description": human-readable interpretation answering whether modern humans have higher AMTL after controls
    """

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the model function.")

    if 'glm_result' not in model_output or model_output['glm_result'] is None:
        raise ValueError("model_output must contain a non-null 'glm_result' entry.")

    # Prefer clustered-robust results if provided
    used_clustered = False
    clustered = model_output.get('glm_result_clustered_by_pop')
    res = clustered if clustered is not None else model_output.get('glm_result')
    if clustered is not None:
        used_clustered = True

    # Ensure the results object has the expected attributes/methods
    required_attrs = ('params', 'bse', 'pvalues')
    for attr in required_attrs:
        if not hasattr(res, attr):
            raise ValueError(f"The results object is missing expected attribute '{attr}'.")

    # 'conf_int' may be either a method or attribute; ensure we can call it or access it
    if not (hasattr(res, 'conf_int') or hasattr(res, 'conf_int')):
        # redundant check kept for clarity; if conf_int doesn't exist we'll handle later
        pass

    # Check that 'is_human' is a fitted coefficient
    params = res.params
    try:
        index_names = list(params.index)
    except Exception:
        # params may be a plain ndarray with no index
        raise ValueError("The results.params object does not expose an index; cannot find 'is_human' coefficient.")

    if 'is_human' not in index_names:
        raise KeyError("The fitted model does not contain a coefficient named 'is_human'.")

    # Extract estimates
    coef = float(params['is_human'])

    # standard error: try .bse (Series) first
    se = None
    try:
        se = float(res.bse['is_human'])
    except Exception:
        # fallback: compute from cov_params if available
        try:
            cov = res.cov_params()
            idx = index_names.index('is_human')
            se = float(np.sqrt(np.diag(cov))[idx])
        except Exception:
            se = None

    # p-value
    p_value = None
    try:
        p_value = float(res.pvalues['is_human'])
    except Exception:
        p_value = None

    # 95% confidence interval on coefficient (log-odds)
    ci_lower = ci_upper = None
    try:
        # conf_int may be a method or an attribute returning a DataFrame/ndarray
        ci_obj = res.conf_int() if callable(getattr(res, "conf_int", None)) else getattr(res, "conf_int")
        # conf_int may return a DataFrame or ndarray; handle both
        if hasattr(ci_obj, 'loc'):
            ci_lower = float(ci_obj.loc['is_human', 0])
            ci_upper = float(ci_obj.loc['is_human', 1])
        else:
            # assume same ordering as params
            idx = index_names.index('is_human')
            ci_lower = float(ci_obj[idx, 0])
            ci_upper = float(ci_obj[idx, 1])
    except Exception:
        ci_lower = None
        ci_upper = None

    # Convert to odds ratio and CI on odds ratio scale (exp)
    try:
        or_est = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        or_est = or_ci_lower = or_ci_upper = None

    # Formulate a simple conclusion: positive coef + statistically significant => humans have higher AMTL.
    sig_level = 0.05
    if p_value is None:
        conclusion = ("Could not determine statistical significance (p-value unavailable). "
                      "Reporting estimates but no definitive yes/no conclusion.")
    else:
        if coef > 0 and p_value < sig_level:
            conclusion = ("Yes — the coefficient for 'is_human' is positive and statistically significant "
                          f"(coef={coef:.4f}, p={p_value:.3g}). This indicates modern humans have higher "
                          "AMTL (higher odds) than the reference non-human primates after adjusting for the "
                          "included covariates.")
        elif coef > 0 and p_value >= sig_level:
            conclusion = ("No strong evidence — the coefficient for 'is_human' is positive but not "
                          f"statistically significant (coef={coef:.4f}, p={p_value:.3g}). The point estimate "
                          "suggests higher AMTL in modern humans, but it is not reliably different from zero "
                          "at the 0.05 level after adjustment.")
        elif coef < 0 and p_value < sig_level:
            conclusion = ("No — the coefficient for 'is_human' is negative and statistically significant "
                          f"(coef={coef:.4f}, p={p_value:.3g}), indicating modern humans have lower AMTL "
                          "than the non-human primates after adjustment.")
        else:
            conclusion = ("No evidence of a difference — the coefficient for 'is_human' is negative or near zero "
                          f"and not statistically significant (coef={coef:.4f}, p={p_value:.3g}).")

    # Prepare human-readable strings for the numeric parts
    se_str = f"{se:.4f}" if se is not None else "NA"
    if ci_lower is not None and ci_upper is not None:
        ci_str = f"({ci_lower:.4f}, {ci_upper:.4f})"
    else:
        ci_str = "NA"

    or_str = f"{or_est:.4f}" if or_est is not None else "NA"
    or_ci_str = (f"({or_ci_lower:.4f}, {or_ci_upper:.4f})"
                 if or_ci_lower is not None and or_ci_upper is not None else "NA")

    # Package the numeric object to return
    result_object = {
        'coef_log_odds': coef,
        'se_log_odds': se,
        'p_value': p_value,
        'ci_log_odds_95': (ci_lower, ci_upper),
        'odds_ratio': or_est,
        'odds_ratio_ci_95': (or_ci_lower, or_ci_upper),
        'used_clustered_cov': bool(used_clustered)
    }

    description = (
        f"Extracted effect of 'is_human' from a binomial GLM (logit link). "
        f"Covariance used: {'cluster-robust by pop' if used_clustered else 'model (non-clustered) covariance'}. "
        f"Coefficient (log-odds) = {coef:.4f}, SE = {se_str}, 95% CI (log-odds) = {ci_str}. "
        f"Odds ratio = {or_str}, 95% CI (OR) = {or_ci_str}. "
        + conclusion
    )

    return {
        "object": result_object,
        "description": description
    }