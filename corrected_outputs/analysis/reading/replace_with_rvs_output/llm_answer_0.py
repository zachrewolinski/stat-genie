def extract_final_answer(model_output):
    """
    Extracts the marginal effect of Reader View for readers WITH dyslexia
    from a fitted statsmodels MixedLMResults (or wrapper).

    Returns a dict with:
      - "object": dict of extracted numeric results (estimate on log scale,
                  SE, z, p-value, 95% CI, percent change on original scale)
      - "description": short interpretation in context

    Notes:
      - The marginal effect for dyslexic readers = coef(reader_view) + coef(reader_view:dyslexia_bin)
      - SE and p-value are computed using the variance-covariance matrix of the fixed effects:
        var(a+b) = var(a) + var(b) + 2 cov(a,b)
      - 95% CI is approximated with normal-critical value 1.96 (large-sample approximation).
    """
    import math
    import numpy as np

    res = model_output

    # Pull parameters and covariance matrix
    try:
        params = res.params  # pandas Series
        cov = res.cov_params()  # DataFrame
    except Exception as e:
        raise ValueError("model_output does not appear to have .params or .cov_params(): %s" % e)

    # Helper to find parameter names robustly
    param_names = list(params.index)

    # Find reader_view main effect name (prefer exact match)
    name_rv = None
    if 'reader_view' in param_names:
        name_rv = 'reader_view'
    else:
        # choose a parameter that contains 'reader_view' but not 'dyslexia'
        for n in param_names:
            if 'reader_view' in n and 'dyslexia' not in n:
                name_rv = n
                break

    # Find interaction term name (contains both substrings)
    name_inter = None
    for n in param_names:
        if 'reader_view' in n and 'dyslexia' in n:
            name_inter = n
            break

    if name_rv is None or name_inter is None:
        raise ValueError(
            "Could not find expected parameter names for 'reader_view' and its interaction with 'dyslexia_bin'. "
            "Available parameter names: %s" % (param_names,)
        )

    # Extract coefficients
    beta_rv = float(params[name_rv])
    beta_int = float(params[name_inter])

    # Marginal effect for dyslexia==1
    est = beta_rv + beta_int

    # Variance of the sum
    try:
        var_rv = float(cov.loc[name_rv, name_rv])
        var_int = float(cov.loc[name_inter, name_inter])
        cov_rv_int = float(cov.loc[name_rv, name_inter])
    except Exception as e:
        raise ValueError("Could not extract covariances for parameters: %s" % e)

    var_sum = var_rv + var_int + 2.0 * cov_rv_int
    if var_sum < 0:
        # numerical safety
        se = float(np.sqrt(max(var_sum, 0.0)))
    else:
        se = float(math.sqrt(var_sum))

    # z-stat and two-sided p-value using normal approx
    if se == 0:
        z = float('inf') if est != 0 else 0.0
        p_value = 0.0 if z == float('inf') else 1.0
    else:
        z = est / se
        # two-sided p-value from normal: p = 1 - erf(|z|/sqrt(2))
        p_value = 1.0 - math.erf(abs(z) / math.sqrt(2.0))

    # 95% CI using normal approx (critical z ≈ 1.96)
    z_crit = 1.96
    ci_lower = est - z_crit * se
    ci_upper = est + z_crit * se

    # Transform back to percent change in reading speed (since DV is log speed):
    # multiplicative factor = exp(est); percent change = (exp(est)-1)*100
    percent_change = float((math.exp(est) - 1.0) * 100.0)

    result_object = {
        "estimate_log_scale": est,
        "se_log_scale": se,
        "z_value": z,
        "p_value": p_value,
        "95ci_log_scale": (ci_lower, ci_upper),
        "percent_change_speed": percent_change,
        "reader_view_param_name": name_rv,
        "interaction_param_name": name_inter
    }

    # Interpretation sentence
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    direction = "increases" if est > 0 else ("decreases" if est < 0 else "no change in")
    description = (
        f"The estimated marginal effect of turning Reader View ON for readers with dyslexia "
        f"(log scale) is {est:.4f} (SE = {se:.4f}, z = {z:.2f}, p = {p_value:.3g}). "
        f"The 95% CI on the log scale is [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"This corresponds to a {percent_change:.1f}% {direction} in reading speed. "
        f"The effect is {significance} at the 0.05 level. "
        f"Parameters used: reader_view='{name_rv}', interaction='{name_inter}'."
    )

    return {"object": result_object, "description": description}