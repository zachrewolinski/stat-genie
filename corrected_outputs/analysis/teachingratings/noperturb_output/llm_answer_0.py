def extract_final_answer(model_output):
    """
    Extracts estimates and interprets the effect of instructor attractiveness (Beauty)
    and its squared term on course evaluation scores from a statsmodels RegressionResultsWrapper.

    Returns a dictionary with:
      - "object": a dictionary of numeric results (coefficients, SEs, p-values, 95% CIs,
                  marginal effect at mean, turning point of quadratic, and SE/CI/p for turning point)
      - "description": a concise plain-language interpretation of those results in context.
    """
    import numpy as np
    from math import sqrt, erf

    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    # Validate expected attributes exist on the model output
    required_attrs = ['params', 'bse', 'pvalues', 'conf_int', 'cov_params']
    for attr in required_attrs:
        if not hasattr(model_output, attr):
            raise AttributeError(f"model_output is missing required attribute/method: {attr}")

    params = model_output.params
    bse = model_output.bse
    pvals = model_output.pvalues
    ci = model_output.conf_int()  # DataFrame or ndarray with two columns [lower, upper]
    cov = model_output.cov_params()  # covariance matrix used for reported SEs (clustered cov)

    # Ensure the key coefficients exist
    for name in ['Beauty', 'Beauty_sq']:
        if name not in params.index:
            raise KeyError(f"Expected coefficient '{name}' not found in model output params.")

    # Extract stats for Beauty and Beauty_sq
    beta1 = float(params['Beauty'])
    se1 = float(bse['Beauty'])
    p1 = float(pvals['Beauty'])
    ci1 = ci.loc['Beauty'] if hasattr(ci, 'loc') else ci[ list(params.index).index('Beauty') ]
    ci1 = (float(ci1[0]), float(ci1[1]))

    beta2 = float(params['Beauty_sq'])
    se2 = float(bse['Beauty_sq'])
    p2 = float(pvals['Beauty_sq'])
    ci2 = ci.loc['Beauty_sq'] if hasattr(ci, 'loc') else ci[ list(params.index).index('Beauty_sq') ]
    ci2 = (float(ci2[0]), float(ci2[1]))

    # Marginal effect of Beauty on Eval at mean-centered Beauty = 0 is simply beta1
    me_mean = beta1
    me_mean_se = se1
    me_mean_ci = (me_mean - 1.96 * me_mean_se, me_mean + 1.96 * me_mean_se)
    me_mean_t = me_mean / me_mean_se if me_mean_se != 0 else float('nan')
    me_mean_p = p1  # already two-sided p-value for beta1

    # Compute turning point of quadratic (argmax/min): -beta1 / (2 * beta2), if beta2 != 0
    turn_point = None
    turn_point_se = None
    turn_point_ci = (None, None)
    turn_point_p = None
    if beta2 != 0:
        tp = -beta1 / (2.0 * beta2)
        turn_point = float(tp)
        # Delta method to get SE of tp = g(b1,b2) where g = -b1/(2 b2)
        # gradient g_b1 = -1/(2 b2); g_b2 = b1/(2 b2^2)
        g_b1 = -1.0 / (2.0 * beta2)
        g_b2 = beta1 / (2.0 * (beta2 ** 2))
        # Retrieve covariance submatrix for Beauty and Beauty_sq in correct order
        try:
            cov_b1b2 = np.array([[cov.loc['Beauty', 'Beauty'], cov.loc['Beauty', 'Beauty_sq']],
                                 [cov.loc['Beauty_sq', 'Beauty'], cov.loc['Beauty_sq', 'Beauty_sq']]])
        except Exception:
            # fallback if cov is ndarray with same ordering as params.index
            idx_b1 = list(params.index).index('Beauty')
            idx_b2 = list(params.index).index('Beauty_sq')
            cov_arr = np.asarray(cov)
            cov_b1b2 = cov_arr[[idx_b1, idx_b2]][:,[idx_b1, idx_b2]]

        grad = np.array([g_b1, g_b2])
        var_tp = float(grad @ cov_b1b2 @ grad.T)
        turn_point_se = sqrt(var_tp) if var_tp >= 0 else float('nan')
        turn_point_ci = (turn_point - 1.96 * turn_point_se, turn_point + 1.96 * turn_point_se) if turn_point_se == turn_point_se else (None, None)
        # approximate two-sided p-value for turning point != 0 (normal approximation)
        if turn_point_se is not None and turn_point_se != 0 and turn_point_se == turn_point_se:
            t_stat = turn_point / turn_point_se
            turn_point_p = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
        else:
            turn_point_p = None

    # Build numeric result object
    result_object = {
        'Beauty': {
            'coef': beta1,
            'se': se1,
            'p_value': p1,
            'ci_95': ci1
        },
        'Beauty_sq': {
            'coef': beta2,
            'se': se2,
            'p_value': p2,
            'ci_95': ci2
        },
        'marginal_effect_at_mean_centered_Beauty_0': {
            'effect': me_mean,
            'se': me_mean_se,
            'p_value': me_mean_p,
            'ci_95': me_mean_ci,
            'interpretation': "This is the instantaneous change in Eval for a one-unit increase in (mean-centered) Beauty at the mean of Beauty."
        },
        'turning_point_Beauty': {
            'value': turn_point,
            'se': turn_point_se,
            'p_value_approx': turn_point_p,
            'ci_95': turn_point_ci,
            'interpretation': "Value of mean-centered Beauty where predicted Eval is extremal (max if Beauty_sq < 0, min if > 0)."
        }
    }

    # Short interpretation string
    # Determine statistical significance at alpha=0.05 for the linear term at mean and squared term
    sig1 = "statistically significant (p < 0.05)" if (p1 < 0.05) else "not statistically significant (p >= 0.05)"
    sig2 = "statistically significant (p < 0.05)" if (p2 < 0.05) else "not statistically significant (p >= 0.05)"

    # Direction and shape interpretation
    shape = "concave (negative quadratic)" if beta2 < 0 else ("convex (positive quadratic)" if beta2 > 0 else "linear (no quadratic effect)")
    desc_lines = [
        f"Estimated effect of mean-centered Beauty on Eval (linear term): coef = {beta1:.4f}, SE = {se1:.4f}, p = {p1:.4g} -> {sig1}.",
        f"Estimated quadratic term (Beauty_sq): coef = {beta2:.6f}, SE = {se2:.6f}, p = {p2:.4g} -> {sig2}; shape = {shape}.",
        f"Marginal effect at mean-centered Beauty = 0: {me_mean:.4f} (95% CI [{me_mean_ci[0]:.4f}, {me_mean_ci[1]:.4f}]).",
    ]
    if turn_point is not None:
        desc_lines.append(f"Turning point (mean-centered Beauty) = {turn_point:.4f} (SE ≈ {turn_point_se:.4f}, 95% CI [{turn_point_ci[0]:.4f}, {turn_point_ci[1]:.4f}], approx. p = {turn_point_p:.4g}).")
        if beta2 < 0:
            desc_lines.append("Because the quadratic coefficient is negative, predicted Eval peaks at this turning point and then declines for larger Beauty.")
        elif beta2 > 0:
            desc_lines.append("Because the quadratic coefficient is positive, predicted Eval is minimized at this turning point and then increases for larger Beauty.")
    else:
        desc_lines.append("No quadratic turning point computed because the quadratic coefficient is exactly zero.")

    # Combine into one description string
    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}