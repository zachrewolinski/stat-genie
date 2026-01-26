def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of instructor beauty on evaluations
    from a fitted statsmodels OLS result or from the wrapper dict produced by
    the modeling function.

    Returns a dict with:
      - "object": dict of numeric results (coefficients, SEs, p-values, CIs,
                    marginal effects, vertex of quadratic, boolean significance)
      - "description": brief textual interpretation of those results

    Expected input:
      - Either the statsmodels RegressionResultsWrapper object, or
      - A dict like {'model_fit': RegressionResultsWrapper, 'coef_table': ..., 'n_obs': ...}
    """
    import numpy as np

    # Accept either the plain model or the dict wrapper used in the task
    if isinstance(model_output, dict) and 'model_fit' in model_output:
        res = model_output['model_fit']
    else:
        res = model_output

    # Basic safety checks
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not appear to contain a fitted statsmodels result (no .params).")

    # Extract coefficients, SEs, p-values, and confidence intervals
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    try:
        ci = res.conf_int()  # DataFrame with columns [0,1]
    except Exception:
        # fallback: try to get conf_int from summary table if available
        ci = None

    # Helper to safely get a value (return NaN if missing)
    def get_val(series, name):
        return float(series[name]) if name in series.index else float('nan')

    # Coefficients of interest
    coef_linear = get_val(params, 'beauty_z')
    se_linear = get_val(bse, 'beauty_z')
    p_linear = get_val(pvals, 'beauty_z')
    if ci is not None and 'beauty_z' in ci.index:
        ci_linear = (float(ci.loc['beauty_z', 0]), float(ci.loc['beauty_z', 1]))
    else:
        ci_linear = (np.nan, np.nan)

    coef_quad = get_val(params, 'beauty_sq')
    se_quad = get_val(bse, 'beauty_sq')
    p_quad = get_val(pvals, 'beauty_sq')
    if ci is not None and 'beauty_sq' in ci.index:
        ci_quad = (float(ci.loc['beauty_sq', 0]), float(ci.loc['beauty_sq', 1]))
    else:
        ci_quad = (np.nan, np.nan)

    # Marginal effect at mean (beauty_z = 0) for quadratic: derivative = coef_linear + 2*coef_quad*beauty_z
    marginal_at_mean = coef_linear  # because beauty_z is standardized with mean 0

    # Predicted change in eval for a one-standard-deviation increase from 0 to +1:
    # delta = coef_linear*(1 - 0) + coef_quad*(1^2 - 0^2) = coef_linear + coef_quad
    effect_plus_1sd = coef_linear + coef_quad

    # Vertex of the quadratic (value of beauty_z where predicted eval is maximized/minimized)
    vertex = None
    predicted_change_at_vertex = None
    if not np.isnan(coef_quad) and coef_quad != 0:
        vertex = -coef_linear / (2.0 * coef_quad)
        # predicted difference (relative to beauty_z=0) at that vertex:
        predicted_change_at_vertex = coef_linear * vertex + coef_quad * (vertex ** 2)

    # Decide statistical significance: consider beauty effect significant if either term is significant at alpha=0.05
    alpha = 0.05
    linear_signif = (not np.isnan(p_linear)) and (p_linear < alpha)
    quad_signif = (not np.isnan(p_quad)) and (p_quad < alpha)
    any_significant = linear_signif or quad_signif

    # Build numeric object to return
    numeric_result = {
        'coef_beauty_z': coef_linear,
        'se_beauty_z': se_linear,
        'p_beauty_z': p_linear,
        'ci95_beauty_z': ci_linear,
        'coef_beauty_sq': coef_quad,
        'se_beauty_sq': se_quad,
        'p_beauty_sq': p_quad,
        'ci95_beauty_sq': ci_quad,
        'marginal_effect_at_mean': marginal_at_mean,
        'effect_for_+1SD_in_beauty': effect_plus_1sd,
        'vertex_beauty_z': vertex,
        'predicted_change_at_vertex': predicted_change_at_vertex,
        'any_beauty_term_significant_at_0.05': bool(any_significant),
        'linear_term_significant': bool(linear_signif),
        'quadratic_term_significant': bool(quad_signif),
    }

    # Short interpretation in context
    if any_significant:
        desc = (
            "The model indicates a statistically significant relationship between instructor beauty and "
            "student evaluations (at least one beauty term p < 0.05). See numeric results for coefficients, "
            "standard errors, p-values, and confidence intervals. Interpret the sign and magnitude of the "
            "linear and quadratic coefficients to determine direction and non-linearity of the effect."
        )
    else:
        # Use estimated magnitudes to provide concrete statement
        desc = (
            "There is no evidence of a statistically significant effect of instructor beauty on course evaluations: "
            f"the linear beauty coefficient = {coef_linear:.4g} (SE = {se_linear:.4g}, p = {p_linear:.3g}), "
            f"the quadratic term = {coef_quad:.4g} (SE = {se_quad:.4g}, p = {p_quad:.3g}). "
            f"A one-standard-deviation increase in standardized beauty is associated with an estimated change in "
            f"the evaluation score of {effect_plus_1sd:.4g} points (95% CIs reported above), which is small and "
            "not statistically significant. The vertex of the quadratic (if meaningful) is at beauty_z = "
            f"{vertex:.4g} and implies at most a negligible change in predicted evaluations."
        )

    return {
        "object": numeric_result,
        "description": desc
    }