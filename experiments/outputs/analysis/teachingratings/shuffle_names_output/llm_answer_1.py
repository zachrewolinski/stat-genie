def extract_final_answer(model_output):
    """
    Extract key statistics for the beauty coefficient from a fitted statsmodels
    regression results object (or from the provided summary dictionary).

    Returns a dictionary with:
      - "object": dict with numeric results (coef, se, t, p, 95% CI, significance)
      - "description": brief plain-language interpretation of the coefficient
    """
    import numpy as np

    model = model_output.get('model') if isinstance(model_output, dict) else None

    # Try to retrieve parameter name (common variants)
    param_candidates = ['BeautyScore_c', 'BeautyScore']

    coef = se = t_val = p_val = ci_lower = ci_upper = None
    param_name = None

    # If we have a statsmodels result object, extract directly
    if model is not None:
        # find which param name exists in the model
        try:
            for pname in param_candidates:
                if pname in model.params.index:
                    param_name = pname
                    break
        except Exception:
            param_name = None

        if param_name is not None:
            coef = float(model.params[param_name])
            se = float(model.bse[param_name])
            t_val = float(model.tvalues[param_name])
            p_val = float(model.pvalues[param_name])
            ci = model.conf_int().loc[param_name]
            ci_lower, ci_upper = float(ci[0]), float(ci[1])

    # Fallback: use provided numeric entries in model_output dict (if present)
    if coef is None:
        coef = model_output.get('coef_beauty')
        se = model_output.get('se_beauty_clustered')
        # Use normal-approximation if no model object to provide t/p/CI
        if coef is not None and se is not None:
            t_val = float(coef) / float(se) if se != 0 else np.nan
            try:
                from scipy import stats as _stats
                p_val = 2 * (1 - _stats.norm.cdf(abs(t_val)))
            except Exception:
                # if scipy is not available, approximate with large-sample normal tail
                p_val = float(2 * (1 - 0.5 * (1 + np.math.erf(abs(t_val) / np.sqrt(2)))))
            ci_lower = float(coef) - 1.96 * float(se)
            ci_upper = float(coef) + 1.96 * float(se)
            param_name = param_candidates[0]  # best guess

    # If still missing required values, return informative message
    if coef is None or se is None or p_val is None:
        return {
            "object": None,
            "description": "Could not locate the beauty coefficient and its standard error in the provided model_output."
        }

    significant_0_05 = (p_val < 0.05)

    # Build human-readable description
    desc = (
        f"Estimated effect of beauty ({param_name}): coefficient = {coef:.4f}, "
        f"SE = {se:.4f}, t = {t_val:.2f}, p = {p_val:.3g}. "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"Interpretation: a one-unit increase in the mean-centered beauty score is associated "
        f"with a {coef:.3f}-point change in the course evaluation score. "
        + ("This effect is statistically significant at alpha=0.05."
           if significant_0_05 else "This effect is NOT statistically significant at alpha=0.05.")
    )

    return {
        "object": {
            "param": param_name,
            "coef": coef,
            "se": se,
            "t": t_val,
            "p_value": p_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "significant_0.05": significant_0_05
        },
        "description": desc
    }