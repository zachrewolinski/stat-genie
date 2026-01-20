def extract_final_answer(model_output):
    """
    Extracts the estimated effect of 'HasChildren' from a fitted statsmodels GLMResultsWrapper
    (Negative Binomial) and returns key statistics plus a short interpretation.

    Returns a dictionary:
      - "object": dict with numeric results (coef, se, z, p-value, 95% CI on log scale,
                                incidence rate ratio (IRR) and its 95% CI)
      - "description": a plain-language interpretation of the effect in the context of the task.
    """
    import numpy as np

    res = model_output

    var = 'HasChildren'
    # Basic checks
    try:
        params_index = res.params.index
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a fitted statsmodels results object.") from e

    if var not in params_index:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(params_index)}")

    # Extract statistics
    coef = float(res.params[var])
    se = float(res.bse[var])
    z = float(coef / se) if se != 0 else float('nan')
    p_value = float(res.pvalues[var])
    ci = res.conf_int()
    # conf_int returns a DataFrame (or array); use .loc for label
    ci_low, ci_high = map(float, ci.loc[var])

    # Convert log-coefficient to incidence rate ratio (IRR)
    irr = float(np.exp(coef))
    irr_ci_low, irr_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    # Percent change interpretation
    percent_change = (irr - 1.0) * 100.0

    # Decide statistical significance at alpha = 0.05
    alpha = 0.05
    significant = p_value < alpha

    # Build description
    if significant:
        sig_text = "statistically significant"
    else:
        sig_text = "not statistically significant"

    effect_direction = "decrease" if irr < 1 else "increase" if irr > 1 else "no change"

    description = (
        f"HasChildren coefficient (log scale) = {coef:.4f} (SE = {se:.4f}), z = {z:.3f}, p = {p_value:.4g}. "
        f"95% CI for coefficient = [{ci_low:.4f}, {ci_high:.4f}].\n"
        f"Incidence Rate Ratio (IRR) = exp(coef) = {irr:.4f}; 95% CI for IRR = [{irr_ci_low:.4f}, {irr_ci_high:.4f}].\n"
        f"This implies a {percent_change:.2f}% {effect_direction} in the expected number of reported extramarital affairs "
        f"for respondents with children vs. without children. The effect is {sig_text} at alpha = {alpha}."
    )

    result_object = {
        "variable": var,
        "coef_log": coef,
        "se": se,
        "z": z,
        "p_value": p_value,
        "95%_CI_log": [ci_low, ci_high],
        "IRR": irr,
        "95%_CI_IRR": [irr_ci_low, irr_ci_high],
        "percent_change": percent_change,
        "significant_at_0.05": bool(significant),
    }

    return {"object": result_object, "description": description}