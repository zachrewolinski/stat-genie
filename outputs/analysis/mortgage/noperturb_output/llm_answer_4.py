def extract_final_answer(model_output):
    """
    Extract key statistics for the 'female' coefficient from a fitted statsmodels
    binary logistic regression (BinaryResultsWrapper).

    Returns a dict with:
      - "object": dict containing coefficient, p-value, odds ratio, 95% CI for OR,
                  significance flag, and a short interpretation string.
      - "description": brief explanation of the returned object.

    Expected input: statsmodels.discrete.discrete_model.BinaryResultsWrapper (the
    result returned by the provided `model` function).
    """
    import numpy as np

    # Ensure the model output has the necessary attributes
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not have 'params' attribute.")
    if not hasattr(model_output, "pvalues"):
        raise ValueError("model_output does not have 'pvalues' attribute.")
    if not hasattr(model_output, "conf_int"):
        raise ValueError("model_output does not have 'conf_int' method/attribute.")

    params = model_output.params
    pvalues = model_output.pvalues

    if 'female' not in params.index:
        raise ValueError("The fitted model does not contain a 'female' coefficient.")

    # Coefficient and p-value
    coef = float(params.loc['female'])
    pval = float(pvalues.loc['female'])

    # Confidence interval for the coefficient (log-odds)
    try:
        conf = model_output.conf_int()
        # conf may be a DataFrame with two columns [0,1]
        ci_low_log = float(conf.loc['female'].iloc[0])
        ci_high_log = float(conf.loc['female'].iloc[1])
    except Exception:
        # If conf_int fails for some reason, set to None
        ci_low_log = None
        ci_high_log = None

    # Odds ratio and CI on odds ratio scale
    odds_ratio = float(np.exp(coef))
    if ci_low_log is not None and ci_high_log is not None:
        or_ci_lower = float(np.exp(ci_low_log))
        or_ci_upper = float(np.exp(ci_high_log))
    else:
        or_ci_lower = None
        or_ci_upper = None

    # Statistical significance (using alpha = 0.05)
    significant = pval < 0.05

    # Effect direction and percent change in odds
    pct_change = (odds_ratio - 1.0) * 100.0
    if odds_ratio > 1:
        direction = f"Females have higher odds of loan acceptance (≈{pct_change:.1f}% higher odds)."
    elif odds_ratio < 1:
        direction = f"Females have lower odds of loan acceptance (≈{abs(pct_change):.1f}% lower odds)."
    else:
        direction = "No change in odds for females compared to males (OR ≈ 1)."

    significance_text = "statistically significant (p < 0.05)." if significant else "not statistically significant (p ≥ 0.05)."

    interpretation = (
        f"The female coefficient (log-odds) = {coef:.4f}, p = {pval:.4g}. "
        f"Odds ratio = {odds_ratio:.3f}"
    )
    if or_ci_lower is not None and or_ci_upper is not None:
        interpretation += f" (95% CI for OR: [{or_ci_lower:.3f}, {or_ci_upper:.3f}]). "
    else:
        interpretation += ". "
    interpretation += f"Interpretation: {direction} This effect is {significance_text}"

    result_object = {
        "coef": coef,
        "p_value": pval,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": [or_ci_lower, or_ci_upper],
        "significant_at_0.05": significant,
        "percent_change_in_odds": pct_change,
        "interpretation": interpretation
    }

    return {
        "object": result_object,
        "description": (
            "Extracted statistics for the independent variable 'female' from the fitted "
            "logistic regression model. 'object' contains numeric summaries (coefficient, "
            "p-value, odds ratio, 95% CI for the OR), a significance flag, and a plain-language "
            "interpretation of how being female affects the odds of loan acceptance relative to males."
        )
    }