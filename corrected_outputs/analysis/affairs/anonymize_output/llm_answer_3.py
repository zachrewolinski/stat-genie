def extract_final_answer(model_output):
    """
    Extracts statistics for the effect of 'HasChildren' from a fitted Negative Binomial GLM result.
    Returns a dict with keys:
      - "object": a dict of numeric results (coefficient, SE, p-value, 95% CI, IRR and IRR CI, percent change)
      - "description": a short text interpretation in the context of whether having children is
                       associated with decreased engagement in extramarital affairs.
    The function prefers robust results if present in model_output['results_robust'].
    """
    import numpy as np

    # Get results objects from the provided model_output dict
    results = model_output.get('results', None)
    results_robust = model_output.get('results_robust', None)

    if results is None:
        raise ValueError("model_output must contain a 'results' entry with the fitted model object.")

    # Prefer robust results when available
    res = results_robust if (results_robust is not None) else results
    used_robust = results_robust is not None

    var = 'HasChildren'
    # Ensure variable is present
    try:
        coef = float(res.params[var])
        se = float(res.bse[var])
        pval = float(res.pvalues[var])
        ci_lower, ci_upper = res.conf_int().loc[var].astype(float)
    except Exception as e:
        raise KeyError(f"Could not extract statistics for variable '{var}': {e}")

    # Exponentiate coefficient to get incidence rate ratio (IRR) for count model
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))
    percent_change = (irr - 1.0) * 100.0  # percent change in expected count

    # Prepare numeric object to return
    numeric_output = {
        'variable': var,
        'model_used': 'NegativeBinomial GLM',
        'used_robust_se': bool(used_robust),
        'coef': round(coef, 6),
        'std_err': round(se, 6),
        'p_value': round(pval, 6),
        '95%_CI_coef': [round(ci_lower, 6), round(ci_upper, 6)],
        'IRR': round(irr, 6),
        '95%_CI_IRR': [round(irr_ci_lower, 6), round(irr_ci_upper, 6)],
        'percent_change_in_expected_count': round(percent_change, 4)
    }

    # Build a concise interpretation
    sig_text = "statistically significant (p < 0.05)" if pval < 0.05 else "not statistically significant (p >= 0.05)"
    if irr < 1.0:
        direction_text = f"Holding controls constant, having children is associated with a lower expected frequency of extramarital intercourse: IRR = {numeric_output['IRR']} (95% CI {numeric_output['95%_CI_IRR']})."
        pct_text = f"This corresponds to an estimated {abs(round(percent_change,2))}% decrease in the expected count of affairs."
    else:
        direction_text = f"Holding controls constant, having children is associated with a higher expected frequency of extramarital intercourse: IRR = {numeric_output['IRR']} (95% CI {numeric_output['95%_CI_IRR']})."
        pct_text = f"This corresponds to an estimated {round(percent_change,2)}% increase in the expected count of affairs."

    description = (
        f"{direction_text} {pct_text} The regression coefficient on 'HasChildren' is {numeric_output['coef']} "
        f"(SE = {numeric_output['std_err']}, 95% CI for coef = {numeric_output['95%_CI_coef']}, p = {numeric_output['p_value']}). "
        f"Result is {sig_text}. Model: Negative Binomial GLM; robust SEs used: {used_robust}."
    )

    return {
        "object": numeric_output,
        "description": description
    }