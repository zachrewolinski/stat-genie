def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted statsmodels Logit result.

    Returns a dictionary with:
      - "object": a dict containing numeric results (log-odds coef, SE, p-value, OR, OR 95% CI)
      - "description": a short plain-language interpretation of the effect of being female
    """
    import numpy as np

    # Expect model_output to be a dict with at least 'model_result' (a statsmodels results object).
    if not isinstance(model_output, dict) or 'model_result' not in model_output:
        raise ValueError("model_output must be a dict containing key 'model_result' (a statsmodels results object).")

    res = model_output['model_result']

    # Ensure the required 'female' parameter exists
    if 'female' not in res.params.index:
        raise ValueError("The fitted model does not contain a parameter named 'female'.")

    # Extract statistics
    coef = float(res.params['female'])
    se = float(res.bse['female'])
    pval = float(res.pvalues['female'])

    # Confidence interval for the log-odds coefficient, then convert to OR scale
    conf = res.conf_int()
    ci_log_lower = float(conf.loc['female', 0])
    ci_log_upper = float(conf.loc['female', 1])

    or_value = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_log_lower))
    or_ci_upper = float(np.exp(ci_log_upper))

    object_out = {
        'coefficient_log_odds': coef,
        'std_error': se,
        'p_value': pval,
        'odds_ratio': or_value,
        'OR_95_CI': [or_ci_lower, or_ci_upper]
    }

    # Plain-language description
    significance = "statistically significant" if pval < 0.05 else "not statistically significant"
    description = (
        f"The estimated log-odds coefficient for female = {coef:.3f} (SE = {se:.3f}, p = {pval:.3f}). "
        f"This corresponds to an odds ratio of {or_value:.3f} with 95% CI [{or_ci_lower:.3f}, {or_ci_upper:.3f}]. "
        f"Interpretation: holding the included controls constant, female applicants have about {or_value:.2f}× the odds "
        f"of mortgage approval compared with male applicants. The effect is {significance} at the 0.05 level."
    )

    return {"object": object_out, "description": description}