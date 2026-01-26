def extract_final_answer(model_output):
    """
    Extracts key statistics for the 'female' coefficient from a fitted statsmodels Logit
    model (or a dict containing the model under the 'model' key).

    Returns a dict with:
      - "object": a dict with numeric results (coef, se, p-value, OR, OR 95% CI,
                  percent change in odds, and a boolean for significance at alpha=0.05)
      - "description": a short plain-language interpretation in the context of the task
    """
    import numpy as np

    # Accept either the raw model object or the dict returned by the model function
    if isinstance(model_output, dict) and 'model' in model_output:
        res = model_output['model']
    else:
        res = model_output

    # Basic validation
    if not hasattr(res, 'params'):
        raise ValueError("Provided model_output does not appear to be a fitted statsmodels result object.")
    if 'female' not in res.params.index:
        raise ValueError("The model does not contain a 'female' coefficient.")

    # Extract statistics
    coef = float(res.params['female'])
    se = float(res.bse['female'])
    p_value = float(res.pvalues['female'])
    ci_series = res.conf_int().loc['female']  # typically a Series with two entries [lower, upper]
    ci_lower = float(ci_series.iloc[0])
    ci_upper = float(ci_series.iloc[1])

    # Odds ratio and its CI
    OR = float(np.exp(coef))
    OR_ci_lower = float(np.exp(ci_lower))
    OR_ci_upper = float(np.exp(ci_upper))

    # Percent change in odds associated with being female
    percent_change_odds = (OR - 1.0) * 100.0

    # Significance at alpha = 0.05
    significant_at_0_05 = (p_value < 0.05)

    # Build output object
    output_object = {
        'coef': coef,
        'se': se,
        'p_value': p_value,
        'OR': OR,
        'OR_ci_lower': OR_ci_lower,
        'OR_ci_upper': OR_ci_upper,
        'percent_change_in_odds': percent_change_odds,
        'significant_at_0.05': significant_at_0_05
    }

    # Plain-language description
    if significant_at_0_05:
        interpretation = (
            f"Controlling for the listed covariates, the 'female' coefficient is {coef:.4f} "
            f"(SE={se:.4f}), corresponding to an odds ratio of {OR:.3f} "
            f"(95% CI: {OR_ci_lower:.3f}–{OR_ci_upper:.3f}), p = {p_value:.4f}. "
            f"This indicates female applicants have about a {percent_change_odds:.1f}% "
            f"{'increase' if percent_change_odds>0 else 'decrease'} in odds of mortgage approval "
            "relative to male applicants, statistically significant at alpha = 0.05. "
            "This is an association (adjusted for covariates), not proof of causation."
        )
    else:
        interpretation = (
            f"Controlling for the listed covariates, the 'female' coefficient is {coef:.4f} "
            f"(SE={se:.4f}), corresponding to an odds ratio of {OR:.3f} "
            f"(95% CI: {OR_ci_lower:.3f}–{OR_ci_upper:.3f}), p = {p_value:.4f}. "
            "The confidence interval includes 1 and the effect is not statistically significant "
            "at alpha = 0.05, so there is no strong evidence that gender affects approval probability "
            "after adjustment. This is an association (adjusted for covariates), not proof of causation."
        )

    return {
        "object": output_object,
        "description": interpretation
    }