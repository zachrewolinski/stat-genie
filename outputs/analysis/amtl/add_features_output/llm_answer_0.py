def extract_final_answer(model_output):
    """
    Extracts the coefficient, p-value, odds ratio and 95% CI for the 'is_human'
    predictor from a fitted statsmodels GLM result (or a dict containing it).
    Returns a dict with "object" (numeric results) and "description" (textual
    interpretation answering whether modern humans have higher AMTL).
    """
    import numpy as np

    # Accept either the raw results object or the dict returned by the model() function
    if isinstance(model_output, dict) and 'model' in model_output:
        res = model_output['model']
    else:
        res = model_output

    # Ensure necessary attributes exist
    if not hasattr(res, 'params') or not hasattr(res, 'pvalues') or not hasattr(res, 'conf_int'):
        raise ValueError("Input does not look like a fitted statsmodels results object or dict containing one.")

    params = res.params
    pvalues = res.pvalues
    conf = res.conf_int()

    if 'is_human' not in params.index:
        raise KeyError("The model does not contain a parameter named 'is_human'.")

    # Extract values
    coef = float(params['is_human'])
    pval = float(pvalues['is_human'])

    # Confidence interval extraction robust to index style
    try:
        ci_lower_log, ci_upper_log = conf.loc['is_human', 0], conf.loc['is_human', 1]
    except Exception:
        # fallback if conf is indexed differently
        pos = list(params.index).index('is_human')
        ci_lower_log, ci_upper_log = conf.iloc[pos, 0], conf.iloc[pos, 1]

    # Exponentiate to get odds ratio and CI on OR scale
    OR = float(np.exp(coef))
    OR_CI_lower = float(np.exp(ci_lower_log))
    OR_CI_upper = float(np.exp(ci_upper_log))

    significant = (pval < 0.05)

    # Short conclusion based on sign and significance of coefficient
    if coef > 0 and significant:
        conclusion = "Yes — modern humans have significantly higher AMTL than non-human primates after accounting for age, sex, and tooth class."
    elif coef > 0 and not significant:
        conclusion = "No strong evidence — the estimated effect indicates higher AMTL in modern humans, but it is not statistically significant."
    elif coef < 0 and significant:
        conclusion = "No — modern humans have significantly lower AMTL than non-human primates after adjustment."
    else:
        conclusion = "No strong evidence of a difference in AMTL between modern humans and non-human primates."

    result_object = {
        'coef_log_odds': coef,
        'p_value': pval,
        'odds_ratio': OR,
        'OR_CI_lower': OR_CI_lower,
        'OR_CI_upper': OR_CI_upper,
        'significant_at_0.05': significant,
        'conclusion': conclusion
    }

    description = (
        "From the fitted binomial (logit) GLM: the coefficient for is_human = {:.4f} "
        "(p = {:.4g}), corresponding to an odds ratio = {:.3f} "
        "with 95% CI [{:.3f}, {:.3f}]. {}\n\n"
        "Interpretation: on the odds scale, after controlling for age (age_c), "
        "sex probability (prob_male_c), and tooth class, modern humans (Homo sapiens) "
        "have {} AMTL compared to non-human primates."
    ).format(coef, pval, OR, OR_CI_lower, OR_CI_upper,
             ("This is statistically significant." if significant else "This is not statistically significant."),
             ("higher" if coef > 0 else "lower" if coef < 0 else "similar levels of"))

    return {
        "object": result_object,
        "description": description
    }