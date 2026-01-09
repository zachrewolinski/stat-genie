def extract_final_answer(model_output):
    """
    Extracts key statistics about the effect of 'female' on mortgage acceptance
    from the provided model_output dict.

    Returns a dictionary with keys:
      - "object": dict of extracted numeric results (coefficient, OR, CI, p-value,
                  predicted probabilities at mean covariates, probability difference, n)
      - "description": brief interpretation of the results in context
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing at least 'model_result' or 'odds_ratios'.")

    results = model_output.get('model_result', None)
    or_table = model_output.get('odds_ratios', None)

    # Try to extract from odds_ratios table if present
    if or_table is not None and 'female' in list(or_table.index):
        try:
            or_female = float(or_table.loc['female', 'OR'])
            ci_lower = float(or_table.loc['female', 'CI_lower'])
            ci_upper = float(or_table.loc['female', 'CI_upper'])
            p_value = float(or_table.loc['female', 'p_value'])
        except Exception:
            # Fallback to model_result if table indexing differs
            or_table = None

    if (or_table is None or 'female' not in list(or_table.index)) and results is None:
        raise ValueError("model_output must contain either 'odds_ratios' with a 'female' row or a 'model_result' statsmodels object.")

    # If we still have a statsmodels result, extract from it
    if results is not None:
        params = results.params  # pandas Series
        conf = results.conf_int()  # DataFrame with lower/upper
        # Basic stats
        coef = float(params['female'])
        try:
            or_female = float(np.exp(coef))
        except Exception:
            or_female = float(np.exp(params.loc['female']))
        try:
            ci_lower = float(np.exp(conf.loc['female', 0]))
            ci_upper = float(np.exp(conf.loc['female', 1]))
        except Exception:
            # If conf uses different indexing, try positional
            idx = list(results.model.exog_names).index('female')
            ci_lower = float(np.exp(conf.iloc[idx, 0]))
            ci_upper = float(np.exp(conf.iloc[idx, 1]))
        p_value = float(results.pvalues['female'])

        # Predicted probability difference at mean covariate values
        exog = results.model.exog  # numpy array (n_obs x k)
        exog_names = list(results.model.exog_names)
        if 'female' not in exog_names:
            raise ValueError("'female' not found in fitted model's exog names.")
        female_idx = exog_names.index('female')

        mean_exog = np.mean(exog, axis=0)
        exog_f1 = mean_exog.copy()
        exog_f0 = mean_exog.copy()
        exog_f1[female_idx] = 1.0
        exog_f0[female_idx] = 0.0

        beta = np.asarray(params)  # parameter vector aligned with exog_names

        def _logistic(x):
            return 1.0 / (1.0 + np.exp(-x))

        prob_f1 = float(_logistic(np.dot(exog_f1, beta)))
        prob_f0 = float(_logistic(np.dot(exog_f0, beta)))
        prob_diff = prob_f1 - prob_f0

        n_obs = int(getattr(results, 'nobs', np.shape(exog)[0]))

        coef_value = float(coef)
    else:
        # If only odds table present (and no results), return OR/p and no predicted probs
        coef_value = None
        prob_f1 = prob_f0 = prob_diff = None
        n_obs = None
        # or_female, ci_lower, ci_upper, p_value should already be set above

    # Assemble the object to return
    object_dict = {
        'coefficient_logit': coef_value,               # log-odds coefficient (None if unavailable)
        'odds_ratio': float(or_female),
        'CI_lower': float(ci_lower),
        'CI_upper': float(ci_upper),
        'p_value': float(p_value),
        'pred_prob_female_1_at_means': prob_f1,        # predicted probability if female=1 (other covariates at sample means)
        'pred_prob_female_0_at_means': prob_f0,        # predicted probability if female=0 (other covariates at sample means)
        'probability_difference': prob_diff,           # absolute difference (f1 - f0)
        'n_obs': n_obs
    }

    # Short interpretation
    # Round numbers in description for readability
    def _fmt(x, digits=3):
        return "NA" if x is None else str(round(x, digits))

    description = (
        "Effect of being female on mortgage acceptance:\n"
        f"- Odds ratio = {_fmt(object_dict['odds_ratio'])} "
        f"(95% CI {_fmt(object_dict['CI_lower'])} to {_fmt(object_dict['CI_upper'])}), "
        f"p = {_fmt(object_dict['p_value'], 4)}.\n"
        f"- Interpretation: female applicants have an estimated { _fmt(object_dict['odds_ratio']) }x odds "
        f"of acceptance compared with male applicants, controlling for the listed covariates. "
    )

    if object_dict['probability_difference'] is not None:
        description += (
            f"At the sample mean of other covariates, predicted acceptance probability is "
            f"{_fmt(object_dict['pred_prob_female_1_at_means'],3)} for females vs "
            f"{_fmt(object_dict['pred_prob_female_0_at_means'],3)} for males, an absolute difference of "
            f"{_fmt(object_dict['probability_difference'],3)} (in probability points). "
        )

    if object_dict['p_value'] < 0.05:
        description += "The effect is statistically significant at the 5% level."
    else:
        description += "The effect is not statistically significant at the 5% level."

    return {"object": object_dict, "description": description}