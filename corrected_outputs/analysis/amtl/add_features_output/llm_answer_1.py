def extract_final_answer(model_output):
    """
    Extracts statistics for the effect of IsHomo from a fitted statsmodels GLMResultsWrapper.

    Returns a dictionary with keys:
      - "object": dict containing coefficient, robust SE, test statistic, p-value,
                  95% CI (log-odds), odds ratio and its 95% CI, and a boolean flag
                  'humans_higher' indicating whether coef>0 and p<0.05.
      - "description": brief interpretation of the result in the context of the task.
    """
    import numpy as np

    res = model_output

    # Find the parameter name corresponding to IsHomo (handles IsHomo or IsHomo[T.True] etc.)
    param_candidates = [name for name in res.params.index if 'IsHomo' in name]
    if len(param_candidates) == 0:
        raise KeyError("No parameter matching 'IsHomo' found in model_output.params.index. "
                       "Available parameters: {}".format(list(res.params.index)))
    # Prefer exact match if present
    param_name = 'IsHomo' if 'IsHomo' in res.params.index else param_candidates[0]

    # Extract statistics
    coef = float(res.params[param_name])
    se = float(res.bse[param_name])
    stat = float(res.tvalues[param_name]) if hasattr(res, 'tvalues') else float(res.zvalues[param_name])
    pvalue = float(res.pvalues[param_name])
    conf = res.conf_int().loc[param_name].values.astype(float)  # [lower, upper]

    # Transform to odds ratio scale
    or_est = float(np.exp(coef))
    or_ci = list(np.exp(conf).astype(float))

    # Decision: positive coef and p < 0.05 indicates higher AMTL in modern humans
    humans_higher = (coef > 0) and (pvalue < 0.05)

    result_object = {
        "param_name": param_name,
        "coef_log_odds": coef,
        "robust_se": se,
        "test_stat": stat,
        "p_value": pvalue,
        "conf_int_log_odds": [float(conf[0]), float(conf[1])],
        "odds_ratio": or_est,
        "odds_ratio_95ci": or_ci,
        "humans_higher": humans_higher
    }

    # Build short interpretation
    if humans_higher:
        conclusion = (
            "The estimated coefficient for '{}' is positive (coef = {:+.4f}), with p = {:.4g}. "
            "On the odds-ratio scale this corresponds to OR = {:.3f} (95% CI [{:.3f}, {:.3f}]). "
            "This provides statistical evidence (alpha=0.05) that modern humans (Homo sapiens) have "
            "higher AMTL frequency than the non-human primates in the sample, "
            "controlling for age, sex (prob_male), and tooth class."
        ).format(param_name, coef, pvalue, or_est, or_ci[0], or_ci[1])
    else:
        conclusion = (
            "The estimated coefficient for '{}' is {:+.4f} with p = {:.4g}. "
            "On the odds-ratio scale OR = {:.3f} (95% CI [{:.3f}, {:.3f}]). "
            "There is no statistically significant evidence (alpha=0.05) that modern humans have higher AMTL "
            "than the non-human primates after controlling for age, sex (prob_male), and tooth class."
        ).format(param_name, coef, pvalue, or_est, or_ci[0], or_ci[1])

    return {"object": result_object, "description": conclusion}