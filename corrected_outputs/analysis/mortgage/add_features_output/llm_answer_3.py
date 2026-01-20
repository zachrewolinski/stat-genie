def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of being female on mortgage acceptance
    from the model output dict produced by model(...).

    Returns a dict with keys:
      - "object": dict with numeric results (coefficient, se, p-value, CI, odds ratio, AME, significance)
      - "description": plain-English interpretation of the result in context
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict as returned by the modeling function.")

    if 'model_result' not in model_output:
        raise KeyError("model_output must contain key 'model_result'")

    result = model_output['model_result']
    ame = model_output.get('avg_marginal_effect_female', None)

    # Ensure result has params
    try:
        params = result.params
    except Exception as e:
        raise ValueError("The provided model_result does not expose .params") from e

    # Check female present
    if 'female' not in params.index:
        raise KeyError("'female' not found among model parameters")

    # Extract coefficient, se, p-value, conf int
    coef = float(params['female'])
    # bse and pvalues should exist for statsmodels results
    try:
        se = float(result.bse['female'])
    except Exception:
        se = None
    try:
        pval = float(result.pvalues['female'])
    except Exception:
        pval = None

    try:
        conf = result.conf_int().loc['female']
        conf_low = float(conf[0])
        conf_high = float(conf[1])
    except Exception:
        conf_low = conf_high = None

    # Odds ratio and CI (exponentiated coefficient and CI)
    try:
        odds_ratio = float(np.exp(coef))
        odds_ratio_ci = (
            float(np.exp(conf_low)) if conf_low is not None else None,
            float(np.exp(conf_high)) if conf_high is not None else None
        )
    except Exception:
        odds_ratio = None
        odds_ratio_ci = (None, None)

    # Average marginal effect (AME) provided by the model function: probability difference
    avg_marginal_effect = float(ame) if ame is not None else None

    # Statistical significance at conventional alpha=0.05 (if pval available)
    significant = None
    if pval is not None:
        significant = bool(pval < 0.05)

    # Build the return object
    obj = {
        'coef_log_odds_female': coef,
        'std_error': se,
        'p_value': pval,
        'conf_int_95_log_odds': (conf_low, conf_high),
        'odds_ratio': odds_ratio,
        'odds_ratio_95_CI': odds_ratio_ci,
        'avg_marginal_effect_female_prob_diff': avg_marginal_effect,
        'significant_at_0.05': significant
    }

    # Construct readable description
    # Interpret direction and magnitude using AME when available, otherwise use coef/odds ratio.
    if avg_marginal_effect is not None:
        # express as percentage points
        ame_pct = avg_marginal_effect * 100
        ame_text = f"On average, being female is associated with a {ame_pct:.2f} percentage-point " \
                   f"{'increase' if avg_marginal_effect > 0 else 'decrease'} in the probability of mortgage approval."
    else:
        ame_text = "Average marginal effect was not provided."

    if pval is not None:
        sig_text = ("This effect is statistically significant (p = "
                    f"{pval:.3g})" if significant else
                    f"This effect is not statistically significant (p = {pval:.3g}).")
    else:
        sig_text = "No p-value available to assess statistical significance."

    or_text = ""
    if odds_ratio is not None:
        or_text = (f"The log-odds coefficient is {coef:.4f}, corresponding to an odds ratio of "
                   f"{odds_ratio:.3f} (95% CI: {odds_ratio_ci[0]:.3f} to {odds_ratio_ci[1]:.3f}).")

    description = (
        f"{ame_text} {sig_text} {or_text} "
        "Conclusion: Based on the fitted logistic model, "
        + ("there is evidence of an association between applicant gender and approval probability."
           if significant else
           "there is no strong evidence of an association between applicant gender and approval probability.")
    )

    return {
        "object": obj,
        "description": description
    }