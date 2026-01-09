def extract_final_answer(model_output):
    """
    Extracts statistics about the 'is_female' coefficient from a fitted statsmodels Logit result
    (BinaryResultsWrapper). Returns a dictionary with:
      - "object": a dict of numeric results (coefficient, se, z, p, CI, odds ratio, marginal effect if available)
      - "description": short plain-language interpretation and how to draw a yes/no conclusion.

    Expected input: a statsmodels.discrete.discrete_model.BinaryResultsWrapper (the object returned by sm.Logit(...).fit()).
    """
    import numpy as np

    res = model_output

    var = 'is_female'
    # Prepare container for results
    out = {}
    try:
        params = res.params
    except Exception as e:
        raise ValueError("model_output does not look like a fitted statsmodels results object: " + str(e))

    if var not in params.index:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    # Extract coefficient, se, p-value, z-stat (if available), and conf int
    coef = float(params[var])
    try:
        se = float(res.bse[var])
    except Exception:
        se = None
    try:
        pval = float(res.pvalues[var])
    except Exception:
        pval = None
    try:
        zstat = float(coef / se) if se not in (None, 0) else None
    except Exception:
        zstat = None
    try:
        ci_row = res.conf_int().loc[var].values
        ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
    except Exception:
        ci_lower = ci_upper = None

    # Odds ratio and CI on odds ratio scale
    try:
        odds_ratio = float(np.exp(coef))
        odds_ci = [float(np.exp(ci_lower)) if ci_lower is not None else None,
                   float(np.exp(ci_upper)) if ci_upper is not None else None]
    except Exception:
        odds_ratio = None
        odds_ci = [None, None]

    # Try to get marginal effect (average/overall)
    me = None
    me_se = None
    me_p = None
    me_ci = [None, None]
    try:
        # get_margeff() returns a MarginsResults; summary_frame() or .summary_frame() yields dy/dx etc.
        margeff = res.get_margeff()
        me_df = margeff.summary_frame()
        if var in me_df.index:
            me = float(me_df.loc[var, 'dy/dx'])
            me_se = float(me_df.loc[var, 'Std. Err.'])
            me_p = float(me_df.loc[var, 'P>|z|'])
            # approx 95% CI for marginal effect
            me_ci = [me - 1.96 * me_se, me + 1.96 * me_se]
    except Exception:
        # If get_margeff fails, leave marginal effects as None (not essential)
        pass

    out = {
        'variable': var,
        'coef_log_odds': coef,
        'std_err': se,
        'z_value': zstat,
        'p_value': pval,
        'conf_int_95_log_odds': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'odds_ratio_95_CI': odds_ci,
        'average_marginal_effect_on_probability': me,
        'marginal_effect_se': me_se,
        'marginal_effect_p_value': me_p,
        'marginal_effect_95_CI': me_ci
    }

    # Description explaining the meaning and the decision rule for yes/no question
    # (Do women have different approval probability than men, controlling for covariates?)
    # We do not presume significance; we explain how to interpret the extracted numbers.
    desc_lines = []
    desc_lines.append("This returns statistics for the coefficient on 'is_female' from the fitted logistic regression.")
    desc_lines.append("Interpretation:")
    desc_lines.append("- 'coef_log_odds' is the estimated change in log-odds of mortgage approval for females vs males, holding controls constant.")
    desc_lines.append("- 'odds_ratio' = exp(coef_log_odds) is the multiplicative change in the odds of approval for females compared with males.")
    desc_lines.append("- The 95% CI fields give the confidence interval for the log-odds and the odds ratio respectively.")
    desc_lines.append("- 'average_marginal_effect_on_probability' (if present) is the change in predicted probability of approval associated with being female (holding covariates as in the sample average).")
    desc_lines.append("")
    desc_lines.append("Decision rule to answer the question 'Does gender affect approval?':")
    desc_lines.append("- If p_value < 0.05, conclude there is a statistically significant association between gender and approval (reject null of no effect).")
    desc_lines.append("- If coef_log_odds > 0 (and significant), being female is associated with higher odds/probability of approval; if < 0 (and significant), being female is associated with lower odds/probability.")
    desc_lines.append("- If p_value >= 0.05, there is no statistically significant evidence of a gender effect after controlling for the listed covariates.")
    desc_lines.append("")
    desc_lines.append("Use the numeric outputs in 'object' to see the estimated effect size, its uncertainty, and whether it is statistically significant.")

    description = "\n".join(desc_lines)

    return {"object": out, "description": description}