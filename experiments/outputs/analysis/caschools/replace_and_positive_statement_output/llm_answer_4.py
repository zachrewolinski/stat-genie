def extract_final_answer(model_output):
    """
    Extracts statistics about the STRatio effect from a regression model output dict and
    returns a concise numeric object plus a short interpretation.

    Returns a dict with keys:
      - "object": dict with numeric results (coefficient, p-value, 95% CI, VIF if available, n_obs)
      - "description": short plain-language interpretation answering whether a lower
                       student-teacher ratio is associated with higher academic performance.
    """
    # Initialize placeholders
    coef = pval = ci_lower = ci_upper = None
    vif_stratio = None
    n_obs = None

    # Try to extract from model_output in a few ways for robustness
    # 1) Prefer the fitted model object if present
    model = model_output.get('model', None)

    if model is not None:
        # params and pvalues should be accessible
        try:
            coef = float(model.params['STRatio'])
        except Exception:
            coef = None
        try:
            pval = float(model.pvalues['STRatio'])
        except Exception:
            pval = None
        # confidence interval (should reflect cov_type used when fitting)
        try:
            ci = model.conf_int().loc['STRatio']
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower = ci_upper = None
        # number of observations
        try:
            # statsmodels stores nobs as.attr or property
            n_obs = int(model.nobs)
        except Exception:
            n_obs = None
    else:
        # 2) Fall back to 'params' and 'pvalues' entries if present
        params = model_output.get('params', None)
        pvalues = model_output.get('pvalues', None)
        if params is not None and 'STRatio' in params:
            coef = float(params['STRatio'])
        if pvalues is not None and 'STRatio' in pvalues:
            pval = float(pvalues['STRatio'])
        # conf int fallback: try 'summary' or not available
        try:
            # If a 'summary' or other fields contain conf ints this is complex;
            # skip if not directly available.
            pass
        except Exception:
            pass

    # VIF extraction if present
    vif_list = model_output.get('vif', None)
    if isinstance(vif_list, (list, tuple)):
        for entry in vif_list:
            try:
                if entry.get('variable') == 'STRatio':
                    vif_stratio = float(entry.get('VIF'))
                    break
            except Exception:
                continue

    # Decision rule:
    # The research question asks: "Is a lower student-teacher ratio associated with higher academic performance?"
    # That corresponds to a negative coefficient for STRatio (fewer students per teacher -> higher scores)
    # and statistical significance (alpha = 0.05).
    conclusion_boolean = None
    conclusion_text = ""
    if coef is None or pval is None:
        conclusion_text = "Could not extract coefficient and/or p-value for STRatio from the model output."
        conclusion_boolean = None
    else:
        significant = (pval < 0.05)
        if (coef < 0) and significant:
            conclusion_boolean = True
            conclusion_text = (
                "Yes — the estimated coefficient for STRatio is negative and statistically significant "
                "(coef = {coef:.3g}, p = {p:.3g}, 95% CI [{lo:.3g}, {hi:.3g}]). "
                "This indicates that lower student-teacher ratios are associated with higher average scores."
            ).format(coef=coef, p=pval, lo=(ci_lower if ci_lower is not None else float('nan')),
                     hi=(ci_upper if ci_upper is not None else float('nan')))
        else:
            # Not the case: either positive or not significant (or both)
            conclusion_boolean = False
            # Build a precise explanation based on sign and significance
            sign_desc = "positive" if coef > 0 else ("zero" if coef == 0 else "negative")
            conclusion_text = (
                "No strong evidence that lower student-teacher ratios are associated with higher academic performance. "
                "Estimated STRatio coefficient = {coef:.3g} ({sign}), p = {p:.3g}, 95% CI [{lo:.3g}, {hi:.3g}]. "
            ).format(coef=coef, sign=sign_desc, p=pval,
                     lo=(ci_lower if ci_lower is not None else float('nan')),
                     hi=(ci_upper if ci_upper is not None else float('nan')))
            # Add note about direction when not significant
            if not significant:
                conclusion_text += "The effect is not statistically significant at alpha=0.05."
            else:
                # significant but coefficient has the "wrong" sign (positive)
                conclusion_text += "The effect is statistically significant but in the direction opposite to 'lower ratio -> higher scores'."

    # Assemble the object to return
    result_object = {
        'coefficient': coef,
        'p_value': pval,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'vif_STRatio': vif_stratio,
        'n_obs': n_obs,
        'conclusion_boolean_lower_ratio_associated_with_higher_scores': conclusion_boolean
    }

    return {
        "object": result_object,
        "description": conclusion_text
    }