def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator on mortgage acceptance from a fitted model output.
    Returns a dictionary with:
      - "object": a dict containing coefficient, odds ratio, 95% CI (OR scale), p-value, significance flag, and a short conclusion.
      - "description": a brief plain-language interpretation of what the statistics mean.
    """
    import numpy as np
    import pandas as pd

    # Initialize placeholders
    coef = None
    or_val = None
    ci_lower = None
    ci_upper = None
    p_value = None

    # Accept either the dict produced by the modeling function or a statsmodels results object directly
    res = None
    or_table = None
    if isinstance(model_output, dict):
        res = model_output.get('model_results', None)
        or_table = model_output.get('odds_ratios', None)
    else:
        # assume it's a statsmodels results object
        res = model_output

    # Try extracting from odds_ratios table first (easier if present)
    if isinstance(or_table, (pd.DataFrame,)) and 'female' in or_table.index:
        row = or_table.loc['female']
        try:
            or_val = float(row['OR'])
            ci_lower = float(row['CI_lower'])
            ci_upper = float(row['CI_upper'])
            p_value = float(row['p_value'])
            # attempt to get log-odds coefficient if available from model_results
            if res is not None and hasattr(res, 'params') and 'female' in res.params.index:
                coef = float(res.params['female'])
            else:
                # approximate log-odds from OR
                coef = float(np.log(or_val))
        except Exception as e:
            raise RuntimeError(f"Failed to read odds_ratios table for 'female': {e}")
    elif res is not None:
        # Try extracting from statsmodels results object
        try:
            # params and pvalues may be pandas Series
            coef = float(res.params['female'])
            p_value = float(res.pvalues['female'])
            conf = res.conf_int().loc['female']  # gives [lower, upper] for coefficient (log-odds)
            coef_ci_lower = float(conf[0])
            coef_ci_upper = float(conf[1])
            # convert to OR scale
            or_val = float(np.exp(coef))
            ci_lower = float(np.exp(coef_ci_lower))
            ci_upper = float(np.exp(coef_ci_upper))
        except Exception as e:
            raise RuntimeError(f"Failed to extract 'female' from model_results: {e}")
    else:
        raise ValueError("model_output does not contain recognizable model results or odds_ratios table.")

    # Determine statistical significance at alpha = 0.05
    significant = False
    if p_value is not None:
        significant = (p_value < 0.05)

    # Build a concise conclusion
    if p_value is None:
        conclusion = "Could not determine statistical significance (p-value missing)."
    else:
        if significant:
            conclusion = (
                f"There is a statistically significant association between applicant gender and mortgage approval "
                f"(OR = {or_val:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}], p = {p_value:.3g})."
            )
        else:
            conclusion = (
                f"No evidence that applicant gender predicts mortgage approval: estimated odds ratio ≈ {or_val:.3f} "
                f"(95% CI [{ci_lower:.3f}, {ci_upper:.3f}]), p = {p_value:.3g}; the CI includes 1 and the effect is not statistically significant."
            )

    # Prepare return object with both raw numbers and human-friendly rounded values
    result_object = {
        'coef_log_odds': float(coef) if coef is not None else None,
        'odds_ratio': float(or_val) if or_val is not None else None,
        'OR_95CI_lower': float(ci_lower) if ci_lower is not None else None,
        'OR_95CI_upper': float(ci_upper) if ci_upper is not None else None,
        'p_value': float(p_value) if p_value is not None else None,
        'significant_at_0.05': bool(significant),
        'conclusion': conclusion
    }

    description = (
        "Extracted statistics refer to the coefficient for the 'female' indicator in the logistic regression predicting "
        "mortgage acceptance. The odds_ratio is exp(coefficient); the 95% CI is on the OR scale. The 'significant_at_0.05' "
        "flag indicates whether p < 0.05. The conclusion summarizes whether there is evidence that gender affects approval."
    )

    return {"object": result_object, "description": description}