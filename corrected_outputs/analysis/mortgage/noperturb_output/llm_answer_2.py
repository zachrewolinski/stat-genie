def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted logistic model output.
    Returns a dictionary with:
      - "object": dict of numeric statistics
      - "description": plain-English interpretation in context
    """
    import numpy as np
    import pandas as pd

    # Defensive retrieval of model results and odds-ratios table
    results = None
    or_table = None
    if isinstance(model_output, dict):
        results = model_output.get('results', None)
        or_table = model_output.get('odds_ratios', None)
    else:
        # If user passed the statsmodels results object directly
        results = model_output

    # Prepare containers
    coef = None
    p_value = None
    ci_lower_log = None
    ci_upper_log = None

    # Try to extract from statsmodels results if available
    if results is not None:
        try:
            params = results.params
            pvalues = results.pvalues
            conf = results.conf_int()  # log-odds CI
            if 'female' in params.index:
                coef = float(params.loc['female'])
                p_value = float(pvalues.loc['female'])
                ci_lower_log = float(conf.loc['female', 0])
                ci_upper_log = float(conf.loc['female', 1])
        except Exception:
            # If extraction fails, reset to None and try odds table below
            coef = p_value = ci_lower_log = ci_upper_log = None

    # If we couldn't get from results, try odds_ratios table (contains OR and CI on OR scale)
    or_value = None
    ci_lower_or = None
    ci_upper_or = None
    if or_table is not None and 'female' in or_table.index:
        try:
            or_value = float(or_table.loc['female', 'OR'])
            ci_lower_or = float(or_table.loc['female', 'CI_lower'])
            ci_upper_or = float(or_table.loc['female', 'CI_upper'])
            # If log-scale CI not available but OR available, compute log-scale equivalents
            if coef is None:
                coef = float(np.log(or_value))
                # derive log CI
                if ci_lower_or > 0 and ci_upper_or > 0:
                    ci_lower_log = float(np.log(ci_lower_or))
                    ci_upper_log = float(np.log(ci_upper_or))
        except Exception:
            or_value = ci_lower_or = ci_upper_or = None

    # If we have coef but not OR, compute OR and OR CI from log-scale values
    if coef is not None and or_value is None:
        or_value = float(np.exp(coef))
        if (ci_lower_log is not None) and (ci_upper_log is not None):
            ci_lower_or = float(np.exp(ci_lower_log))
            ci_upper_or = float(np.exp(ci_upper_log))

    # Prepare return object (numeric)
    numeric_result = {
        'coef_log_odds': float(coef) if coef is not None else None,
        'odds_ratio': float(or_value) if or_value is not None else None,
        'ci_lower_or': float(ci_lower_or) if ci_lower_or is not None else None,
        'ci_upper_or': float(ci_upper_or) if ci_upper_or is not None else None,
        'p_value': float(p_value) if p_value is not None else None,
        'significant_at_0.05': (float(p_value) < 0.05) if p_value is not None else None
    }

    # Build human-readable description / interpretation
    if numeric_result['odds_ratio'] is None:
        description = ("Could not extract statistics for 'female' from the provided model_output. "
                       "Please pass a dict with keys 'results' (statsmodels results) or 'odds_ratios' DataFrame.")
    else:
        or_pct = (numeric_result['odds_ratio'] - 1.0) * 100.0
        sig_text = ("statistically significant (p = {:.3g})".format(numeric_result['p_value'])
                    if numeric_result['p_value'] is not None and numeric_result['p_value'] < 0.05
                    else ("not statistically significant (p = {:.3g})".format(numeric_result['p_value'])
                          if numeric_result['p_value'] is not None else "of unknown significance"))
        description = (
            "Controlling for the listed covariates, the estimated effect of being female on mortgage approval:\n"
            "- Log-odds coefficient = {coef:.4f}\n"
            "- Odds ratio = {or_val:.3f}, meaning females have about {pct:.1f}% {increase_or_decrease} in odds of approval vs males.\n"
            "- 95% CI for odds ratio = [{ci_low:.3f}, {ci_high:.3f}].\n"
            "- The effect is {sig}.\n"
            "Interpretation: women appear to have higher odds of mortgage approval compared to otherwise similar men "
            "by approximately {pct:.1f}%, conditional on the included controls. This is an association, not a causal claim."
        ).format(
            coef=numeric_result['coef_log_odds'],
            or_val=numeric_result['odds_ratio'],
            pct=abs(or_pct),
            increase_or_decrease=("increase" if or_pct > 0 else "decrease"),
            ci_low=numeric_result['ci_lower_or'],
            ci_high=numeric_result['ci_upper_or'],
            sig=sig_text
        )

    return {
        "object": numeric_result,
        "description": description
    }