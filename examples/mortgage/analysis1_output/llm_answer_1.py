def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted logistic model output.
    Expects model_output to be the dictionary produced by the provided modeling function:
      {'results': <statsmodels results>, 'odds_ratio_table': pd.DataFrame}
    Returns a dict with keys:
      - "object": dict with numeric results (coef, odds_ratio, 95% CI, p-value, significance, percent change)
      - "description": plain-language interpretation in context
    """
    import math

    # Try to get odds-ratio table and results object if present
    or_table = None
    results = None
    if isinstance(model_output, dict):
        or_table = model_output.get('odds_ratio_table', None)
        results = model_output.get('results', None)
    else:
        # If the user passed a bare results object, use it directly
        results = model_output

    # Extract values (prefer odds_ratio_table if available because it already contains transformed values)
    try:
        if or_table is not None and 'female' in or_table.index:
            row = or_table.loc['female']
            coef = float(row['coef'])
            odds_ratio = float(row['odds_ratio'])
            ci_lower = float(row['ci_lower'])
            ci_upper = float(row['ci_upper'])
            p_value = float(row['p_value'])
        elif results is not None:
            # Use statsmodels results to compute
            coef = float(results.params['female'])
            odds_ratio = float(math.exp(coef))
            ci = results.conf_int().loc['female']  # [lower, upper] on log-odds scale
            ci_lower = float(math.exp(ci[0]))
            ci_upper = float(math.exp(ci[1]))
            p_value = float(results.pvalues['female'])
        else:
            raise KeyError("Could not find 'odds_ratio_table' or 'results' in model_output.")
    except Exception as e:
        raise RuntimeError(f"Error extracting 'female' statistics: {e}")

    alpha = 0.05
    significant = p_value < alpha
    percent_change_in_odds = (odds_ratio - 1.0) * 100.0

    interpretation = (
        f"Holding the listed controls constant, female applicants have an estimated "
        f"odds ratio of {odds_ratio:.3f} for mortgage acceptance (95% CI {ci_lower:.3f}–{ci_upper:.3f}). "
        f"This corresponds to a {percent_change_in_odds:+.1f}% change in the odds of acceptance "
        f"relative to male applicants. The two-sided p-value is {p_value:.3g}, "
        f"which is {'<' if significant else '>='} {alpha}, so this effect is "
        f"{'statistically significant' if significant else 'not statistically significant'} at the {alpha} level."
    )

    return {
        "object": {
            "coef": coef,
            "odds_ratio": odds_ratio,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "alpha": alpha,
            "significant": significant,
            "percent_change_in_odds": percent_change_in_odds
        },
        "description": interpretation
    }