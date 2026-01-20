def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and odds ratios
    for the two focal predictors (RelSize_z and HomeRangeAdv_z) from a
    statsmodels GLM/Results-like object.

    Returns a dict with keys:
      - "object": dict mapping each focal predictor to a dict of statistics
      - "description": a short, plain-language interpretation of those stats
    """
    import numpy as np
    import pandas as pd

    focal_vars = ['RelSize_z', 'HomeRangeAdv_z']

    # Validate model_output exposes expected attributes
    required_attrs = ['params', 'pvalues', 'bse', 'conf_int']
    missing_attrs = [a for a in required_attrs if not hasattr(model_output, a)]
    if missing_attrs:
        raise ValueError(f"model_output is missing required attributes: {missing_attrs}")

    params = model_output.params            # pandas Series (index = param names)
    pvalues = model_output.pvalues
    bse = model_output.bse

    # conf_int() -> array-like (n_params x 2) or DataFrame; ensure DataFrame with param index
    conf = model_output.conf_int()
    conf_df = pd.DataFrame(conf, index=params.index, columns=['ci_lower', 'ci_upper'])

    results = {}
    for v in focal_vars:
        if v not in params.index:
            raise ValueError(f"Predictor '{v}' not found in model parameters. Available params: {list(params.index)}")

        coef = float(params[v])
        se = float(bse[v]) if v in bse.index else float(np.nan)
        p = float(pvalues[v]) if v in pvalues.index else float(np.nan)
        ci_low = float(conf_df.loc[v, 'ci_lower'])
        ci_high = float(conf_df.loc[v, 'ci_upper'])

        # Odds ratio and its 95% CI (because model is logit)
        or_point = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))

        results[v] = {
            'coef_log_odds': coef,
            'std_error': se,
            'p_value': p,
            'ci_95_log_odds': (ci_low, ci_high),
            'odds_ratio': or_point,
            'ci_95_odds_ratio': (or_ci_low, or_ci_high),
        }

    # Build a concise description that interprets the coefficients
    def interpret_entry(name, stats):
        sign = "increase" if stats['coef_log_odds'] > 0 else ("decrease" if stats['coef_log_odds'] < 0 else "no change")
        pct_change = (stats['odds_ratio'] - 1) * 100
        signif = "statistically significant" if (not np.isnan(stats['p_value']) and stats['p_value'] < 0.05) else "not statistically significant"
        return (
            f"{name}: coefficient (log-odds) = {stats['coef_log_odds']:.3f}, SE = {stats['std_error']:.3f}, "
            f"p = {stats['p_value']:.3g}. 95% CI (log-odds) = [{stats['ci_95_log_odds'][0]:.3f}, {stats['ci_95_log_odds'][1]:.3f}]. "
            f"Odds ratio = {stats['odds_ratio']:.3f} (95% CI = [{stats['ci_95_odds_ratio'][0]:.3f}, {stats['ci_95_odds_ratio'][1]:.3f}]). "
            f"Interpretation: a one-SD increase in {name} is associated with a {pct_change:.1f}% {sign} in the odds of the focal group winning. "
            f"This effect is {signif}."
        )

    description_lines = []
    for var in focal_vars:
        description_lines.append(interpret_entry(var, results[var]))

    summary_description = " ".join(description_lines)

    return {
        "object": results,
        "description": summary_description
    }