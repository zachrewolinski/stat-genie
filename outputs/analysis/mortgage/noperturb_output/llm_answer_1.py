def extract_final_answer(model_output):
    """
    Extracts statistics for the primary independent variable 'female' from a fitted model output.

    Returns a dictionary with:
      - "object": dict with coefficient (log-odds), standard error, p-value, odds ratio, and 95% CI for the odds ratio
      - "description": brief interpretation of the result in plain language

    Accepts model_output as the dict returned by the modeling function (expected keys: 'result' and/or 'odds_ratios_table').
    """
    import math
    import numpy as _np
    import pandas as _pd

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict (as returned by the modeling function).")

    result = model_output.get('result', None)
    or_table = model_output.get('odds_ratios_table', None)

    if result is None:
        raise ValueError("model_output does not contain 'result' (fitted model result object).")

    # Helper to safely extract by label or by position
    def _get_series_value(series_like, key):
        # series_like likely a pandas Series; try label access then fallback to positional if available
        try:
            return float(series_like.loc[key])
        except Exception:
            try:
                # If label not found, try to find index position from model exog names
                exog_names = getattr(result.model, "exog_names", None)
                if exog_names and key in exog_names:
                    pos = list(exog_names).index(key)
                    return float(series_like.iloc[pos])
            except Exception:
                pass
        raise KeyError(f"Could not extract key '{key}' from the provided series-like object.")

    # Extract coefficient, se, p-value, and conf-int (log-odds scale)
    try:
        coef = _get_series_value(result.params, 'female')
        se = _get_series_value(result.bse, 'female')
        pval = _get_series_value(result.pvalues, 'female')
        conf_int = result.conf_int()
        # conf_int likely a DataFrame with two columns [lower, upper]
        try:
            ci_lower_log = float(conf_int.loc['female', 0])
            ci_upper_log = float(conf_int.loc['female', 1])
        except Exception:
            # fallback by position
            exog_names = getattr(result.model, "exog_names", None)
            if exog_names and 'female' in exog_names:
                pos = list(exog_names).index('female')
                ci_lower_log = float(conf_int.iloc[pos, 0])
                ci_upper_log = float(conf_int.iloc[pos, 1])
            else:
                raise
    except KeyError:
        # As a fallback, try to use odds_ratios_table if provided
        if or_table is None:
            raise
        if 'female' not in or_table.index:
            raise KeyError("Unable to find 'female' in model result or odds_ratios_table.")
        or_val = float(or_table.loc['female', 'OR'])
        ci_lower = float(or_table.loc['female', 'CI_lower'])
        ci_upper = float(or_table.loc['female', 'CI_upper'])
        pval = float(or_table.loc['female', 'pvalue'])
        # Convert odds ratio back to log-odds for consistency
        coef = math.log(or_val) if or_val > 0 else float("nan")
        se = float('nan')
        ci_lower_log = math.log(ci_lower) if ci_lower > 0 else float("nan")
        ci_upper_log = math.log(ci_upper) if ci_upper > 0 else float("nan")

    # Compute odds ratio and its CI by exponentiating log-odds CI
    odds_ratio = float(_np.exp(coef))
    ci_lower_or = float(_np.exp(ci_lower_log))
    ci_upper_or = float(_np.exp(ci_upper_log))

    # Build result object (raw numeric values)
    result_object = {
        "coefficient_log_odds": coef,
        "std_error": se,
        "p_value": pval,
        "odds_ratio": odds_ratio,
        "odds_ratio_CI_lower": ci_lower_or,
        "odds_ratio_CI_upper": ci_upper_or
    }

    # Construct human-readable description
    sig_text = "statistically significant" if (pval is not None and pval < 0.05) else "not statistically significant"
    percent_change = (odds_ratio - 1.0) * 100.0
    description = (
        f"Effect of being female on mortgage approval (adjusted for controls):\n"
        f"- Odds ratio = {odds_ratio:.3f} (95% CI: {ci_lower_or:.3f} to {ci_upper_or:.3f}); "
        f"this corresponds to a {percent_change:.1f}% change in odds compared to male applicants.\n"
        f"- p-value = {pval:.4g}, so the effect is {sig_text} at conventional levels (alpha=0.05).\n"
        f"Interpretation: Being female is associated with {'higher' if odds_ratio>1 else 'lower' if odds_ratio<1 else 'similar'} "
        f"odds of mortgage acceptance relative to being male, controlling for the listed covariates."
    )

    return {
        "object": result_object,
        "description": description
    }