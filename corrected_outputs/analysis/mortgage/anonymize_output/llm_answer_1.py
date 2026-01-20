def extract_final_answer(model_output):
    """
    Extract the estimated effect of 'Female' from a fitted statsmodels binary logit result.
    Returns a dict with:
      - "object": dict with numeric results (coefficient, p-value, odds ratio, 95% CI, significance flag)
      - "description": plain-language interpretation of the effect in context

    This function is defensive: it tries common ways to obtain params, p-values and confidence
    intervals from statsmodels result objects (BinaryResultsWrapper / Results).
    """
    import numpy as np
    import math

    # Helper to raise a clear error if variable not found
    def _not_found(var):
        raise KeyError(f"Variable '{var}' not found in model output (params/pvalues/conf_int).")

    var = 'Female'

    # Try to get coefficient (log-odds)
    try:
        params = model_output.params  # pandas Series-like
    except Exception:
        # maybe model_output is the raw results object inside wrapper
        try:
            params = model_output._results.params
        except Exception:
            raise ValueError("Could not read params from model_output.")

    if var not in params.index:
        # try lowercase/upper variants
        var_candidates = [v for v in params.index if v.lower() == var.lower()]
        if var_candidates:
            var = var_candidates[0]
        else:
            _not_found(var)

    coef = float(params[var])

    # p-value
    try:
        pvalues = model_output.pvalues
    except Exception:
        try:
            pvalues = model_output._results.pvalues
        except Exception:
            pvalues = None

    p_value = None
    if pvalues is not None:
        if var in pvalues.index:
            p_value = float(pvalues[var])
        else:
            # try match ignoring case
            matches = [v for v in pvalues.index if v.lower() == var.lower()]
            if matches:
                p_value = float(pvalues[matches[0]])

    # odds ratio
    # model code may have attached odds_ratios attribute; else compute exp(coef)
    odds_ratio = None
    if hasattr(model_output, 'odds_ratios'):
        try:
            ors = model_output.odds_ratios
            if hasattr(ors, 'loc') and var in ors.index:
                odds_ratio = float(ors.loc[var])
            elif var in ors:
                odds_ratio = float(ors[var])
        except Exception:
            odds_ratio = None

    if odds_ratio is None:
        try:
            odds_ratio = float(np.exp(coef))
        except Exception:
            odds_ratio = None

    # 95% CI for odds ratio: prefer conf_odds if attached, else compute from conf_int()
    ci_lower = ci_upper = None
    # try conf_odds attribute first
    if hasattr(model_output, 'conf_odds'):
        try:
            conf_odds = model_output.conf_odds
            if hasattr(conf_odds, 'loc') and var in conf_odds.index:
                ci_lower = float(conf_odds.loc[var, 0])
                ci_upper = float(conf_odds.loc[var, 1])
            else:
                # conf_odds might be an array-like in same order as params
                try:
                    idx = list(conf_odds.index).index(var)
                    ci_lower = float(conf_odds.iloc[idx, 0])
                    ci_upper = float(conf_odds.iloc[idx, 1])
                except Exception:
                    pass
        except Exception:
            pass

    if ci_lower is None or ci_upper is None:
        # fallback: use conf_int() on model_output, which returns linear coef CI, then exponentiate
        try:
            conf = model_output.conf_int()
            # conf may be DataFrame-like
            if var in conf.index:
                l, u = conf.loc[var].iloc[0], conf.loc[var].iloc[1]
            else:
                matches = [v for v in conf.index if v.lower() == var.lower()]
                if matches:
                    l, u = conf.loc[matches[0]].iloc[0], conf.loc[matches[0]].iloc[1]
                else:
                    raise KeyError
            ci_lower = float(np.exp(l))
            ci_upper = float(np.exp(u))
        except KeyError:
            # last resort: compute approximate CI using coef +/- 1.96*se
            try:
                bse = model_output.bse
                if var in bse.index:
                    se = float(bse[var])
                else:
                    matches = [v for v in bse.index if v.lower() == var.lower()]
                    if matches:
                        se = float(bse[matches[0]])
                    else:
                        raise KeyError
                l, u = coef - 1.96 * se, coef + 1.96 * se
                ci_lower = float(np.exp(l))
                ci_upper = float(np.exp(u))
            except Exception:
                ci_lower = ci_upper = None

    # Determine significance at alpha=0.05 if p-value available
    significant_0_05 = None
    if p_value is not None and not math.isnan(p_value):
        significant_0_05 = (p_value < 0.05)

    # Interpretation sentence
    if odds_ratio is None:
        interpretation = "Could not compute odds ratio for 'Female'."
    else:
        pct_change = (odds_ratio - 1.0) * 100.0
        if significant_0_05 is True:
            if odds_ratio > 1:
                direction = f"Women have higher odds of mortgage approval: about {pct_change:.1f}% higher odds (OR={odds_ratio:.3f})."
            else:
                direction = f"Women have lower odds of mortgage approval: about {abs(pct_change):.1f}% lower odds (OR={odds_ratio:.3f})."
            sig_phrase = f"This effect is statistically significant at the 5% level (p = {p_value:.3g})."
        elif significant_0_05 is False:
            direction = f"The estimated effect is OR={odds_ratio:.3f}, corresponding to {pct_change:.1f}% change in odds for women versus men."
            sig_phrase = f"This effect is NOT statistically significant at the 5% level (p = {p_value:.3g})."
        else:
            direction = f"The estimated effect is OR={odds_ratio:.3f}, corresponding to {pct_change:.1f}% change in odds for women versus men."
            sig_phrase = "No p-value available to judge statistical significance."

        if ci_lower is not None and ci_upper is not None:
            ci_phrase = f"95% CI for OR: [{ci_lower:.3f}, {ci_upper:.3f}]."
        else:
            ci_phrase = "95% CI for OR could not be determined."

        interpretation = " ".join([direction, ci_phrase, sig_phrase])

    # Build the object to return (numeric values + interpretation)
    result_object = {
        "variable": var,
        "coefficient_log_odds": None if coef is None else round(coef, 6),
        "p_value": None if p_value is None else float(p_value),
        "odds_ratio": None if odds_ratio is None else float(round(odds_ratio, 6)),
        "odds_ratio_ci_lower": None if ci_lower is None else float(round(ci_lower, 6)),
        "odds_ratio_ci_upper": None if ci_upper is None else float(round(ci_upper, 6)),
        "significant_at_0.05": significant_0_05
    }

    return {
        "object": result_object,
        "description": interpretation
    }