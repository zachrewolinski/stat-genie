def extract_final_answer(model_output):
    """
    Extracts the Female effect from a fitted logistic model output dict.

    Returns a dict with:
      - "object": dict with numeric extraction (coef, p_value, odds_ratio, 95% CI)
      - "description": short plain-language interpretation in the context of mortgage approval
    """
    import numpy as np

    # Initialize placeholders
    coef = pval = oratio = ci_low = ci_high = None

    # If model_output is the dict returned by the modeling function, try preferred entries first
    if isinstance(model_output, dict):
        # Try to extract from statsmodels result object if available
        result = model_output.get('result', None)

        # 1) Prefer direct statsmodels result (gives coef, p-value, conf_int)
        if result is not None:
            try:
                coef = float(result.params['Female'])
            except Exception:
                try:
                    coef = float(result.params.loc['Female'])
                except Exception:
                    coef = None
            try:
                pval = float(result.pvalues['Female'])
            except Exception:
                try:
                    pval = float(result.pvalues.loc['Female'])
                except Exception:
                    pval = None
            try:
                # odds ratio and CI from parameter and conf_int
                oratio = float(np.exp(coef)) if coef is not None else None
                conf = result.conf_int()
                # conf can be a DataFrame with rows indexed by variable name
                try:
                    ci_low = float(np.exp(conf.loc['Female', 0]))
                    ci_high = float(np.exp(conf.loc['Female', 1]))
                except Exception:
                    # fallback if conf indexing differs
                    ci_low = ci_high = None
            except Exception:
                pass

        # 2) If odds_ratios/conf_int_odds were precomputed in the dict, use them if needed
        if (oratio is None or ci_low is None or ci_high is None) and 'odds_ratios' in model_output:
            try:
                ors = model_output['odds_ratios']
                oratio = float(ors['Female'])
            except Exception:
                try:
                    oratio = float(ors.loc['Female'])
                except Exception:
                    pass
            # try confidence intervals on odds scale
            conf_odds = model_output.get('conf_int_odds', None)
            if conf_odds is not None:
                try:
                    ci_low = float(conf_odds.loc['Female', 0])
                    ci_high = float(conf_odds.loc['Female', 1])
                except Exception:
                    try:
                        # if conf_odds is an ndarray-like with index alignment
                        ci_low = float(conf_odds['Female'][0])
                        ci_high = float(conf_odds['Female'][1])
                    except Exception:
                        pass

    # If we still lack the coefficient or p-value but have odds ratio, convert back to coef where possible
    if coef is None and oratio is not None:
        try:
            coef = float(np.log(oratio))
        except Exception:
            coef = None

    # Prepare returned numeric object
    numeric_object = {
        'coef_Female_log_odds': None if coef is None else coef,
        'p_value_Female': None if pval is None else pval,
        'odds_ratio_Female': None if oratio is None else oratio,
        'odds_ratio_95ci_lower': None if ci_low is None else ci_low,
        'odds_ratio_95ci_upper': None if ci_high is None else ci_high
    }

    # Build plain-language description
    desc_parts = []
    if numeric_object['odds_ratio_Female'] is not None:
        desc_parts.append(
            f"Female odds ratio = {numeric_object['odds_ratio_Female']:.3f} "
            f"(95% CI {numeric_object['odds_ratio_95ci_lower']:.3f}–{numeric_object['odds_ratio_95ci_upper']:.3f})"
        )
    elif numeric_object['coef_Female_log_odds'] is not None:
        # show coef on log-odds if OR not available
        desc_parts.append(f"Female log-odds coef = {numeric_object['coef_Female_log_odds']:.3f}")

    if numeric_object['p_value_Female'] is not None:
        desc_parts.append(f"p = {numeric_object['p_value_Female']:.3f}")
        signif = numeric_object['p_value_Female'] < 0.05
    else:
        signif = None

    # Interpretation sentence
    if numeric_object['odds_ratio_Female'] is not None and signif is not None:
        if signif:
            interp = (
                "Controlling for the listed covariates, female applicants have higher odds of mortgage approval "
                "than male applicants, and this difference is statistically significant at the 0.05 level."
            )
        else:
            interp = (
                "Controlling for the listed covariates, the estimated difference in approval odds for female vs male "
                "applicants is not statistically significant at the 0.05 level."
            )
    else:
        interp = "Could not fully determine statistical significance or effect size from the provided model output."

    description = " ; ".join(desc_parts + [interp])

    return {
        "object": numeric_object,
        "description": description
    }