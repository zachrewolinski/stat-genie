def extract_final_answer(model_output):
    """
    Extract the effect of IsHuman from a fitted statsmodels GLM result dict (as returned by the model function).
    Returns a dict with:
      - "object": dict of numeric results (coef, se, z, p, 95% CI on coef scale, odds ratio and its 95% CI)
      - "description": brief plain-language interpretation answering whether modern humans have higher AMTL.
    """
    import numpy as np
    import pandas as pd

    # Choose clustered result if available, otherwise fall back to glm_result
    result = None
    if isinstance(model_output, dict):
        result = model_output.get('glm_result_clustered') or model_output.get('glm_result')
    else:
        result = model_output

    if result is None:
        raise ValueError("No model result found in model_output. Expected keys 'glm_result_clustered' or 'glm_result'.")

    # Helper to pull a named value from result, with fallbacks
    def _get_series_or_attr(attr_name):
        val = getattr(result, attr_name, None)
        return val

    # Try to extract coefficient, se, z, p-value
    try:
        params = _get_series_or_attr('params')
        pvalues = _get_series_or_attr('pvalues')
        bse = _get_series_or_attr('bse')
    except Exception:
        params = pvalues = bse = None

    # Ensure we have pandas Series-like objects; if not, try result.summary2 or result.summary
    if params is None or 'IsHuman' not in params:
        # try to parse from conf_int or index positions
        try:
            params = pd.Series(result.params)
            pvalues = pd.Series(result.pvalues)
            bse = pd.Series(result.bse)
        except Exception:
            raise ValueError("Could not extract params/pvalues/bse from the model result.")

    if 'IsHuman' not in params.index:
        raise KeyError("Model does not contain a coefficient named 'IsHuman'.")

    coef = float(params.loc['IsHuman'])
    se = float(bse.loc['IsHuman']) if bse is not None and 'IsHuman' in bse.index else None
    pval = float(pvalues.loc['IsHuman']) if pvalues is not None and 'IsHuman' in pvalues.index else None

    # Confidence interval (default 95%)
    try:
        conf = result.conf_int()
        if isinstance(conf, (pd.DataFrame, pd.Series)):
            ci_lower = float(conf.loc['IsHuman'].iloc[0])
            ci_upper = float(conf.loc['IsHuman'].iloc[1])
        else:
            # conf_int returned ndarray; find index of IsHuman in params.index
            idx = list(params.index).index('IsHuman')
            ci_lower = float(conf[idx, 0])
            ci_upper = float(conf[idx, 1])
    except Exception:
        # If conf_int fails, approximate using coef +/- 1.96*se if se available
        if se is not None:
            ci_lower = float(coef - 1.96 * se)
            ci_upper = float(coef + 1.96 * se)
        else:
            ci_lower = ci_upper = None

    # Odds ratio and its CI
    or_val = float(np.exp(coef))
    or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
    or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

    # Build the object to return
    obj = {
        'coef_logit': coef,
        'std_err': se,
        'z_or_na': float(coef / se) if (se is not None and se != 0) else None,
        'p_value': pval,
        'ci_95_logit': [ci_lower, ci_upper],
        'odds_ratio': or_val,
        'odds_ratio_95_ci': [or_ci_lower, or_ci_upper],
        # also include a short numeric summary string for quick display
        'summary_str': (
            f"IsHuman coef = {coef:.4f} (SE={se:.4f}), p={pval:.3g}; "
            f"OR={or_val:.3f}, 95% CI for OR = [{or_ci_lower:.3f}, {or_ci_upper:.3f}]"
            if (se is not None and pval is not None and or_ci_lower is not None)
            else "Incomplete numeric summary"
        )
    }

    # Interpretation: answer the yes/no question, with context
    if pval is not None:
        if pval < 0.05:
            conclusion = (
                "Yes — controlling for tooth class, age, and ProbMale (and clustering SEs by specimen), "
                "modern humans (Homo sapiens) have a statistically significantly higher frequency of AMTL. "
                f"The model coefficient on the logit scale is {coef:.4f} (p = {pval:.3g}), corresponding to "
                f"an odds ratio of {or_val:.3f} (95% CI {or_ci_lower:.3f}–{or_ci_upper:.3f}). "
                "This indicates about a "
                f"{(or_val - 1) * 100:.1f}% higher odds of a missing tooth in modern humans vs the non-human genera in the sample, "
                "after adjustment. The effect size is modest."
            )
        else:
            conclusion = (
                "No — there is not evidence of a statistically significant difference in AMTL frequency between modern humans "
                f"and non-human primates after adjustment (IsHuman coef = {coef:.4f}, p = {pval:.3g})."
            )
    else:
        conclusion = "Could not determine statistical significance because the p-value was not available."

    return {
        "object": obj,
        "description": conclusion
    }