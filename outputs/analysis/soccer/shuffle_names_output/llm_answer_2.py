def extract_final_answer(model_output):
    """
    Extracts statistics for the 'DarkSkin' predictor from a fitted negative-binomial model output.
    Expects model_output to be a dict with either:
      - 'irr_table': a pandas DataFrame containing rows for coefficients (including 'DarkSkin'),
        with columns ['coef','irr','ci_lower','ci_upper','pvalue'], OR
      - 'results': a statsmodels GLMResultsWrapper from which coef/conf/pvalues can be computed.
    Returns a dict with:
      - "object": dict of numeric results (coef, irr, ci, pvalue) and boolean flags,
      - "description": short plain-English interpretation of the result in context.
    """
    import numpy as np
    import pandas as pd

    # Helper to format numeric values as native Python floats
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    # Try using precomputed IRR table if available
    irr_table = model_output.get('irr_table')
    if irr_table is not None:
        if not isinstance(irr_table, pd.DataFrame):
            # try to convert
            irr_table = pd.DataFrame(irr_table)
        if 'DarkSkin' not in irr_table.index:
            raise KeyError("irr_table present but does not contain a 'DarkSkin' row.")
        row = irr_table.loc['DarkSkin']
        coef = _f(row.get('coef'))
        irr = _f(row.get('irr'))
        ci_lower = _f(row.get('ci_lower'))
        ci_upper = _f(row.get('ci_upper'))
        pvalue = _f(row.get('pvalue'))
    else:
        # Fall back to statsmodels results object
        results = model_output.get('results')
        if results is None:
            raise KeyError("model_output must contain either 'irr_table' or 'results'.")
        params = results.params
        conf = results.conf_int()
        pvals = results.pvalues
        if 'DarkSkin' not in params.index:
            raise KeyError("'DarkSkin' not found in model parameters.")
        coef = _f(params['DarkSkin'])
        irr = _f(np.exp(coef) if coef is not None else None)
        # conf is a DataFrame with two columns (lower, upper)
        try:
            ci_lower = _f(np.exp(conf.loc['DarkSkin'].iloc[0]))
            ci_upper = _f(np.exp(conf.loc['DarkSkin'].iloc[1]))
        except Exception:
            ci_lower, ci_upper = None, None
        pvalue = _f(pvals['DarkSkin'])

    # Determine statistical interpretation
    significant = (pvalue is not None) and (pvalue < 0.05)
    ci_excludes_one = (ci_lower is not None and ci_upper is not None) and ((ci_lower > 1.0) or (ci_upper < 1.0))
    direction = None
    if irr is not None:
        if irr > 1.0:
            direction = "higher"
        elif irr < 1.0:
            direction = "lower"
        else:
            direction = "no difference"

    # Simple yes/no answer to the question "Are dark-skinned players more likely to receive red cards?"
    if irr is not None:
        if (irr > 1.0) and significant:
            simple_answer = "Yes"
        elif (irr > 1.0) and (not significant):
            simple_answer = "No (point estimate >1 but not statistically significant)"
        elif (irr <= 1.0) and significant:
            # statistically significantly lower or equal
            simple_answer = "No"
        else:
            simple_answer = "No"
    else:
        simple_answer = "Insufficient information"

    # Build the numeric object to return
    object_dict = {
        'coef': coef,
        'irr': irr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'pvalue': pvalue,
        'significant_at_0.05': bool(significant),
        'ci_excludes_one': bool(ci_excludes_one),
        'direction': direction,
        'simple_answer': simple_answer
    }

    # Construct human-readable description
    if irr is not None and pvalue is not None and ci_lower is not None and ci_upper is not None:
        description = (
            f"The estimated incidence rate ratio (IRR) for DarkSkin is {irr:.3f} "
            f"(coef = {coef:.3f}), 95% CI = [{ci_lower:.3f}, {ci_upper:.3f}], "
            f"p = {pvalue:.4f}. This implies that players rated as Dark receive red cards at an "
            f"estimated {100*(irr-1):.1f}% {direction} rate compared to Light players, "
            f"controlling for covariates and exposure. "
        )
        if significant:
            description += "This effect is statistically significant at alpha = 0.05."
        else:
            description += "This effect is not statistically significant at alpha = 0.05."
    else:
        description = "Could not extract complete statistics for 'DarkSkin' from the provided model_output."

    return {"object": object_dict, "description": description}