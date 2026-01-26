def extract_final_answer(model_output):
    """
    Extract key statistics for age_years, is_female, and received_help from a fitted statsmodels
    results object (MixedLMResultsWrapper, RegressionResultsWrapper, or similar).

    Returns a dict with:
      - "object": dict mapping each target variable to its estimated coefficient, standard error,
                  p-value, 95% CI, and a boolean 'significant' (p < 0.05). Missing variables
                  are reported as None.
      - "description": a concise human-readable interpretation of the estimates in the study context.
    """
    import numpy as np

    targets = ['age_years', 'is_female', 'received_help']

    # Helper to safely get attributes from different statsmodels result wrappers
    def safe_attr(res, name):
        return getattr(res, name, None)

    # Try to access params, bse, pvalues, conf_int
    params = safe_attr(model_output, 'params')
    pvalues = safe_attr(model_output, 'pvalues')
    bse = safe_attr(model_output, 'bse')
    # conf_int might be a method
    try:
        conf_int = model_output.conf_int()
    except Exception:
        conf_int = None

    results = {}
    desc_lines = []
    for var in targets:
        if params is None or var not in params.index:
            results[var] = {
                'coefficient': None,
                'std_error': None,
                'p_value': None,
                'ci_lower': None,
                'ci_upper': None,
                'significant': None,
                'note': f"Variable '{var}' not found in model output."
            }
            desc_lines.append(f"{var}: not estimated (missing from model output).")
            continue

        coef = float(params.loc[var])
        se = float(bse.loc[var]) if (bse is not None and var in bse.index) else None
        pval = float(pvalues.loc[var]) if (pvalues is not None and var in pvalues.index) else None

        if conf_int is not None and var in conf_int.index:
            ci_lower = float(conf_int.loc[var, 0])
            ci_upper = float(conf_int.loc[var, 1])
        else:
            ci_lower = None
            ci_upper = None

        significant = None
        if pval is not None:
            significant = bool(pval < 0.05)

        results[var] = {
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': significant,
            'note': None
        }

        # Build interpretation line for this variable
        if var == 'age_years':
            meaning = "Each additional year of age is associated with"
        elif var == 'is_female':
            meaning = "Being female (is_female=1) compared to male (0) is associated with"
        elif var == 'received_help':
            meaning = "Receiving help (received_help=1) compared to no help (0) is associated with"
        else:
            meaning = f"{var}:"

        # Format numbers for description
        coef_s = f"{coef:.6g}"
        if ci_lower is not None and ci_upper is not None:
            ci_s = f"95% CI [{ci_lower:.6g}, {ci_upper:.6g}]"
        else:
            ci_s = "95% CI not available"

        p_s = f"p = {pval:.3g}" if pval is not None else "p-value not available"
        sig_s = "statistically significant (p < 0.05)" if significant else "not statistically significant (p ≥ 0.05)" if significant is not None else "significance unknown"

        desc_lines.append(f"{meaning} a change of {coef_s} nuts/sec ({ci_s}; {p_s}) — {sig_s}.")

    description = " ".join(desc_lines)

    return {
        "object": results,
        "description": description
    }