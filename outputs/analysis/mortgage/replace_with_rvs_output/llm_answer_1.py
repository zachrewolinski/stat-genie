def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted binary/logit model output.

    Accepts either:
      - a statsmodels result object (BinaryResultsWrapper or GLMResults), or
      - a dict containing 'model_result' (as produced by the modeling function) or
        precomputed 'odds_ratios' / 'odds_CI_lower' / 'odds_CI_upper'.

    Returns a dictionary with:
      - "object": dict with numeric results: coefficient (log-odds), p-value,
                  odds_ratio, CI_lower, CI_upper, and a boolean 'significant' (at alpha=0.05)
      - "description": short human-readable interpretation of the effect of being female
                       on mortgage acceptance.
    """
    import numpy as np

    # Helper to format numeric safely
    def _fmt(x, dig=4):
        try:
            return float(np.round(x, dig))
        except Exception:
            return x

    # Try to obtain a statsmodels result object
    result = None
    if hasattr(model_output, 'params') and hasattr(model_output, 'pvalues'):
        # model_output looks like a statsmodels result object
        result = model_output
    elif isinstance(model_output, dict) and 'model_result' in model_output:
        result = model_output['model_result']
    else:
        # May still contain precomputed odds ratios and CIs
        result = None

    # Initialize return fields
    output = {
        'coefficient': None,
        'p_value': None,
        'odds_ratio': None,
        'CI_lower': None,
        'CI_upper': None,
        'significant': None
    }

    try:
        if result is not None:
            # Extract from statsmodels result object
            # Use .params, .pvalues, .conf_int()
            params = result.params
            pvalues = result.pvalues
            conf = result.conf_int()

            if 'female' not in params.index:
                raise KeyError("The model result does not contain a 'female' coefficient")

            coef = float(params['female'])
            pval = float(pvalues['female']) if 'female' in pvalues.index else None
            ci_lower_log, ci_upper_log = float(conf.loc['female', 0]), float(conf.loc['female', 1])

            odds = float(np.exp(coef))
            ci_lower = float(np.exp(ci_lower_log))
            ci_upper = float(np.exp(ci_upper_log))

            output.update({
                'coefficient': _fmt(coef, 6),
                'p_value': _fmt(pval, 6) if pval is not None else None,
                'odds_ratio': _fmt(odds, 6),
                'CI_lower': _fmt(ci_lower, 6),
                'CI_upper': _fmt(ci_upper, 6),
                'significant': (pval is not None) and (pval < 0.05)
            })
        else:
            # Fall back to reading precomputed odds ratios and CIs from the dict
            if not isinstance(model_output, dict):
                raise ValueError("model_output must be a statsmodels result or a dict with precomputed values")

            or_dict = model_output.get('odds_ratios', {})
            ci_low_dict = model_output.get('odds_CI_lower', {})
            ci_up_dict = model_output.get('odds_CI_upper', {})

            if 'female' not in or_dict:
                raise KeyError("No 'female' entry found in provided odds ratios")

            odds = float(or_dict['female'])
            ci_lower = float(ci_low_dict['female']) if 'female' in ci_low_dict else None
            ci_upper = float(ci_up_dict['female']) if 'female' in ci_up_dict else None

            # Coefficient and p-value not available in this branch
            output.update({
                'coefficient': None,
                'p_value': None,
                'odds_ratio': _fmt(odds, 6),
                'CI_lower': _fmt(ci_lower, 6) if ci_lower is not None else None,
                'CI_upper': _fmt(ci_upper, 6) if ci_upper is not None else None,
                'significant': None
            })
    except Exception as e:
        # Return a clear message if extraction failed
        return {
            "object": None,
            "description": f"Failed to extract 'female' statistics from model_output: {e}"
        }

    # Build interpretation string
    if output['odds_ratio'] is None:
        description = "Could not extract odds ratio for 'female' from the provided model output."
    else:
        or_val = output['odds_ratio']
        ci_l = output['CI_lower']
        ci_u = output['CI_upper']
        coef = output['coefficient']
        pval = output['p_value']
        sig = output['significant']

        # Percent change in odds
        try:
            pct_change = (float(or_val) - 1.0) * 100.0
            pct_str = f"{_fmt(pct_change,3)}%"
        except Exception:
            pct_str = "N/A"

        # Significance wording
        if sig is True:
            sig_text = "statistically significant (p < 0.05)"
        elif sig is False:
            sig_text = "not statistically significant (p >= 0.05)"
        else:
            sig_text = "statistical significance could not be determined (p-value not available)"

        # Compose description
        description = (
            f"Female coefficient (log-odds) = {coef}. "
            f"Odds ratio = {or_val} (95% CI: {ci_l} to {ci_u}). "
            f"This implies female applicants have about {pct_str} change in the odds of mortgage acceptance "
            f"compared to male applicants. The effect is {sig_text}."
        )
        if pval is not None:
            description += f" (p = {_fmt(pval,6)})"

    return {
        "object": output,
        "description": description
    }