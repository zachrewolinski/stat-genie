def extract_final_answer(model_output):
    import numpy as np
    from math import exp, isnan, isfinite

    def safe_float(x):
        try:
            x = float(x)
            return x if isfinite(x) else None
        except Exception:
            return None

    # Try to get the fitted result object first
    result = model_output.get('model_result_object') if isinstance(model_output, dict) else None

    coef = pvalue = ci_low = ci_high = odds_ratio = odds_ci_low = odds_ci_high = n_obs = None

    if result is not None:
        # Preferred path: use statsmodels result attributes (params, pvalues, conf_int, nobs)
        try:
            params = result.params
            if 'is_female' in params.index:
                coef = safe_float(params['is_female'])
            else:
                # try dictionary-like access
                coef = safe_float(params.get('is_female'))
        except Exception:
            coef = None

        try:
            pvals = result.pvalues
            pvalue = safe_float(pvals.get('is_female') if hasattr(pvals, 'get') else pvals['is_female'])
        except Exception:
            pvalue = None

        try:
            conf = result.conf_int()
            # conf may be a DataFrame with rows indexed by variable names
            if 'is_female' in conf.index:
                row = conf.loc['is_female']
                # row might be array-like [low, high]
                ci_low = safe_float(row.iloc[0]) if hasattr(row, 'iloc') else safe_float(row[0])
                ci_high = safe_float(row.iloc[1]) if hasattr(row, 'iloc') else safe_float(row[1])
        except Exception:
            ci_low = ci_high = None

        try:
            n_obs = int(result.nobs)
        except Exception:
            n_obs = None

    # Fallbacks: if we couldn't get values from the result object, try the provided summaries
    if coef is None and isinstance(model_output, dict) and 'coefficients' in model_output:
        try:
            coef = safe_float(model_output['coefficients'].get('is_female'))
        except Exception:
            coef = None

    if (ci_low is None or ci_high is None) and isinstance(model_output, dict) and 'conf_int' in model_output:
        try:
            ci_entry = model_output['conf_int'].get('is_female')
            if isinstance(ci_entry, dict):
                ci_low = safe_float(ci_entry.get('ci_lower'))
                ci_high = safe_float(ci_entry.get('ci_upper'))
        except Exception:
            ci_low = ci_high = None

    if n_obs is None and isinstance(model_output, dict) and 'n_obs' in model_output:
        try:
            n_obs = int(model_output['n_obs'])
        except Exception:
            n_obs = None

    # Compute odds ratio and CI for odds ratio if we have coef/ci
    if coef is not None:
        try:
            odds_ratio = float(np.exp(coef))
        except Exception:
            odds_ratio = None
    if ci_low is not None:
        try:
            odds_ci_low = float(np.exp(ci_low))
        except Exception:
            odds_ci_low = None
    if ci_high is not None:
        try:
            odds_ci_high = float(np.exp(ci_high))
        except Exception:
            odds_ci_high = None

    # Determine simple significance statement
    significance = None
    significance_reason = None
    if pvalue is None:
        significance = None
        significance_reason = ("p-value is not available (NaN/missing). This may indicate "
                               "complete or quasi-complete separation, perfect prediction, or "
                               "that standard errors could not be computed reliably.")
    else:
        significance = (pvalue < 0.05)
        significance_reason = (f"p-value = {pvalue:.3g}; "
                               + ("statistically significant at alpha=0.05." if significance else "not statistically significant at alpha=0.05."))

    # Build the object to return
    output_object = {
        'variable': 'is_female',
        'coef_log_odds': coef,
        'odds_ratio': odds_ratio,
        'odds_ratio_ci_95': (odds_ci_low, odds_ci_high) if (odds_ci_low is not None or odds_ci_high is not None) else None,
        'conf_int_log_odds_95': (ci_low, ci_high) if (ci_low is not None or ci_high is not None) else None,
        'p_value': pvalue,
        'n_obs': n_obs,
        'significant_at_0.05': significance,
        'significance_explanation': significance_reason
    }

    # Short human-readable description / interpretation
    if coef is None:
        desc = ("Could not extract a numeric estimate for the effect of gender ('is_female') from the model result. "
                "Either the model result object is missing expected attributes or the estimate is undefined.")
    else:
        direction = "higher" if coef > 0 else "lower" if coef < 0 else "no change"
        desc = (f"Estimated effect of being female on mortgage approval: log-odds coef = {coef:.4g}, "
                f"corresponding odds ratio = {odds_ratio:.4g}." if odds_ratio is not None else
                f"Estimated effect of being female on mortgage approval: log-odds coef = {coef:.4g}.")
        if odds_ci_low is not None and odds_ci_high is not None:
            desc += (f" 95% CI for odds ratio = ({odds_ci_low:.4g}, {odds_ci_high:.4g}).")
        if pvalue is None:
            desc += " p-value is unavailable; this often indicates perfect or quasi-complete separation or unstable estimates, so the result should not be trusted for inference."
        else:
            desc += f" {significance_reason}"
        desc += f" Directionally, a {direction} probability of approval is implied for female applicants compared to male applicants (holding controls constant)."

    return {
        "object": output_object,
        "description": desc
    }