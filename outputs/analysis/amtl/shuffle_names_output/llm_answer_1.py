def extract_final_answer(model_output):
    """
    Extracts the coefficient, SE, z, p-value, 95% CI, and odds-ratio for the 'is_human'
    variable from the provided model_output dict.

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Concise interpretation of the results in context"
      }
    The function will prefer clustered-robust results if present (model_output['clustered_result']),
    otherwise it falls back to the original model_result.
    """
    import numpy as np

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Prefer clustered result if available
    res = model_output.get('clustered_result') or model_output.get('model_result')
    cluster_error = model_output.get('cluster_error')

    if res is None:
        raise ValueError("No results object found in model_output under 'clustered_result' or 'model_result'.")

    varname = 'is_human'

    # Try multiple ways to extract statistics (handles both result wrappers and plain summaries)
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        # conf_int() returns a DataFrame or array-like
        conf = res.conf_int()
    except Exception as e:
        # If the result object does not have the usual attributes, attempt to parse model_summary text
        summary_text = model_output.get('model_summary', '')
        raise RuntimeError(f"Unable to extract numeric results from the result object: {e}. "
                           f"Model summary (if any) provided for debugging:\n{summary_text}")

    if varname not in params.index:
        raise KeyError(f"Variable '{varname}' not found in model results. Available variables: {list(params.index)}")

    coef = float(params.loc[varname])
    se = float(bse.loc[varname]) if varname in bse.index else float(np.nan)
    pval = float(pvalues.loc[varname]) if varname in pvalues.index else float(np.nan)

    # Extract confidence interval for the coefficient
    # conf may be a DataFrame with columns [0,1] or named; handle accordingly
    try:
        if hasattr(conf, 'loc'):
            ci_lower = float(conf.loc[varname, 0])
            ci_upper = float(conf.loc[varname, 1])
        else:
            # conf is array-like with same ordering as params
            idx = list(params.index).index(varname)
            ci_lower = float(conf[idx, 0])
            ci_upper = float(conf[idx, 1])
    except Exception:
        # fallback to NaNs
        ci_lower = float(np.nan)
        ci_upper = float(np.nan)

    # Exponentiate to get odds ratio and CI on OR scale (since logit model)
    try:
        odds_ratio = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else float(np.nan)
        or_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else float(np.nan)
    except Exception:
        odds_ratio = or_ci_lower = or_ci_upper = float(np.nan)

    # Number of observations (if available)
    nobs = None
    try:
        # statsmodels result objects sometimes expose nobs or .model.endog / .model.exog
        nobs = int(getattr(res, 'nobs', getattr(res.model, 'nobs', None)))
    except Exception:
        nobs = None

    # Build the numeric object to return
    results_object = {
        'variable': varname,
        'coef': coef,
        'std_err': se,
        'z_or_t': float(getattr(res, 'tvalues', {}).get(varname, np.nan)) if hasattr(res, 'tvalues') else float(getattr(res, 'zvalues', {}).get(varname, np.nan) if hasattr(res, 'zvalues') else np.nan),
        'p_value': pval,
        'conf_int_coef': [ci_lower, ci_upper],
        'odds_ratio': odds_ratio,
        'conf_int_odds_ratio': [or_ci_lower, or_ci_upper],
        'nobs': nobs,
        'used_clustered_results': model_output.get('clustered_result') is not None,
        'cluster_error': cluster_error
    }

    # Short interpretation in context
    # Decision rule: conventional alpha = 0.05
    if np.isnan(pval):
        conclusion = "p-value unavailable; cannot make a statistical conclusion."
    elif pval < 0.05:
        # direction: check sign of coef
        direction = "higher" if coef > 0 else "lower"
        conclusion = (f"Statistically significant effect (p = {pval:.3g}). "
                      f"Modern humans (is_human=1) have {direction} AMTL frequency compared to non-human primates, "
                      f"estimated OR = {odds_ratio:.3g} (95% CI {or_ci_lower:.3g}–{or_ci_upper:.3g}).")
    else:
        conclusion = (f"No evidence of a difference in AMTL frequency between modern humans and the non-human primate genera "
                      f"after adjusting for age, sex, and tooth class (coef = {coef:.3g}, p = {pval:.3g}). "
                      f"Estimated odds ratio = {odds_ratio:.3g} (95% CI {or_ci_lower:.3g}–{or_ci_upper:.3g}).")

    description = ("Extracted coefficient and inference for 'is_human'. " + conclusion +
                   (f" Results used clustered-robust SEs: {results_object['used_clustered_results']}. "
                    f"Cluster error (if any): {cluster_error}"))

    return {"object": results_object, "description": description}