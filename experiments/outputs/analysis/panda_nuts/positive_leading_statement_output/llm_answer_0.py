def extract_final_answer(model_output):
    """
    Extract coefficients, p-values, 95% CIs and interpretive percent-change
    (on the original 1 + rate scale) for the focal predictors:
      - age
      - Sex_Male
      - Help_Received

    Returns a dictionary with keys:
      - "object": dict mapping each predictor to its stats
      - "description": brief explanation of which model was used and how to interpret
    """
    import math

    predictors = ['age', 'Sex_Male', 'Help_Received']

    # Prefer the mixed model result if available; otherwise fall back to cluster-robust OLS
    mixed = model_output.get('mixedlm_result', None)
    ols = model_output.get('cluster_robust_ols', None)
    used_model = None
    used_name = None

    if mixed is not None:
        used_model = mixed
        used_name = 'mixedlm_result'
    elif ols is not None:
        used_model = ols
        used_name = 'cluster_robust_ols'
    else:
        raise ValueError("No model result found in model_output (expected keys 'mixedlm_result' or 'cluster_robust_ols').")

    # Helper accessors with safe fallbacks
    params = getattr(used_model, 'params', None)
    pvalues = getattr(used_model, 'pvalues', None)
    bse = getattr(used_model, 'bse', None)
    # conf_int may be a method
    conf_int = None
    try:
        conf_int = used_model.conf_int()
    except Exception:
        conf_int = None

    results = {}
    for var in predictors:
        entry = {
            'coef': None,
            'p_value': None,
            'ci_95': (None, None),
            'percent_change_estimate': None,
            'percent_change_ci_95': (None, None),
            'significant_at_0_05': None,
            'note': ''
        }

        if params is None or var not in params.index:
            entry['note'] = f"Variable '{var}' not found in the fitted model's fixed-effects parameters."
            results[var] = entry
            continue

        coef = float(params.loc[var])
        entry['coef'] = coef

        # p-value
        if (pvalues is not None) and (var in pvalues.index):
            entry['p_value'] = float(pvalues.loc[var])
            entry['significant_at_0_05'] = entry['p_value'] < 0.05
        else:
            entry['note'] += "p-value not available. "
            entry['significant_at_0_05'] = None

        # 95% CI: try conf_int first, else use coef +/- 1.96*bse
        ci_low = None
        ci_high = None
        try:
            if conf_int is not None and var in conf_int.index:
                # conf_int may be a DataFrame with columns [0,1]
                ci_row = conf_int.loc[var]
                # Support both DataFrame or numpy array shapes
                ci_low = float(ci_row.iloc[0])
                ci_high = float(ci_row.iloc[1])
            elif (bse is not None) and (var in bse.index):
                se = float(bse.loc[var])
                ci_low = coef - 1.96 * se
                ci_high = coef + 1.96 * se
            else:
                entry['note'] += "Could not compute 95% CI (no conf_int or bse available). "
        except Exception as e:
            entry['note'] += f"Error computing CI: {e}. "

        entry['ci_95'] = (ci_low, ci_high)

        # Interpret on original scale:
        # Outcome was log(1 + rate). A 1-unit increase in predictor multiplies (1 + rate) by exp(coef).
        try:
            pct = (math.exp(coef) - 1.0) * 100.0
            entry['percent_change_estimate'] = pct
            if (ci_low is not None) and (ci_high is not None):
                pct_low = (math.exp(ci_low) - 1.0) * 100.0
                pct_high = (math.exp(ci_high) - 1.0) * 100.0
                entry['percent_change_ci_95'] = (pct_low, pct_high)
        except Exception:
            entry['note'] += "Could not back-transform coefficient to percent-change scale. "

        # Short human-readable interpretation fragment
        interp = []
        if entry['significant_at_0_05'] is True:
            interp.append("statistically significant (p < 0.05)")
        elif entry['significant_at_0_05'] is False:
            interp.append("not statistically significant (p >= 0.05)")
        else:
            interp.append("significance unknown (p-value not available)")

        if entry['percent_change_estimate'] is not None:
            interp.append(f"estimated change in (1 + nuts/sec): {entry['percent_change_estimate']:.1f}% per unit increase")
        else:
            interp.append("no percent-change estimate available")

        entry['interpretation'] = "; ".join(interp)

        results[var] = entry

    description = (
        f"Extracted fixed-effect estimates for predictors from the fitted model '{used_name}'.\n"
        "Coefficients are on the log(1 + nuts/sec) scale. For interpretability, the code also\n"
        "reports the approximate percent change in (1 + nuts/sec) = exp(coef)-1 (×100%).\n"
        "Each predictor entry includes coef, p-value (if available), 95% CI (if available),\n"
        "percent-change interpretation, and whether it's statistically significant at alpha=0.05."
    )

    return {'object': results, 'description': description}