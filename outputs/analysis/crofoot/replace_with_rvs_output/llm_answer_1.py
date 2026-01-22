def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLM/ResultsWrapper (possibly
    cluster-robust results) to answer whether relative group size and contest
    location influence the probability of focal group winning.

    Returns:
      dict with keys:
        - "object": dict of extracted numeric results (coef, SE, p, 95% CI,
                    odds ratios, and marginal effects for RelSizeRatio at
                    RelDistance = -1, 0, +1)
        - "description": short plain-English interpretation of the results
    """
    import numpy as np
    import pandas as pd

    # Helper to try multiple possible names for the interaction term
    possible_interactions = ['RelSizeRatio:RelDistance', 'RelDistance:RelSizeRatio']

    # Obtain parameter estimates, std errors, p-values, conf_int, covariance matrix robustly
    try:
        params = model_output.params
    except Exception:
        raise ValueError("Could not extract params from model_output")

    # Ensure params is a pandas Series for convenient .get/.index usage
    if not isinstance(params, pd.Series):
        params = pd.Series(params, index=getattr(model_output, 'param_names', None))

    # p-values
    pvalues = None
    try:
        pvalues = model_output.pvalues
        if not isinstance(pvalues, pd.Series):
            pvalues = pd.Series(pvalues, index=params.index)
    except Exception:
        # may not be available
        pvalues = pd.Series({k: np.nan for k in params.index})

    # Confidence intervals
    try:
        ci = model_output.conf_int()
        # conf_int may return an ndarray or DataFrame
        if isinstance(ci, np.ndarray):
            ci = pd.DataFrame(ci, index=params.index, columns=[0,1])
        elif isinstance(ci, pd.DataFrame):
            # ensure columns are numeric 0,1
            pass
        else:
            ci = pd.DataFrame(ci, index=params.index, columns=[0,1])
    except Exception:
        # fallback to NA intervals
        ci = pd.DataFrame(index=params.index, columns=[0,1], data=np.nan)

    # Covariance matrix (for marginal effect SEs)
    cov_ok = True
    try:
        cov = model_output.cov_params()
        cov = pd.DataFrame(cov, index=params.index, columns=params.index)
    except Exception:
        cov_ok = False
        cov = None

    # Terms of interest
    terms = ['RelSizeRatio', 'RelDistance']
    # find which interaction name is present
    interaction_term = None
    for name in possible_interactions:
        if name in params.index:
            interaction_term = name
            break
    if interaction_term is None:
        # Interaction not present (maybe model was refit without it)
        interaction_term = possible_interactions[0]  # still report attempted name
    terms.append(interaction_term)

    results = {}
    for t in terms:
        if t in params.index:
            coef = float(params[t])
            se = float(model_output.bse[t]) if hasattr(model_output, 'bse') and t in model_output.bse.index else (float(ci.loc[t,1] - ci.loc[t,0]) / (2*1.96) if t in ci.index else np.nan)
            p = float(pvalues[t]) if t in pvalues.index else np.nan
            ci_low = float(ci.loc[t, 0]) if (t in ci.index and 0 in ci.columns) else np.nan
            ci_high = float(ci.loc[t, 1]) if (t in ci.index and 1 in ci.columns) else np.nan
            or_val = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low)) if not np.isnan(ci_low) else np.nan
            or_ci_high = float(np.exp(ci_high)) if not np.isnan(ci_high) else np.nan
            results[t] = {
                'term': t,
                'coef': coef,
                'se': se,
                'pvalue': p,
                'ci_95': (ci_low, ci_high),
                'odds_ratio': or_val,
                'odds_ratio_95': (or_ci_low, or_ci_high),
                'significant_at_0.05': bool((not np.isnan(p)) and (p < 0.05))
            }
        else:
            results[t] = {
                'term': t,
                'note': 'term not present in model',
            }

    # Compute marginal effect (slope of log-odds w.r.t RelSizeRatio) at several values of RelDistance
    # slope = beta_size + beta_interaction * RelDistance_value
    marg_points = {'RelDistance=-1': -1.0, 'RelDistance=0': 0.0, 'RelDistance=+1': 1.0}
    marg_results = {}
    if ('RelSizeRatio' in params.index) and (interaction_term in params.index):
        beta_size = float(params['RelSizeRatio'])
        beta_int = float(params[interaction_term])
        for lab, val in marg_points.items():
            slope = beta_size + beta_int * val
            # compute SE of slope if covariance available
            if cov_ok and ('RelSizeRatio' in cov.index) and (interaction_term in cov.index):
                var_size = float(cov.loc['RelSizeRatio', 'RelSizeRatio'])
                var_int = float(cov.loc[interaction_term, interaction_term])
                covar = float(cov.loc['RelSizeRatio', interaction_term])
                var_slope = var_size + (val**2) * var_int + 2 * val * covar
                se_slope = float(np.sqrt(max(var_slope, 0.0)))
                ci_low = slope - 1.96 * se_slope
                ci_high = slope + 1.96 * se_slope
                or_slope = float(np.exp(slope))
                or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
            else:
                se_slope = np.nan
                ci_low = np.nan
                ci_high = np.nan
                or_slope = float(np.exp(slope))
                or_ci = (np.nan, np.nan)
            marg_results[lab] = {
                'RelDistance_value': val,
                'slope_log_odds_per_unit_RelSizeRatio': slope,
                'se_slope': se_slope,
                'slope_95_CI': (ci_low, ci_high),
                'odds_ratio_per_unit_RelSizeRatio': or_slope,
                'or_95_CI': or_ci,
                'significant_at_0.05': (not np.isnan(se_slope)) and (abs(slope / se_slope) > 1.96)
            }
    else:
        marg_results = {'note': 'Cannot compute marginal effects because one or both terms missing.'}

    # Short textual interpretation
    # Determine conclusions about main effects and interaction
    def sig_label(term_dict):
        if 'significant_at_0.05' in term_dict:
            return 'significant' if term_dict['significant_at_0.05'] else 'not significant'
        return 'not available'

    interpret_lines = []
    # RelSizeRatio
    rs = results.get('RelSizeRatio', {})
    if 'coef' in rs:
        interpret_lines.append(
            f"RelSizeRatio: coef={rs['coef']:.3f}, OR={rs['odds_ratio']:.3f}, p={rs['pvalue']:.3g} ({sig_label(rs)})."
        )
    else:
        interpret_lines.append("RelSizeRatio: result not available in model output.")

    # RelDistance
    rd = results.get('RelDistance', {})
    if 'coef' in rd:
        interpret_lines.append(
            f"RelDistance (location advantage): coef={rd['coef']:.3f}, OR={rd['odds_ratio']:.3f}, p={rd['pvalue']:.3g} ({sig_label(rd)})."
        )
    else:
        interpret_lines.append("RelDistance: result not available in model output.")

    # Interaction
    ri = results.get(interaction_term, {})
    if 'coef' in ri:
        # interpret sign: positive interaction -> numerical advantage more effective when contest is closer to focal (RelDistance positive)
        sign_desc = ("positive -> numerical advantage becomes stronger when contest is closer to focal"
                     if ri['coef'] > 0 else
                     "negative -> numerical advantage becomes weaker when contest is closer to focal")
        interpret_lines.append(
            f"Interaction ({interaction_term}): coef={ri['coef']:.3f}, OR={ri['odds_ratio']:.3f}, p={ri['pvalue']:.3g} ({sig_label(ri)}). Interpretation: {sign_desc}."
        )
    else:
        interpret_lines.append(f"Interaction term {interaction_term} not present in model output.")

    # Combine description
    description = (
        "Extracted model coefficients and uncertainty for RelSizeRatio, RelDistance, and their interaction.\n"
        "Key numeric outputs are in 'object'.\n"
        "Summary:\n" + "\n".join(interpret_lines) +
        "\n\nMarginal effects (how the log-odds change per unit increase in RelSizeRatio) are provided for RelDistance = -1, 0, +1 in 'object'['marginal_effects']."
    )

    # Assemble final object to return
    object_dict = {
        'terms': results,
        'marginal_effects': marg_results,
        'notes': {
            'interaction_term_used': interaction_term,
            'covariance_available_for_marginal_SEs': cov_ok
        }
    }

    return {'object': object_dict, 'description': description}