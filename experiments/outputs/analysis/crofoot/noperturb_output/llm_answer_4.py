def extract_final_answer(model_output):
    """
    Extract coefficients, cluster-robust standard errors, z-stats, p-values,
    95% CIs, and odds ratios for the predictors of interest from the model output.

    Returns a dict with keys:
      - "object": dict mapping term -> statistics (coef, bse_cluster, z, p, CI, OR, OR_CI)
      - "description": brief explanation of the contents and how to interpret them

    The function uses the cluster-robust results object if present
    (model_output['cluster_robust_results']), otherwise falls back to the GLM results.
    """
    import numpy as np
    import pandas as pd
    try:
        from scipy import stats as _stats
    except Exception:
        _stats = None

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")

    # Prefer cluster-robust results if provided
    results = model_output.get('cluster_robust_results') or model_output.get('glm_results')
    if results is None:
        raise ValueError("model_output must contain 'cluster_robust_results' or 'glm_results'.")

    # Extract parameter vector
    params = getattr(results, 'params', None)
    if params is None:
        # Try to obtain from glm_results if wrapper doesn't expose it
        glm = model_output.get('glm_results')
        if glm is None or not hasattr(glm, 'params'):
            raise ValueError("Could not find parameter estimates in model_output.")
        params = glm.params

    # Extract covariance matrix (clustered if available)
    try:
        cov = results.cov_params()
    except Exception:
        # fallback: try glm_results.cov_params()
        glm = model_output.get('glm_results')
        if glm is None:
            raise
        cov = glm.cov_params()

    # Normalize cov to numpy array and align with params index
    if hasattr(cov, "values"):
        cov_mat = np.asarray(cov.values)
    else:
        cov_mat = np.asarray(cov)

    # Ensure params is a pandas Series for index access
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            raise ValueError("Could not coerce params to pandas Series.")

    # Compute cluster-robust standard errors
    try:
        # If the wrapper exposes bse, use it (it should already use cluster SEs)
        bse = getattr(results, 'bse', None)
        if bse is None:
            raise AttributeError
        # bse might be a numpy array or Series; coerce to Series aligned with params
        if not isinstance(bse, pd.Series):
            bse = pd.Series(np.asarray(bse), index=params.index)
    except Exception:
        # Compute from covariance matrix diagonal
        if cov_mat.shape[0] != cov_mat.shape[1]:
            raise ValueError("Covariance matrix has unexpected shape.")
        se_diag = np.sqrt(np.diag(cov_mat))
        bse = pd.Series(se_diag, index=params.index)

    # Compute z-statistics and p-values using normal approximation
    z_stats = params / bse
    if _stats is not None:
        p_values = 2.0 * (1.0 - _stats.norm.cdf(np.abs(z_stats)))
    else:
        # approximate normal CDF using error function if scipy unavailable
        from math import erf, sqrt
        def _norm_cdf(x):
            return 0.5 * (1.0 + erf(x / sqrt(2.0)))
        p_values = 2.0 * (1.0 - np.array([_norm_cdf(abs(v)) for v in z_stats]))

    # 95% CI
    z_crit = 1.959963984540054  # approx 1.96 for 95% CI
    ci_lower = params - z_crit * bse
    ci_upper = params + z_crit * bse

    # Odds ratios and their CIs
    or_vals = np.exp(params)
    or_ci_lower = np.exp(ci_lower)
    or_ci_upper = np.exp(ci_upper)

    # Terms of interest
    candidate_terms = [
        'SizeRatio_z',
        'LocationAdvantage_z',
        'SizeRatio_z:LocationAdvantage_z',  # typical statsmodels interaction name
        'SizeRatio_z*LocationAdvantage_z'   # alternate possible name (rare)
    ]

    # Build output for whichever of the above terms are present
    out = {}
    for term in candidate_terms:
        if term in params.index:
            out_term = {
                'coef': float(params.loc[term]),
                'bse_cluster': float(bse.loc[term]),
                'z': float(z_stats.loc[term]),
                'p_value': float(p_values[params.index.get_loc(term)]),
                'ci_95_lower': float(ci_lower.loc[term]),
                'ci_95_upper': float(ci_upper.loc[term]),
                'odds_ratio': float(or_vals.loc[term]),
                'or_ci_95_lower': float(or_ci_lower.loc[term]),
                'or_ci_95_upper': float(or_ci_upper.loc[term])
            }
            out[term] = out_term

    # If interaction was named with ':' but not found, try the other representation
    # (handle case where only one of the two candidate interaction names exists)
    if 'SizeRatio_z:LocationAdvantage_z' not in out and 'SizeRatio_z*LocationAdvantage_z' in out:
        # rename the '*' entry to the ':' canonical name for clarity
        out['SizeRatio_z:LocationAdvantage_z'] = out.pop('SizeRatio_z*LocationAdvantage_z')

    # Also include the full parameter table for reference (safe conversion)
    full_table = pd.DataFrame({
        'coef': params,
        'bse_cluster': bse,
        'z': z_stats,
        'p_value': p_values,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'odds_ratio': or_vals,
        'or_ci_95_lower': or_ci_lower,
        'or_ci_95_upper': or_ci_upper
    })

    # Coerce full_table to plain Python types (for JSON-friendly output if needed)
    full_table_dict = full_table.fillna(np.nan).to_dict(orient='index')

    description_lines = [
        "Returned object contains cluster-robust coefficient estimates and statistics",
        "for the predictors of interest (SizeRatio_z, LocationAdvantage_z, and their interaction).",
        "Interpretation:",
        "- 'coef' is the log-odds effect: positive => increases probability of focal group winning.",
        "- 'odds_ratio' = exp(coef): >1 means higher odds of winning per 1 SD increase in the predictor.",
        "- 'bse_cluster' are cluster-robust standard errors (clustered by dyad).",
        "- 'p_value' is computed using the normal (Wald) approximation with cluster SEs; p < 0.05 is a common threshold for statistical significance.",
        "- 'ci_95_lower' / 'ci_95_upper' are the 95% confidence limits for the log-odds; exponentiate to get OR CIs (provided).",
        "",
        "If the interaction term is statistically significant, the effect of relative group size depends on contest location (and vice versa).",
        "Use the provided coefficients and their interaction to compute conditional effects (e.g., effect of SizeRatio_z at specific values of LocationAdvantage_z).",
        "",
        "The 'object' key returns the extracted per-term statistics; 'full_table' contains the full parameter table."
    ]
    description = "\n".join(description_lines)

    return {
        "object": {
            "terms": out,
            "full_table": full_table_dict,
            "n_obs": int(model_output.get('n_obs')) if model_output.get('n_obs') is not None else None,
            "formula": model_output.get('formula')
        },
        "description": description
    }