def extract_final_answer(model_output):
    """
    Extract key coefficients, standard errors, test statistics, p-values, 95% CIs,
    and odds ratios for the predictors related to relative group size and contest
    location from the modeling output produced by the provided `model()` function.

    Returns a dictionary with:
      - "object": dict mapping predictor names to their extracted statistics
      - "description": short interpretation of what the numbers mean in context
    """
    import numpy as np
    import pandas as pd

    # Safety checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")
    if 'results' not in model_output:
        raise ValueError("model_output does not contain 'results' key.")

    results = model_output['results']
    # Expected attributes on the ClusterResultsWrapper: params, bse, tvalues, pvalues
    params = getattr(results, 'params', None)
    bse = getattr(results, 'bse', None)
    tvalues = getattr(results, 'tvalues', None)
    pvalues = getattr(results, 'pvalues', None)

    # Variables of interest (as used in the model formula)
    predictors = [
        'relsize_z',
        'focal_adv_location',
        'relsize_x_loc',     # interaction relsize * focal_adv_location
        'distance_diff_z',
        'male_diff_z',
        'female_diff_z'
    ]

    stats = {}
    for var in predictors:
        # Initialize with NA defaults
        coef = np.nan
        se = np.nan
        t = np.nan
        p = np.nan
        ci_lower = np.nan
        ci_upper = np.nan
        odds_ratio = np.nan
        or_ci_lower = np.nan
        or_ci_upper = np.nan

        try:
            if params is not None and var in params.index:
                coef = float(params.loc[var])
        except Exception:
            coef = np.nan

        try:
            if bse is not None and var in bse.index:
                se = float(bse.loc[var])
        except Exception:
            se = np.nan

        try:
            if tvalues is not None and var in tvalues.index:
                t = float(tvalues.loc[var])
        except Exception:
            t = np.nan

        try:
            if pvalues is not None and var in pvalues.index:
                p = float(pvalues.loc[var])
        except Exception:
            p = np.nan

        # 95% CI using normal approximation if se and coef available and finite
        if np.isfinite(coef) and np.isfinite(se):
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
            # Odds ratio and CI on OR scale
            try:
                odds_ratio = float(np.exp(coef))
                or_ci_lower = float(np.exp(ci_lower))
                or_ci_upper = float(np.exp(ci_upper))
            except Exception:
                odds_ratio = np.nan
                or_ci_lower = np.nan
                or_ci_upper = np.nan

        stats[var] = {
            'coef_log_odds': coef,
            'std_err': se,
            'z_or_t': t,
            'p_value': p,
            'ci_95_log_odds': (ci_lower, ci_upper),
            'odds_ratio': odds_ratio,
            'ci_95_odds_ratio': (or_ci_lower, or_ci_upper),
            # convenience flag
            'significant_p_lt_0.05': bool(np.isfinite(p) and (p < 0.05))
        }

    # Build a short interpretation
    # Determine whether any of the relsize/location terms are statistically significant
    sig_terms = [v for v, s in stats.items() if s['significant_p_lt_0.05']]
    if len(sig_terms) == 0:
        interpretation = (
            "No strong evidence that relative group size (relsize_z), contest-location advantage "
            "(focal_adv_location or distance_diff_z), or their interaction (relsize_x_loc) "
            "predict the focal group's probability of winning in this fitted model: "
            "none of the tested predictors have p < 0.05. "
        )
    else:
        interpretation = (
            "The following predictor(s) showed p < 0.05 in the clustered estimates: "
            + ", ".join(sig_terms)
            + ". Interpret their effect sizes (log-odds and odds ratios) below."
        )

    # Warn if there are NaNs in key statistics (indicating estimation problems)
    nan_warnings = []
    for var, s in stats.items():
        if not np.isfinite(s['std_err']) or not np.isfinite(s['p_value']):
            nan_warnings.append(var)
    if nan_warnings:
        interpretation += (
            " Note: some parameters ({} ) have missing or non-finite standard errors or p-values, "
            "which indicates numerical/identification problems (e.g., separation, perfect collinearity, "
            "or issues with cluster-robust covariance estimation). Treat inference for those terms as unreliable."
        ).format(", ".join(nan_warnings))

    # Also include the textual summary if available for reference
    summary_text = model_output.get('summary_text', None)

    return {
        "object": {
            "predictor_stats": stats,
            "summary_text": summary_text,
            # include the predictions df if present (useful for downstream checks)
            "data_with_predictions": model_output.get('data_with_predictions', None)
        },
        "description": interpretation
    }