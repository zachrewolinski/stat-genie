def extract_final_answer(model_output):
    """
    Extracts coefficient table and tests the effect of relative group size (RelSize_z)
    across contest locations (Neutral, FocalHome, OtherHome) from a fitted statsmodels
    Logit BinaryResultsWrapper.

    Returns a dict with keys:
      - "object": dict containing:
          - "coef_table": pandas.DataFrame with coefficients, SE, z, p, 95% CI, odds ratios and OR CI
            for each model term.
          - "simple_slopes": pandas.DataFrame with the slope of RelSize_z (log-odds per SD)
            evaluated at each Location (Neutral, FocalHome, OtherHome), including SE, z, p,
            95% CI and odds ratio (and OR CI).
      - "description": short explanation of the results and how to interpret them.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output  # statsmodels BinaryResultsWrapper

    # Extract basic coefficient table
    params = res.params.copy()
    bse = res.bse.copy()
    zvals = params / bse
    # Prefer using res.pvalues if available for individual coefficients
    try:
        pvals = res.pvalues.copy()
    except Exception:
        pvals = 2 * (1 - stats.norm.cdf(np.abs(zvals)))
    ci = res.conf_int(alpha=0.05)
    ci.columns = ['ci_lower', 'ci_upper']

    coef_table = pd.DataFrame({
        'coef': params,
        'se': bse,
        'z': zvals,
        'p': pvals,
        'ci_lower': ci['ci_lower'],
        'ci_upper': ci['ci_upper']
    })
    # Add odds ratios and OR CIs
    coef_table['odds_ratio'] = np.exp(coef_table['coef'])
    coef_table['or_ci_lower'] = np.exp(coef_table['ci_lower'])
    coef_table['or_ci_upper'] = np.exp(coef_table['ci_upper'])

    # Prepare covariance matrix for tests of linear combinations (simple slopes)
    cov = res.cov_params()

    # Names used in the model code
    term_R = 'RelSize_z'
    term_R_Focal_inter = 'RelSize_z_x_Loc_FocalHome'
    term_R_Other_inter = 'RelSize_z_x_Loc_OtherHome'

    # Function to compute linear combination stats
    def lincomb_stats(coef_vector):
        """
        coef_vector: pandas Series or 1D array aligned with params.index,
                     containing weights for linear combination.
        Returns dict with coef, se, z, p, ci_lower, ci_upper, odds_ratio, or_ci_lower, or_ci_upper
        """
        # Align with params index
        a = np.zeros(len(params))
        index = list(params.index)
        for i, name in enumerate(index):
            a[i] = float(coef_vector.get(name, 0.0))
        est = float(np.dot(a, params.values))
        se = float(np.sqrt(np.dot(a, np.dot(cov.values, a))))
        # If se is zero (degenerate), set z/p to nan
        if se > 0:
            z = est / se
            p = 2 * (1 - stats.norm.cdf(abs(z)))
            ci_low = est - 1.96 * se
            ci_high = est + 1.96 * se
        else:
            z = np.nan
            p = np.nan
            ci_low = np.nan
            ci_high = np.nan
        or_est = np.exp(est) if np.isfinite(est) else np.nan
        or_ci_low = np.exp(ci_low) if np.isfinite(ci_low) else np.nan
        or_ci_high = np.exp(ci_high) if np.isfinite(ci_high) else np.nan

        return {
            'coef': est,
            'se': se,
            'z': z,
            'p': p,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'odds_ratio': or_est,
            'or_ci_lower': or_ci_low,
            'or_ci_upper': or_ci_high
        }

    # Simple slopes for RelSize_z at each location:
    # Neutral: slope = RelSize_z
    vec_neutral = {term_R: 1.0}
    neutral_stats = lincomb_stats(vec_neutral)

    # FocalHome: slope = RelSize_z + RelSize_z_x_Loc_FocalHome
    vec_focal = {term_R: 1.0, term_R_Focal_inter: 1.0}
    focal_stats = lincomb_stats(vec_focal)

    # OtherHome: slope = RelSize_z + RelSize_z_x_Loc_OtherHome
    vec_other = {term_R: 1.0, term_R_Other_inter: 1.0}
    other_stats = lincomb_stats(vec_other)

    simple_slopes = pd.DataFrame({
        'Neutral': neutral_stats,
        'FocalHome': focal_stats,
        'OtherHome': other_stats
    }).T[['coef', 'se', 'z', 'p', 'ci_lower', 'ci_upper', 'odds_ratio', 'or_ci_lower', 'or_ci_upper']]

    # Build description explaining interpretation
    description = (
        "This output gives (1) the model coefficient table for each model term, including\n"
        "    log-odds coefficients (coef), standard errors, z-statistics, p-values,\n"
        "    95% confidence intervals, and corresponding odds ratios (and their CIs);\n"
        "and (2) the simple slopes of RelSize_z (the effect of being relatively larger by 1 SD)\n"
        "    evaluated at each contest location (Neutral, FocalHome, OtherHome).\n\n"
        "Interpretation notes:\n"
        "- The coef for RelSize_z in the 'coef_table' is the effect of relative group size\n"
        "  when Location == 'Neutral' (the reference). A positive coefficient means that\n"
        "  as the focal group is relatively larger (per SD), its log-odds of winning increase;\n"
        "  exp(coef) gives the multiplicative change in odds of winning per SD increase in RelSize_z.\n"
        "- The interaction coefficients (RelSize_z_x_Loc_FocalHome and RelSize_z_x_Loc_OtherHome)\n"
        "  modify the RelSize_z slope in those locations. The 'simple_slopes' table reports the\n"
        "  combined slope (RelSize_z + interaction) for FocalHome and OtherHome, with SE, p-value,\n"
        "  and odds ratios. Use these to see whether the effect of relative size is stronger,\n"
        "  weaker, or reversed in different locations.\n"
        "- Coefficients for Loc_FocalHome and Loc_OtherHome (in coef_table) represent the effect\n"
        "  of being at that location (versus Neutral) when RelSize_z == 0 (an average/centered value).\n\n"
        "To answer the research question, inspect the sign, magnitude, and statistical significance\n"
        "of the simple slopes and their odds ratios: if the slope is positive and significant,\n"
        "being relatively larger increases win probability at that location; if the slope differs\n"
        "across locations, the interaction is present (check interaction p-values in coef_table).\n"
    )

    return {
        "object": {
            "coef_table": coef_table,
            "simple_slopes": simple_slopes
        },
        "description": description
    }