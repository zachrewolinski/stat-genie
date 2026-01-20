def extract_final_answer(model_output):
    """
    Extracts and summarizes the effect of relative group size (SizeDiff_z) on the
    probability that the focal group wins, separately for each contest location,
    including standard errors, z-statistics, p-values, 95% CIs, and odds ratios.
    Also extracts p-values for interaction terms and runs a joint test of both
    interaction coefficients = 0.

    Returns a dict with:
      - "object": a nested dict containing numeric results
      - "description": a short interpretation of the results in the task context
    """
    import numpy as np
    from scipy import stats
    import pandas as pd

    res = model_output  # statsmodels GLMResultsWrapper

    # Required coefficient names in the fitted model (as used in the provided code)
    coef_names = list(res.params.index)

    # Helper to compute linear combination estimate, se, z, p, CI, OR
    def linear_combination(weights):
        """
        weights: dict mapping coefficient name -> multiplier
        returns dict with beta, se, z, p, CI_beta, OR, CI_OR
        """
        # create weights series aligned with params
        w = pd.Series(0.0, index=coef_names)
        for k, v in weights.items():
            if k not in w.index:
                # If a coef is missing, treat its weight as 0 (coefficient effectively 0)
                w[k] = 0.0
            else:
                w[k] = v
        beta = float((w * res.params).sum())
        cov = res.cov_params()
        var = float(w.values @ cov.values @ w.values)  # w' * Cov * w
        se = float(np.sqrt(var)) if var >= 0 else float(np.nan)
        z = float(beta / se) if se and not np.isnan(se) else float('nan')
        p = float(2 * (1 - stats.norm.cdf(abs(z)))) if not np.isnan(z) else float('nan')
        ci_low = beta - 1.96 * se
        ci_high = beta + 1.96 * se
        or_est = float(np.exp(beta))
        or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
        return {
            'beta': beta,
            'se': se,
            'z': z,
            'p_value': p,
            'CI95_beta': (ci_low, ci_high),
            'odds_ratio': or_est,
            'CI95_odds_ratio': or_ci
        }

    # Effects of SizeDiff_z by contest location:
    # Reference (FocalHome) occurs when both dummies are 0.
    # Effect when FocalHome: coefficient on SizeDiff_z
    # Effect when OtherHome: SizeDiff_z + Size_x_OtherHome
    # Effect when Neutral: SizeDiff_z + Size_x_Neutral

    effects = {}

    # FocalHome
    effects['FocalHome'] = linear_combination({'SizeDiff_z': 1.0})

    # OtherHome
    effects['OtherHome'] = linear_combination({
        'SizeDiff_z': 1.0,
        'Size_x_OtherHome': 1.0
    })

    # Neutral
    effects['Neutral'] = linear_combination({
        'SizeDiff_z': 1.0,
        'Size_x_Neutral': 1.0
    })

    # Extract raw coefficients and p-values for key terms
    def safe_get(series, name):
        return float(series[name]) if name in series.index else None

    raw = {
        'params': res.params.to_dict(),
        'pvalues': res.pvalues.to_dict(),
        'bse': res.bse.to_dict()
    }

    # Interaction p-values
    p_inter_other = raw['pvalues'].get('Size_x_OtherHome', None)
    p_inter_neutral = raw['pvalues'].get('Size_x_Neutral', None)

    # Joint test: both interaction coefficients = 0
    # Build restriction matrix R for [Size_x_OtherHome, Size_x_Neutral]
    try:
        R = np.zeros((2, len(coef_names)))
        idx_map = {name: i for i, name in enumerate(coef_names)}
        if 'Size_x_OtherHome' in idx_map:
            R[0, idx_map['Size_x_OtherHome']] = 1.0
        if 'Size_x_Neutral' in idx_map:
            R[1, idx_map['Size_x_Neutral']] = 1.0
        wtest = res.wald_test(R)
        p_joint = float(wtest.pvalue) if hasattr(wtest, 'pvalue') else None
    except Exception:
        p_joint = None

    result_object = {
        'effects_by_location': effects,
        'raw_coefficients_and_tests': raw,
        'interaction_pvalues': {
            'Size_x_OtherHome': p_inter_other,
            'Size_x_Neutral': p_inter_neutral,
            'joint_interaction_pvalue': p_joint
        }
    }

    # Short interpretation string
    description = (
        "This output gives the estimated effect (log-odds coefficient) of relative group size "
        "(SizeDiff_z) on the probability that the focal group wins, computed separately for contests "
        "near the focal group's home (FocalHome, the reference), near the other group's home (OtherHome), "
        "and at neutral locations (Neutral). For each location you get the coefficient (beta), its SE, z-stat, "
        "two-sided p-value, a 95% CI on the beta scale, and the corresponding odds ratio with 95% CI. "
        "Interaction term p-values are provided for the two location interaction coefficients, and a joint "
        "Wald test p-value tests whether both interaction effects are simultaneously zero (i.e., whether the "
        "effect of relative group size differs by location). A positive beta (or OR>1) means that as the focal "
        "group becomes larger relative to the other group, the probability that the focal group wins increases. "
        "Interpret significance using the p-values: small p (e.g., <0.05) indicates evidence of a non-zero effect."
    )

    return {"object": result_object, "description": description}