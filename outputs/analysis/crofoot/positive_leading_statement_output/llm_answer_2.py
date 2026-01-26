def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, odds ratios,
    and tests the effect of relative group size (SizeDiff_z) overall and conditional on
    contest being at the focal group's home (AtHome = 1) using the provided GEE results object.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coefficients, se, z, p, 95% CI, OR, OR_CI,
                  slope of SizeDiff when AtHome=0 and AtHome=1, their SEs and p-values)
      - "description": plain-language interpretation of those statistics in context
    """
    import numpy as np
    from scipy import stats

    res = model_output  # expected to be a statsmodels ResultsWrapper (GEE results)

    # Extract base quantities
    params = res.params.copy()        # pandas Series
    bse = res.bse.copy()              # standard errors
    pvalues = res.pvalues.copy()
    try:
        conf = res.conf_int()         # DataFrame with lower/upper bounds
    except Exception:
        # Some result objects use different method name
        conf = res.conf_int()
    cov = res.cov_params()

    # Helper to find parameter names robustly
    idx = list(params.index)

    def find_exact_or_contains(key):
        # Prefer exact match, otherwise first name that contains the key (but not interaction)
        if key in idx:
            return key
        for name in idx:
            if key in name and ':' not in name:
                return name
        # fallback: any name containing key
        for name in idx:
            if key in name:
                return name
        return None

    size_name = find_exact_or_contains('SizeDiff_z')
    at_name = find_exact_or_contains('AtHome')
    # Find interaction name containing both tokens
    interaction_name = None
    for name in idx:
        if ('SizeDiff_z' in name) and ('AtHome' in name):
            interaction_name = name
            break

    # Collect term-level results (if present)
    def term_dict(name):
        if name is None:
            return None
        return {
            'name': name,
            'coef': float(params[name]),
            'se': float(bse[name]),
            'z': float(params[name] / bse[name]) if bse[name] != 0 else None,
            'p': float(pvalues[name]) if name in pvalues.index else None,
            'ci95_low': float(conf.loc[name, 0]) if name in conf.index else None,
            'ci95_high': float(conf.loc[name, 1]) if name in conf.index else None,
            'odds_ratio': float(np.exp(params[name])),
            'or_ci95_low': float(np.exp(conf.loc[name, 0])) if name in conf.index else None,
            'or_ci95_high': float(np.exp(conf.loc[name, 1])) if name in conf.index else None,
        }

    size_res = term_dict(size_name)
    at_res = term_dict(at_name)
    inter_res = term_dict(interaction_name)

    # Compute conditional slopes for SizeDiff_z:
    # - When AtHome = 0: slope = coef(SizeDiff_z)
    # - When AtHome = 1: slope = coef(SizeDiff_z) + coef(interaction)
    slope_at0 = None
    slope_at0_se = None
    slope_at0_z = None
    slope_at0_p = None
    slope_at1 = None
    slope_at1_se = None
    slope_at1_z = None
    slope_at1_p = None

    if size_name is not None:
        slope_at0 = float(params[size_name])
        slope_at0_se = float(bse[size_name])
        slope_at0_z = slope_at0 / slope_at0_se if slope_at0_se != 0 else None
        slope_at0_p = float(2 * (1 - stats.norm.cdf(abs(slope_at0_z)))) if slope_at0_z is not None else None

    if size_name is not None and interaction_name is not None:
        # slope at AtHome=1
        slope_at1 = float(params[size_name] + params[interaction_name])
        # var(sum) = var(size) + var(inter) + 2*cov(size,inter)
        var_sum = cov.loc[size_name, size_name] + cov.loc[interaction_name, interaction_name] + 2 * cov.loc[size_name, interaction_name]
        slope_at1_se = float(np.sqrt(var_sum)) if var_sum >= 0 else None
        if slope_at1_se is not None and slope_at1_se != 0:
            slope_at1_z = slope_at1 / slope_at1_se
            slope_at1_p = float(2 * (1 - stats.norm.cdf(abs(slope_at1_z))))
    elif size_name is not None and interaction_name is None:
        # No interaction term present; slope at AtHome=1 equals slope_at0
        slope_at1 = slope_at0
        slope_at1_se = slope_at0_se
        slope_at1_z = slope_at0_z
        slope_at1_p = slope_at0_p

    # Build the returned object
    out_object = {
        'terms': {
            'SizeDiff': size_res,
            'AtHome': at_res,
            'SizeDiff:AtHome': inter_res
        },
        'slope_SizeDiff': {
            'AtHome=0': {
                'slope_logodds': slope_at0,
                'se': slope_at0_se,
                'z': slope_at0_z,
                'p': slope_at0_p,
                'odds_ratio_per_sd': float(np.exp(slope_at0)) if slope_at0 is not None else None,
                'or_ci95_low': float(np.exp(slope_at0 - 1.96 * slope_at0_se)) if slope_at0 is not None and slope_at0_se is not None else None,
                'or_ci95_high': float(np.exp(slope_at0 + 1.96 * slope_at0_se)) if slope_at0 is not None and slope_at0_se is not None else None,
            },
            'AtHome=1': {
                'slope_logodds': slope_at1,
                'se': slope_at1_se,
                'z': slope_at1_z,
                'p': slope_at1_p,
                'odds_ratio_per_sd': float(np.exp(slope_at1)) if slope_at1 is not None else None,
                'or_ci95_low': float(np.exp(slope_at1 - 1.96 * slope_at1_se)) if slope_at1 is not None and slope_at1_se is not None else None,
                'or_ci95_high': float(np.exp(slope_at1 + 1.96 * slope_at1_se)) if slope_at1 is not None and slope_at1_se is not None else None,
            }
        },
        'notes': {
            'param_names_found': list(params.index),
            'size_param_name': size_name,
            'at_param_name': at_name,
            'interaction_param_name': interaction_name
        }
    }

    # Short, plain-language interpretation
    # We explain what a positive slope means and whether it is statistically significant
    if slope_at0 is None:
        description = ("Could not find a parameter for SizeDiff_z in the model output; "
                       "no inference about the effect of relative group size can be made.")
    else:
        def sig_str(p):
            if p is None:
                return "p-value unavailable"
            elif p < 0.001:
                return "p < 0.001"
            else:
                return f"p = {p:.3f}"

        s0_sig = sig_str(slope_at0_p)
        s1_sig = sig_str(slope_at1_p)

        desc_lines = []
        desc_lines.append(f"Effect of relative group size (SizeDiff_z) when contest is not at the focal group's home (AtHome=0): slope on log-odds = {slope_at0:.3f}, SE = {slope_at0_se:.3f}, {s0_sig}.")
        desc_lines.append(f"This corresponds to an odds ratio per 1 SD increase in SizeDiff of {np.exp(slope_at0):.3f} (approx. 95% CI: [{np.exp(slope_at0 - 1.96 * slope_at0_se):.3f}, {np.exp(slope_at0 + 1.96 * slope_at0_se):.3f}]).")

        if interaction_name is not None:
            desc_lines.append(f"When the contest occurs closer to the focal group's home (AtHome=1), the slope = {slope_at1:.3f}, SE = {slope_at1_se:.3f}, {s1_sig}.")
            desc_lines.append(f"Odds ratio per 1 SD SizeDiff at AtHome=1 = {np.exp(slope_at1):.3f} (approx. 95% CI: [{np.exp(slope_at1 - 1.96 * slope_at1_se):.3f}, {np.exp(slope_at1 + 1.96 * slope_at1_se):.3f}]).")
            desc_lines.append("A statistically significant positive slope indicates that larger focal groups are more likely to win; a significant interaction indicates that this size advantage differs depending on whether the contest is nearer the focal group's home.")
        else:
            desc_lines.append("No SizeDiff_z:AtHome interaction term was found in the model, so the effect of SizeDiff_z is the same regardless of AtHome in this fitted model.")

        description = " ".join(desc_lines)

    return {'object': out_object, 'description': description}