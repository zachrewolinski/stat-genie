def extract_final_answer(model_output):
    """
    Extracts and summarizes the effects of relative group size (SizeAdv_z),
    relative contest location (DistDiff_z), and their interaction on the
    probability that the focal group wins (logistic regression), using
    cluster-robust covariance supplied in model_output.

    Returns a dictionary with:
      - "object": a dict containing numeric summaries (coefficients, clustered SE,
                  z, p, odds ratios, 95% CI) for:
                    * SizeAdv_z effect at DistDiff_z = -1, 0, +1 (approx. other-home, neutral, focal-home)
                    * DistDiff_z effect at SizeAdv_z = -1, 0, +1
                    * raw coefficients, clustered SEs, z and p from clustered_summary
      - "description": a short interpretation of what the numbers mean
    """
    import numpy as np
    import scipy.stats

    # Unpack
    glm_res = model_output.get('glm_results')
    clustered_cov = model_output.get('clustered_cov')
    clustered_summary = model_output.get('clustered_summary')

    if glm_res is None or clustered_cov is None or clustered_summary is None:
        raise ValueError("model_output must contain 'glm_results', 'clustered_cov', and 'clustered_summary'")

    # Parameter names and values
    params = glm_res.params.copy()
    param_names = list(params.index)

    # Helper to find parameter index robustly
    def idx(name):
        if name in param_names:
            return param_names.index(name)
        # try common alternative interaction naming with reversed order
        alt = ':'.join(name.split(':')[::-1])
        if alt in param_names:
            return param_names.index(alt)
        raise KeyError(f"Parameter '{name}' not found in model parameters: {param_names}")

    # Required parameter names (as in the provided output)
    name_size = 'SizeAdv_z'
    name_dist = 'DistDiff_z'
    name_inter = 'SizeAdv_z:DistDiff_z'  # as in the clustered_summary provided

    # Ensure parameters exist
    for nm in (name_size, name_dist, name_inter):
        alt_nm = ':'.join(nm.split(':')[::-1])
        if nm not in param_names and alt_nm not in param_names:
            raise KeyError(f"Expected parameter '{nm}' not found. Available: {param_names}")

    i_size = idx(name_size)
    i_dist = idx(name_dist)
    i_inter = idx(name_inter)

    # Function to compute linear combination stats using clustered covariance
    def lincomb_stats(coef_vector):
        """
        coef_vector: 1D array of same length as params, containing weights for linear combination.
        Returns: dict with 'log_odds', 'se', 'z', 'p', 'odds_ratio', 'ci95_or' (tuple)
        """
        coef_vector = np.asarray(coef_vector).reshape(-1)
        est = float(np.dot(coef_vector, params.values))
        # Ensure covariance is a numpy array for correct matrix multiplication
        cov = np.asarray(clustered_cov)
        try:
            var = float(coef_vector @ cov @ coef_vector)
        except Exception:
            # fallback: use dot for compatibility
            var = float(np.dot(coef_vector, np.dot(cov, coef_vector)))
        se = np.sqrt(var) if var >= 0 else np.nan
        z = est / se if (se is not None and se != 0 and not np.isnan(se)) else np.nan
        p = 2 * (1 - scipy.stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        or_ = np.exp(est)
        ci_low = np.exp(est - 1.96 * se) if not np.isnan(se) else np.nan
        ci_high = np.exp(est + 1.96 * se) if not np.isnan(se) else np.nan
        return {
            'log_odds': est,
            'se': se,
            'z': z,
            'p': p,
            'odds_ratio': or_,
            'ci95_or': (ci_low, ci_high)
        }

    n_params = len(params)

    # Helper to build consistent keys (no trailing .0)
    def key_for(name, val):
        try:
            ival = int(val)
            return f"{name}={ival}"
        except Exception:
            # fallback to general formatting
            return f"{name}={val}"

    # Compute effect of SizeAdv_z (marginal effect) at DistDiff_z = -1, 0, +1
    size_effects = {}
    for d in (-1.0, 0.0, 1.0):
        vec = np.zeros(n_params)
        vec[i_size] = 1.0
        vec[i_inter] = d
        size_effects[key_for('DistDiff', d)] = lincomb_stats(vec)

    # Compute effect of DistDiff_z at SizeAdv_z = -1, 0, +1
    dist_effects = {}
    for s in (-1.0, 0.0, 1.0):
        vec = np.zeros(n_params)
        vec[i_dist] = 1.0
        vec[i_inter] = s
        dist_effects[key_for('SizeAdv', s)] = lincomb_stats(vec)

    # Also return raw clustered summary (for reference)
    # Convert clustered_summary to a plain dict of dicts
    raw_summary = {}
    try:
        _ = clustered_summary.iterrows
    except Exception:
        # If clustered_summary is already a dict-like structure
        for k, v in clustered_summary.items():
            raw_summary[k] = {
                'coef': float(v.get('coef', np.nan)),
                'se_cluster': float(v.get('se_cluster', np.nan)),
                'z_cluster': float(v.get('z_cluster', np.nan)),
                'p_cluster': float(v.get('p_cluster', np.nan))
            }
    else:
        for rowname, row in clustered_summary.iterrows():
            # row may be a Series or dict-like
            get = getattr(row, 'get', None)
            if callable(get):
                coef = row.get('coef', np.nan)
                se_cl = row.get('se_cluster', np.nan)
                z_cl = row.get('z_cluster', np.nan)
                p_cl = row.get('p_cluster', np.nan)
            else:
                # for pandas Series, use direct indexing with fallback
                coef = row['coef'] if 'coef' in row.index else np.nan
                se_cl = row['se_cluster'] if 'se_cluster' in row.index else np.nan
                z_cl = row['z_cluster'] if 'z_cluster' in row.index else np.nan
                p_cl = row['p_cluster'] if 'p_cluster' in row.index else np.nan
            raw_summary[rowname] = {
                'coef': float(coef) if not (coef is None) else np.nan,
                'se_cluster': float(se_cl) if not (se_cl is None) else np.nan,
                'z_cluster': float(z_cl) if not (z_cl is None) else np.nan,
                'p_cluster': float(p_cl) if not (p_cl is None) else np.nan
            }

    # Short interpretation: check significance for key effects (SizeAdv at neutral and DistDiff at neutral, and interaction)
    size_neutral = size_effects.get(key_for('DistDiff', 0))
    dist_neutral = dist_effects.get(key_for('SizeAdv', 0))

    # Robustly retrieve interaction coefficient from params
    interaction_coef = None
    if name_inter in params.index:
        interaction_coef = float(params[name_inter])
    else:
        alt_inter = ':'.join(name_inter.split(':')[::-1])
        if alt_inter in params.index:
            interaction_coef = float(params[alt_inter])
        else:
            # try .get with possible None
            val = params.get(name_inter)
            if val is None:
                val = params.get(alt_inter)
            interaction_coef = float(val) if val is not None else np.nan

    # get se for interaction directly from clustered_summary if present
    interaction_se = raw_summary.get(name_inter, {}).get('se_cluster', None)
    if interaction_se is None:
        # try reversed interaction name
        alt_inter = ':'.join(name_inter.split(':')[::-1])
        interaction_se = raw_summary.get(alt_inter, {}).get('se_cluster', None)

    # Determine significance (alpha = 0.05)
    def sig(p):
        return (p is not None) and (not np.isnan(p)) and (p < 0.05)

    interaction_significant = False
    if (interaction_se is not None) and (not np.isnan(interaction_se)) and (interaction_se > 0):
        z_inter = interaction_coef / interaction_se if not np.isnan(interaction_coef) else np.nan
        p_inter = 2 * (1 - scipy.stats.norm.cdf(abs(z_inter))) if not np.isnan(z_inter) else np.nan
        interaction_significant = (p_inter is not None) and (not np.isnan(p_inter)) and (p_inter < 0.05)

    conclusions = {
        'SizeAdv_at_neutral_significant': sig(size_neutral['p']) if size_neutral is not None else False,
        'DistDiff_at_neutral_significant': sig(dist_neutral['p']) if dist_neutral is not None else False,
        'Interaction_significant': interaction_significant
    }

    result_object = {
        'raw_clustered_summary': raw_summary,
        'size_effects_by_location': size_effects,
        'dist_effects_by_size': dist_effects,
        'conclusions_flags': conclusions
    }

    # Human-readable short description
    # Based on p-values in clustered_summary and the computed marginal effects, provide concise interpretation.
    if (not conclusions['SizeAdv_at_neutral_significant'] and
        not conclusions['DistDiff_at_neutral_significant'] and
        not conclusions['Interaction_significant']):
        short_desc = (
            "Model estimates indicate no statistically significant effect (cluster-robust SEs) of relative group size "
            "(SizeAdv_z), relative contest location (DistDiff_z), or their interaction on the probability that the focal "
            "group wins. Point estimates (log-odds) and odds ratios are returned in 'object'. Interpret cautiously: "
            "the sign of the SizeAdv_z coefficient is negative in the fitted model (larger focal group associated with "
            "lower odds of winning in this sample), but it is not statistically different from zero once clustering by dyad "
            "is accounted for."
        )
    else:
        short_desc = (
            "Model suggests some statistically significant effects (cluster-robust SEs) among SizeAdv_z, DistDiff_z, "
            "or their interaction. See the numeric summaries in 'object' for direction, magnitude (odds ratios), and CIs."
        )

    return {
        "object": result_object,
        "description": short_desc
    }