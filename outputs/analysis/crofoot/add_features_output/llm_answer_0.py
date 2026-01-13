def extract_final_answer(model_output):
    """
    Extract relevant statistics from the fitted model output to answer how
    relative group size (SizeDiff_c), location advantage (LocAdv_c), and
    their interaction (SizeDiff_c:LocAdv_c) influence the probability that the
    focal group wins.

    Returns a dictionary with:
      - "object": a dict containing coefficients, standard errors, z/stats,
                  p-values, 95% CIs, odds ratios and odds-ratio CIs, and a
                  boolean 'significant' flag for each focal term.
      - "description": a concise interpretive summary of the results in plain text.

    The function handles models where a cluster-robust result is provided under
    'cluster_robust_result' or falls back to 'glm_result'.
    """
    import numpy as np

    # Choose cluster-robust result if available; otherwise use glm result
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('cluster_robust_result') or model_output.get('glm_result')
    else:
        res = model_output

    if res is None:
        raise ValueError("No model result found in model_output (expected keys 'cluster_robust_result' or 'glm_result').")

    # Terms of interest (interaction name matches statsmodels convention for ':' )
    terms = ['SizeDiff_c', 'LocAdv_c', 'SizeDiff_c:LocAdv_c']

    # Extract parameter table: params, bse, pvalues, conf_int
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        conf = res.conf_int()  # returns DataFrame with two columns (lower, upper)
    except Exception as e:
        raise RuntimeError(f"Failed to extract stats from model result: {e}")

    # Build output stats for each term of interest
    stats = {}
    missing_terms = [t for t in terms if t not in params.index]
    if missing_terms:
        # If interaction or main terms are named differently, try alternative names
        # (no further alternatives expected given the formula used). Warn the user.
        raise KeyError(f"The following expected terms are missing from the model results: {missing_terms}. "
                       "Check model formula / parameter names in the fitted model.")

    alpha = 0.05
    for t in terms:
        coef = float(params.loc[t])
        se = float(bse.loc[t])
        p = float(pvalues.loc[t]) if (t in pvalues.index) else None
        # For GLM, statsmodels uses z-statistics; some result types expose tvalues instead.
        stat_name = 'z' if hasattr(res, 'zvalues') and t in getattr(res, 'zvalues', {}).index else 'stat'
        stat_value = None
        if hasattr(res, 'tvalues') and t in getattr(res, 'tvalues', {}).index:
            stat_value = float(res.tvalues.loc[t])
            stat_name = 't'
        elif hasattr(res, 'zvalues') and t in getattr(res, 'zvalues', {}).index:
            stat_value = float(res.zvalues.loc[t])
            stat_name = 'z'
        else:
            # fallback compute coef / se
            stat_value = float(coef / se) if se != 0 else None

        ci_lower, ci_upper = None, None
        try:
            row = conf.loc[t]
            # conf might be DataFrame with numeric column labels
            ci_lower, ci_upper = float(row.iloc[0]), float(row.iloc[1])
        except Exception:
            ci_lower, ci_upper = None, None

        # Odds ratio and CI
        or_val = float(np.exp(coef))
        or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

        significant = (p is not None) and (p < alpha)

        stats[t] = {
            'coefficient': coef,
            'se': se,
            stat_name: stat_value,
            'p_value': p,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'odds_ratio': or_val,
            'odds_ratio_ci_95_lower': or_ci_lower,
            'odds_ratio_ci_95_upper': or_ci_upper,
            'significant_at_0.05': significant
        }

    # Prepare concise interpretation
    # Interpret main effects in context of mean-centering: main effect of SizeDiff_c
    # gives effect of size when LocAdv_c == 0 (average location advantage), and vice versa.
    lines = []
    used_res_name = 'cluster-robust result' if model_output.get('cluster_robust_result') is not None else 'GLM result (no cluster-robust SE available)'
    lines.append(f"Using the {used_res_name}:")
    # Size effect
    s = stats['SizeDiff_c']
    lines.append(f"- Relative group size (SizeDiff_c): coef = {s['coefficient']:.3f}, OR = {s['odds_ratio']:.3f}, p = {s['p_value']:.3g}. "
                 + ("Statistically significant (p < 0.05)." if s['significant_at_0.05'] else "Not statistically significant (p >= 0.05)."))
    # Location effect
    l = stats['LocAdv_c']
    lines.append(f"- Location advantage (LocAdv_c): coef = {l['coefficient']:.3f}, OR = {l['odds_ratio']:.3f}, p = {l['p_value']:.3g}. "
                 + ("Statistically significant." if l['significant_at_0.05'] else "Not statistically significant."))
    # Interaction
    it = stats['SizeDiff_c:LocAdv_c']
    if it['significant_at_0.05']:
        lines.append(f"- Interaction (SizeDiff_c:LocAdv_c) is statistically significant (coef = {it['coefficient']:.3f}, p = {it['p_value']:.3g}). "
                     "This indicates the effect of relative group size on winning depends on contest location advantage (i.e., the size effect differs when focal group is closer vs farther from its home-range center).")
    else:
        lines.append(f"- Interaction (SizeDiff_c:LocAdv_c) is not statistically significant (coef = {it['coefficient']:.3f}, p = {it['p_value']:.3g}). "
                     "This suggests no evidence that the effect of relative group size on winning depends on location advantage; main effects can be interpreted additively (with main-effect interpretations at the mean of the other variable due to mean-centering).")

    # Short overall conclusion
    # Decide a simple yes/no answer about whether relative size and contest location influence winning:
    size_influences = s['significant_at_0.05']
    loc_influences = l['significant_at_0.05']
    interaction_influences = it['significant_at_0.05']

    if interaction_influences:
        overall = "There is evidence that the effect of relative group size on winning depends on contest location (significant interaction). Interpret main effects conditional on the other variable."
    else:
        parts = []
        if size_influences:
            parts.append("relative group size influences the probability of winning")
        else:
            parts.append("no evidence that relative group size influences winning")
        if loc_influences:
            parts.append("location advantage influences the probability of winning")
        else:
            parts.append("no evidence that location advantage influences winning")
        overall = " and ".join(parts).capitalize() + " (based on p < 0.05 threshold)."

    lines.append("Overall conclusion: " + overall)

    description = " ".join(lines)

    return {
        "object": stats,
        "description": description
    }