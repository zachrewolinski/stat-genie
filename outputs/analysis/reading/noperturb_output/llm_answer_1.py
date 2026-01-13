def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of Reader View on reading speed for:
      - non-dyslexic readers (main effect of reader_view)
      - dyslexic readers (main effect + interaction)
      - the interaction term (difference in Reader View effect between dyslexic and non-dyslexic)

    Returns a dict with:
      - "object": a dictionary of numeric results (estimates, SEs, t-stats, p-values, 95% CIs,
                  and percent-change interpretation on original speed scale)
      - "description": a short human-readable interpretation about whether Reader View
                       improves reading speed for dyslexic readers (based on alpha=0.05).
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # pull params and covariance matrix (should reflect cluster-robust cov if model was fit that way)
    params = res.params
    cov = res.cov_params()

    # Try to robustly find the parameter names for the reader_view main effect and interaction
    # Expected names: 'reader_view' and 'reader_view:dyslexia_bin' but try alternatives if needed.
    # Find reader_view name
    reader_view_name = None
    for n in params.index:
        if n == 'reader_view':
            reader_view_name = n
            break
    if reader_view_name is None:
        # fallback: any param that equals 'reader_view' substring but not the interaction
        cand = [n for n in params.index if 'reader_view' in n and 'dyslexia_bin' not in n]
        if cand:
            reader_view_name = cand[0]
    if reader_view_name is None:
        raise KeyError("Could not find a parameter corresponding to the main effect 'reader_view' in model parameters.")

    # Find interaction name
    interaction_name = None
    # common expected form
    candidates = [
        'reader_view:dyslexia_bin',
        'reader_view*dyslexia_bin',
        'dyslexia_bin:reader_view',
        'dyslexia_bin*reader_view'
    ]
    for c in candidates:
        if c in params.index:
            interaction_name = c
            break
    if interaction_name is None:
        # fallback: any param that contains both substrings
        cand = [n for n in params.index if 'reader_view' in n and 'dyslexia_bin' in n]
        if cand:
            interaction_name = cand[0]
    if interaction_name is None:
        raise KeyError("Could not find an interaction parameter for 'reader_view' and 'dyslexia_bin' in model parameters.")

    # Helper to compute statistics for a single parameter
    def single_param_stats(name):
        est = float(params[name])
        var = float(cov.loc[name, name])
        se = np.sqrt(var)
        t_stat = est / se if se > 0 else np.nan
        # use residual df for t-distribution if available; otherwise use normal approx
        df = getattr(res, 'df_resid', None)
        if df is None or np.isinf(df):
            pval = 2 * (1 - stats.norm.cdf(abs(t_stat)))
            t_crit = stats.norm.ppf(0.975)
        else:
            pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
            t_crit = stats.t.ppf(0.975, df=df)
        ci_lower = est - t_crit * se
        ci_upper = est + t_crit * se
        return {
            'name': name,
            'estimate_log': est,
            'se': se,
            't': t_stat,
            'p': pval,
            'ci95_log': (ci_lower, ci_upper),
            # percent change on original speed scale: exp(estimate)-1
            'percent_change': (np.exp(est) - 1) if not np.isnan(est) and np.isfinite(est) else np.nan,
            'percent_change_ci95': (np.exp(ci_lower) - 1, np.exp(ci_upper) - 1)
        }

    # Main effect for non-dyslexic readers (reader_view coefficient)
    non_dys_stats = single_param_stats(reader_view_name)

    # Interaction statistics
    interaction_stats = single_param_stats(interaction_name)

    # Effect for dyslexic readers = reader_view + interaction
    est_dys = float(params[reader_view_name]) + float(params[interaction_name])
    # variance = var(rv) + var(interaction) + 2*cov(rv, interaction)
    cov_rv_inter = float(cov.loc[reader_view_name, interaction_name])
    var_dys = float(cov.loc[reader_view_name, reader_view_name]) + float(cov.loc[interaction_name, interaction_name]) + 2.0 * cov_rv_inter
    se_dys = np.sqrt(var_dys) if var_dys >= 0 else np.nan
    df = getattr(res, 'df_resid', None)
    if df is None or np.isinf(df):
        p_dys = 2 * (1 - stats.norm.cdf(abs(est_dys / se_dys))) if se_dys > 0 else np.nan
        t_crit = stats.norm.ppf(0.975)
    else:
        p_dys = 2 * (1 - stats.t.cdf(abs(est_dys / se_dys), df=df)) if se_dys > 0 else np.nan
        t_crit = stats.t.ppf(0.975, df=df)
    ci_dys = (est_dys - t_crit * se_dys, est_dys + t_crit * se_dys) if se_dys > 0 else (np.nan, np.nan)
    dys_stats = {
        'name': f"{reader_view_name} + {interaction_name} (effect for dyslexic readers)",
        'estimate_log': est_dys,
        'se': se_dys,
        't': est_dys / se_dys if se_dys > 0 else np.nan,
        'p': p_dys,
        'ci95_log': ci_dys,
        'percent_change': (np.exp(est_dys) - 1) if not np.isnan(est_dys) and np.isfinite(est_dys) else np.nan,
        'percent_change_ci95': (np.exp(ci_dys[0]) - 1, np.exp(ci_dys[1]) - 1) if se_dys > 0 else (np.nan, np.nan)
    }

    results_object = {
        'non_dyslexic_reader_view': non_dys_stats,
        'interaction_term': interaction_stats,
        'dyslexic_reader_view': dys_stats
    }

    # Interpretation using alpha = 0.05 on dyslexic effect
    sig = dys_stats['p'] < 0.05 if (dys_stats['p'] is not None and not np.isnan(dys_stats['p'])) else False
    direction = "increase" if dys_stats['estimate_log'] > 0 else ("decrease" if dys_stats['estimate_log'] < 0 else "no change")
    desc = (
        f"Estimated effect of Reader View for dyslexic readers: log-change = {dys_stats['estimate_log']:.3f} "
        f"(SE = {dys_stats['se']:.3f}, t = {dys_stats['t']:.3f}, p = {dys_stats['p']:.3g}).\n"
        f"This corresponds to a {dys_stats['percent_change']*100:.1f}% {direction} in reading speed "
        f"(95% CI: {dys_stats['percent_change_ci95'][0]*100:.1f}% to {dys_stats['percent_change_ci95'][1]*100:.1f}%).\n"
    )
    if sig:
        desc += "Conclusion at alpha=0.05: Reader View significantly affects reading speed for individuals with dyslexia."
    else:
        desc += "Conclusion at alpha=0.05: No statistically significant effect of Reader View on reading speed for individuals with dyslexia."

    return {
        "object": results_object,
        "description": desc
    }