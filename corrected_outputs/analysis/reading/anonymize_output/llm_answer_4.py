def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the effect of ReaderView on ReadingSpeed_wps,
    including the ReaderView:Dyslexia interaction and the combined effect of ReaderView
    for readers with dyslexia (i.e., ReaderView + ReaderView:Dyslexia).

    Returns:
      {
        "object": {
          "readerview": {coef, se, t, p, ci_lower, ci_upper},
          "interaction": {coef, se, t, p, ci_lower, ci_upper},
          "readerview_for_dyslexia": {coef, se, t, p, ci_lower, ci_upper},
          "significant_for_dyslexia": bool,
          "alpha": 0.05
        },
        "description": "Plain-language interpretation of the above results."
      }

    Notes:
      - This function tries to be robust to slight differences in parameter naming
        (e.g., 'ReaderView:Dyslexia' vs 'Dyslexia:ReaderView'). It assumes the model
        object is a statsmodels RegressionResultsWrapper (or similar) with attributes
        .params, .bse, .tvalues, .pvalues, .conf_int(), .cov_params(), and .df_resid,
        and which supports .t_test().
    """
    import numpy as np

    res = model_output

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params).")

    params = res.params
    param_names = [str(n) for n in params.index]

    # Helper to find exact main ReaderView param and the interaction param
    def find_main_readerview_name(names):
        # Prefer exact match
        if 'ReaderView' in names:
            return 'ReaderView'
        # Fallback: any name containing ReaderView but not colon (i.e., not an interaction)
        for n in names:
            if 'ReaderView' in n and ':' not in n:
                return n
        # Last resort: any name that contains ReaderView
        for n in names:
            if 'ReaderView' in n:
                return n
        return None

    def find_interaction_name(names):
        # Look for a name that contains both ReaderView and Dyslexia
        for n in names:
            if 'ReaderView' in n and 'Dyslexia' in n:
                return n
        # Also accept either order with colon
        for n in names:
            if ':' in n and ('ReaderView' in n or 'Dyslexia' in n) and ('ReaderView' in n and 'Dyslexia' in n):
                return n
        return None

    main_name = find_main_readerview_name(param_names)
    interaction_name = find_interaction_name(param_names)

    if main_name is None:
        raise ValueError("Could not find a parameter name for the main ReaderView effect in model parameters.")
    if interaction_name is None:
        # It's possible the interaction term was omitted or named unexpectedly; we'll still return main effect only.
        interaction_present = False
    else:
        interaction_present = True

    # Extract stats for main effect
    coef_main = float(params.loc[main_name])
    # Try to get standard error and t/p/CI reliably
    try:
        se_main = float(res.bse.loc[main_name])
    except Exception:
        # fallback: use covariance diagonal
        se_main = float(np.sqrt(res.cov_params().loc[main_name, main_name]))
    # t and p for main param (may be in res.tvalues/res.pvalues)
    try:
        t_main = float(res.tvalues.loc[main_name])
        p_main = float(res.pvalues.loc[main_name])
    except Exception:
        # fallback compute
        t_main = coef_main / se_main if se_main != 0 else np.nan
        # approximate two-sided p using normal (fallback)
        from math import erf, sqrt
        z = abs(t_main)
        p_main = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))

    # Confidence interval for main (try res.conf_int())
    try:
        ci_df = res.conf_int()
        # conf_int may be a DataFrame or ndarray; handle both
        if hasattr(ci_df, 'loc'):
            ci_main = ci_df.loc[main_name].astype(float).tolist()
        else:
            # ci_df is ndarray; need to find index of main_name
            idx = param_names.index(main_name)
            ci_main = [float(ci_df[idx, 0]), float(ci_df[idx, 1])]
    except Exception:
        # fallback normal approx (use df_resid if available)
        df = getattr(res, 'df_resid', np.nan)
        try:
            from scipy import stats
            tcrit = stats.t.ppf(1 - 0.025, df) if not np.isnan(df) else 1.96
        except Exception:
            tcrit = 1.96
        ci_main = [coef_main - tcrit * se_main, coef_main + tcrit * se_main]

    main_stats = {
        'name': main_name,
        'coef': coef_main,
        'se': se_main,
        't': t_main,
        'p': p_main,
        'ci_lower': float(ci_main[0]),
        'ci_upper': float(ci_main[1])
    }

    interaction_stats = None
    if interaction_present:
        coef_int = float(params.loc[interaction_name])
        try:
            se_int = float(res.bse.loc[interaction_name])
        except Exception:
            se_int = float(np.sqrt(res.cov_params().loc[interaction_name, interaction_name]))
        try:
            t_int = float(res.tvalues.loc[interaction_name])
            p_int = float(res.pvalues.loc[interaction_name])
        except Exception:
            t_int = coef_int / se_int if se_int != 0 else np.nan
            from math import erf, sqrt
            z = abs(t_int)
            p_int = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        try:
            ci_df = res.conf_int()
            if hasattr(ci_df, 'loc'):
                ci_int = ci_df.loc[interaction_name].astype(float).tolist()
            else:
                idx = param_names.index(interaction_name)
                ci_int = [float(ci_df[idx, 0]), float(ci_df[idx, 1])]
        except Exception:
            df = getattr(res, 'df_resid', np.nan)
            try:
                from scipy import stats
                tcrit = stats.t.ppf(1 - 0.025, df) if not np.isnan(df) else 1.96
            except Exception:
                tcrit = 1.96
            ci_int = [coef_int - tcrit * se_int, coef_int + tcrit * se_int]

        interaction_stats = {
            'name': interaction_name,
            'coef': coef_int,
            'se': se_int,
            't': t_int,
            'p': p_int,
            'ci_lower': float(ci_int[0]),
            'ci_upper': float(ci_int[1])
        }

    # Combined effect for dyslexia = ReaderView + ReaderView:Dyslexia (if interaction present).
    # Construct contrast vector for linear combination
    params_index = list(params.index)
    k = len(params_index)
    contrast = np.zeros(k, dtype=float)
    idx_main = params_index.index(main_name)
    contrast[idx_main] = 1.0
    if interaction_present:
        idx_int = params_index.index(interaction_name)
        contrast[idx_int] = 1.0

    # Use res.t_test to get effect, sd, t, p for the linear combination
    try:
        ct = res.t_test(contrast)
        # ct.effect may be shape (1,1) or (1,) etc.
        effect_comb = float(np.asarray(ct.effect).reshape(-1)[0])
        se_comb = float(np.asarray(ct.sd).reshape(-1)[0])
        t_comb = float(np.asarray(ct.tvalue).reshape(-1)[0])
        p_comb = float(np.asarray(ct.pvalue).reshape(-1)[0])
        # Try to get CI from ct
        try:
            ci_comb = ct.conf_int(alpha=0.05)
            ci_comb = [float(ci_comb[0, 0]), float(ci_comb[0, 1])]
        except Exception:
            # fallback compute using df_resid
            df = getattr(res, 'df_resid', np.nan)
            try:
                from scipy import stats
                tcrit = stats.t.ppf(1 - 0.025, df) if not np.isnan(df) else 1.96
            except Exception:
                tcrit = 1.96
            ci_comb = [effect_comb - tcrit * se_comb, effect_comb + tcrit * se_comb]
    except Exception:
        # If t_test fails, compute via covariance matrix
        cov = res.cov_params()
        var_comb = float(contrast @ cov.values @ contrast)
        se_comb = float(np.sqrt(var_comb))
        effect_comb = float((contrast @ params.values).astype(float))
        df = getattr(res, 'df_resid', np.nan)
        try:
            from scipy import stats
            t_comb = effect_comb / se_comb if se_comb != 0 else np.nan
            p_comb = 2 * stats.t.sf(abs(t_comb), df) if not np.isnan(df) else 2 * (1 - stats.norm.cdf(abs(t_comb)))
            tcrit = stats.t.ppf(1 - 0.025, df) if not np.isnan(df) else 1.96
        except Exception:
            from math import erf, sqrt
            t_comb = effect_comb / se_comb if se_comb != 0 else np.nan
            z = abs(t_comb)
            p_comb = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
            tcrit = 1.96
        ci_comb = [effect_comb - tcrit * se_comb, effect_comb + tcrit * se_comb]

    combined_stats = {
        'coef': effect_comb,
        'se': se_comb,
        't': t_comb,
        'p': p_comb,
        'ci_lower': float(ci_comb[0]),
        'ci_upper': float(ci_comb[1]),
        'contrast_vector': contrast.tolist()
    }

    alpha = 0.05
    significant_for_dyslexia = (combined_stats['p'] < alpha) and (combined_stats['coef'] > 0)

    # Build plain-language description
    if significant_for_dyslexia:
        conclusion = (
            f"The combined effect of ReaderView for readers with dyslexia is "
            f"+{combined_stats['coef']:.4g} words/sec (SE={combined_stats['se']:.4g}), "
            f"95% CI [{combined_stats['ci_lower']:.4g}, {combined_stats['ci_upper']:.4g}], "
            f"p = {combined_stats['p']:.4g}. This indicates a statistically significant "
            f"increase in reading speed for readers with dyslexia when ReaderView is ON "
            f"(alpha = {alpha})."
        )
    else:
        conclusion = (
            f"The combined effect of ReaderView for readers with dyslexia is "
            f"{combined_stats['coef']:.4g} words/sec (SE={combined_stats['se']:.4g}), "
            f"95% CI [{combined_stats['ci_lower']:.4g}, {combined_stats['ci_upper']:.4g}], "
            f"p = {combined_stats['p']:.4g}. This does not provide evidence of a "
            f"statistically significant improvement for readers with dyslexia at alpha = {alpha}."
        )

    # Package object to return
    output_object = {
        'readerview_main': main_stats,
        'interaction': interaction_stats,
        'readerview_for_dyslexia': combined_stats,
        'significant_for_dyslexia': bool(significant_for_dyslexia),
        'alpha': alpha
    }

    description = (
        "Extracted statistics for the effect of ReaderView on reading speed.\n"
        "- 'readerview_main' is the estimated effect of turning ReaderView ON for the reference group (Dyslexia=0).\n"
        "- 'interaction' is the ReaderView:Dyslexia interaction coefficient (how much the ReaderView effect differs for dyslexic readers).\n"
        "- 'readerview_for_dyslexia' is the combined effect (ReaderView + ReaderView:Dyslexia) representing the effect of ReaderView for readers with dyslexia.\n\n"
        + conclusion
    )

    return {"object": output_object, "description": description}