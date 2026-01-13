def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, confidence intervals, z-stats, and p-values
    for:
      - the main effect of reader_view (interpreted as the effect when dyslexia_bin = 0)
      - the reader_view:dyslexia_bin interaction
      - the combined effect of reader_view when dyslexia_bin = 1 (main + interaction)

    Returns:
      {
        "object": { ... detailed numeric results ... },
        "description": "<brief interpretation in plain English>"
      }
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Ensure required attributes exist
    if not hasattr(res, "params") or not hasattr(res, "cov_params"):
        raise ValueError("model_output does not look like a fitted statsmodels results object with .params and .cov_params()")

    params = res.params
    cov = res.cov_params()

    # Find parameter names related to reader_view
    param_names = list(params.index)

    # Identify main reader_view term (contains 'reader_view' but not a colon)
    main_name = next((n for n in param_names if ('reader_view' in n) and (':' not in n)), None)
    # Identify interaction term (contains both 'reader_view' and ':' typically 'reader_view:dyslexia_bin')
    interaction_name = next((n for n in param_names if ('reader_view' in n) and (':' in n)), None)

    if main_name is None:
        raise ValueError("Could not find a parameter corresponding to the main effect 'reader_view' in model params.")
    if interaction_name is None:
        # It's possible the interaction has a different naming; try any param that contains both substrings
        interaction_name = next((n for n in param_names if ('reader_view' in n and 'dyslexia' in n) or ('reader_view' in n and 'dyslexia_bin' in n)), None)

    if interaction_name is None:
        raise ValueError("Could not find an interaction parameter for reader_view * dyslexia_bin in model params.")

    # Helper to compute stats for a single coefficient name
    def coef_stats(name):
        coef = float(params[name])
        var = float(cov.loc[name, name])
        se = float(np.sqrt(var))
        z = coef / se if se > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_lo = coef - 1.96 * se
        ci_hi = coef + 1.96 * se
        return {"name": name, "coef": coef, "se": se, "z": z, "p": p, "ci_95": (ci_lo, ci_hi)}

    main_stats = coef_stats(main_name)
    inter_stats = coef_stats(interaction_name)

    # Combined effect for dyslexia_bin = 1: main + interaction
    coef_comb = main_stats["coef"] + inter_stats["coef"]
    # variance of sum = var(main) + var(interaction) + 2*cov(main, interaction)
    cov_main_inter = float(cov.loc[main_name, interaction_name]) if (main_name in cov.index and interaction_name in cov.columns) else 0.0
    var_comb = float(cov.loc[main_name, main_name]) + float(cov.loc[interaction_name, interaction_name]) + 2.0 * cov_main_inter
    se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else np.nan
    z_comb = coef_comb / se_comb if se_comb and not np.isnan(se_comb) else np.nan
    p_comb = 2 * (1 - stats.norm.cdf(abs(z_comb))) if not np.isnan(z_comb) else np.nan
    ci_comb = (coef_comb - 1.96 * se_comb, coef_comb + 1.96 * se_comb) if not np.isnan(se_comb) else (np.nan, np.nan)

    combined_stats = {
        "name": f"{main_name} + {interaction_name} (effect when dyslexia_bin=1)",
        "coef": coef_comb,
        "se": se_comb,
        "z": z_comb,
        "p": p_comb,
        "ci_95": ci_comb
    }

    # Build result object
    result_object = {
        "reader_view_effect_dyslexia0": main_stats,
        "interaction_term_reader_view_x_dyslexia": inter_stats,
        "reader_view_effect_dyslexia1": combined_stats,
        # include raw param names for traceability
        "param_names": {"main": main_name, "interaction": interaction_name}
    }

    # Short interpretation / answer to the yes/no question
    # We interpret "improves reading speed for individuals with dyslexia" as:
    #   whether the combined effect for dyslexia_bin = 1 is positive and statistically significant (two-sided p < 0.05).
    comb_coef = combined_stats["coef"]
    comb_p = combined_stats["p"]
    if np.isnan(comb_coef) or np.isnan(comb_p):
        conclusion = "Could not compute combined effect for dyslexic individuals due to missing/invalid variance information."
    else:
        if (comb_coef > 0) and (comb_p < 0.05):
            conclusion = (
                "Yes — Reader View appears to improve reading speed for individuals with dyslexia. "
                f"Estimated increase = {comb_coef:.2f} WPM (95% CI [{combined_stats['ci_95'][0]:.2f}, {combined_stats['ci_95'][1]:.2f}]), "
                f"p = {comb_p:.3f}."
            )
        elif (comb_coef <= 0) and (comb_p < 0.05):
            conclusion = (
                "No — Reader View appears to decrease reading speed for individuals with dyslexia. "
                f"Estimated change = {comb_coef:.2f} WPM (95% CI [{combined_stats['ci_95'][0]:.2f}, {combined_stats['ci_95'][1]:.2f}]), "
                f"p = {comb_p:.3f}."
            )
        else:
            conclusion = (
                "No strong evidence that Reader View improves reading speed for individuals with dyslexia. "
                f"Estimated change = {comb_coef:.2f} WPM (95% CI [{combined_stats['ci_95'][0]:.2f}, {combined_stats['ci_95'][1]:.2f}]), "
                f"p = {comb_p:.3f}. The interaction term (difference in effect between dyslexic and non-dyslexic) is "
                f"{inter_stats['coef']:.3f} (p = {inter_stats['p']:.3f})."
            )

    description = (
        "Extracted coefficients and statistics for the Reader View effect and its interaction with dyslexia.\n"
        "Interpretation: " + conclusion
    )

    return {"object": result_object, "description": description}