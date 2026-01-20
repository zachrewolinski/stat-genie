def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the effect of ReaderView on LogReadingWPM,
    including whether that effect differs for dyslexic readers (interaction).

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Concise interpretation in context"
      }

    The returned "object" contains:
      - main_effect: effect of ReaderView for non-dyslexic readers (log change),
                     with se, p-value, 95% CI (log), and percent change (and CI).
      - interaction: coefficient for ReaderView:Dyslexic (log), with se, p-value, CI.
      - combined_effect_dyslexic: total effect for dyslexic readers (log = main + interaction),
                     with se, p-value (test of sum=0), 95% CI (log), and percent change (and CI).
      - conclusion: boolean-like summary whether ReaderView increases reading speed
                    for dyslexic readers at alpha=0.05 and short note.
    """
    import numpy as np
    from scipy import stats

    # Basic checks
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not look like a fitted statsmodels result (missing .params).")

    params = model_output.params
    cov = model_output.cov_params()  # robust covariance used in fitting
    param_names = list(params.index)

    # Helper to find parameter names robustly
    def find_main_reader_param(names):
        # Prefer exact match
        if 'ReaderView' in names:
            return 'ReaderView'
        # Otherwise try variants (e.g., if categorical encoding produced other names)
        for n in names:
            if n == 'ReaderView':
                return n
        # fallback: find a param that contains ReaderView but is not the interaction
        for n in names:
            if 'ReaderView' in n and ':' not in n and 'Dyslexic' not in n:
                return n
        return None

    def find_interaction_param(names):
        # Typical name: 'ReaderView:Dyslexic'
        for n in names:
            if ':' in n and 'ReaderView' in n and 'Dyslexic' in n:
                return n
        # try other separators (unlikely)
        for n in names:
            if 'ReaderView' in n and 'Dyslexic' in n and n not in ( 'ReaderView', 'Dyslexic'):
                return n
        return None

    reader_param = find_main_reader_param(param_names)
    interaction_param = find_interaction_param(param_names)

    if reader_param is None:
        raise ValueError("Could not find a parameter corresponding to 'ReaderView' in model params: " + ", ".join(param_names))
    # interaction may be absent (e.g., removed by collinearity) - handle that case
    if interaction_param is None:
        # Treat interaction as zero if absent (but warn in description)
        interaction_coef = 0.0
        interaction_se = 0.0
        interaction_p = None
        interaction_ci = (None, None)
        interaction_present = False
    else:
        interaction_present = True
        interaction_coef = float(params[interaction_param])
        interaction_se = float(np.sqrt(cov.loc[interaction_param, interaction_param]))
        # p-value and CI for interaction
        t_int = interaction_coef / (interaction_se if interaction_se > 0 else np.nan)
        df = getattr(model_output, "df_resid", None)
        if df is None:
            interaction_p = 2 * (1 - stats.norm.cdf(abs(t_int)))
            crit = stats.norm.ppf(0.975)
        else:
            interaction_p = 2 * stats.t.sf(abs(t_int), df)
            crit = stats.t.ppf(0.975, df)
        interaction_ci = (interaction_coef - crit * interaction_se, interaction_coef + crit * interaction_se)

    # Main ReaderView effect (for Dyslexic==0 baseline)
    main_coef = float(params[reader_param])
    main_se = float(np.sqrt(cov.loc[reader_param, reader_param]))
    df = getattr(model_output, "df_resid", None)
    if df is None:
        main_t = main_coef / (main_se if main_se > 0 else np.nan)
        main_p = 2 * (1 - stats.norm.cdf(abs(main_t)))
        crit = stats.norm.ppf(0.975)
    else:
        main_t = main_coef / (main_se if main_se > 0 else np.nan)
        main_p = 2 * stats.t.sf(abs(main_t), df)
        crit = stats.t.ppf(0.975, df)
    main_ci = (main_coef - crit * main_se, main_coef + crit * main_se)

    # Combined effect for dyslexic readers: main_coef + interaction_coef
    total_coef = main_coef + (interaction_coef if interaction_present else 0.0)

    # Variance of sum = var(main) + var(interaction) + 2*cov(main, interaction)
    if interaction_present:
        var_main = cov.loc[reader_param, reader_param]
        var_int = cov.loc[interaction_param, interaction_param]
        cov_main_int = cov.loc[reader_param, interaction_param]
        var_total = var_main + var_int + 2.0 * cov_main_int
        se_total = float(np.sqrt(var_total)) if var_total >= 0 else np.nan
    else:
        # if no interaction parameter available, total effect = main, same se
        se_total = main_se
        var_total = main_se ** 2

    # t-stat and p-value for combined effect (test total_coef = 0)
    if df is None:
        t_total = total_coef / (se_total if se_total > 0 else np.nan)
        p_total = 2 * (1 - stats.norm.cdf(abs(t_total)))
        crit = stats.norm.ppf(0.975)
    else:
        t_total = total_coef / (se_total if se_total > 0 else np.nan)
        p_total = 2 * stats.t.sf(abs(t_total), df)
        crit = stats.t.ppf(0.975, df)
    total_ci = (total_coef - crit * se_total, total_coef + crit * se_total)

    # Convert log-coefficients to percent changes: (exp(coef) - 1) * 100
    def pct_change_from_log(coef):
        return (np.exp(coef) - 1.0) * 100.0

    main_pct = pct_change_from_log(main_coef)
    main_ci_pct = (pct_change_from_log(main_ci[0]), pct_change_from_log(main_ci[1]))

    if interaction_present:
        interaction_pct = pct_change_from_log(interaction_coef)
        interaction_ci_pct = (pct_change_from_log(interaction_ci[0]), pct_change_from_log(interaction_ci[1]))
    else:
        interaction_pct = None
        interaction_ci_pct = (None, None)

    total_pct = pct_change_from_log(total_coef)
    total_ci_pct = (pct_change_from_log(total_ci[0]), pct_change_from_log(total_ci[1]))

    # Conclusion: does ReaderView improve reading speed for dyslexic readers?
    # We'll call it "improves" if the combined effect is positive and p < 0.05.
    improves = None
    if p_total is not None:
        improves = (total_coef > 0) and (p_total < 0.05)
    else:
        improves = None  # could not compute p-value

    result_object = {
        "reader_param_name": reader_param,
        "interaction_param_name": interaction_param if interaction_present else None,
        "main_effect_non_dyslexic": {
            "coef_log": main_coef,
            "se": main_se,
            "t": main_t,
            "p_value": main_p,
            "ci_log": main_ci,
            "percent_change": main_pct,
            "ci_percent_change": main_ci_pct,
        },
        "interaction": {
            "present": interaction_present,
            "coef_log": interaction_coef if interaction_present else None,
            "se": interaction_se if interaction_present else None,
            "p_value": interaction_p if interaction_present else None,
            "ci_log": interaction_ci if interaction_present else (None, None),
            "percent_change": interaction_pct,
            "ci_percent_change": interaction_ci_pct,
        },
        "combined_effect_dyslexic": {
            "coef_log": total_coef,
            "se": se_total,
            "t": t_total,
            "p_value": p_total,
            "ci_log": total_ci,
            "percent_change": total_pct,
            "ci_percent_change": total_ci_pct,
        },
        "conclusion": {
            "improves_for_dyslexic_at_alpha_0_05": bool(improves) if improves is not None else None,
            "p_value_combined": p_total,
            "note": "Decision is True if combined effect > 0 and p < 0.05. See numeric results for exact estimates."
        }
    }

    # Human-readable short description
    if interaction_present:
        desc = (
            f"The model estimates a main ReaderView effect (for non-dyslexic readers) of "
            f"{main_coef:.4f} log-units (≈ {main_pct:.1f}% change; 95% CI {main_ci_pct[0]:.1f}% to {main_ci_pct[1]:.1f}%), "
            f"and an interaction ReaderView:Dyslexic of {interaction_coef:.4f} log-units (≈ {interaction_pct:.1f}%; "
            f"p = {interaction_p:.3g}). For dyslexic readers the combined effect is {total_coef:.4f} log-units "
            f"(≈ {total_pct:.1f}% change; 95% CI {total_ci_pct[0]:.1f}% to {total_ci_pct[1]:.1f}%; p = {p_total:.3g}). "
            f"Conclusion: {'ReaderView appears to improve reading speed for dyslexic readers (statistically significant).' if improves else 'No statistically significant improvement for dyslexic readers at alpha=0.05.'}"
        )
    else:
        desc = (
            f"No interaction parameter (ReaderView:Dyslexic) was found in the model output. "
            f"The main ReaderView effect (applies to all readers under the model as specified) is {main_coef:.4f} log-units "
            f"(≈ {main_pct:.1f}% change; 95% CI {main_ci_pct[0]:.1f}% to {main_ci_pct[1]:.1f}%; p = {main_p:.3g}). "
            "Because the interaction term is missing, we cannot conclude a differential effect for dyslexic readers from this fit."
        )

    return {"object": result_object, "description": desc}