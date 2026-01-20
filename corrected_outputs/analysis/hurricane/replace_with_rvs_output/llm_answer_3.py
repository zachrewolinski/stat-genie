def extract_final_answer(model_output):
    """
    Extracts statistics for the femininity predictor (masfem_z) on log(1 + alldeaths)
    from the provided statsmodels fitted models dictionary.

    Returns a dictionary with:
      - "object": nested dicts containing coefficient, SE, t, p, 95% CI (log scale),
                  percent-change interpretation and sample size for:
            * 'deaths_primary'       -> model_output['death_model'] (masfem_z)
            * 'deaths_mturk_robust' -> model_output['death_model_mturk_iv'] (masfem_mturk_z)
            * 'damage_primary'       -> model_output['damage_model'] (masfem_z on log_ndam15)
      - "description": short plain-language interpretation of the primary result
                       and whether it provides evidence for the hypothesis.

    Notes:
      - masfem_z is mean-centered and scaled -> coefficient is per 1 SD increase in femininity.
      - Outcomes are log(1 + outcome). Percent-change = (exp(beta) - 1) * 100.
    """
    import numpy as np

    def summarize(result, varname):
        if result is None:
            return None
        try:
            params = result.params
        except Exception:
            return None
        if varname not in params.index:
            return None
        beta = float(params[varname])
        # Some result objects may not have tvalues if cov_type used; handle gracefully
        t = float(result.tvalues[varname]) if hasattr(result, 'tvalues') and varname in result.tvalues.index else None
        se = float(result.bse[varname]) if hasattr(result, 'bse') and varname in result.bse.index else None
        p = float(result.pvalues[varname]) if hasattr(result, 'pvalues') and varname in result.pvalues.index else None
        ci = result.conf_int(alpha=0.05).loc[varname].values if hasattr(result, 'conf_int') else np.array([np.nan, np.nan])
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
        # Percent change interpretation on original (1 + outcome) scale
        pct_change = (np.exp(beta) - 1.0) * 100.0
        pct_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0
        pct_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0
        nobs = int(result.nobs) if hasattr(result, 'nobs') else None

        return {
            'variable': varname,
            'coef_log_units': beta,
            'se': se,
            't': t,
            'p_value': p,
            'ci_log_lower': ci_lower,
            'ci_log_upper': ci_upper,
            'percent_change_point_estimate': pct_change,
            'percent_change_ci_lower': pct_ci_lower,
            'percent_change_ci_upper': pct_ci_upper,
            'nobs': nobs
        }

    # Pull models safely
    death_model = model_output.get('death_model')
    damage_model = model_output.get('damage_model')
    death_model_mturk = model_output.get('death_model_mturk_iv')

    deaths_primary = summarize(death_model, 'masfem_z')
    deaths_mturk = summarize(death_model_mturk, 'masfem_mturk_z')
    damage_primary = summarize(damage_model, 'masfem_z')

    # Build a concise plain-language description for the primary deaths result
    if deaths_primary is None:
        description = "Could not locate masfem_z coefficient in the provided death_model."
    else:
        beta = deaths_primary['coef_log_units']
        p = deaths_primary['p_value']
        pct = deaths_primary['percent_change_point_estimate']
        ci_low = deaths_primary['percent_change_ci_lower']
        ci_high = deaths_primary['percent_change_ci_upper']
        n = deaths_primary['nobs']

        # Determine statistical significance at alpha = 0.05 (two-sided)
        sig = (p is not None) and (p < 0.05)

        # Direction interpretation
        if beta < 0:
            direction = "More feminine names are associated with fewer deaths (negative coefficient)."
        elif beta > 0:
            direction = "More feminine names are associated with more deaths (positive coefficient)."
        else:
            direction = "No association detected (coefficient is ~0)."

        sig_text = "statistically significant" if sig else "not statistically significant"
        description = (
            f"Primary result (n={n}): masfem_z coef = {beta:.4f}, p = {p:.3g}. "
            f"{direction} This corresponds to an approximate {pct:.2f}% change in (1 + deaths) "
            f"per 1 SD increase in femininity (95% CI: {ci_low:.2f}% to {ci_high:.2f}%). "
            f"The effect is {sig_text} at alpha = 0.05. "
        )

        # Add robustness note if available
        if deaths_mturk is not None:
            beta_r = deaths_mturk['coef_log_units']
            p_r = deaths_mturk['p_value']
            pct_r = deaths_mturk['percent_change_point_estimate']
            sig_r = (p_r is not None) and (p_r < 0.05)
            description += (
                f"Robustness check using MTurk-rated femininity (masfem_mturk_z): coef = {beta_r:.4f}, "
                f"p = {p_r:.3g}, approx {pct_r:.2f}% change; "
                f"{'significant' if sig_r else 'not significant'}."
            )

    result_object = {
        'deaths_primary': deaths_primary,
        'deaths_mturk_robust': deaths_mturk,
        'damage_primary': damage_primary
    }

    return {
        'object': result_object,
        'description': description
    }