def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of gender (Female) on mortgage application acceptance
    from the provided model_output dictionary (expected keys: 'results', 'marginal_effects').

    Returns:
      {
        "object": { ... detailed numeric results ... },
        "description": "Plain-language interpretation of those results"
      }
    """
    import numpy as np
    from math import sqrt
    from scipy.stats import norm

    # Unpack
    results = model_output.get('results', None)
    marg_eff = model_output.get('marginal_effects', None)

    if results is None:
        raise ValueError("model_output must contain 'results' (the fitted GLM results).")

    params = results.params
    bse = results.bse
    pvalues = results.pvalues
    conf = results.conf_int()  # DataFrame or ndarray-like: two columns (lower, upper)
    cov = results.cov_params()

    # Helper to safely get param entries
    def _get(name):
        if name not in params.index:
            raise KeyError(f"Parameter '{name}' not found in model results.")
        coef = float(params[name])
        se = float(bse[name])
        p = float(pvalues[name])
        ci_low = float(conf.loc[name, 0])
        ci_high = float(conf.loc[name, 1])
        z = coef / se if se != 0 else np.nan
        or_val = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))
        return {
            'coef': coef, 'se': se, 'z': z, 'p_value': p,
            'ci_95': (ci_low, ci_high),
            'odds_ratio': or_val,
            'odds_ratio_95ci': (or_ci_low, or_ci_high)
        }

    # Extract Female and Female_Black (interaction) basic stats
    female_stats = _get('Female')
    interaction_stats = None
    if 'Female_Black' in params.index:
        interaction_stats = _get('Female_Black')
    else:
        # If interaction term missing, set to zeros
        interaction_stats = {
            'coef': 0.0, 'se': 0.0, 'z': np.nan, 'p_value': np.nan,
            'ci_95': (0.0, 0.0), 'odds_ratio': 1.0, 'odds_ratio_95ci': (1.0, 1.0)
        }

    # Compute marginal effect / combined effect of Female for Black applicants:
    # effect_black = coef(Female) + coef(Female_Black)
    coef_f = female_stats['coef']
    coef_fb = interaction_stats['coef']
    effect_black_coef = coef_f + coef_fb

    # Get variance for combined coefficient
    try:
        var_f = float(cov.loc['Female', 'Female'])
        var_fb = float(cov.loc['Female_Black', 'Female_Black']) if 'Female_Black' in cov.index else 0.0
        cov_ffb = float(cov.loc['Female', 'Female_Black']) if ('Female' in cov.index and 'Female_Black' in cov.index) else 0.0
        var_sum = var_f + var_fb + 2.0 * cov_ffb
        se_sum = sqrt(var_sum) if var_sum >= 0 else float('nan')
        z_sum = effect_black_coef / se_sum if se_sum != 0 else np.nan
        p_sum = 2.0 * norm.sf(abs(z_sum)) if not np.isnan(z_sum) else np.nan
        ci_low_sum = effect_black_coef - 1.96 * se_sum
        ci_high_sum = effect_black_coef + 1.96 * se_sum
        or_black = float(np.exp(effect_black_coef))
        or_black_ci = (float(np.exp(ci_low_sum)), float(np.exp(ci_high_sum)))
    except Exception:
        # Fallback if covariance structure not accessible
        se_sum = np.nan
        z_sum = np.nan
        p_sum = np.nan
        ci_low_sum = np.nan
        ci_high_sum = np.nan
        or_black = float(np.exp(effect_black_coef))
        or_black_ci = (np.nan, np.nan)

    female_black_stats = {
        'coef': effect_black_coef,
        'se': se_sum,
        'z': z_sum,
        'p_value': p_sum,
        'ci_95': (ci_low_sum, ci_high_sum),
        'odds_ratio': or_black,
        'odds_ratio_95ci': or_black_ci
    }

    # Try to extract average marginal effect for Female if available
    ame_female = None
    try:
        if marg_eff is not None:
            # summary_frame usually contains the margins table; find the row for 'Female'
            me_df = marg_eff.summary_frame()
            # The index usually contains variable names; try several fallback ways
            if 'Female' in me_df.index:
                row = me_df.loc['Female']
                # first column is the marginal effect numeric value
                me_value = float(row.iloc[0])
                me_se = float(row.iloc[1]) if row.shape[0] > 1 else np.nan
                me_z = float(row.iloc[2]) if row.shape[0] > 2 else np.nan
                me_p = float(row.iloc[3]) if row.shape[0] > 3 else np.nan
                ci_low = float(row.iloc[4]) if row.shape[0] > 4 else np.nan
                ci_high = float(row.iloc[5]) if row.shape[0] > 5 else np.nan
                ame_female = {
                    'AME': me_value,
                    'SE': me_se,
                    'z': me_z,
                    'p_value': me_p,
                    'CI_95': (ci_low, ci_high)
                }
            else:
                # If the index is not variable names, try to take the row corresponding to the first effect
                # (best-effort fallback)
                row0 = me_df.iloc[0]
                ame_female = {
                    'AME': float(row0.iloc[0]),
                    'SE': float(row0.iloc[1]) if row0.size > 1 else np.nan,
                    'z': float(row0.iloc[2]) if row0.size > 2 else np.nan,
                    'p_value': float(row0.iloc[3]) if row0.size > 3 else np.nan,
                    'CI_95': (float(row0.iloc[4]) if row0.size > 4 else np.nan,
                              float(row0.iloc[5]) if row0.size > 5 else np.nan)
                }
    except Exception:
        ame_female = None

    # Build a concise textual interpretation
    def signif_label(p):
        return "statistically significant (p < 0.05)" if (p is not None and not np.isnan(p) and p < 0.05) else "not statistically significant (p >= 0.05)"

    desc_lines = []
    desc_lines.append(
        "Effect of being female (reference = male) among non-Black applicants:\n"
        f"  log-odds = {female_stats['coef']:.4f}, SE = {female_stats['se']:.4f}, z = {female_stats['z']:.2f}, p = {female_stats['p_value']:.4g};\n"
        f"  95% CI (log-odds) = [{female_stats['ci_95'][0]:.4f}, {female_stats['ci_95'][1]:.4f}];\n"
        f"  odds ratio = {female_stats['odds_ratio']:.3f}, 95% CI (OR) = [{female_stats['odds_ratio_95ci'][0]:.3f}, {female_stats['odds_ratio_95ci'][1]:.3f}];\n"
        f"  This effect is {signif_label(female_stats['p_value'])}."
    )
    desc_lines.append(
        "Effect of being female among Black applicants (Female + Female_Black interaction):\n"
        f"  combined log-odds = {female_black_stats['coef']:.4f}, SE = {female_black_stats['se']:.4f}, z = {female_black_stats['z']:.2f}, p = {female_black_stats['p_value']:.4g};\n"
        f"  95% CI (log-odds) = [{female_black_stats['ci_95'][0]:.4f}, {female_black_stats['ci_95'][1]:.4f}];\n"
        f"  odds ratio = {female_black_stats['odds_ratio']:.3f}, 95% CI (OR) = [{female_black_stats['odds_ratio_95ci'][0]:.3f}, {female_black_stats['odds_ratio_95ci'][1]:.3f}];\n"
        f"  This combined effect is {signif_label(female_black_stats['p_value'])}."
    )
    desc_lines.append(
        f"Interaction term (Female_Black): log-odds = {interaction_stats['coef']:.4f}, SE = {interaction_stats['se']:.4f}, "
        f"p = {interaction_stats['p_value']:.4g}. This tests whether the effect of Female differs between Black and non-Black applicants."
    )
    if ame_female is not None:
        desc_lines.append(
            f"Average marginal effect of Female (overall): AME = {ame_female['AME']:.4f}, SE = {ame_female['SE']:.4f}, "
            f"p = {ame_female['p_value']:.4g}, 95% CI = [{ame_female['CI_95'][0]:.4f}, {ame_female['CI_95'][1]:.4f}].\n"
            "  (This is the average change in predicted probability of acceptance when changing Female from 0 to 1.)"
        )
    else:
        desc_lines.append("Average marginal effect for 'Female' was not available from the model output.")

    description = "\n\n".join(desc_lines)

    # Compose the return object with numeric details and short textual summary
    result_object = {
        'female_nonblack': female_stats,
        'female_black': female_black_stats,
        'interaction_Female_Black': interaction_stats,
        'AME_female_overall': ame_female,
        'notes': "Coefficients are log-odds. Odds ratios = exp(coef). Statistical significance assessed at alpha=0.05."
    }

    return {
        "object": result_object,
        "description": description
    }