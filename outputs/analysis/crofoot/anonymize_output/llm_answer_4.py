def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and computes
    the simple slopes (effect of LogSizeRatio) when LocationAdv_binary == 0
    and when LocationAdv_binary == 1 (uses the interaction term).

    Returns:
      {
        "object": {
          "coef": { ... },                # raw coefficients & stats for relevant terms
          "simple_slopes": {              # effect of LogSizeRatio when location=0 and =1
            "location_0": { ... },
            "location_1": { ... }
          },
          "interaction": { ... }          # interaction term stats
        },
        "description": "..."              # plain-language interpretation
      }
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output  # statsmodels results wrapper expected
    params = res.params.copy()        # pd.Series
    bse = res.bse.copy()              # pd.Series
    pvalues = res.pvalues.copy()      # pd.Series
    ci = res.conf_int().copy()        # DataFrame with 0 and 1 columns

    # covariance may not be available for some result objects; handle gracefully
    try:
        cov = res.cov_params()
    except Exception:
        cov = None

    # Helper to safe-get a term name (handles slight naming differences)
    def find_term(possible_names):
        for name in possible_names:
            if name in params.index:
                return name
        return None

    # Expected base names
    name_log = find_term(['LogSizeRatio', 'LogSizeRatio[T.1]'])
    # Interaction term may be named in different ways; search generically
    inter_name = None
    for idx in params.index:
        if 'LogSizeRatio' in idx and 'LocationAdv_binary' in idx:
            inter_name = idx
            break
    # Also find location main effect if present (helpful for interpretation)
    loc_name = find_term(['LocationAdv_binary', 'LocationAdv_binary[T.1]'])

    results = {}

    # Extract stats for LogSizeRatio (main)
    if name_log is not None:
        coef_log = float(params[name_log])
        se_log = float(bse[name_log]) if name_log in bse.index else np.nan
        z_log = coef_log / se_log if se_log and (not np.isnan(se_log)) else np.nan
        p_log = float(pvalues[name_log]) if name_log in pvalues.index else np.nan
        try:
            ci_low, ci_high = float(ci.loc[name_log, 0]), float(ci.loc[name_log, 1])
        except Exception:
            ci_low, ci_high = np.nan, np.nan
        or_log = np.exp(coef_log)
        or_ci = (np.exp(ci_low) if not np.isnan(ci_low) else np.nan,
                 np.exp(ci_high) if not np.isnan(ci_high) else np.nan)
        results['coef'] = {
            'term': name_log,
            'coef': coef_log,
            'se': se_log,
            'z': z_log,
            'p': p_log,
            '95% CI coef': (ci_low, ci_high),
            'odds_ratio': or_log,
            '95% CI OR': or_ci
        }
    else:
        results['coef'] = {'error': 'LogSizeRatio term not found in model.'}

    # Extract interaction stats
    if inter_name is not None:
        coef_int = float(params[inter_name])
        se_int = float(bse[inter_name]) if inter_name in bse.index else np.nan
        z_int = coef_int / se_int if se_int and (not np.isnan(se_int)) else np.nan
        p_int = float(pvalues[inter_name]) if inter_name in pvalues.index else np.nan
        try:
            ci_low_i, ci_high_i = float(ci.loc[inter_name, 0]), float(ci.loc[inter_name, 1])
        except Exception:
            ci_low_i, ci_high_i = np.nan, np.nan
        or_int = np.exp(coef_int)
        or_ci_int = (np.exp(ci_low_i) if not np.isnan(ci_low_i) else np.nan,
                     np.exp(ci_high_i) if not np.isnan(ci_high_i) else np.nan)
        results['interaction'] = {
            'term': inter_name,
            'coef': coef_int,
            'se': se_int,
            'z': z_int,
            'p': p_int,
            '95% CI coef': (ci_low_i, ci_high_i),
            'odds_ratio': or_int,
            '95% CI OR': or_ci_int
        }
    else:
        # If no explicit interaction term found, set to zero and warn (model likely omitted it)
        results['interaction'] = {'error': 'Interaction term between LogSizeRatio and LocationAdv_binary not found.'}

    # Compute simple slopes: effect of LogSizeRatio when LocationAdv_binary == 0 and == 1
    simple = {}
    # slope at location = 0 is just the LogSizeRatio main effect (if present)
    if name_log is not None:
        simple['location_0'] = {
            'coef': results['coef']['coef'],
            'se': results['coef']['se'],
            'z': results['coef']['z'],
            'p': results['coef']['p'],
            'odds_ratio': results['coef']['odds_ratio'],
            '95% CI OR': results['coef']['95% CI OR']
        }
    else:
        simple['location_0'] = {'error': 'LogSizeRatio term missing; cannot compute slope for location=0.'}

    # slope at location = 1 is coef_log + coef_interaction; compute its SE using covariance if available
    if (name_log is not None) and (inter_name is not None):
        beta_log_val = float(params[name_log])
        beta_int_val = float(params[inter_name])
        beta_sum = beta_log_val + beta_int_val

        # Compute variance of the sum: use covariance if available, otherwise fall back to sum of variances (assume cov=0)
        try:
            if cov is not None and name_log in cov.index and inter_name in cov.index:
                var_sum = float(cov.loc[name_log, name_log]) + float(cov.loc[inter_name, inter_name]) + 2.0 * float(cov.loc[name_log, inter_name])
            else:
                # fallback: use bse^2 sums (assume covariance = 0)
                se_log_sq = float(bse[name_log]) ** 2 if name_log in bse.index else np.nan
                se_int_sq = float(bse[inter_name]) ** 2 if inter_name in bse.index else np.nan
                if np.isnan(se_log_sq) or np.isnan(se_int_sq):
                    var_sum = np.nan
                else:
                    var_sum = se_log_sq + se_int_sq
        except Exception:
            var_sum = np.nan

        se_sum = float(np.sqrt(var_sum)) if (not np.isnan(var_sum)) and (var_sum >= 0) else np.nan
        z_sum = float(beta_sum / se_sum) if se_sum and (not np.isnan(se_sum)) else np.nan
        p_sum = float(2.0 * (1.0 - norm.cdf(abs(z_sum)))) if (not np.isnan(z_sum)) else np.nan

        if not np.isnan(se_sum):
            ci_low_sum = float(beta_sum - 1.96 * se_sum)
            ci_high_sum = float(beta_sum + 1.96 * se_sum)
        else:
            ci_low_sum, ci_high_sum = np.nan, np.nan

        or_sum = float(np.exp(beta_sum)) if (not np.isnan(beta_sum)) else np.nan
        or_ci_sum = (float(np.exp(ci_low_sum)) if not np.isnan(ci_low_sum) else np.nan,
                     float(np.exp(ci_high_sum)) if not np.isnan(ci_high_sum) else np.nan)

        simple['location_1'] = {
            'coef': float(beta_sum),
            'se': se_sum,
            'z': z_sum,
            'p': p_sum,
            '95% CI coef': (ci_low_sum, ci_high_sum),
            'odds_ratio': or_sum,
            '95% CI OR': or_ci_sum
        }
    else:
        simple['location_1'] = {'error': 'Cannot compute slope for location=1 because LogSizeRatio or interaction term is missing.'}

    results['simple_slopes'] = simple

    # Optionally include location main effect stats for context
    if loc_name is not None:
        try:
            loc_ci_low, loc_ci_high = float(ci.loc[loc_name, 0]), float(ci.loc[loc_name, 1])
        except Exception:
            loc_ci_low, loc_ci_high = np.nan, np.nan
        loc_stats = {
            'term': loc_name,
            'coef': float(params[loc_name]),
            'se': float(bse[loc_name]) if loc_name in bse.index else np.nan,
            'p': float(pvalues[loc_name]) if loc_name in pvalues.index else np.nan,
            '95% CI coef': (loc_ci_low, loc_ci_high),
            'odds_ratio': float(np.exp(params[loc_name])),
            '95% CI OR': (float(np.exp(loc_ci_low)) if not np.isnan(loc_ci_low) else np.nan,
                          float(np.exp(loc_ci_high)) if not np.isnan(loc_ci_high) else np.nan)
        }
        results['LocationAdv_binary'] = loc_stats

    # Build description (concise)
    # Interpret whether LogSizeRatio increases focal group's winning probability, and whether this depends on location
    desc_lines = []
    if name_log is not None:
        coef = results['coef']['coef']
        pval = results['coef']['p']
        desc_lines.append(
            f"The coefficient for LogSizeRatio (effect when contest is closer to the other group's home-range, i.e. LocationAdv_binary=0) is {coef:.3f} (p = {pval:.3g})."
        )
    if inter_name is not None and 'coef' in results.get('interaction', {}):
        coef_i = results['interaction']['coef']
        p_i = results['interaction']['p']
        desc_lines.append(
            f"The interaction term {inter_name} has coefficient {coef_i:.3f} (p = {p_i:.3g}), which tests whether the effect of LogSizeRatio differs when the contest is nearer the focal group's home-range."
        )
    if 'location_0' in simple and 'error' not in simple['location_0']:
        or0 = simple['location_0']['odds_ratio']
        p0 = simple['location_0']['p']
        desc_lines.append(
            f"When the contest is nearer the other group's home-range (LocationAdv_binary=0), a one-unit increase in log size ratio multiplies the odds that the focal group wins by ~{or0:.3f} (p = {p0:.3g})."
        )
    if 'location_1' in simple and 'error' not in simple['location_1']:
        or1 = simple['location_1']['odds_ratio']
        p1 = simple['location_1']['p']
        desc_lines.append(
            f"When the contest is nearer the focal group's home-range (LocationAdv_binary=1), a one-unit increase in log size ratio multiplies the odds that the focal group wins by ~{or1:.3f} (p = {p1:.3g})."
        )

    # Final concise verdict about moderation
    interaction_p = results.get('interaction', {}).get('p', None)
    if (inter_name is not None) and (interaction_p is not None) and (not np.isnan(interaction_p)) and (interaction_p < 0.05):
        desc_lines.append("Conclusion: There is evidence that the location moderates the effect of relative group size (significant interaction).")
    elif inter_name is not None:
        desc_lines.append("Conclusion: No strong evidence that location moderates the effect of relative group size (interaction not statistically significant).")
    else:
        desc_lines.append("Conclusion: Interaction term not available in model output, cannot assess moderation formally.")

    description = " ".join(desc_lines)

    return {"object": results, "description": description}