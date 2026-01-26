def extract_final_answer(model_output):
    """
    Extracts coefficients, robust SEs, p-values, confidence intervals, odds-ratios,
    and simple-slope tests for the effect of relative group size and its interaction
    with location advantage from a fitted statsmodels GLM results object stored in
    model_output['results_clustered'].

    Returns a dictionary with keys:
      - "object": dict containing numeric results (coef table, ORs, simple slopes)
      - "description": human-readable interpretation / conclusion about whether
                       relative group size and/or location influence contest outcome.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Pull results object
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary (the function's returned object).")
    if 'results_clustered' not in model_output:
        raise ValueError("model_output missing 'results_clustered' key.")
    res = model_output['results_clustered']

    # Extract parameter table (robust results wrapper should expose these)
    try:
        params = res.params.copy()
        bse = res.bse.copy()
        pvalues = res.pvalues.copy()
        conf = res.conf_int()  # DataFrame-like with two cols
        cov = res.cov_params()
    except Exception as e:
        raise RuntimeError(f"Could not extract statistics from results object: {e}")

    # Identify parameter names of interest
    name_size = 'z_RelSize_log'
    name_loc = 'z_LocationAdv'
    # interaction may be named as 'z_RelSize_log:z_LocationAdv' (typical for patsy/statsmodels)
    # try to find the actual interaction name present
    param_names = list(params.index.astype(str))
    interaction_name = None
    for nm in param_names:
        if (name_size in nm) and (name_loc in nm) and (nm != name_size) and (nm != name_loc):
            interaction_name = nm
            break
    if interaction_name is None:
        # fallback to explicit candidate
        candidate = f'{name_size}:{name_loc}'
        if candidate in param_names:
            interaction_name = candidate

    # Helper to safely extract a term's stats if present
    def get_term_stats(term):
        if term not in params.index:
            return None
        coef = float(params.loc[term])
        se = float(bse.loc[term])
        p = float(pvalues.loc[term])
        ci_low, ci_high = float(conf.loc[term, 0]), float(conf.loc[term, 1])
        or_est = float(np.exp(coef))
        or_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))
        return {
            'term': term,
            'coef': coef,
            'se': se,
            'p_value': p,
            'ci_95': (ci_low, ci_high),
            'odds_ratio': or_est,
            'odds_ratio_95ci': or_ci
        }

    stats_size = get_term_stats(name_size)
    stats_loc = get_term_stats(name_loc)
    stats_inter = get_term_stats(interaction_name) if interaction_name is not None else None

    # Prepare coefficient table
    coef_table = {}
    if stats_size:
        coef_table[name_size] = stats_size
    if stats_loc:
        coef_table[name_loc] = stats_loc
    if stats_inter:
        coef_table[interaction_name] = stats_inter

    # Simple-slope test: effect of relative size at selected values of location advantage
    # (location advantage is standardized, so use -1, 0, +1 SD)
    simple_slopes = {}
    if stats_size:
        beta_size = stats_size['coef']
        var_size = float(cov.loc[name_size, name_size]) if name_size in cov.index else np.nan

        if stats_inter:
            beta_inter = stats_inter['coef']
            # cov between size and interaction
            cov_si = float(cov.loc[name_size, interaction_name]) if (name_size in cov.index and interaction_name in cov.index) else np.nan
            var_inter = float(cov.loc[interaction_name, interaction_name]) if interaction_name in cov.index else np.nan

            for adv in [-1.0, 0.0, 1.0]:
                # slope = beta_size + adv * beta_inter
                slope = beta_size + adv * beta_inter
                # variance (delta method)
                # Var(slope) = Var(beta_size) + adv^2 Var(beta_inter) + 2*adv*Cov(beta_size,beta_inter)
                var_slope = var_size + (adv**2) * var_inter + 2.0 * adv * cov_si
                se_slope = np.sqrt(var_slope) if var_slope >= 0 else np.nan
                z = slope / se_slope if (se_slope and not np.isnan(se_slope)) else np.nan
                p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
                ci_low = slope - 1.96 * se_slope if not np.isnan(se_slope) else np.nan
                ci_high = slope + 1.96 * se_slope if not np.isnan(se_slope) else np.nan
                simple_slopes[f'z_RelSize_at_z_LocationAdv_{adv:+.1f}'] = {
                    'adv_value': adv,
                    'slope_logit': slope,
                    'se': se_slope,
                    'z': z,
                    'p_value': p,
                    'ci_95_logit': (ci_low, ci_high),
                    'odds_ratio': float(np.exp(slope)) if not np.isnan(slope) else np.nan,
                    'odds_ratio_95ci': (float(np.exp(ci_low)), float(np.exp(ci_high))) if not np.isnan(ci_low) else (np.nan, np.nan)
                }
        else:
            # No interaction: simple slope is just beta_size (same for all adv)
            slope = beta_size
            se_slope = np.sqrt(var_size) if not np.isnan(var_size) else np.nan
            z = slope / se_slope if (se_slope and not np.isnan(se_slope)) else np.nan
            p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ci_low = slope - 1.96 * se_slope if not np.isnan(se_slope) else np.nan
            ci_high = slope + 1.96 * se_slope if not np.isnan(se_slope) else np.nan
            simple_slopes['z_RelSize_at_z_LocationAdv_any'] = {
                'adv_value': None,
                'slope_logit': slope,
                'se': se_slope,
                'z': z,
                'p_value': p,
                'ci_95_logit': (ci_low, ci_high),
                'odds_ratio': float(np.exp(slope)) if not np.isnan(slope) else np.nan,
                'odds_ratio_95ci': (float(np.exp(ci_low)), float(np.exp(ci_high))) if not np.isnan(ci_low) else (np.nan, np.nan)
            }

    # Summarize / form conclusion
    alpha = 0.05
    messages = []
    if stats_inter:
        if stats_inter['p_value'] < alpha:
            messages.append("There is a statistically significant interaction between relative group size and location advantage (p < 0.05). "
                            "This means the effect of relative group size on win probability depends on location advantage.")
        else:
            messages.append("The interaction between relative group size and location advantage is not statistically significant (p >= 0.05).")
    # main effects
    if stats_size:
        if stats_size['p_value'] < alpha:
            messages.append("Relative group size has a statistically significant main effect on win probability (p < 0.05): larger focal groups are more likely to win overall.")
        else:
            messages.append("Relative group size does not show a statistically significant main effect (p >= 0.05).")
    if stats_loc:
        if stats_loc['p_value'] < alpha:
            messages.append("Location advantage has a statistically significant main effect on win probability (p < 0.05): contests closer to the focal group's center increase their win probability.")
        else:
            messages.append("Location advantage does not show a statistically significant main effect (p >= 0.05).")

    # Check simple slopes for practical significance (any significant slopes?)
    sig_slopes = []
    for k, v in simple_slopes.items():
        if ('p_value' in v) and (not np.isnan(v['p_value'])) and (v['p_value'] < alpha):
            sig_slopes.append((k, v))
    if sig_slopes:
        messages.append("Simple-slope tests show that the effect of relative group size is statistically significant at some values of location advantage (see simple_slopes results).")
    else:
        messages.append("Simple-slope tests do not show statistically significant effects of relative group size at the tested location-advantage values.")

    conclusion = " ".join(messages)

    result_object = {
        'coef_table': coef_table,
        'simple_slopes': simple_slopes,
        'cov_matrix': cov,  # might be large; included for completeness
        'conclusion': conclusion
    }

    description = (
        "This output contains coefficient estimates (log-odds scale), robust standard errors, p-values, "
        "95% confidence intervals and odds-ratios for the main predictors (relative group size, location advantage) "
        "and their interaction (if present). It also includes simple-slope tests for the effect of relative group "
        "size at location advantage = -1, 0, +1 standard deviations (when an interaction is present). "
        "The 'conclusion' string summarizes whether the main effects or their interaction are statistically significant "
        "at alpha = 0.05 and whether any simple slopes are significant."
    )

    return {
        'object': result_object,
        'description': description + " " + conclusion
    }