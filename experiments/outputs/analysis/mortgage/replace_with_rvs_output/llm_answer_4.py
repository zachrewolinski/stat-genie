def extract_final_answer(model_output):
    """
    Extracts statistics on the effect of gender (female) on mortgage acceptance
    from the model_output produced by the modeling function.

    Returns:
      {
        "object": {
          "female": {coef_log_odds, se, z, p_value, OR, CI_lower_OR, CI_upper_OR, significant_at_0.05},
          "female_black_interaction": {...},
          "female_effect_for_black_applicants": {...}  # combined effect (female + interaction)
        },
        "description": <text explanation of what these numbers mean>
      }
    """
    import numpy as np
    from scipy import stats

    # Expecting model_output to be the dict returned by the model() function
    robust_res = model_output.get('robust_result')
    or_table = model_output.get('odds_ratios_table')

    if robust_res is None:
        raise ValueError("model_output does not contain 'robust_result'")

    # Obtain parameter values
    params_raw = getattr(robust_res, 'params', None)
    # Some result objects provide names via param_names or model.exog_names
    if hasattr(params_raw, 'index'):
        param_names = list(params_raw.index)
        params_dict = {name: float(params_raw[name]) for name in param_names}
    else:
        # params_raw may be ndarray
        if hasattr(robust_res, 'param_names'):
            param_names = list(robust_res.param_names)
        elif hasattr(robust_res, 'model') and hasattr(robust_res.model, 'exog_names'):
            param_names = list(robust_res.model.exog_names)
        else:
            # fallback: try to infer length from params_raw
            try:
                length = len(params_raw)
            except Exception:
                param_names = []
            else:
                param_names = [str(i) for i in range(length)]
        if params_raw is None:
            params_dict = {}
        else:
            try:
                params_list = list(params_raw)
            except Exception:
                params_list = []
            params_dict = {name: float(params_list[i]) if i < len(params_list) else float('nan') for i, name in enumerate(param_names)}

    # p-values: robust_res.pvalues might be Series or ndarray
    pvalues_raw = getattr(robust_res, 'pvalues', None)
    pvals_dict = {}
    if pvalues_raw is not None:
        if hasattr(pvalues_raw, 'index'):
            for name in pvalues_raw.index:
                try:
                    pvals_dict[name] = float(pvalues_raw[name])
                except Exception:
                    pvals_dict[name] = float('nan')
        else:
            # ndarray or list-like
            try:
                p_list = list(pvalues_raw)
            except Exception:
                p_list = []
            for i, name in enumerate(param_names):
                pvals_dict[name] = float(p_list[i]) if i < len(p_list) else float('nan')

    # Standard errors: try bse, then se
    se_raw = getattr(robust_res, 'bse', None)
    if se_raw is None:
        se_raw = getattr(robust_res, 'se', None)
    se_dict = {}
    if se_raw is not None:
        if hasattr(se_raw, 'index'):
            for name in se_raw.index:
                try:
                    se_dict[name] = float(se_raw[name])
                except Exception:
                    se_dict[name] = float('nan')
        else:
            try:
                se_list = list(se_raw)
            except Exception:
                se_list = []
            for i, name in enumerate(param_names):
                se_dict[name] = float(se_list[i]) if i < len(se_list) else float('nan')

    # Confidence intervals: robust_res.conf_int() may be callable; try to get DataFrame/array
    conf = None
    try:
        if hasattr(robust_res, 'conf_int') and callable(robust_res.conf_int):
            conf = robust_res.conf_int()
        else:
            conf = getattr(robust_res, 'conf_int', None)
    except Exception:
        conf = None

    # Covariance matrix: try cov_params(), then cov, then covariance attribute
    cov = None
    try:
        if hasattr(robust_res, 'cov_params') and callable(robust_res.cov_params):
            cov = robust_res.cov_params()
        else:
            cov = getattr(robust_res, 'cov', None)
            if cov is None:
                cov = getattr(robust_res, 'cov_params', None)
    except Exception:
        cov = getattr(robust_res, 'cov', None)

    # Normalize cov to numpy array if it's a DataFrame-like
    cov_array = None
    if cov is not None:
        try:
            if hasattr(cov, 'values'):
                cov_array = np.asarray(cov.values)
            else:
                cov_array = np.asarray(cov)
        except Exception:
            cov_array = None

    # Normalize conf to a structure we can index by name
    conf_is_df = False
    conf_array = None
    conf_dict = {}
    if conf is not None:
        if hasattr(conf, 'loc') and hasattr(conf, 'columns'):
            # DataFrame-like
            conf_is_df = True
        else:
            try:
                conf_array = np.asarray(conf)
            except Exception:
                conf_array = None
        # If conf_array and param_names available, build dict
        if conf_array is not None and len(conf_array) == len(param_names):
            for i, name in enumerate(param_names):
                try:
                    low = float(conf_array[i, 0])
                    high = float(conf_array[i, 1])
                except Exception:
                    low = float('nan')
                    high = float('nan')
                conf_dict[name] = (low, high)
        elif conf_is_df:
            # fill conf_dict from DataFrame-like using loc
            for name in param_names:
                try:
                    row = conf.loc[name]
                    low = float(row.iloc[0])
                    high = float(row.iloc[1])
                except Exception:
                    low = float('nan')
                    high = float('nan')
                conf_dict[name] = (low, high)

    # Helper to get index of a parameter
    def index_of(name):
        try:
            return param_names.index(name)
        except ValueError:
            return None

    def get_conf(name):
        if name in conf_dict:
            return conf_dict[name]
        if conf_is_df:
            try:
                row = conf.loc[name]
                return (float(row.iloc[0]), float(row.iloc[1]))
            except Exception:
                return (float('nan'), float('nan'))
        if conf_array is not None:
            idx = index_of(name)
            if idx is not None and idx < conf_array.shape[0]:
                try:
                    return (float(conf_array[idx, 0]), float(conf_array[idx, 1]))
                except Exception:
                    return (float('nan'), float('nan'))
        return (float('nan'), float('nan'))

    # Build term stats robustly
    def term_stats(term):
        if term not in param_names:
            return None
        coef = float(params_dict.get(term, float('nan')))
        # standard error: prefer se_dict if present, else sqrt of cov diagonal
        if term in se_dict:
            se = float(se_dict[term])
        elif cov_array is not None:
            idx = index_of(term)
            if idx is not None and idx < cov_array.shape[0]:
                try:
                    se = float(np.sqrt(cov_array[idx, idx]))
                except Exception:
                    se = float('nan')
            else:
                se = float('nan')
        else:
            se = float('nan')
        z = float(coef / se) if (se and not np.isnan(se) and se != 0) else float('nan')
        p = float(pvals_dict.get(term, float('nan')))
        ci_low, ci_high = get_conf(term)
        OR = float(np.exp(coef)) if (coef is not None and not np.isnan(coef)) else float('nan')
        OR_low = float(np.exp(ci_low)) if (not np.isnan(ci_low)) else float('nan')
        OR_high = float(np.exp(ci_high)) if (not np.isnan(ci_high)) else float('nan')
        significant = (p < 0.05) if (not np.isnan(p)) else False
        return {
            'coef_log_odds': coef,
            'se': se,
            'z': z,
            'p_value': p,
            'OR': OR,
            'CI_lower_OR': OR_low,
            'CI_upper_OR': OR_high,
            'significant_at_0.05': bool(significant)
        }

    female_stats = term_stats('female')
    female_black_stats = term_stats('female_black')

    # Combined effect for Black applicants: coef_sum = female + female_black
    combined_stats = None
    if ('female' in param_names) and ('female_black' in param_names):
        coef_f = float(params_dict.get('female', float('nan')))
        coef_fb = float(params_dict.get('female_black', float('nan')))
        coef_sum = coef_f + coef_fb
        # compute variance of sum if covariance matrix available
        se_sum = float('nan')
        if cov_array is not None:
            idx_f = index_of('female')
            idx_fb = index_of('female_black')
            if idx_f is not None and idx_fb is not None and idx_f < cov_array.shape[0] and idx_fb < cov_array.shape[0]:
                try:
                    var_sum = cov_array[idx_f, idx_f] + cov_array[idx_fb, idx_fb] + 2.0 * cov_array[idx_f, idx_fb]
                    se_sum = float(np.sqrt(var_sum)) if var_sum >= 0 else float('nan')
                except Exception:
                    se_sum = float('nan')
        z_sum = float(coef_sum / se_sum) if (not np.isnan(se_sum) and se_sum != 0) else float('nan')
        p_sum = float(2 * (1 - stats.norm.cdf(abs(z_sum)))) if (not np.isnan(z_sum)) and (not np.isnan(se_sum)) else float('nan')
        if not np.isnan(se_sum):
            ci_low_sum = float(coef_sum - stats.norm.ppf(0.975) * se_sum)
            ci_high_sum = float(coef_sum + stats.norm.ppf(0.975) * se_sum)
        else:
            ci_low_sum = float('nan')
            ci_high_sum = float('nan')
        OR_sum = float(np.exp(coef_sum)) if (coef_sum is not None and not np.isnan(coef_sum)) else float('nan')
        OR_low_sum = float(np.exp(ci_low_sum)) if (not np.isnan(ci_low_sum)) else float('nan')
        OR_high_sum = float(np.exp(ci_high_sum)) if (not np.isnan(ci_high_sum)) else float('nan')
        combined_stats = {
            'coef_log_odds': coef_sum,
            'se': se_sum,
            'z': z_sum,
            'p_value': p_sum,
            'OR': OR_sum,
            'CI_lower_OR': OR_low_sum,
            'CI_upper_OR': OR_high_sum,
            'significant_at_0.05': (p_sum < 0.05) if (not np.isnan(p_sum)) else False,
            'note': "This is the effect of being female (vs male) for applicants who are Black (female + female_black interaction)."
        }

    # Build a concise object to return
    result_object = {
        'female': female_stats,
        'female_black_interaction': female_black_stats,
        'female_effect_for_black_applicants': combined_stats
    }

    # Short interpretation text
    description_lines = []
    description_lines.append(
        "Interpretation: The model includes a gender main effect ('female') and an interaction ('female_black'). "
        "The 'female' coefficient describes the effect of being female (vs male) when black==0 (i.e., non-Black applicants)."
    )
    description_lines.append(
        "For Black applicants the effect of female is the sum of 'female' and 'female_black' (provided above as 'female_effect_for_black_applicants')."
    )
    # Add succinct numeric summary if available
    if female_stats is not None:
        try:
            OR = female_stats.get('OR', float('nan'))
            CI_low = female_stats.get('CI_lower_OR', float('nan'))
            CI_high = female_stats.get('CI_upper_OR', float('nan'))
            pval = female_stats.get('p_value', float('nan'))
            sig = female_stats.get('significant_at_0.05', False)
            description_lines.append(
                f"Non-Black applicants: OR = {OR:.3f} "
                f"(95% CI: {CI_low:.3f} - {CI_high:.3f}), "
                f"p = {pval:.3f}. "
                f"{'Statistically significant.' if sig else 'Not statistically significant.'}"
            )
        except Exception:
            # If formatting fails, provide a fallback brief note
            description_lines.append("Non-Black applicants: statistics available in the 'object' output.")
    if combined_stats is not None:
        try:
            OR = combined_stats.get('OR', float('nan'))
            CI_low = combined_stats.get('CI_lower_OR', float('nan'))
            CI_high = combined_stats.get('CI_upper_OR', float('nan'))
            pval = combined_stats.get('p_value', float('nan'))
            sig = combined_stats.get('significant_at_0.05', False)
            ptext = f"p = {pval:.3f}" if not np.isnan(pval) else "p = NA (covariance missing)"
            sig_text = "Statistically significant." if sig else "Not statistically significant."
            description_lines.append(
                f"Black applicants: OR = {OR:.3f} "
                f"(95% CI: {CI_low:.3f} - {CI_high:.3f}), {ptext}. {sig_text}"
            )
        except Exception:
            description_lines.append("Black applicants: combined statistics available in the 'object' output.")

    description_lines.append(
        "Conclusion: Based on the model output, interpret the reported odds ratios and p-values to assess whether gender "
        "is associated with mortgage acceptance for non-Black applicants and whether the gender effect differs for Black applicants (interaction)."
    )

    description = " ".join(description_lines)

    return {"object": result_object, "description": description}