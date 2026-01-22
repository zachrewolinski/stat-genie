def extract_final_answer(model_output):
    """
    Extract relevant statistics from a fitted statsmodels GLM (logistic) result object.
    
    Returns a dictionary with keys:
      - "object": dict of extracted numeric results (coefficients, SEs, z, p, odds ratios, 95% CIs)
      - "description": short textual interpretation of the extracted results in the context
                       of the question (how relative group size and location advantage
                       influence the probability that the focal group wins).
    """
    import numpy as np
    from scipy import stats
    
    res = model_output  # expected to be a statsmodels GLMResultsWrapper
    
    params = res.params
    cov = res.cov_params()
    conf = res.conf_int()
    idx = list(params.index)
    
    # Helper to find a parameter name containing all tokens (robust to factor naming)
    def find_param(*tokens):
        for name in idx:
            if all(t in name for t in tokens):
                return name
        return None
    
    # Locate parameter names
    name_size = find_param('size_diff_z')
    name_loc = find_param('loc_adv_z')
    name_inter = find_param('size_diff_z', 'homefield')  # interaction term
    
    if name_size is None or name_loc is None:
        raise ValueError("Could not find expected parameter names 'size_diff_z' or 'loc_adv_z' in the model parameters: {}".format(idx))
    
    def summarize_param(name):
        coef = float(params[name])
        se = float(np.sqrt(cov.loc[name, name]))
        z = coef / se if se != 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low, ci_high = map(float, conf.loc[name].values)
        or_ = float(np.exp(coef))
        or_ci = [float(np.exp(ci_low)), float(np.exp(ci_high))]
        return {
            'param_name': name,
            'coef': coef,
            'se': se,
            'z': z,
            'p_value': p,
            'ci95': [ci_low, ci_high],
            'odds_ratio': or_,
            'odds_ratio_ci95': or_ci
        }
    
    size_summary = summarize_param(name_size)
    loc_summary = summarize_param(name_loc)
    
    # Combined effect of size when homefield == 1 (if interaction exists)
    combined_summary = None
    if name_inter is not None:
        coef_size = params[name_size]
        coef_inter = params[name_inter]
        coef_comb = float(coef_size + coef_inter)
        # variance of sum: var(a)+var(b)+2cov(a,b)
        var_comb = float(cov.loc[name_size, name_size] + cov.loc[name_inter, name_inter] + 2 * cov.loc[name_size, name_inter])
        se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else float('nan')
        z_comb = coef_comb / se_comb if se_comb != 0 else np.nan
        p_comb = 2 * (1 - stats.norm.cdf(abs(z_comb))) if not np.isnan(z_comb) else np.nan
        ci_low_comb = coef_comb - 1.96 * se_comb
        ci_high_comb = coef_comb + 1.96 * se_comb
        or_comb = float(np.exp(coef_comb))
        or_ci_comb = [float(np.exp(ci_low_comb)), float(np.exp(ci_high_comb))]
        combined_summary = {
            'param_name': f"{name_size} + {name_inter}  (effect of size_diff_z when homefield=1)",
            'coef': float(coef_comb),
            'se': se_comb,
            'z': z_comb,
            'p_value': p_comb,
            'ci95': [ci_low_comb, ci_high_comb],
            'odds_ratio': or_comb,
            'odds_ratio_ci95': or_ci_comb
        }
    
    # Build a readable description string
    desc_lines = []
    desc_lines.append("Extracted results from logistic regression predicting focal-group win (1 = focal won):")
    desc_lines.append(f"- Relative group size (parameter '{size_summary['param_name']}'): coef = {size_summary['coef']:.4f}, SE = {size_summary['se']:.4f}, z = {size_summary['z']:.3f}, p = {size_summary['p_value']:.4g}")
    desc_lines.append(f"  95% CI (coef): [{size_summary['ci95'][0]:.4f}, {size_summary['ci95'][1]:.4f}]; OR = {size_summary['odds_ratio']:.3f}, OR 95% CI = [{size_summary['odds_ratio_ci95'][0]:.3f}, {size_summary['odds_ratio_ci95'][1]:.3f}]")
    if combined_summary is not None:
        desc_lines.append(f"- Interaction: (size_diff_z : homefield) found as '{name_inter}'.")
        desc_lines.append(f"  Effect of relative size when focal is at home (homefield=1): coef = {combined_summary['coef']:.4f}, SE = {combined_summary['se']:.4f}, z = {combined_summary['z']:.3f}, p = {combined_summary['p_value']:.4g}")
        desc_lines.append(f"  95% CI (coef): [{combined_summary['ci95'][0]:.4f}, {combined_summary['ci95'][1]:.4f}]; OR = {combined_summary['odds_ratio']:.3f}, OR 95% CI = [{combined_summary['odds_ratio_ci95'][0]:.3f}, {combined_summary['odds_ratio_ci95'][1]:.3f}]")
    else:
        desc_lines.append("- No interaction term between size_diff_z and homefield was found in the model parameters.")
    desc_lines.append(f"- Location advantage (parameter '{loc_summary['param_name']}'): coef = {loc_summary['coef']:.4f}, SE = {loc_summary['se']:.4f}, z = {loc_summary['z']:.3f}, p = {loc_summary['p_value']:.4g}")
    desc_lines.append(f"  95% CI (coef): [{loc_summary['ci95'][0]:.4f}, {loc_summary['ci95'][1]:.4f}]; OR = {loc_summary['odds_ratio']:.3f}, OR 95% CI = [{loc_summary['odds_ratio_ci95'][0]:.3f}, {loc_summary['odds_ratio_ci95'][1]:.3f}]")
    desc_lines.append("")
    desc_lines.append("Interpretation notes:")
    desc_lines.append("- A positive coefficient for size_diff_z means that when the focal group is relatively larger, the log-odds of winning increase. The odds ratio > 1 indicates the multiplicative change in odds for a one SD increase in relative size.")
    desc_lines.append("- The combined effect (size + interaction) gives the effect of size when the focal group is on its homefield (homefield=1).")
    desc_lines.append("- A positive coefficient for loc_adv_z means that being relatively closer to home increases the probability of winning; odds ratios interpret similarly.")
    desc = "\n".join(desc_lines)
    
    # Pack numeric outputs into "object"
    result_object = {
        'size_summary': size_summary,
        'loc_summary': loc_summary,
        'size_when_homefield_summary': combined_summary,
        # include which parameter names were used (helpful for traceability)
        'param_names_used': {
            'size_param': name_size,
            'loc_param': name_loc,
            'interaction_param': name_inter
        }
    }
    
    return {
        "object": result_object,
        "description": desc
    }