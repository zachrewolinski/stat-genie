def extract_final_answer(model_output):
    """
    Extracts the estimated effect of 'children_binary' on 'affair_count' from a fitted
    statsmodels GLMResultsWrapper (Negative Binomial / Poisson). Also evaluates the
    interaction with gender ('children_gender_interaction') so we can report the effect
    for females (gender_male = 0) and males (gender_male = 1).

    Returns:
      {
        "object": {
          "children_binary": {coef, se, z, pvalue, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper, pct_change},
          "children_gender_interaction": {...},
          "children_effect_male": {...},   # combined coef (children + interaction) and stats
          "conclusion": "..."
        },
        "description": "Human readable interpretation..."
      }
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output  # alias

    # Ensure needed parameters are present
    params = res.params
    required = ['children_binary', 'children_gender_interaction']
    missing = [p for p in required if p not in params.index]
    if missing:
        raise ValueError(f"Required parameter(s) not found in model results: {missing}")

    # Helper to get conf int for a given param robustly
    ci_array = res.conf_int()
    def get_ci(param):
        # ci_array may be DataFrame or ndarray
        try:
            # If DataFrame
            row = ci_array.loc[param]
            return float(row[0]), float(row[1])
        except Exception:
            # Fallback to index-based
            idx = list(params.index).index(param)
            return float(ci_array[idx, 0]), float(ci_array[idx, 1])

    # Extract main components for children_binary
    coef_c = float(params['children_binary'])
    se_c = float(res.bse['children_binary'])
    z_c = coef_c / se_c if se_c > 0 else np.nan
    p_c = float(2 * (1 - norm.cdf(abs(z_c)))) if not np.isnan(z_c) else np.nan
    ci_lower_c, ci_upper_c = get_ci('children_binary')
    irr_c = float(np.exp(coef_c))
    irr_ci_lower_c = float(np.exp(ci_lower_c))
    irr_ci_upper_c = float(np.exp(ci_upper_c))
    pct_change_c = (irr_c - 1.0) * 100.0

    # Interaction parameter
    coef_int = float(params['children_gender_interaction'])
    se_int = float(res.bse['children_gender_interaction'])
    z_int = coef_int / se_int if se_int > 0 else np.nan
    p_int = float(2 * (1 - norm.cdf(abs(z_int)))) if not np.isnan(z_int) else np.nan
    ci_lower_int, ci_upper_int = get_ci('children_gender_interaction')
    irr_int = float(np.exp(coef_int))
    irr_ci_lower_int = float(np.exp(ci_lower_int))
    irr_ci_upper_int = float(np.exp(ci_upper_int))
    pct_change_int = (irr_int - 1.0) * 100.0

    # Combined effect for males: children_binary + children_gender_interaction
    coef_male = coef_c + coef_int
    # Var(sum) = var(c) + var(int) + 2*cov(c,int)
    cov = res.cov_params()
    try:
        var_c = float(cov.loc['children_binary', 'children_binary'])
        var_int = float(cov.loc['children_gender_interaction', 'children_gender_interaction'])
        cov_c_int = float(cov.loc['children_binary', 'children_gender_interaction'])
    except Exception:
        # fallback if cov is ndarray
        idx_c = list(params.index).index('children_binary')
        idx_int = list(params.index).index('children_gender_interaction')
        var_c = float(cov[idx_c, idx_c])
        var_int = float(cov[idx_int, idx_int])
        cov_c_int = float(cov[idx_c, idx_int])

    var_male = var_c + var_int + 2.0 * cov_c_int
    se_male = float(np.sqrt(var_male)) if var_male >= 0 else np.nan
    z_male = coef_male / se_male if se_male > 0 else np.nan
    p_male = float(2 * (1 - norm.cdf(abs(z_male)))) if not np.isnan(z_male) else np.nan

    # Confidence interval for combined coef (approx using normal)
    # Use coef_male +/- 1.96 * se_male
    ci_lower_male = float(coef_male - 1.96 * se_male) if not np.isnan(se_male) else np.nan
    ci_upper_male = float(coef_male + 1.96 * se_male) if not np.isnan(se_male) else np.nan
    irr_male = float(np.exp(coef_male))
    irr_ci_lower_male = float(np.exp(ci_lower_male)) if not np.isnan(ci_lower_male) else np.nan
    irr_ci_upper_male = float(np.exp(ci_upper_male)) if not np.isnan(ci_upper_male) else np.nan
    pct_change_male = (irr_male - 1.0) * 100.0

    # Build output object
    out = {
        'children_binary': {
            'coef': coef_c,
            'se': se_c,
            'z': z_c,
            'pvalue': p_c,
            'ci_lower': ci_lower_c,
            'ci_upper': ci_upper_c,
            'irr': irr_c,
            'irr_ci_lower': irr_ci_lower_c,
            'irr_ci_upper': irr_ci_upper_c,
            'pct_change': pct_change_c
        },
        'children_gender_interaction': {
            'coef': coef_int,
            'se': se_int,
            'z': z_int,
            'pvalue': p_int,
            'ci_lower': ci_lower_int,
            'ci_upper': ci_upper_int,
            'irr': irr_int,
            'irr_ci_lower': irr_ci_lower_int,
            'irr_ci_upper': irr_ci_upper_int,
            'pct_change': pct_change_int
        },
        'children_effect_male': {
            'coef': coef_male,
            'se': se_male,
            'z': z_male,
            'pvalue': p_male,
            'ci_lower': ci_lower_male,
            'ci_upper': ci_upper_male,
            'irr': irr_male,
            'irr_ci_lower': irr_ci_lower_male,
            'irr_ci_upper': irr_ci_upper_male,
            'pct_change': pct_change_male
        }
    }

    # Interpret results in plain language
    def interpret_entry(entry, label):
        sig = entry['pvalue'] < 0.05 if (entry['pvalue'] == entry['pvalue']) else False
        direction = 'decrease' if entry['coef'] < 0 else ('increase' if entry['coef'] > 0 else 'no change')
        pct = entry['pct_change']
        irr = entry['irr']
        if np.isnan(entry['pvalue']):
            return f"For {label}: estimated log-IRR = {entry['coef']:.4f} (IRR = {irr:.3f}), approximate percent change = {pct:.2f}%. p-value unavailable."
        if sig:
            return (f"For {label}: coefficient = {entry['coef']:.4f} (IRR = {irr:.3f}), which corresponds to an "
                    f"{abs(pct):.2f}% {direction} in expected affair count. This effect is statistically significant (p = {entry['pvalue']:.3g}).")
        else:
            return (f"For {label}: coefficient = {entry['coef']:.4f} (IRR = {irr:.3f}), approximately {pct:.2f}% {direction}, "
                    f"but this effect is NOT statistically significant (p = {entry['pvalue']:.3g}).")

    interp_female = interpret_entry(out['children_binary'], "females (gender_male=0)")
    interp_male = interpret_entry(out['children_effect_male'], "males (gender_male=1)")

    # Overall conclusion logic
    female_sig = (out['children_binary']['pvalue'] < 0.05) if (out['children_binary']['pvalue'] == out['children_binary']['pvalue']) else False
    male_sig = (out['children_effect_male']['pvalue'] < 0.05) if (out['children_effect_male']['pvalue'] == out['children_effect_male']['pvalue']) else False

    if female_sig and out['children_binary']['coef'] < 0 and male_sig and out['children_effect_male']['coef'] < 0:
        overall = ("Yes — having children is associated with a statistically significant decrease in reported extramarital affairs for both females "
                   f"and males. Females: {out['children_binary']['pct_change']:.2f}% change; Males: {out['children_effect_male']['pct_change']:.2f}% change.")
    elif female_sig and out['children_binary']['coef'] < 0 and (not male_sig or out['children_effect_male']['coef'] >= 0):
        overall = ("Partially: having children is associated with a significant decrease in affairs for females (gender_male=0), "
                   "but there is no significant evidence of a decrease for males.")
    elif male_sig and out['children_effect_male']['coef'] < 0 and (not female_sig or out['children_binary']['coef'] >= 0):
        overall = ("Partially: having children is associated with a significant decrease in affairs for males (gender_male=1), "
                   "but there is no significant evidence of a decrease for females.")
    else:
        overall = ("No clear evidence that having children decreases engagement in extramarital affairs in this model. "
                   "Estimated effects may be small or not statistically significant; the effect may differ by gender.")

    description = (
        "Extracted statistics for the effect of having children on reported affair counts (Negative Binomial GLM).\n"
        + interp_female + "\n" + interp_male + "\n\nOverall conclusion: " + overall
    )

    return {"object": out, "description": description}