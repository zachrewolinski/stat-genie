def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals,
    odds ratios, and tests the moderated effect of relative group size by location
    from a fitted statsmodels GLM (clustered-robust) result object.

    Returns a dict with keys:
      - "object": dict with extracted numeric results for focal predictors and marginal effects
      - "description": human-readable interpretation of those results
    """
    import numpy as np
    import math

    # Names of focal predictors
    focal_vars = ['size_diff_z', 'focal_closer', 'size_x_loc']

    # Prepare containers
    vars_stats = {}
    missing_vars = []

    # Access basic outputs (params, bse, pvalues, conf_int, cov_params)
    try:
        params = model_output.params  # pandas Series
        pvalues = model_output.pvalues
        bse = model_output.bse
        conf = model_output.conf_int()  # DataFrame with [lower, upper]
        cov = model_output.cov_params()  # covariance matrix DataFrame
    except Exception as e:
        raise ValueError(f"Model output does not have expected attributes: {e}")

    # Extract stats for focal variables (if present)
    for v in focal_vars:
        if v in params.index:
            coef = float(params.loc[v])
            se = float(bse.loc[v]) if v in bse.index else None
            p = float(pvalues.loc[v]) if v in pvalues.index else None
            ci_lower = float(conf.loc[v, 0]) if (v in conf.index) else None
            ci_upper = float(conf.loc[v, 1]) if (v in conf.index) else None
            # odds ratio and CI
            try:
                or_val = float(math.exp(coef))
                or_ci_lower = float(math.exp(ci_lower)) if ci_lower is not None else None
                or_ci_upper = float(math.exp(ci_upper)) if ci_upper is not None else None
            except Exception:
                or_val = or_ci_lower = or_ci_upper = None

            vars_stats[v] = {
                'coef': coef,
                'se': se,
                'p_value': p,
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'odds_ratio': or_val,
                'odds_ratio_ci_95_lower': or_ci_lower,
                'odds_ratio_ci_95_upper': or_ci_upper,
                'significant_at_0.05': (p is not None and p < 0.05)
            }
        else:
            missing_vars.append(v)

    # Calculate marginal effect of size_diff_z when focal_closer = 0 and =1
    marginal = {}
    if 'size_diff_z' in params.index:
        beta_size = float(params.loc['size_diff_z'])
        # base effect when focal_closer = 0
        marginal['size_when_focal_closer_0'] = {
            'log_odds_coef': beta_size,
            'odds_ratio': float(math.exp(beta_size))
        }
        # effect when focal_closer = 1: beta_size + beta_interaction
        if 'size_x_loc' in params.index:
            beta_int = float(params.loc['size_x_loc'])
            combined_coef = beta_size + beta_int
            # compute SE for combined coef using covariance matrix if available
            try:
                # cov may be DataFrame or ndarray; prefer DataFrame with labeled rows
                if hasattr(cov, 'loc'):
                    var_sum = float(cov.loc['size_diff_z', 'size_diff_z']) + \
                              float(cov.loc['size_x_loc', 'size_x_loc']) + \
                              2.0 * float(cov.loc['size_diff_z', 'size_x_loc'])
                else:
                    # fallback: try to index by position
                    cov_arr = np.asarray(cov)
                    idx = list(params.index).index('size_diff_z')
                    jdx = list(params.index).index('size_x_loc')
                    var_sum = float(cov_arr[idx, idx] + cov_arr[jdx, jdx] + 2.0 * cov_arr[idx, jdx])
                se_combined = math.sqrt(max(var_sum, 0.0))
                ci_low = combined_coef - 1.96 * se_combined
                ci_high = combined_coef + 1.96 * se_combined
                or_combined = float(math.exp(combined_coef))
                or_ci_low = float(math.exp(ci_low))
                or_ci_high = float(math.exp(ci_high))
            except Exception:
                # if covariance unavailable, return coef-only
                se_combined = None
                ci_low = ci_high = or_combined = or_ci_low = or_ci_high = None

            marginal['size_when_focal_closer_1'] = {
                'log_odds_coef': combined_coef,
                'se': se_combined,
                'ci_95_lower': ci_low,
                'ci_95_upper': ci_high,
                'odds_ratio': or_combined,
                'odds_ratio_ci_95_lower': or_ci_low,
                'odds_ratio_ci_95_upper': or_ci_high,
                'interaction_coef': beta_int,
                'interaction_significant_at_0.05': (('size_x_loc' in params.index) and
                                                   (float(pvalues.loc['size_x_loc']) < 0.05))
            }
        else:
            marginal['size_when_focal_closer_1'] = None
    else:
        marginal = None

    # Prepare a concise human-readable description dynamically using p-values
    def sig_text(p):
        if p is None:
            return "p-value unavailable"
        return f"p = {p:.3f} ({'significant' if p < 0.05 else 'not significant'} at α=0.05)"

    desc_lines = []
    desc_lines.append("Extracted model estimates for predictors of focal group winning an intergroup contest.")
    for v in focal_vars:
        if v in vars_stats:
            s = vars_stats[v]
            sign = "positive" if s['coef'] > 0 else ("zero" if s['coef'] == 0 else "negative")
            desc_lines.append(
                f"- {v}: coef = {s['coef']:.3f}, SE = {s['se']:.3f} if available, "
                f"95%CI = [{s['ci_95_lower']:.3f}, {s['ci_95_upper']:.3f}] if available, "
                f"OR = {s['odds_ratio']:.3f} (CI [{s['odds_ratio_ci_95_lower']:.3f}, {s['odds_ratio_ci_95_upper']:.3f}] if available); "
                f"effect is {sign}; {sig_text(s['p_value'])}."
            )
        else:
            desc_lines.append(f"- {v}: NOT in model output.")

    # Interpret moderation/marginal effect
    if marginal:
        if marginal.get('size_when_focal_closer_1') is not None:
            mm0 = marginal['size_when_focal_closer_0']
            mm1 = marginal['size_when_focal_closer_1']
            desc_lines.append(
                f"- Marginal effect of relative group size (size_diff_z): when focal_closer = 0, "
                f"log-odds coef = {mm0['log_odds_coef']:.3f}, OR = {mm0['odds_ratio']:.3f}."
            )
            if mm1['se'] is not None:
                desc_lines.append(
                    f"  When focal_closer = 1, combined log-odds coef = {mm1['log_odds_coef']:.3f} "
                    f"(SE ≈ {mm1['se']:.3f}), 95%CI = [{mm1['ci_95_lower']:.3f}, {mm1['ci_95_upper']:.3f}], "
                    f"OR = {mm1['odds_ratio']:.3f} (CI [{mm1['odds_ratio_ci_95_lower']:.3f}, {mm1['odds_ratio_ci_95_upper']:.3f}]); "
                    f"interaction term {('is' if mm1['interaction_significant_at_0.05'] else 'is not')} statistically significant."
                )
            else:
                desc_lines.append(
                    f"  When focal_closer = 1, combined log-odds coef = {mm1['log_odds_coef']:.3f}; "
                    f"standard error / CI for the combined effect unavailable (covariance matrix missing)."
                )
        else:
            desc_lines.append("- Could not compute marginal effect for focal_closer=1 because interaction term missing.")
    else:
        desc_lines.append("- Could not compute marginal effects because size_diff_z absent.")

    # Compose result object
    result_object = {
        'variables': vars_stats,
        'marginal_effects': marginal,
        'missing_variables': missing_vars
    }

    description = "\n".join(desc_lines)

    return {'object': result_object, 'description': description}