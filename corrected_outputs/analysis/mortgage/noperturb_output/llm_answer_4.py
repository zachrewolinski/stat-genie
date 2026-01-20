def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of gender on mortgage approval from the
    fitted model objects returned by the provided modeling function.

    Input:
      model_output: dict with keys 'model_with_interaction' and 'model_without_interaction'
                    each being a statsmodels fitted results object (Logit/GLM).

    Returns:
      dict with keys:
        - "object": a dictionary containing numeric summaries (coefficients, SEs,
                    p-values, 95% CIs, odds ratios and OR CIs) for:
                      * 'female' from the model without interaction
                      * 'female' (main effect when black=0) from the model with interaction
                      * 'female_black' interaction term (if present)
                      * 'female_effect_if_black' (combined effect = female + female_black),
                        with its SE, z, p-value, CI, and odds ratio (only if interaction present)
        - "description": a short interpretation of what these quantities mean and how to
                         interpret them (statistical significance threshold = 0.05).
    """
    import numpy as np
    import math

    out = {}
    desc_lines = []

    # Helper: normal two-sided p-value from z
    def two_sided_p_from_z(z):
        # standard normal cdf using erf
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        return 2.0 * (1.0 - cdf) if z >= 0 else 2.0 * cdf

    def summarize_model(results):
        # Extract common quantities
        params = results.params
        bse = results.bse
        pvalues = results.pvalues
        conf = results.conf_int()  # DataFrame: columns [0,1]
        cov = results.cov_params()
        summary = {}
        for var in params.index:
            # Skip const
            if var == 'const':
                continue
            coef = float(params[var])
            se = float(bse[var]) if var in bse.index else None
            pval = float(pvalues[var]) if var in pvalues.index else None
            ci_low = float(conf.loc[var, 0]) if var in conf.index else None
            ci_high = float(conf.loc[var, 1]) if var in conf.index else None
            or_val = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
            or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

            summary[var] = {
                'coef': coef,
                'se': se,
                'pvalue': pval,
                'ci_95': (ci_low, ci_high),
                'odds_ratio': or_val,
                'odds_ratio_ci_95': (or_ci_low, or_ci_high)
            }
        return summary, params, cov

    # Fetch models
    model_with = model_output.get('model_with_interaction')
    model_without = model_output.get('model_without_interaction')

    if model_without is None and model_with is None:
        raise ValueError("Both 'model_with_interaction' and 'model_without_interaction' are missing from model_output")

    # Summarize model without interaction (gives main female effect)
    if model_without is not None:
        summary_no_inter, params_no_inter, cov_no_inter = summarize_model(model_without)
        out['model_without_interaction'] = summary_no_inter
        if 'female' in summary_no_inter:
            coef = summary_no_inter['female']['coef']
            pval = summary_no_inter['female']['pvalue']
            desc_lines.append(
                f"Model without interaction: 'female' coef = {coef:.4f}, p = {pval:.4g}. "
                "This coefficient is the change in log-odds of approval for female vs male "
                "(holding controls fixed). Odds ratio = "
                f"{summary_no_inter['female']['odds_ratio']:.3f} "
                f"(95% CI {summary_no_inter['female']['odds_ratio_ci_95'][0]:.3f}, {summary_no_inter['female']['odds_ratio_ci_95'][1]:.3f})."
            )
        else:
            desc_lines.append("Model without interaction: no 'female' coefficient found in the model.")

    # Summarize model with interaction (gives main effect and interaction)
    if model_with is not None:
        summary_with, params_with, cov_with = summarize_model(model_with)
        out['model_with_interaction'] = summary_with

        if 'female' in summary_with:
            coef_f = summary_with['female']['coef']
            pval_f = summary_with['female']['pvalue']
            desc_lines.append(
                f"Model with interaction: 'female' (main) coef = {coef_f:.4f}, p = {pval_f:.4g}. "
                "This is the effect of being female when 'black' = 0 (non-Black applicants). "
                f"Odds ratio (female vs male, among non-Black) = {summary_with['female']['odds_ratio']:.3f}."
            )
        else:
            desc_lines.append("Model with interaction: no 'female' main effect found.")

        if 'female_black' in summary_with:
            coef_int = summary_with['female_black']['coef']
            pval_int = summary_with['female_black']['pvalue']
            desc_lines.append(
                f"Interaction term 'female_black' coef = {coef_int:.4f}, p = {pval_int:.4g}. "
                "A statistically significant interaction indicates the effect of gender differs for Black applicants."
            )

            # Compute combined effect for female when black=1
            if 'female' in params_with:
                beta_f = float(params_with['female'])
                beta_fb = float(params_with['female_black'])
                combined = beta_f + beta_fb

                # variance of sum
                var_f = float(cov_with.loc['female', 'female'])
                var_fb = float(cov_with.loc['female_black', 'female_black'])
                cov_ffb = float(cov_with.loc['female', 'female_black'])
                var_combined = var_f + var_fb + 2.0 * cov_ffb
                se_combined = math.sqrt(var_combined) if var_combined >= 0 else float('nan')
                z_combined = combined / se_combined if se_combined and not math.isnan(se_combined) else None
                p_combined = two_sided_p_from_z(z_combined) if z_combined is not None else None
                ci_low = combined - 1.96 * se_combined
                ci_high = combined + 1.96 * se_combined
                or_combined = float(np.exp(combined))
                or_ci_low = float(np.exp(ci_low))
                or_ci_high = float(np.exp(ci_high))

                out['model_with_interaction']['female_effect_if_black'] = {
                    'coef': float(combined),
                    'se': float(se_combined),
                    'z': float(z_combined) if z_combined is not None else None,
                    'pvalue': float(p_combined) if p_combined is not None else None,
                    'ci_95': (float(ci_low), float(ci_high)),
                    'odds_ratio': or_combined,
                    'odds_ratio_ci_95': (or_ci_low, or_ci_high)
                }

                desc_lines.append(
                    "Combined effect for female when black=1 (Black female vs Black male): "
                    f"coef = {combined:.4f}, se = {se_combined:.4f}, z = {z_combined:.3f}, p = {p_combined:.4g}. "
                    f"Odds ratio = {or_combined:.3f} (95% CI {or_ci_low:.3f}, {or_ci_high:.3f})."
                )
            else:
                desc_lines.append("Cannot compute combined female effect for Black applicants because 'female' term missing.")
        else:
            desc_lines.append("Model with interaction: no 'female_black' interaction term found; interpret 'female' as average effect across races in that model.")

    # Final interpretive guidance
    desc_lines.append(
        "Interpretation notes: coefficients are on the log-odds scale. Positive coef => higher odds of approval for the group coded 1. "
        "Odds ratio > 1 indicates higher odds; < 1 indicates lower odds. Statistical significance is commonly assessed at alpha = 0.05."
    )

    return {
        "object": out,
        "description": " ".join(desc_lines)
    }