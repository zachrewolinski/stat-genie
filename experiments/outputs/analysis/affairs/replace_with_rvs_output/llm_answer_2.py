def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of having children on extramarital affairs
    from two fitted Negative Binomial GLM models contained in model_output:
      - 'nb_main': main-effects model (Children effect for reference group, Female=0)
      - 'nb_with_interaction': model with Children x Female interaction

    Returns a dict with:
      - "object": dictionary of numeric results (coefficients, SEs, p-values,
                  95% CIs, incidence rate ratios and percent change)
      - "description": brief plain-language interpretation about whether having
                       children decreases engagement in extramarital affairs,
                       overall and by gender (based on the interaction model).
    """
    import numpy as np
    from scipy.stats import norm

    # Helper to safe-get model
    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dict containing 'nb_main' and 'nb_with_interaction' keys.")

    if 'nb_main' not in model_output or 'nb_with_interaction' not in model_output:
        raise KeyError("model_output must contain keys 'nb_main' and 'nb_with_interaction'.")

    m_main = model_output['nb_main']
    m_int = model_output['nb_with_interaction']

    # Extract main-model Children effect (this represents effect when Female=0)
    params_main = m_main.params
    b_children_main = float(params_main.get('Children', np.nan))
    try:
        se_children_main = float(m_main.bse['Children'])
    except Exception:
        se_children_main = float(np.nan)
    # two-sided p-value (statsmodels already has pvalues)
    p_children_main = float(m_main.pvalues.get('Children', np.nan))
    # 95% CI
    try:
        ci_main = m_main.conf_int().loc['Children'].astype(float).tolist()
    except Exception:
        ci_main = [np.nan, np.nan]

    irr_children_main = float(np.exp(b_children_main))
    irr_children_main_ci = [float(np.exp(ci_main[0])), float(np.exp(ci_main[1]))]
    pct_change_main = (irr_children_main - 1.0) * 100.0

    # Interaction model: extract components
    params_int = m_int.params
    b_children_int = float(params_int.get('Children', np.nan))  # effect for males (Female=0)
    b_children_female_inter = float(params_int.get('Children_Female', np.nan))  # difference for females
    # Standard errors
    try:
        se_children_int = float(m_int.bse['Children'])
    except Exception:
        se_children_int = float(np.nan)
    try:
        se_children_fem_int = float(m_int.bse['Children_Female'])
    except Exception:
        se_children_fem_int = float(np.nan)
    # p-values for individual terms
    p_children_int = float(m_int.pvalues.get('Children', np.nan))
    p_children_fem_int = float(m_int.pvalues.get('Children_Female', np.nan))

    # Combined effect for females: beta_children + beta_children_female
    b_children_female = b_children_int + b_children_female_inter

    # Compute SE for combined effect using covariance matrix
    try:
        cov = m_int.cov_params()
        var_children = cov.loc['Children', 'Children']
        var_children_fem = cov.loc['Children_Female', 'Children_Female']
        cov_term = cov.loc['Children', 'Children_Female']
        se_children_female = float(np.sqrt(var_children + var_children_fem + 2.0 * cov_term))
    except Exception:
        se_children_female = float(np.nan)

    # z and p for combined female effect
    if not np.isnan(se_children_female) and se_children_female > 0:
        z_female = b_children_female / se_children_female
        p_children_female = 2.0 * norm.sf(abs(z_female))
    else:
        z_female = float('nan')
        p_children_female = float(np.nan)

    # 95% CI for combined female effect (on log scale) and IRR
    try:
        ci_low_f = b_children_female - 1.96 * se_children_female
        ci_high_f = b_children_female + 1.96 * se_children_female
        irr_children_female = float(np.exp(b_children_female))
        irr_children_female_ci = [float(np.exp(ci_low_f)), float(np.exp(ci_high_f))]
        pct_change_female = (irr_children_female - 1.0) * 100.0
    except Exception:
        irr_children_female = float(np.nan)
        irr_children_female_ci = [float(np.nan), float(np.nan)]
        pct_change_female = float(np.nan)

    # Summary decision logic: consider effect "decrease" if IRR < 1 and p < .05
    def interpret(irr, p):
        if np.isnan(irr) or np.isnan(p):
            return "insufficient information"
        if (irr < 1.0) and (p < 0.05):
            return "statistically significant decrease"
        if (irr < 1.0) and (p >= 0.05):
            return "decrease (not statistically significant)"
        if (irr > 1.0) and (p < 0.05):
            return "statistically significant increase"
        if (irr > 1.0) and (p >= 0.05):
            return "increase (not statistically significant)"
        return "no change detected"

    interpretation_main = interpret(irr_children_main, p_children_main)
    interpretation_males_int = interpret(float(np.exp(b_children_int)), p_children_int)
    interpretation_females_int = interpret(irr_children_female, p_children_female)

    # Build result object
    result_object = {
        'main_model_children': {
            'coef_log': b_children_main,
            'se': se_children_main,
            'p_value': p_children_main,
            '95ci_log': ci_main,
            'incidence_rate_ratio': irr_children_main,
            '95ci_irr': irr_children_main_ci,
            'percent_change': pct_change_main,
            'interpretation': interpretation_main,
            'note': "In the main model this coefficient represents the effect of Children for the reference group (Female=0)."
        },
        'interaction_model': {
            'coef_children_males_log': b_children_int,
            'se_children_males': se_children_int,
            'p_children_males': p_children_int,
            'irr_children_males': float(np.exp(b_children_int)) if not np.isnan(b_children_int) else float('nan'),
            'coef_children_female_diff_log': b_children_female_inter,
            'se_children_female_diff': se_children_fem_int,
            'p_children_female_diff': p_children_fem_int,
            'combined_coef_children_fem_log': b_children_female,
            'se_combined_children_fem': se_children_female,
            'p_combined_children_fem': p_children_female,
            'irr_children_female': irr_children_female,
            '95ci_irr_children_female': irr_children_female_ci,
            'percent_change_female': pct_change_female,
            'interpretation_males': interpretation_males_int,
            'interpretation_females': interpretation_females_int,
            'note': "In the interaction model, 'Children' is the effect for males (Female=0); 'Children_Female' is the difference in effect for females. Combined female effect = Children + Children_Female."
        }
    }

    # Plain-language description
    desc_lines = []
    desc_lines.append("Main model (no interaction):")
    desc_lines.append(
        f"  - Coef (log) for Children = {b_children_main:.4f}, SE = {se_children_main:.4f}, p = {p_children_main:.3g}."
        f" IRR = {irr_children_main:.3f} (95% CI [{irr_children_main_ci[0]:.3f}, {irr_children_main_ci[1]:.3f}]),"
        f" change = {pct_change_main:.1f}%."
    )
    desc_lines.append(f"  - Interpretation: {interpretation_main} for the reference group (usually males).")

    desc_lines.append("Interaction model (Children x Female):")
    desc_lines.append(
        f"  - For males (Female=0): Coef (log) = {b_children_int:.4f}, p = {p_children_int:.3g}, IRR = {np.exp(b_children_int):.3f}."
    )
    desc_lines.append(
        f"  - Difference for females (Children_Female): Coef = {b_children_female_inter:.4f}, p = {p_children_fem_int:.3g}."
    )
    desc_lines.append(
        f"  - Combined effect for females: Coef (log) = {b_children_female:.4f}, SE = {se_children_female:.4f}, p = {p_children_female:.3g},"
        f" IRR = {irr_children_female:.3f} (95% CI [{irr_children_female_ci[0]:.3f}, {irr_children_female_ci[1]:.3f}]), change = {pct_change_female:.1f}%."
    )
    desc_lines.append(f"  - Interpretations: males -> {interpretation_males_int}; females -> {interpretation_females_int}.")

    # Final summary statement
    # Decide overall: if either group shows statistically significant decrease, note it
    overall_findings = []
    if interpretation_males_int == "statistically significant decrease":
        overall_findings.append("For males, having children is associated with a statistically significant decrease in affairs.")
    if interpretation_females_int == "statistically significant decrease":
        overall_findings.append("For females, having children is associated with a statistically significant decrease in affairs.")
    if (interpretation_males_int.startswith("decrease") or interpretation_females_int.startswith("decrease")) and not overall_findings:
        overall_findings.append("There is some evidence of decreased affairs with children, but effects are not statistically significant.")
    if not overall_findings and not (interpretation_males_int.startswith("decrease") or interpretation_females_int.startswith("decrease")):
        overall_findings.append("No evidence that having children decreases engagement in extramarital affairs; effects are absent or indicate increase/non-significance.")

    desc_lines.append("Overall conclusion: " + " ".join(overall_findings))

    description = "\n".join(desc_lines)

    return {
        "object": result_object,
        "description": description
    }