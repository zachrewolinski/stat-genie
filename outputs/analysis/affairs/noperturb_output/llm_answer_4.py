def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of ChildrenBinary on AffairsCount from a
    fitted statsmodels ZeroInflatedNegativeBinomialResultsWrapper.

    Returns a dictionary with:
      - "object": a dict of numeric results (count-model coef, IRR, p-values, CIs;
                  interaction; marginal effects for males/females; inflation-model
                  coef and odds-ratio)
      - "description": a short plain-language interpretation of those statistics
                       in context (whether having children appears to decrease
                       engagement in extramarital affairs, and how that may differ
                       by gender).
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output

    params = res.params
    bse = res.bse
    pvals = res.pvalues
    try:
        conf = res.conf_int()
    except Exception:
        # fallback if conf_int fails
        conf = None
    cov = res.cov_params()

    # Helper to find parameter names robustly
    def find_param(name_substr, inflation=False):
        for name in params.index:
            if name_substr in name:
                if inflation:
                    if name.startswith('inflate') or 'inflate' in name:
                        return name
                else:
                    if not (name.startswith('inflate') or 'inflate' in name):
                        return name
        return None

    # Locate relevant parameter names
    count_children_name = find_param('ChildrenBinary', inflation=False)
    interaction_name = find_param('Children_GenderInteraction', inflation=False)
    infl_children_name = find_param('ChildrenBinary', inflation=True)
    infl_gender_name = find_param('GenderMale', inflation=True)

    output = {}
    desc_lines = []

    # Extract count-model children coefficient
    if count_children_name is not None:
        coef = float(params[count_children_name])
        se = float(bse[count_children_name]) if count_children_name in bse.index else None
        p = float(pvals[count_children_name]) if count_children_name in pvals.index else None
        if conf is not None and count_children_name in conf.index:
            ci_low, ci_high = map(float, conf.loc[count_children_name])
        else:
            ci_low = coef - 1.96 * se if se is not None else None
            ci_high = coef + 1.96 * se if se is not None else None
        irr = float(np.exp(coef))
        irr_ci = [float(np.exp(ci_low)), float(np.exp(ci_high))] if (ci_low is not None and ci_high is not None) else [None, None]

        output['count_children'] = {
            'param_name': count_children_name,
            'coef': coef,
            'se': se,
            'p_value': p,
            '95ci_coef': [ci_low, ci_high],
            'incidence_rate_ratio': irr,
            '95ci_irr': irr_ci
        }

        desc_lines.append(
            f"Count model (among those at-risk of affairs): '{count_children_name}' coef={coef:.4f}, "
            f"IRR={irr:.4f}, p={p:.3g}."
        )
    else:
        desc_lines.append("Count-model ChildrenBinary parameter not found in model output.")
    
    # Extract interaction parameter (children x gender)
    if interaction_name is not None:
        coef_int = float(params[interaction_name])
        se_int = float(bse[interaction_name]) if interaction_name in bse.index else None
        p_int = float(pvals[interaction_name]) if interaction_name in pvals.index else None
        if conf is not None and interaction_name in conf.index:
            ci_low_int, ci_high_int = map(float, conf.loc[interaction_name])
        else:
            ci_low_int = coef_int - 1.96 * se_int if se_int is not None else None
            ci_high_int = coef_int + 1.96 * se_int if se_int is not None else None

        output['count_children_gender_interaction'] = {
            'param_name': interaction_name,
            'coef': coef_int,
            'se': se_int,
            'p_value': p_int,
            '95ci_coef': [ci_low_int, ci_high_int]
        }

        desc_lines.append(
            f"Interaction '{interaction_name}' coef={coef_int:.4f}, p={p_int:.3g} "
            "(this modifies the children effect for males)."
        )
    else:
        desc_lines.append("Interaction parameter (Children x Gender) not found in model output.")

    # Compute marginal effects for females and males in the count model
    # Female (GenderMale=0): effect = coef_children
    # Male (GenderMale=1): effect = coef_children + coef_interaction
    try:
        female_effect = None
        male_effect = None
        if count_children_name is not None:
            female_coef = float(params[count_children_name])
            female_se = float(bse[count_children_name]) if count_children_name in bse.index else None
            female_irr = float(np.exp(female_coef))
            female_ci = [float(np.exp(output['count_children']['95ci_coef'][0])),
                         float(np.exp(output['count_children']['95ci_coef'][1]))] if output.get('count_children') else [None, None]
            output['marginal_effect_female'] = {
                'coef': female_coef,
                'se': female_se,
                'incidence_rate_ratio': female_irr,
                '95ci_irr': female_ci,
                'p_value': float(pvals[count_children_name]) if count_children_name in pvals.index else None
            }
            desc_lines.append(
                f"For females (GenderMale=0): children coef={female_coef:.4f}, IRR={female_irr:.4f}, p={pvals[count_children_name]:.3g}."
            )

            if interaction_name is not None:
                # male effect = coef_children + coef_interaction
                male_coef = float(params[count_children_name]) + float(params[interaction_name])
                # variance
                try:
                    var_c = cov.loc[count_children_name, count_children_name]
                    var_i = cov.loc[interaction_name, interaction_name]
                    cov_ci = cov.loc[count_children_name, interaction_name]
                    var_sum = var_c + var_i + 2.0 * cov_ci
                    male_se = float(np.sqrt(var_sum))
                except Exception:
                    male_se = None
                male_irr = float(np.exp(male_coef))
                # p-value for linear combination
                if male_se is not None:
                    z = male_coef / male_se
                    p_male = float(2 * (1 - norm.cdf(abs(z))))
                    ci_low_male = male_coef - 1.96 * male_se
                    ci_high_male = male_coef + 1.96 * male_se
                    ci_irr_male = [float(np.exp(ci_low_male)), float(np.exp(ci_high_male))]
                else:
                    p_male = None
                    ci_irr_male = [None, None]

                output['marginal_effect_male'] = {
                    'coef': male_coef,
                    'se': male_se,
                    'incidence_rate_ratio': male_irr,
                    '95ci_irr': ci_irr_male,
                    'p_value': p_male
                }
                desc_lines.append(
                    f"For males (GenderMale=1): children marginal coef={male_coef:.4f}, IRR={male_irr:.4f}, p={p_male:.3g if p_male is not None else 'NA'}."
                )
            else:
                desc_lines.append("Cannot compute male marginal effect because interaction parameter was not found.")
    except Exception as e:
        desc_lines.append(f"Error computing marginal effects: {e}")

    # Extract inflation-model children coefficient (odds of being an "always-zero" case)
    if infl_children_name is not None:
        coef_infl = float(params[infl_children_name])
        se_infl = float(bse[infl_children_name]) if infl_children_name in bse.index else None
        p_infl = float(pvals[infl_children_name]) if infl_children_name in pvals.index else None
        if conf is not None and infl_children_name in conf.index:
            ci_low_infl, ci_high_infl = map(float, conf.loc[infl_children_name])
        else:
            ci_low_infl = coef_infl - 1.96 * se_infl if se_infl is not None else None
            ci_high_infl = coef_infl + 1.96 * se_infl if se_infl is not None else None
        or_infl = float(np.exp(coef_infl))
        or_ci = [float(np.exp(ci_low_infl)), float(np.exp(ci_high_infl))] if (ci_low_infl is not None and ci_high_infl is not None) else [None, None]

        output['inflation_children'] = {
            'param_name': infl_children_name,
            'coef': coef_infl,
            'se': se_infl,
            'p_value': p_infl,
            '95ci_coef': [ci_low_infl, ci_high_infl],
            'odds_ratio': or_infl,
            '95ci_or': or_ci
        }

        desc_lines.append(
            f"Inflation model (probability of structural zero): '{infl_children_name}' coef={coef_infl:.4f}, "
            f"OR={or_infl:.4f}, p={p_infl:.3g}."
        )
    else:
        desc_lines.append("Inflation-model ChildrenBinary parameter not found in model output.")

    # Brief interpretation: does having children decrease engagement in extramarital affairs?
    # We'll synthesize a cautious statement based on sign and significance of count and inflation effects.
    interpret = []
    # check count effect significance
    try:
        cnt = output.get('count_children')
        if cnt:
            if cnt['p_value'] is not None and cnt['p_value'] < 0.05:
                if cnt['coef'] < 0:
                    interpret.append("In the count model, having children is associated with a statistically significant decrease in the expected number of affairs (IRR<1).")
                else:
                    interpret.append("In the count model, having children is associated with a statistically significant increase in the expected number of affairs (IRR>1).")
            else:
                interpret.append("In the count model, the coefficient for having children is not statistically significant.")
    except Exception:
        pass

    # check inflation effect significance
    try:
        infl = output.get('inflation_children')
        if infl:
            if infl['p_value'] is not None and infl['p_value'] < 0.05:
                if infl['coef'] > 0:
                    interpret.append("In the inflation model, having children significantly increases the odds of being in the 'always zero' group (i.e., more likely to report zero affairs), which implies fewer people with any affairs.")
                else:
                    interpret.append("In the inflation model, having children significantly decreases the odds of being an 'always zero' (i.e., less likely to report zero affairs).")
            else:
                interpret.append("In the inflation model, the coefficient for having children is not statistically significant.")
    except Exception:
        pass

    if not interpret:
        interpret.append("No clear evidence from either submodel (count or inflation) was found in the model output to claim a statistically significant effect of children on affairs.")

    # Combine description
    description = (
        "Extracted statistics for ChildrenBinary (count and inflation parts) and marginal effects by gender.\n"
        + "\n".join(desc_lines)
        + "\n\nSummary interpretation:\n- " + "\n- ".join(interpret)
    )

    return {"object": output, "description": description}