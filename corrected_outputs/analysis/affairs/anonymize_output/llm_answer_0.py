def extract_final_answer(model_output):
    """
    Extracts statistics about the effect of 'Children' on 'AffairCount' from a
    statsmodels ZeroInflatedNegativeBinomialResultsWrapper.

    Returns a dictionary with keys:
      - "object": dict with numeric results for:
            * female_effect: coefficient, se, pvalue, CI, IRR, IRR_CI, pct_change
            * male_effect: same as above (computed as Children + Children_Male) or None if no interaction
            * inflation_effect (if present): coefficient, se, pvalue, CI, odds_ratio, odds_ratio_CI
      - "description": short plain-language interpretation of the results.

    Notes:
      - Female effect corresponds to Male=0 (i.e., the 'Children' coefficient).
      - Male effect = Children + Children_Male (if 'Children_Male' present).
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output

    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    ci_df = res.conf_int()
    cov = res.cov_params()

    def safe_get(name):
        if name in params.index:
            return params[name], bse[name], pvalues[name], ci_df.loc[name].tolist()
        else:
            return None

    # Get count-model effect for Children (this is for Female / Male=0)
    children_info = safe_get('Children')
    if children_info is None:
        # try alternative name variants
        alt_names = [n for n in params.index if n.lower().endswith('children')]
        if alt_names:
            name = alt_names[0]
            children_info = (params[name], bse[name], pvalues[name], ci_df.loc[name].tolist())
            children_name = name
        else:
            raise KeyError("Could not find a parameter named 'Children' in model params.")
    else:
        children_name = 'Children'

    coef_children, se_children, p_children, ci_children = children_info
    coef_children = float(coef_children); se_children = float(se_children); p_children = float(p_children)
    ci_children = [float(ci_children[0]), float(ci_children[1])]

    irr_children = float(np.exp(coef_children))
    irr_ci_children = [float(np.exp(ci_children[0])), float(np.exp(ci_children[1]))]
    pct_change_children = (irr_children - 1.0) * 100.0  # percent change in expected count

    female_effect = {
        'coef': coef_children,
        'se': se_children,
        'pvalue': p_children,
        'ci_lower': ci_children[0],
        'ci_upper': ci_children[1],
        'irr (exp(coef))': irr_children,
        'irr_ci_lower': irr_ci_children[0],
        'irr_ci_upper': irr_ci_children[1],
        'pct_change_in_count': pct_change_children
    }

    # Compute male effect if interaction present
    male_effect = None
    if 'Children_Male' in params.index:
        coef_int = float(params['Children_Male'])
        # variance of sum = var(a)+var(b)+2cov(a,b)
        try:
            var_children = float(cov.loc[children_name, children_name])
            var_int = float(cov.loc['Children_Male', 'Children_Male'])
            cov_ch_int = float(cov.loc[children_name, 'Children_Male'])
        except Exception:
            # fallback to diagonal only (conservative)
            var_children = se_children ** 2
            var_int = float(bse['Children_Male']) ** 2
            cov_ch_int = 0.0
        var_sum = var_children + var_int + 2.0 * cov_ch_int
        se_sum = float(np.sqrt(var_sum)) if var_sum >= 0 else float(np.nan)
        coef_sum = coef_children + coef_int
        z = coef_sum / se_sum if se_sum != 0 else float('nan')
        p_sum = float(2.0 * (1.0 - norm.cdf(abs(z)))) if not np.isnan(z) else float('nan')
        ci_lower = coef_sum - 1.96 * se_sum
        ci_upper = coef_sum + 1.96 * se_sum
        irr_sum = float(np.exp(coef_sum))
        irr_ci = [float(np.exp(ci_lower)), float(np.exp(ci_upper))]
        pct_change_male = (irr_sum - 1.0) * 100.0

        male_effect = {
            'coef': coef_sum,
            'se': se_sum,
            'pvalue': p_sum,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'irr (exp(coef))': irr_sum,
            'irr_ci_lower': irr_ci[0],
            'irr_ci_upper': irr_ci[1],
            'pct_change_in_count': pct_change_male,
            'notes': "This is the marginal effect of Children for Male=1 (Children + Children_Male)."
        }

    # If Children appears in inflation model, extract that too (affects probability of structural zero)
    inflation_effect = None
    inflate_name = 'inflate_Children'
    if inflate_name in params.index:
        coef_infl, se_infl, p_infl, ci_infl = safe_get(inflate_name)
        coef_infl = float(coef_infl); se_infl = float(se_infl); p_infl = float(p_infl)
        ci_infl = [float(ci_infl[0]), float(ci_infl[1])]
        # inflation uses logit; exponentiated coef is odds ratio for being in the inflate (zero) state
        or_infl = float(np.exp(coef_infl))
        or_ci = [float(np.exp(ci_infl[0])), float(np.exp(ci_infl[1]))]
        inflation_effect = {
            'coef (logit)': coef_infl,
            'se': se_infl,
            'pvalue': p_infl,
            'ci_lower': ci_infl[0],
            'ci_upper': ci_infl[1],
            'odds_ratio': or_infl,
            'odds_ratio_ci_lower': or_ci[0],
            'odds_ratio_ci_upper': or_ci[1],
            'interpretation': "Positive coef => higher odds of being in structural-zero (i.e., no chance of any affairs)."
        }

    # Build human-readable description
    desc_lines = []
    # Female
    desc_lines.append(
        "For respondents coded Male=0 (women), the 'Children' coefficient in the count model is "
        f"{coef_children:.4f} (SE={se_children:.4f}, p={p_children:.3g}), 95% CI [{ci_children[0]:.4f}, {ci_children[1]:.4f}]."
    )
    desc_lines.append(
        f"Exponentiated: IRR = {irr_children:.4f}, 95% CI [{irr_ci_children[0]:.4f}, {irr_ci_children[1]:.4f}], "
        f"which corresponds to a {pct_change_children:.1f}% change in expected affair count."
    )
    # Male
    if male_effect is not None:
        desc_lines.append(
            "For respondents coded Male=1 (men), the marginal effect of 'Children' (Children + Children_Male) is "
            f"{male_effect['coef']:.4f} (SE={male_effect['se']:.4f}, p={male_effect['pvalue']:.3g}), "
            f"95% CI [{male_effect['ci_lower']:.4f}, {male_effect['ci_upper']:.4f}]."
        )
        desc_lines.append(
            f"Exponentiated: IRR = {male_effect['irr (exp(coef))']:.4f}, 95% CI "
            f"[{male_effect['irr_ci_lower']:.4f}, {male_effect['irr_ci_upper']:.4f}], "
            f"≈ {male_effect['pct_change_in_count']:.1f}% change in expected affair count for men."
        )
    else:
        desc_lines.append("No interaction term 'Children_Male' found; the 'Children' coefficient applies to all respondents (no gender moderation).")

    # Inflation
    if inflation_effect is not None:
        desc_lines.append(
            "In the zero-inflation (logit) part, 'Children' has coef "
            f"{inflation_effect['coef (logit)']:.4f} (p={inflation_effect['pvalue']:.3g}); "
            f"odds ratio = {inflation_effect['odds_ratio']:.4f}, 95% CI "
            f"[{inflation_effect['odds_ratio_ci_lower']:.4f}, {inflation_effect['odds_ratio_ci_upper']:.4f}]. "
            "A positive value suggests children increase the odds of being in the structural-zero group (i.e., no chance of affairs)."
        )

    # Conclude directionality succinctly
    conclusion = "Overall interpretation: "
    if coef_children < 0:
        conclusion += "Having children is associated with a lower reported frequency of extramarital intercourse (negative coefficient / IRR < 1) for women."
    elif coef_children > 0:
        conclusion += "Having children is associated with a higher reported frequency of extramarital intercourse (positive coefficient / IRR > 1) for women."
    else:
        conclusion += "No effect of children observed for women."

    if male_effect is not None:
        if male_effect['coef'] < 0:
            conclusion += " For men the marginal effect is negative (children associated with fewer reported affairs)."
        elif male_effect['coef'] > 0:
            conclusion += " For men the marginal effect is positive (children associated with more reported affairs)."
        else:
            conclusion += " For men there is no marginal effect."

    desc_lines.append(conclusion)

    # Assemble object payload with numeric outputs
    object_payload = {
        'female_effect': female_effect,
        'male_effect': male_effect,
        'inflation_effect': inflation_effect
    }

    return {
        "object": object_payload,
        "description": " ".join(desc_lines)
    }