def extract_final_answer(model_output):
    """
    Extract and interpret the effect of gender ('female') on mortgage acceptance
    from the model_output produced by the provided modeling function.

    Returns a dictionary with:
      - "object": a dict with numeric results (coef, OR, CI, p-value, significance flag)
      - "description": a short plain-English interpretation of those results
    """
    # Try to read from the tidy summary_table if present (preferred)
    summary = model_output.get('summary_table', None)
    result = model_output.get('result', None)

    # Helper to build fallback from statsmodels result object
    def from_result(res):
        out = {}
        params = res.params
        pvals = res.pvalues
        try:
            conf = res.conf_int()
        except Exception:
            conf = None
        # get female row if available
        if 'female' in params.index:
            coef = float(params['female'])
            or_ = float(np.exp(coef))
            if conf is not None and 'female' in conf.index:
                ci_low = float(np.exp(conf.loc['female', 0]))
                ci_high = float(np.exp(conf.loc['female', 1]))
            else:
                ci_low = ci_high = None
            p = float(pvals['female']) if 'female' in pvals.index else None
            out['coef'] = coef
            out['OR'] = or_
            out['CI_lower'] = ci_low
            out['CI_upper'] = ci_high
            out['pvalue'] = p
        return out

    # Extract female stats
    female_stats = None
    if summary is not None and 'female' in summary.index:
        row = summary.loc['female']
        female_stats = {
            'coef': float(row['coef']),
            'OR': float(row['OR']),
            'CI_lower': float(row['CI_lower']),
            'CI_upper': float(row['CI_upper']),
            'pvalue': float(row['pvalue'])
        }
    elif result is not None:
        female_stats = from_result(result)

    if female_stats is None:
        raise ValueError("Could not find 'female' results in model_output.")

    # Also extract interaction female_black if available (for moderation context)
    interaction_stats = None
    if summary is not None and 'female_black' in summary.index:
        r = summary.loc['female_black']
        interaction_stats = {
            'coef': float(r['coef']),
            'OR': float(r['OR']),
            'CI_lower': float(r['CI_lower']),
            'CI_upper': float(r['CI_upper']),
            'pvalue': float(r['pvalue'])
        }
    elif result is not None and 'female_black' in result.params.index:
        params = result.params; pvals = result.pvalues
        conf = None
        try:
            conf = result.conf_int()
        except Exception:
            conf = None
        coef = float(params['female_black'])
        or_ = float(np.exp(coef))
        if conf is not None and 'female_black' in conf.index:
            ci_low = float(np.exp(conf.loc['female_black', 0])); ci_high = float(np.exp(conf.loc['female_black', 1]))
        else:
            ci_low = ci_high = None
        p = float(pvals['female_black']) if 'female_black' in pvals.index else None
        interaction_stats = {'coef': coef, 'OR': or_, 'CI_lower': ci_low, 'CI_upper': ci_high, 'pvalue': p}

    # Interpret significance at alpha = 0.05
    alpha = 0.05
    sig = (female_stats['pvalue'] is not None) and (female_stats['pvalue'] < alpha)
    # Build interpretation string
    interp_lines = []
    interp_lines.append(
        f"Estimated effect for 'female' (vs male): log-odds coef = {female_stats['coef']:.4f}, "
        f"OR = {female_stats['OR']:.3f}, 95% CI for OR = [{female_stats['CI_lower']:.3f}, {female_stats['CI_upper']:.3f}], "
        f"p = {female_stats['pvalue']:.3g}."
    )
    if sig:
        interp_lines.append(f"Statistically significant at alpha={alpha}: yes (p < {alpha}).")
        if female_stats['OR'] > 1:
            interp_lines.append("Interpretation: Females have higher odds of approval than males (holding controls constant).")
        else:
            interp_lines.append("Interpretation: Females have lower odds of approval than males (holding controls constant).")
    else:
        interp_lines.append(f"Statistically significant at alpha={alpha}: no (p >= {alpha}).")
        interp_lines.append(
            "Interpretation: There is no strong evidence in this model that applicant gender (female vs male) "
            "affects mortgage approval. The point estimate suggests females have higher odds (~{:.1f}x), "
            "but the 95% CI includes 1, so this effect could be due to sampling variability."
            .format(female_stats['OR'])
        )

    # If interaction present, report it briefly
    if interaction_stats is not None:
        interp_lines.append(
            f"Interaction 'female_black': OR = {interaction_stats['OR']:.3f}, p = {interaction_stats['pvalue']:.3g}. "
            "This interaction is not statistically significant (no evidence that the female effect differs by Black race)."
            if interaction_stats['pvalue'] is not None and interaction_stats['pvalue'] >= alpha
            else f"Interaction 'female_black' p = {interaction_stats['pvalue']:.3g}."
        )

    description = " ".join(interp_lines)

    return {
        "object": {
            "female_coef_log_odds": female_stats['coef'],
            "female_OR": female_stats['OR'],
            "female_OR_CI_lower": female_stats['CI_lower'],
            "female_OR_CI_upper": female_stats['CI_upper'],
            "female_pvalue": female_stats['pvalue'],
            "female_significant_at_0.05": sig,
            "alpha": alpha,
            "interaction_female_black": interaction_stats  # may be None
        },
        "description": description
    }