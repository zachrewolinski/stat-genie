def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of relative group size (size_ratio) on the probability
    that the focal group wins, and how that effect is modified by contest location
    (interaction terms size_ratio_x_contest_location_*).

    Returns:
      {
        "object": dict with numeric summaries (coefficients, SE, z, p, 95% CI, odds ratio and CI)
                  for:
                    - the baseline effect of size_ratio (reference location: FocalHome)
                    - the marginal effect of size_ratio in each contest location (combination of main + interaction)
                    - the raw interaction coefficients (for completeness)
        "description": short interpretation of what those numbers mean in context
      }
    """
    import numpy as np
    try:
        from scipy import stats
    except Exception:
        # Fallback to normal cdf via math.erf if scipy not available
        import math
        class _Norm:
            @staticmethod
            def cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        stats = _Norm()

    res = model_output  # statsmodels GLMResultsWrapper expected

    # Extract params, covariance, and basic summaries
    params = res.params
    cov = res.cov_params()
    # conf_int may depend on method; we'll compute CIs ourselves using normal approx
    bse = np.sqrt(np.diag(cov))

    # Helper to compute p-value from z
    def two_sided_p(z):
        try:
            return float(2 * (1 - stats.norm.cdf(abs(z))))
        except Exception:
            # stats may be fallback object with cdf method
            return float(2 * (1 - stats.cdf(abs(z))))

    # Prepare output structure
    summary = {}
    # Main size_ratio coefficient (baseline = reference contest location, presumably FocalHome)
    if 'size_ratio' not in params.index:
        raise KeyError("Model output does not contain a 'size_ratio' coefficient. Check model specification.")

    coef_size = float(params['size_ratio'])
    se_size = float(bse[params.index.get_loc('size_ratio')])
    z_size = coef_size / se_size if se_size != 0 else np.nan
    p_size = two_sided_p(z_size)
    ci_low_size = coef_size - 1.96 * se_size
    ci_high_size = coef_size + 1.96 * se_size
    or_size = float(np.exp(coef_size))
    or_ci_low = float(np.exp(ci_low_size))
    or_ci_high = float(np.exp(ci_high_size))

    summary['size_ratio_baseline'] = {
        'location': 'FocalHome (reference)',
        'coef_log_odds_per_unit_size_ratio': coef_size,
        'se': se_size,
        'z': z_size,
        'p_value': p_size,
        '95ci_log_odds': [ci_low_size, ci_high_size],
        'odds_ratio': or_size,
        '95ci_odds_ratio': [or_ci_low, or_ci_high],
        'interpretation_brief': (
            "Baseline effect of focal:other size ratio on log-odds of focal winning "
            "(applicable when contest_location == FocalHome). Positive coef -> higher "
            "size_ratio increases probability focal wins."
        )
    }

    # Find interaction terms of the form 'size_ratio_x_contest_location_<LocationName>'
    interaction_prefix = 'size_ratio_x_contest_location_'
    interaction_terms = [name for name in params.index if name.startswith(interaction_prefix)]

    # Compute marginal effects (combined coefficient) for each location:
    marginal_effects = {}
    # Baseline location entry
    marginal_effects['FocalHome'] = {
        'combined_coef_log_odds': coef_size,
        'se': se_size,
        'z': z_size,
        'p_value': p_size,
        '95ci_log_odds': [ci_low_size, ci_high_size],
        'odds_ratio': or_size,
        '95ci_odds_ratio': [or_ci_low, or_ci_high]
    }

    # For each interaction, compute combined coefficient = size_ratio + interaction_coef
    for inter in interaction_terms:
        # infer location name
        location = inter.replace(interaction_prefix, '')
        inter_coef = float(params[inter])
        # Build contrast vector: 1 for size_ratio, 1 for this interaction, 0 elsewhere
        p_index = list(params.index)
        contrast = np.zeros(len(p_index))
        contrast[p_index.index('size_ratio')] = 1.0
        contrast[p_index.index(inter)] = 1.0
        # variance of contrast
        var = float(contrast @ cov.values @ contrast)
        se_comb = float(np.sqrt(var)) if var >= 0 else float(np.nan)
        coef_comb = float(coef_size + inter_coef)
        z_comb = coef_comb / se_comb if se_comb != 0 else np.nan
        p_comb = two_sided_p(z_comb)
        ci_low = coef_comb - 1.96 * se_comb
        ci_high = coef_comb + 1.96 * se_comb
        or_comb = float(np.exp(coef_comb))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))

        marginal_effects[location] = {
            'combined_coef_log_odds': coef_comb,
            'se': se_comb,
            'z': z_comb,
            'p_value': p_comb,
            '95ci_log_odds': [ci_low, ci_high],
            'odds_ratio': or_comb,
            '95ci_odds_ratio': [or_ci_low, or_ci_high],
            'interpretation_brief': (
                f"Effect of size_ratio on log-odds of focal winning when contest_location == {location}."
            )
        }

    # Also include raw interaction coefficients (to see whether the interaction itself is significant)
    raw_interactions = {}
    for inter in interaction_terms:
        coef_i = float(params[inter])
        se_i = float(bse[params.index.get_loc(inter)])
        z_i = coef_i / se_i if se_i != 0 else np.nan
        p_i = two_sided_p(z_i)
        ci_low_i = coef_i - 1.96 * se_i
        ci_high_i = coef_i + 1.96 * se_i
        raw_interactions[inter] = {
            'coef': coef_i,
            'se': se_i,
            'z': z_i,
            'p_value': p_i,
            '95ci': [ci_low_i, ci_high_i],
            'interpretation_brief': (
                "Interaction term: how much the slope of size_ratio (log-odds per unit) differs "
                "in this location compared to the reference (FocalHome)."
            )
        }

    # Package final object
    output_obj = {
        'baseline_size_ratio': summary['size_ratio_baseline'],
        'marginal_effects_by_location': marginal_effects,
        'raw_interaction_coefficients': raw_interactions,
        # also include full params & pvalues for completeness (converted to native types)
        'all_coefficients': {name: float(val) for name, val in params.items()},
        'all_pvalues': {name: float(val) for name, val in res.pvalues.items()}
    }

    # Short text description
    # We'll highlight how to read the results:
    desc_lines = [
        "This object gives (1) the baseline effect of size_ratio on the log-odds of focal winning",
        "   (reference contest location: FocalHome), (2) the marginal effect of size_ratio in each",
        "   contest location (computed as baseline + interaction), and (3) the raw interaction coefficients.",
        "",
        "Interpretation guidance:",
        "- A positive combined coef means that increasing the focal:other size ratio increases the",
        "  log-odds (and hence probability) that the focal group wins. The odds_ratio > 1 corresponds",
        "  to multiplicative change in odds per unit increase in size_ratio.",
        "- Interaction coefficients indicate whether the slope (effect of size_ratio) is stronger",
        "  or weaker in that contest location compared to FocalHome. A positive interaction means the",
        "  effect of size_ratio is larger in that location.",
        "",
        "Review the 'p_value' fields to assess statistical evidence for each effect; note that p-values",
        "are computed using a normal approximation (z-test) from the model covariance supplied by the fitted model."
    ]
    description = "\n".join(desc_lines)

    return {"object": output_obj, "description": description}