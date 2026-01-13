def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLM (possibly robustified)
    that predicted 'win' from RelSizeDiff, FocalHomeAdv, their interaction,
    and controls.

    Returns a dictionary with:
      - "object": a nested dict containing coefficients, SEs, z-stats, p-values,
                  95% CIs, odds-ratios and odds-ratio CIs for relevant terms,
                  and combined effects of RelSizeDiff when FocalHomeAdv = 0 and 1.
      - "description": a short explanation of what the numbers mean for the
                       research question (how relative group size and contest
                       location influence the probability of winning).
    """
    import numpy as np
    from math import sqrt
    try:
        # scipy is usually available; use it for p-values from z
        from scipy import stats
        norm_cdf = stats.norm.cdf
    except Exception:
        # fallback: approximate using numpy (less convenient); define a simple norm cdf
        def norm_cdf(x):
            # Abramowitz and Stegun approximation for normal CDF
            t = 1.0 / (1.0 + 0.2316419 * abs(x))
            d = 0.3989423 * np.exp(-x * x / 2.0)
            prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
            return 1.0 - prob if x > 0 else prob

    res = model_output

    # Pull parameter/index information
    try:
        params = res.params.copy()
        bse = res.bse.copy()
        cov = res.cov_params().copy()
        conf = res.conf_int().copy()
    except Exception as e:
        raise ValueError("Model output does not appear to have the expected attributes "
                         "(params, bse, cov_params, conf_int). Error: " + str(e))

    # Term names we care about
    term_rel = 'RelSizeDiff'
    term_home = 'FocalHomeAdv'
    # interaction naming in statsmodels is 'RelSizeDiff:FocalHomeAdv'
    term_int = f'{term_rel}:{term_home}'

    # Prepare container
    out = {
        'terms': {},
        'combined_effects': {},
    }

    # Helper to populate term info if present
    for term in [term_rel, term_home, term_int]:
        if term in params.index:
            coef = float(params[term])
            se = float(bse[term])
            z = coef / se if se != 0 else np.nan
            p = float(2 * (1.0 - norm_cdf(abs(z)))) if not np.isnan(z) else np.nan
            ci_low = float(conf.loc[term, 0])
            ci_high = float(conf.loc[term, 1])
            or_est = float(np.exp(coef))
            or_ci_low = float(np.exp(ci_low))
            or_ci_high = float(np.exp(ci_high))

            out['terms'][term] = {
                'coef': coef,
                'se': se,
                'z': z,
                'p_value': p,
                '95%_CI_coef': [ci_low, ci_high],
                'odds_ratio': or_est,
                '95%_CI_odds_ratio': [or_ci_low, or_ci_high],
            }
        else:
            out['terms'][term] = None

    # Compute combined effect of RelSizeDiff when FocalHomeAdv = 0 (i.e., main effect)
    # and when FocalHomeAdv = 1 (main effect + interaction).
    # Use covariance matrix to get SE of linear combinations.
    index = list(params.index)

    def linear_combination(vec_dict):
        """
        vec_dict: mapping term -> multiplier (e.g., {'RelSizeDiff':1, 'RelSizeDiff:FocalHomeAdv':1})
        Returns: (estimate, se, z, p, 95% CI for coef, 95% CI for OR)
        """
        L = np.zeros(len(index))
        for t, mul in vec_dict.items():
            if t in index:
                L[index.index(t)] = mul
            else:
                # if term not present, multiplier remains zero (effectively ignored)
                pass
        est = float(np.dot(L, params.values))
        var = float(np.dot(L, np.dot(cov.values, L)))
        se_l = sqrt(var) if var >= 0 else float('nan')
        z = est / se_l if se_l != 0 else float('nan')
        p = float(2 * (1.0 - norm_cdf(abs(z)))) if not np.isnan(z) else float('nan')
        ci_low = est - 1.96 * se_l
        ci_high = est + 1.96 * se_l
        or_est = float(np.exp(est))
        or_ci_low = float(np.exp(ci_low))
        or_ci_high = float(np.exp(ci_high))
        return {
            'coef': est,
            'se': se_l,
            'z': z,
            'p_value': p,
            '95%_CI_coef': [ci_low, ci_high],
            'odds_ratio': or_est,
            '95%_CI_odds_ratio': [or_ci_low, or_ci_high],
        }

    # RelSizeDiff effect when FocalHomeAdv = 0 -> just RelSizeDiff coefficient
    out['combined_effects']['RelSizeDiff_when_FocalHomeAdv_0'] = linear_combination({term_rel: 1.0})

    # RelSizeDiff effect when FocalHomeAdv = 1 -> RelSizeDiff + RelSizeDiff:FocalHomeAdv
    out['combined_effects']['RelSizeDiff_when_FocalHomeAdv_1'] = linear_combination({term_rel: 1.0, term_int: 1.0})

    # Also include main effect of FocalHomeAdv (difference in log-odds when RelSizeDiff = 0)
    out['combined_effects']['FocalHomeAdv_main_effect_at_RelSizeDiff_0'] = linear_combination({term_home: 1.0})

    # Pack final object and human-readable description
    description_lines = []
    description_lines.append(
        "Extracted coefficients, standard errors, z-statistics, two-sided p-values, "
        "and 95% confidence intervals (for coefficients and odds ratios) for:"
    )
    description_lines.append(" - RelSizeDiff (numerical advantage of focal over other group)")
    description_lines.append(" - FocalHomeAdv (1 if contest closer to focal group's home-range center)")
    description_lines.append(" - Their interaction (RelSizeDiff:FocalHomeAdv)")
    description_lines.append("")
    description_lines.append(
        "Interpretation guidance: a positive coefficient for RelSizeDiff means that an "
        "increase of one unit in focal group's numerical advantage increases the log-odds "
        "of the focal group winning; the odds ratio > 1 means multiplicative increase in odds. "
        "The interaction term shows whether the effect of numerical advantage differs when the contest "
        "is nearer the focal group's home range (FocalHomeAdv = 1)."
    )
    description_lines.append(
        "The 'combined_effects' entries give the estimated effect (and inference) of a one-unit increase "
        "in RelSizeDiff when FocalHomeAdv = 0 and when FocalHomeAdv = 1 (i.e., they show how the slope changes "
        "with location). The p-values indicate whether these effects are statistically distinguishable from 0."
    )
    description_lines.append(
        "Use the odds_ratio and its CI to state effect size on the multiplicative scale: e.g., an odds_ratio of 1.5 "
        "means a one-unit increase in the predictor multiplies the odds of focal group winning by 1.5."
    )

    return {
        "object": out,
        "description": "\n".join(description_lines)
    }