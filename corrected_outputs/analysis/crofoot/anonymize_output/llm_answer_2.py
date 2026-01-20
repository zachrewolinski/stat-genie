def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (logistic regression).
    Returns a dictionary with:
      - "object": a dict of extracted numeric results (coefficients, SEs, p-values,
                  95% CIs, odds ratios and CIs) for focal predictors and marginal
                  effects of SizeRatio by InFocalHome (outside vs inside focal home).
      - "description": a short explanation of what the numbers mean for the
                       study question.

    Expected model parameter names (as in the modeling code):
      'SizeRatio', 'InFocalHome', 'SizeRatio_InFocalHome', 'DistanceDiff', ...
    """
    import math

    res = model_output

    # Basic parameter tables
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    try:
        conf = res.conf_int()  # DataFrame or array-like indexed by param name
    except Exception:
        # If conf_int not available, compute approximate using normal approx
        conf = None

    def get_conf_interval(name):
        if conf is None:
            coef = float(params[name])
            se = float(bse[name])
            return [coef - 1.96 * se, coef + 1.96 * se]
        else:
            ci_row = conf.loc[name]
            return [float(ci_row[0]), float(ci_row[1])]

    out = {}

    # List of focal predictors we want to report explicitly
    focal_vars = ['SizeRatio', 'InFocalHome', 'SizeRatio_InFocalHome', 'DistanceDiff']

    for name in focal_vars:
        if name in params.index:
            coef = float(params[name])
            se = float(bse[name])
            p = float(pvals[name])
            ci = get_conf_interval(name)
            odds = math.exp(coef)
            or_ci = [math.exp(ci[0]), math.exp(ci[1])]
            out[name] = {
                'coef': coef,
                'se': se,
                'p_value': p,
                '95ci_coef': ci,
                'odds_ratio': odds,
                '95ci_odds_ratio': or_ci
            }

    # Marginal effects for SizeRatio when InFocalHome = 0 and = 1
    size_exists = 'SizeRatio' in params.index
    int_exists = 'SizeRatio_InFocalHome' in params.index

    if size_exists:
        beta_size = float(params['SizeRatio'])
        se_size = float(bse['SizeRatio'])
        ci_size = get_conf_interval('SizeRatio')
        out['SizeRatio_when_NotInFocalHome'] = {
            'coef': beta_size,
            'se': se_size,
            '95ci_coef': ci_size,
            'odds_ratio': math.exp(beta_size),
            '95ci_odds_ratio': [math.exp(ci_size[0]), math.exp(ci_size[1])],
            'interpretation': 'Effect of a one-unit increase in SizeRatio when InFocalHome=0'
        }

        if int_exists:
            beta_int = float(params['SizeRatio_InFocalHome'])
            # combined effect = beta_size + beta_int
            combined_beta = beta_size + beta_int

            # compute se of combined effect using cov_params
            cov = res.cov_params()
            # protect against missing names in cov matrix
            try:
                var_comb = (cov.loc['SizeRatio', 'SizeRatio']
                            + cov.loc['SizeRatio_InFocalHome', 'SizeRatio_InFocalHome']
                            + 2 * cov.loc['SizeRatio', 'SizeRatio_InFocalHome'])
                se_comb = math.sqrt(float(var_comb))
            except Exception:
                # fallback: approximate by sqrt(se^2 + se_int^2)
                se_comb = math.sqrt(se_size ** 2 + float(bse['SizeRatio_InFocalHome']) ** 2)

            ci_low = combined_beta - 1.96 * se_comb
            ci_high = combined_beta + 1.96 * se_comb
            or_comb = math.exp(combined_beta)
            or_ci_comb = [math.exp(ci_low), math.exp(ci_high)]

            # z and two-sided p-value using normal approximation
            if se_comb > 0:
                z = combined_beta / se_comb
                # standard normal cdf via erf to avoid external deps
                Phi = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
                p_comb = 2.0 * (1.0 - Phi)
            else:
                z = float('nan')
                p_comb = float('nan')

            out['SizeRatio_when_InFocalHome'] = {
                'coef': combined_beta,
                'se': se_comb,
                'z': z,
                'p_value': p_comb,
                '95ci_coef': [ci_low, ci_high],
                'odds_ratio': or_comb,
                '95ci_odds_ratio': or_ci_comb,
                'interpretation': 'Effect of a one-unit increase in SizeRatio when InFocalHome=1'
            }

            # Also provide an explicit test of interaction (beta_int)
            out['Interaction_term'] = {
                'coef': beta_int,
                'se': float(bse['SizeRatio_InFocalHome']),
                'p_value': float(pvals['SizeRatio_InFocalHome']),
                '95ci_coef': get_conf_interval('SizeRatio_InFocalHome'),
                'odds_ratio': math.exp(beta_int),
                '95ci_odds_ratio': [math.exp(x) for x in get_conf_interval('SizeRatio_InFocalHome')],
                'interpretation': 'How much the SizeRatio effect changes when contest is in the focal home (multiplicative on odds)'
            }

    # Small helper summary notes
    description_lines = [
        "Extracted coefficients, standard errors, two-sided p-values, 95% CIs, and odds ratios",
        "for the key predictors: SizeRatio, InFocalHome, their interaction (SizeRatio_InFocalHome), and DistanceDiff.",
        "Interpretation guidance:",
        "- Positive coef for SizeRatio: larger focal group increases log-odds (and odds) of winning.",
        "- Positive coef for InFocalHome: contests nearer focal group's home increase focal group's odds of winning.",
        "- Positive interaction coef: the benefit of a size advantage is larger when the contest is in the focal group's home.",
        "The returned 'object' contains numeric summaries for these quantities and the marginal effect",
        "of SizeRatio when InFocalHome = 0 (outside focal home) and when InFocalHome = 1 (inside/closer to focal home).",
        "Use the reported p-values / CIs to judge statistical evidence: p < 0.05 indicates conventional significance."
    ]

    return {
        "object": out,
        "description": " ".join(description_lines)
    }