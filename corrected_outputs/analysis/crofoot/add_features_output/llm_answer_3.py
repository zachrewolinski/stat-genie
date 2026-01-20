def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals,
    odds ratios, and marginal effects for the key predictors relating to:
      - rel_size_log (relative group size)
      - location_advantage (location advantage indicator)
      - rel_size_log_x_loc (interaction)
    Also computes the marginal effect of rel_size_log when location_advantage = 0 and = 1,
    with standard errors, z-statistics, p-values, and confidence intervals.

    Returns:
      {
        "object": results_dict,   # numeric results (dict)
        "description": text       # short interpretation of what the numbers mean
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output  # statsmodels GLMResultsWrapper

    # Safe retrieval of parameter table info
    params = pd.Series(res.params)
    bse = pd.Series(res.bse)
    pvals = pd.Series(res.pvalues)
    conf = res.conf_int()  # DataFrame with two columns [0]=lower, [1]=upper
    conf.columns = ['2.5%', '97.5%']

    # Keys we care about
    key_vars = ['rel_size_log', 'location_advantage', 'rel_size_log_x_loc']
    results = {}

    for var in key_vars:
        if var in params.index:
            coef = float(params[var])
            se = float(bse[var]) if var in bse.index else None
            p = float(pvals[var]) if var in pvals.index else None
            ci_low = float(conf.loc[var, '2.5%'])
            ci_high = float(conf.loc[var, '97.5%'])
            orr = float(np.exp(coef))
            orr_ci = (float(np.exp(ci_low)), float(np.exp(ci_high)))

            results[var] = {
                'coef': round(coef, 4),
                'se': round(se, 4) if se is not None else None,
                'p_value': round(p, 4) if p is not None else None,
                '95%_CI_coef': (round(ci_low, 4), round(ci_high, 4)),
                'odds_ratio': round(orr, 4),
                '95%_CI_odds_ratio': (round(orr_ci[0], 4), round(orr_ci[1], 4)),
            }
        else:
            results[var] = None

    # Compute marginal effect of rel_size_log when location_advantage = 0 and = 1
    # effect at loc=0 is coef(rel_size_log)
    # effect at loc=1 is coef(rel_size_log) + coef(rel_size_log_x_loc)
    cov = res.cov_params()  # DataFrame

    if ('rel_size_log' in params.index) and ('rel_size_log_x_loc' in params.index):
        b1 = params['rel_size_log']
        b3 = params['rel_size_log_x_loc']
        # effect when loc = 0
        eff0 = float(b1)
        se0 = float(bse['rel_size_log'])
        z0 = eff0 / se0 if se0 != 0 else np.nan
        p0 = float(2 * (1 - (abs(z0) / abs(z0)))) if np.isnan(z0) else float(2 * (1 - \
                (0.5 * (1 + np.math.erf(z0 / np.sqrt(2))))))  # fallback, but we'll compute p properly below

        # effect when loc = 1
        eff1 = float(b1 + b3)
        # variance of sum = var(b1)+var(b3)+2cov(b1,b3)
        var_b1 = float(cov.loc['rel_size_log', 'rel_size_log'])
        var_b3 = float(cov.loc['rel_size_log_x_loc', 'rel_size_log_x_loc'])
        cov_b1b3 = float(cov.loc['rel_size_log', 'rel_size_log_x_loc'])
        var_sum = var_b1 + var_b3 + 2 * cov_b1b3
        se1 = float(np.sqrt(var_sum)) if var_sum >= 0 else float(np.nan)
        # z and p for eff1
        z1 = eff1 / se1 if se1 != 0 else np.nan
        # compute two-sided p-values using normal distribution
        from scipy import stats
        p0 = float(2 * (1 - stats.norm.cdf(abs(eff0 / se0)))) if se0 != 0 else float(np.nan)
        p1 = float(2 * (1 - stats.norm.cdf(abs(z1)))) if se1 != 0 else float(np.nan)
        # 95% CI for effects on log-odds scale
        ci0 = (eff0 - 1.96 * se0, eff0 + 1.96 * se0)
        ci1 = (eff1 - 1.96 * se1, eff1 + 1.96 * se1)
        # convert to odds ratio scale
        or0 = float(np.exp(eff0))
        or1 = float(np.exp(eff1))
        or0_ci = (float(np.exp(ci0[0])), float(np.exp(ci0[1])))
        or1_ci = (float(np.exp(ci1[0])), float(np.exp(ci1[1])))

        results['marginal_effect_rel_size'] = {
            'when_location_advantage_0': {
                'coef_log_odds': round(eff0, 4),
                'se': round(se0, 4),
                'z': round(eff0 / se0, 4),
                'p_value': round(p0, 4),
                '95%_CI_log_odds': (round(ci0[0], 4), round(ci0[1], 4)),
                'odds_ratio': round(or0, 4),
                '95%_CI_odds_ratio': (round(or0_ci[0], 4), round(or0_ci[1], 4)),
            },
            'when_location_advantage_1': {
                'coef_log_odds': round(eff1, 4),
                'se': round(se1, 4),
                'z': round(z1, 4),
                'p_value': round(p1, 4),
                '95%_CI_log_odds': (round(ci1[0], 4), round(ci1[1], 4)),
                'odds_ratio': round(or1, 4),
                '95%_CI_odds_ratio': (round(or1_ci[0], 4), round(or1_ci[1], 4)),
            }
        }
    else:
        results['marginal_effect_rel_size'] = None

    # Package final object and a concise description explaining interpretation
    description_lines = [
        "Extracted coefficients, standard errors, p-values, 95% CIs, and odds ratios",
        "for rel_size_log (relative focal group size), location_advantage (focal closer to its range center),",
        "and their interaction (rel_size_log_x_loc).",
        "Also computed the marginal effect of relative size on the log-odds of winning when the focal group",
        "does not have the location advantage (location_advantage = 0) and when it does (location_advantage = 1).",
        "",
        "Interpretation guidance:",
        "- coef (log-odds): positive => increases probability of focal group winning; negative => decreases.",
        "- odds_ratio > 1 => increases the odds of focal win; < 1 => decreases the odds.",
        "- For the interaction: a significant coefficient indicates the effect of relative size differs depending on",
        "  whether the focal group has the location advantage.",
        "- Use the marginal_effect_rel_size entries to see how the effect of relative size changes with location."
    ]
    description = "\n".join(description_lines)

    return {"object": results, "description": description}