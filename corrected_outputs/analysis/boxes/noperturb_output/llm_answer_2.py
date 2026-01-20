def extract_final_answer(model_output):
    """
    Extracts age-related effects (log-odds slopes and inference) on choosing the majority
    option for the reference site and for each culture/site that has an age interaction.
    Returns a dictionary with detailed numeric results in "object" and a plain-language
    summary in "description".
    """
    import numpy as np
    from scipy import stats

    res = model_output  # statsmodels BinaryResultsWrapper

    # Required parameter names
    age_name = 'age_c'
    interaction_prefix = 'age_x_culture_'
    culture_prefix = 'culture_'  # to check presence of culture dummies if desired

    params = res.params
    cov = res.cov_params()

    if age_name not in params.index:
        raise ValueError(f"Model output does not contain required coefficient '{age_name}'")

    # helper to compute CI and p-value from coef and se
    def summarize_coef(coef, se, alpha=0.05):
        z = coef / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_low = coef - z_crit * se
        ci_high = coef + z_crit * se
        return {
            'coef': float(coef),
            'se': float(se),
            'z': float(z),
            'p': float(p),
            'ci95': (float(ci_low), float(ci_high)),
            # also return odds ratio per 1 year increase and its CI
            'odds_ratio': float(np.exp(coef)),
            'odds_ratio_ci95': (float(np.exp(ci_low)), float(np.exp(ci_high)))
        }

    # Reference site (site 1) slope = coefficient on age_c
    age_coef = params[age_name]
    age_se = np.sqrt(float(cov.loc[age_name, age_name]))
    site_results = {}
    site_results['site_1'] = summarize_coef(age_coef, age_se)

    # For each culture 2..8 compute slope = age_c + age_x_culture_i (if present)
    significant_positive = []
    significant_negative = []
    nonsignificant = []

    for i in range(2, 9):
        inter_name = f'{interaction_prefix}{i}'
        # Only compute if interaction term exists in model
        if inter_name in params.index:
            inter_coef = params[inter_name]
            coef = age_coef + inter_coef
            # variance of sum = var(age) + var(inter) + 2*cov(age,inter)
            var = (
                float(cov.loc[age_name, age_name])
                + float(cov.loc[inter_name, inter_name])
                + 2.0 * float(cov.loc[age_name, inter_name])
            )
            se = np.sqrt(var) if var >= 0 else np.nan
            summary = summarize_coef(coef, se)
            site_results[f'site_{i}'] = summary

            # categorize by significance & direction
            if summary['p'] < 0.05:
                if summary['coef'] > 0:
                    significant_positive.append(f'site_{i}')
                else:
                    significant_negative.append(f'site_{i}')
            else:
                nonsignificant.append(f'site_{i}')
        else:
            # if interaction missing, we cannot compute a site-specific slope;
            # include a note that the interaction term was not present
            site_results[f'site_{i}'] = {
                'note': f"interaction term '{inter_name}' not present in model; cannot estimate site-specific age slope"
            }
            nonsignificant.append(f'site_{i}')

    # Also include reference site categorization
    if site_results['site_1']['p'] < 0.05:
        if site_results['site_1']['coef'] > 0:
            significant_positive.insert(0, 'site_1')
        else:
            significant_negative.insert(0, 'site_1')
    else:
        nonsignificant.insert(0, 'site_1')

    # Build description: plain-language interpretation emphasizing what the numbers mean
    desc_lines = []
    desc_lines.append("Computed per-site age slopes on the log-odds scale for choosing the majority option.")
    desc_lines.append("Interpretation: coef = change in log-odds of choosing the majority per 1 year increase in age.")
    desc_lines.append("Odds ratio = multiplicative change in odds of choosing majority per 1 year of age (exp(coef)).")
    desc_lines.append("")
    desc_lines.append("Results summary:")
    # Mention reference site
    r = site_results['site_1']
    desc_lines.append(
        f"- Reference (site_1): coef={r['coef']:.3f}, se={r['se']:.3f}, z={r['z']:.2f}, p={r['p']:.3f}, "
        f"OR={r['odds_ratio']:.3f}, 95%CI_OR=({r['odds_ratio_ci95'][0]:.3f}, {r['odds_ratio_ci95'][1]:.3f})"
    )
    # Mention other sites succinctly
    for i in range(2, 9):
        key = f'site_{i}'
        entry = site_results[key]
        if 'note' in entry:
            desc_lines.append(f"- {key}: {entry['note']}")
        else:
            desc_lines.append(
                f"- {key}: coef={entry['coef']:.3f}, se={entry['se']:.3f}, p={entry['p']:.3f}, "
                f"OR={entry['odds_ratio']:.3f}, 95%CI_OR=({entry['odds_ratio_ci95'][0]:.3f}, {entry['odds_ratio_ci95'][1]:.3f})"
            )

    # Summarize which sites show significant increases/decreases with age
    desc_lines.append("")
    if significant_positive:
        desc_lines.append(f"Sites with significant positive age effects (greater reliance on majority with age): {', '.join(significant_positive)}")
    else:
        desc_lines.append("No sites showed a significant positive age effect at p < 0.05.")
    if significant_negative:
        desc_lines.append(f"Sites with significant negative age effects (less reliance on majority with age): {', '.join(significant_negative)}")
    else:
        desc_lines.append("No sites showed a significant negative age effect at p < 0.05.")
    desc_lines.append(f"Sites without significant age effects: {', '.join(nonsignificant)}")

    # Optionally include basic model fit info if available
    try:
        nobs = int(res.nobs)
        llf = float(res.llf)
        desc_lines.append(f"Model: n={nobs}, log-likelihood={llf:.3f}")
    except Exception:
        pass

    description = "\n".join(desc_lines)

    return {
        "object": site_results,
        "description": description
    }