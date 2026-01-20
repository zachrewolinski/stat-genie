def extract_final_answer(model_output):
    """
    Extracts age-related developmental effects (main effect and culture-specific slopes)
    from a statsmodels BinaryResultsWrapper (logit) fitted with formula:
      MajorityChoice ~ age_c * C(culture_cat) + is_boy + majority_first

    Returns:
      {
        "object": {
            "age_main": {coef, se, z, p, ci_lower, ci_upper, OR, OR_ci},
            "cultures": {
                "<culture_name_or_reference>": {
                    "slope_logodds", "se", "z", "p", "slope_ci", "OR", "OR_ci"
                }, ...
            },
            "raw_params": {param_name: value, ...},
            "raw_pvalues": {param_name: pval, ...},
            "raw_conf_int": DataFrame-like (2-col) of conf_int for original params (if available)
        },
        "description": "..."
      }
    """
    import re
    import math
    import numpy as np
    import pandas as pd

    res = model_output  # statsmodels BinaryResultsWrapper

    # helper: normal cdf using math.erf (no external packages)
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Collect raw parameter info
    params = res.params.copy()
    pvalues = res.pvalues.copy()
    try:
        conf = res.conf_int()  # returns DataFrame-like indexed by param names with [0]=lower [1]=upper
    except Exception:
        conf = None

    # Ensure covariance matrix available for computing SE of linear combos
    cov = res.cov_params()

    # Find the main age coefficient name (exact match 'age_c')
    if 'age_c' not in params.index:
        raise ValueError("Expected parameter named 'age_c' in model parameters. Found: {}".format(list(params.index)))

    age_coef = float(params['age_c'])
    age_se = float(res.bse['age_c'])
    age_z = age_coef / age_se if age_se != 0 else float('nan')
    age_p = float(pvalues['age_c'])
    if conf is not None and 'age_c' in conf.index:
        age_ci = (float(conf.loc['age_c', 0]), float(conf.loc['age_c', 1]))
    else:
        # approximate 95% CI
        age_ci = (age_coef - 1.96 * age_se, age_coef + 1.96 * age_se)
    age_or = math.exp(age_coef)
    age_or_ci = (math.exp(age_ci[0]), math.exp(age_ci[1]))

    # Identify interaction terms of the form: age_c:C(culture_cat)[T.<LEVEL>]
    interaction_pattern = re.compile(r'^age_c:C\(culture_cat\)\[T\.(.+)\]$')
    interaction_terms = {}
    for name in params.index:
        m = interaction_pattern.match(name)
        if m:
            culture = m.group(1)
            interaction_terms[culture] = name

    # Identify which cultures had main-effect dummies (to list non-reference levels)
    main_dummy_pattern = re.compile(r'^C\(culture_cat\)\[T\.(.+)\]$')
    main_dummies = {}
    for name in params.index:
        m = main_dummy_pattern.match(name)
        if m:
            main_dummies[m.group(1)] = name

    # Prepare output per-culture slopes (log-odds change per unit age_c).
    # For the omitted (reference) culture, slope = age_coef.
    results_by_culture = {}

    # Reference/baseline culture (omitted category) — label as 'reference (omitted)'
    results_by_culture['reference (omitted)'] = {
        'slope_logodds': age_coef,
        'se': age_se,
        'z': age_z,
        'p': age_p,
        'slope_ci': age_ci,
        'OR': age_or,
        'OR_ci': age_or_ci,
        'note': "This is the slope for the omitted (reference) culture used by the model. "
                "If you want a specific culture name for the reference level, check the original data or model encoding."
    }

    # For each culture with an interaction term, compute culture-specific slope = age_coef + interaction_coef
    for culture, term in interaction_terms.items():
        int_coef = float(params[term])
        # variance of (age + interaction) = var(age) + var(interaction) + 2*cov(age, interaction)
        var_age = float(cov.loc['age_c', 'age_c'])
        var_int = float(cov.loc[term, term])
        cov_ai = float(cov.loc['age_c', term])
        slope = age_coef + int_coef
        slope_se = math.sqrt(var_age + var_int + 2.0 * cov_ai)
        slope_z = slope / slope_se if slope_se != 0 else float('nan')
        slope_p = 2.0 * (1.0 - norm_cdf(abs(slope_z)))
        slope_ci = (slope - 1.96 * slope_se, slope + 1.96 * slope_se)
        slope_or = math.exp(slope)
        slope_or_ci = (math.exp(slope_ci[0]), math.exp(slope_ci[1]))

        results_by_culture[culture] = {
            'slope_logodds': slope,
            'se': slope_se,
            'z': slope_z,
            'p': slope_p,
            'slope_ci': slope_ci,
            'OR': slope_or,
            'OR_ci': slope_or_ci,
            'interaction_term': term,
            'interaction_coef': int_coef,
            'interaction_p': float(pvalues.get(term, np.nan))
        }

    # If there were no interaction terms, that means the slope is assumed identical across cultures.
    if len(interaction_terms) == 0:
        # Remove the 'reference' placeholder and instead present a single summary entry
        results_by_culture = {
            'all_cultures (common slope)': {
                'slope_logodds': age_coef,
                'se': age_se,
                'z': age_z,
                'p': age_p,
                'slope_ci': age_ci,
                'OR': age_or,
                'OR_ci': age_or_ci,
                'note': "No age-by-culture interaction terms detected: the model estimates a common age slope across cultures."
            }
        }

    # Package raw params/pvalues and conf_int if available (make JSON-serializable friendly)
    raw_params = {str(k): float(v) for k, v in params.items()}
    raw_pvals = {str(k): float(v) for k, v in pvalues.items()}
    raw_conf = None
    if conf is not None:
        raw_conf = {str(idx): (float(conf.loc[idx, 0]), float(conf.loc[idx, 1])) for idx in conf.index}

    output_object = {
        'age_main': {
            'coef_logodds': age_coef,
            'se': age_se,
            'z': age_z,
            'p': age_p,
            'ci_95': age_ci,
            'OR': age_or,
            'OR_95_CI': age_or_ci
        },
        'cultures': results_by_culture,
        'raw_params': raw_params,
        'raw_pvalues': raw_pvals,
        'raw_conf_int': raw_conf
    }

    description = (
        "This output summarizes the developmental effect of centered age (age_c) on the probability of choosing "
        "the majority-demonstrated option in log-odds units, and provides culture-specific slopes where an "
        "age_c:C(culture_cat)[T.<level>] interaction was estimated. "
        "For the omitted (reference) culture, the slope equals the 'age_c' coefficient. "
        "For any culture with an interaction term, the culture-specific slope = age_c + interaction_coef. "
        "Each slope entry includes its standard error, z-score, two-sided p-value (normal approximation), "
        "95% CI on the log-odds scale, and the exponentiated odds ratio with its 95% CI. "
        "Interpretation: a positive slope_logodds means that as age increases (per unit of centered age), "
        "the log-odds (and thus probability) of choosing the majority option increases; a negative slope means it decreases. "
        "Use the OR (exp(slope)) to interpret multiplicative change in odds per unit increase in centered age."
    )

    return {"object": output_object, "description": description}