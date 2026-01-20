def extract_final_answer(model_output):
    """
    Extracts the age effect (age_c) on the log-odds of choosing the majority option
    for the reference culture and for each other culture (by combining the main
    age_c coefficient with age_c:C(culture)[T.<level>] interaction coefficients).
    Returns a dictionary with keys:
      - "object": pandas.DataFrame with rows for the reference culture and each culture level
                  found in the interaction terms. Columns: coef, se, z, p, ci_lower, ci_upper.
      - "description": brief explanation of what the table means.
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    res = model_output

    # Extract parameters, covariance matrix, and p-values
    params = res.params
    cov = res.cov_params()
    pvalues = res.pvalues

    if 'age_c' not in params.index:
        raise ValueError("The fitted model does not contain an 'age_c' parameter.")

    # Baseline (reference culture) age effect
    age_coef = params['age_c']
    age_var = cov.loc['age_c', 'age_c']
    age_se = np.sqrt(age_var)
    age_z = age_coef / age_se
    age_p = 2 * (1 - norm.cdf(abs(age_z)))
    zcrit = norm.ppf(0.975)
    age_ci = (age_coef - zcrit * age_se, age_coef + zcrit * age_se)

    results = {}
    # Use label 'reference' for the baseline culture (the omitted category of C(culture))
    results['reference'] = {
        'coef': float(age_coef),
        'se': float(age_se),
        'z': float(age_z),
        'p': float(age_p),
        'ci_lower': float(age_ci[0]),
        'ci_upper': float(age_ci[1]),
        'note': 'This is the age effect for the reference (omitted) culture'
    }

    # Find interaction parameter names like 'age_c:C(culture)[T.<level>]'
    interaction_names = [n for n in params.index if n.startswith('age_c:C(culture)')]

    for inter in interaction_names:
        # Parse culture level from the parameter name
        m = re.search(r'\[T\.?(.*)\]', inter)
        level = m.group(1) if m else inter

        inter_coef = params[inter]
        # Combined effect for this culture = age_c + interaction
        comb_coef = age_coef + inter_coef

        # Variance of the combination: Var(age_c) + Var(inter) + 2*Cov(age_c, inter)
        var_inter = cov.loc[inter, inter]
        cov_ai = cov.loc['age_c', inter]
        comb_var = age_var + var_inter + 2.0 * cov_ai

        # Protect against tiny negative variance from numerical issues
        comb_var = float(max(comb_var, 0.0))
        comb_se = np.sqrt(comb_var)
        comb_z = comb_coef / comb_se if comb_se > 0 else np.nan
        comb_p = 2 * (1 - norm.cdf(abs(comb_z))) if comb_se > 0 else np.nan
        comb_ci = (comb_coef - zcrit * comb_se, comb_coef + zcrit * comb_se) if comb_se > 0 else (np.nan, np.nan)

        results[level] = {
            'coef': float(comb_coef),
            'se': float(comb_se),
            'z': float(comb_z),
            'p': float(comb_p),
            'ci_lower': float(comb_ci[0]),
            'ci_upper': float(comb_ci[1]),
            'interaction_coef': float(inter_coef),
            'interaction_p': float(pvalues.get(inter, np.nan)),
            'note': f'age effect for culture level "{level}" (age_c + {inter})'
        }

    # Convert to DataFrame for easy viewing
    df = pd.DataFrame.from_dict(results, orient='index')
    # Reorder columns for readability
    cols_order = ['coef', 'se', 'z', 'p', 'ci_lower', 'ci_upper']
    # If extra columns exist (interaction info), keep them after the main ones
    other_cols = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + other_cols]

    description = (
        "Table gives estimated effect of age (age_c, mean-centered) on the log-odds of choosing "
        "the majority option by cultural context. 'reference' is the omitted (baseline) culture; "
        "other rows give the combined effect (age_c + age_c:C(culture)[T.<level>]) for each listed culture. "
        "Columns: coef = change in log-odds per year of age; se = standard error (cluster-robust if used); "
        "z, p = test statistics and two-sided p-value; ci_lower/ci_upper = 95% confidence interval. "
        "A positive coef indicates increasing reliance on the majority option with age in that culture."
    )

    return {"object": df, "description": description}