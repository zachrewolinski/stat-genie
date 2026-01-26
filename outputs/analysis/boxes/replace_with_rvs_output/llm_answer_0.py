def extract_final_answer(model_output):
    """
    Extract statistics describing how the effect of age (child development)
    relates to choosing the majority option across cultures, using the
    fitted binary logistic model stored under 'logit_majority' in model_output.

    Returns:
      {
        "object": {
           "age_effects_by_culture": pandas.DataFrame with rows for each culture
              (including baseline culture_1). Columns: coef (d log-odds per unit
              centered age at age_c=0), se, z, p, OR, OR_ci_lower, OR_ci_upper,
              sig (True if p < .05).
           "age_quadratic": pandas.Series with coef, se, z, p for age_c2 (the quadratic term)
           "notes": textual notes about how the effects were computed
        },
        "description": short human-readable interpretation of the table
      }
    The function is robust to missing culture interaction columns (it will treat
    missing interactions as zero difference from baseline).
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Basic checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing 'logit_majority' result.")
    if 'logit_majority' not in model_output:
        raise ValueError("model_output does not contain key 'logit_majority'.")

    logit_res = model_output['logit_majority']

    # Extract params and covariance
    params = logit_res.params.copy()          # pd.Series
    cov = logit_res.cov_params().copy()       # pd.DataFrame

    if 'age_c' not in params.index:
        raise ValueError("'age_c' coefficient not found in model parameters.")
    # Prepare list of cultures based on interaction parameter names
    # We consider baseline as culture_1 (no explicit param), and any interactions present
    interaction_prefix = 'age_c:culture_'
    interaction_names = [name for name in params.index if name.startswith(interaction_prefix)]
    # Derive culture numbers from interaction names
    cultures = ['culture_1']  # baseline
    culture_numbers = []
    for name in interaction_names:
        try:
            # parse trailing number
            n = int(name.split(interaction_prefix)[1])
            culture_numbers.append(n)
        except Exception:
            continue
    culture_numbers = sorted(set(culture_numbers))
    for n in culture_numbers:
        cultures.append(f'culture_{n}')

    # Compute baseline age coefficient (at centered age = 0)
    beta_age = params['age_c']
    var_age = cov.loc['age_c', 'age_c']

    rows = []
    for c in cultures:
        if c == 'culture_1':
            coef = beta_age
            se = np.sqrt(var_age)
        else:
            inter_name = f'age_c:culture_{int(c.split("_")[1])}'
            if inter_name in params.index:
                beta_inter = params[inter_name]
                # variance of sum: var(age) + var(inter) + 2*cov(age,inter)
                var_inter = cov.loc[inter_name, inter_name]
                cov_ai = cov.loc['age_c', inter_name]
                coef = beta_age + beta_inter
                se = np.sqrt(var_age + var_inter + 2.0 * cov_ai)
            else:
                # interaction not present (shouldn't happen because we built cultures from interactions),
                # but keep fallback to baseline
                coef = beta_age
                se = np.sqrt(var_age)

        z = coef / se if se > 0 else np.nan
        p = 2.0 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
        OR = np.exp(coef)
        ci_low = np.exp(coef - 1.96 * se)
        ci_high = np.exp(coef + 1.96 * se)
        rows.append({
            'culture': c,
            'coef_age_at_mean_age_c': coef,
            'se': se,
            'z': z,
            'p': p,
            'OR_per_unit_age_at_mean': OR,
            'OR_95CI_lower': ci_low,
            'OR_95CI_upper': ci_high,
            'sig_p_lt_0.05': (p < 0.05) if not np.isnan(p) else False
        })

    age_effects_df = pd.DataFrame(rows).set_index('culture')

    # Also return the quadratic term info (age_c2)
    age2_info = None
    if 'age_c2' in params.index:
        b2 = params['age_c2']
        se2 = np.sqrt(cov.loc['age_c2', 'age_c2'])
        z2 = b2 / se2 if se2 > 0 else np.nan
        p2 = 2.0 * stats.norm.sf(abs(z2)) if not np.isnan(z2) else np.nan
        age2_info = pd.Series({
            'coef_age_c2': b2,
            'se': se2,
            'z': z2,
            'p': p2,
            'note': 'Quadratic term changes how the age effect changes across ages. The age-by-culture effects above are the instantaneous linear effect at centered age = 0.'
        })
    else:
        age2_info = pd.Series({
            'coef_age_c2': np.nan,
            'se': np.nan,
            'z': np.nan,
            'p': np.nan,
            'note': 'No quadratic (age_c2) term found in model.'
        })

    notes = (
        "Each row shows the estimated instantaneous effect of a one-unit increase in centered age "
        "(age_c) on the log-odds of choosing the majority option, evaluated at centered age = 0 "
        "(the sample mean age). For culture_1 (baseline) the value is the model's age_c coefficient; "
        "for other cultures it is age_c + age_c:culture_X. OR = exp(coef) is the multiplicative change in odds "
        "of choosing the majority per one-unit increase in centered age (at the mean). The age_c2 (quadratic) "
        "coefficient is provided because it means the marginal effect of age changes with age; the table above "
        "reports the instantaneous linear effect at age_c=0."
    )

    # Short interpretation string
    # We'll summarize which cultures show a statistically significant increasing or decreasing effect.
    sig_inc = age_effects_df[(age_effects_df['sig_p_lt_0.05']) & (age_effects_df['coef_age_at_mean_age_c'] > 0)].index.tolist()
    sig_dec = age_effects_df[(age_effects_df['sig_p_lt_0.05']) & (age_effects_df['coef_age_at_mean_age_c'] < 0)].index.tolist()
    summary_parts = []
    if sig_inc:
        summary_parts.append(f"Significant positive age effect (greater reliance on majority with age) in: {', '.join(sig_inc)}.")
    if sig_dec:
        summary_parts.append(f"Significant negative age effect (less reliance on majority with age) in: {', '.join(sig_dec)}.")
    if not summary_parts:
        summary_parts.append("No culture shows a statistically significant linear age effect at centered age = 0 (p < 0.05).")
    short_summary = " ".join(summary_parts)

    result_object = {
        'age_effects_by_culture': age_effects_df,
        'age_quadratic': age2_info,
        'notes': notes,
        'short_summary': short_summary
    }

    description = (
        "This output gives, for each culture (including baseline culture_1), the estimated linear effect of age "
        "(per one centered-age unit at the sample mean age) on the odds of choosing the majority option. "
        "Columns: coef (log-odds change), se, z, p, OR (odds ratio), and 95% CI for the OR. "
        "Also includes the age quadratic coefficient (age_c2) because it modifies how effects change across ages."
    )

    return {"object": result_object, "description": description}