def extract_final_answer(model_output):
    """
    Extracts age-related effects (slopes) on preference for the majority option
    across cultural sites from the fitted 'majority_model' (logistic regression
    predicting majority_choice ~ age_z * C(culture_cat) + ...).

    Returns:
      {
        "object": pd.DataFrame (index = culture categories) with columns:
            - slope: estimated effect of a 1-SD increase in age (age_z) on log-odds of choosing the majority
            - se: standard error of the slope estimate
            - z: z-statistic (slope / se)
            - p_value: two-sided p-value
            - ci_lower, ci_upper: 95% Wald confidence interval for the slope
        "description": brief interpretation of the returned table
      }

    The function is robust to the interaction term ordering in parameter names
    (e.g., "age_z:C(culture_cat)[T.X]" or "C(culture_cat)[T.X]:age_z").
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # Helper to create an informative failure return
    def fail(msg):
        return {"object": None, "description": msg}

    if not isinstance(model_output, dict):
        return fail("model_output must be a dict as returned by the modeling function.")

    m = model_output.get('majority_model', None)
    if m is None:
        return fail("No 'majority_model' found or it could not be fit. Cannot extract majority-preference age effects.")

    # Extract parameter estimates, cov matrix, confints
    params = m.params  # pandas Series
    bse = m.bse
    cov = m.cov_params()

    # Get the data frame used to fit the model to find culture levels
    try:
        df_used = m.model.data.frame
        cultures = list(pd.unique(df_used['culture_cat'].astype(str)))
    except Exception:
        # Fallback: try to infer culture levels from parameter names
        cultures = []
        for name in params.index:
            if 'C(culture_cat)' in name and 'T.' in name:
                # parse text between 'T.' and ']'
                start = name.find('T.')
                end = name.find(']', start)
                if start != -1 and end != -1:
                    cultures.append(name[start+2:end])
        cultures = sorted(set(cultures))
        if len(cultures) == 0:
            return fail("Could not determine culture categories from the model object.")

    # Identify base age coefficient and interaction parameters
    # Possible base param name is 'age_z'
    if 'age_z' not in params.index:
        return fail("No main 'age_z' parameter found in the model parameters.")

    base_coef = params['age_z']
    # base variance
    base_var = cov.loc['age_z', 'age_z'] if ('age_z' in cov.index and 'age_z' in cov.columns) else (bse['age_z'] ** 2)

    # Find all interaction parameters that combine age_z and culture
    interaction_params = {}
    for pname in params.index:
        if ('age_z' in pname) and ('C(culture_cat)' in pname):
            # Extract category name after 'T.' up to the closing bracket if present
            cat = None
            tpos = pname.find('T.')
            if tpos != -1:
                # read until ']' or end
                endpos = pname.find(']', tpos)
                if endpos != -1:
                    cat = pname[tpos+2:endpos]
                else:
                    # fallback: take remainder after 'T.'
                    cat = pname[tpos+2:]
            else:
                # as a fallback, try splitting by ':' and taking the non-age piece
                parts = pname.split(':')
                for part in parts:
                    if 'C(culture_cat)' in part and 'T.' in part:
                        tpos = part.find('T.')
                        if tpos != -1:
                            endpos = part.find(']', tpos)
                            if endpos != -1:
                                cat = part[tpos+2:endpos]
                            else:
                                cat = part[tpos+2:]
            if cat is not None:
                interaction_params[cat] = pname

    # Determine the reference (baseline) culture: the one without a main-effect dummy key in params
    # Look for main-effect names like 'C(culture_cat)[T.<cat>]'
    referenced_cats = set()
    for pname in params.index:
        if pname.startswith('C(culture_cat)'):
            # parse 'C(culture_cat)[T.<cat>]'
            tpos = pname.find('T.')
            if tpos != -1:
                endpos = pname.find(']', tpos)
                if endpos != -1:
                    cat = pname[tpos+2:endpos]
                    referenced_cats.add(cat)
    # baseline candidates = those in cultures not referenced
    baseline_cats = [c for c in cultures if c not in referenced_cats]
    baseline = baseline_cats[0] if baseline_cats else cultures[0]

    # For each culture, compute slope = base_coef + interaction(if any)
    rows = []
    for cat in cultures:
        inter_name = interaction_params.get(cat, None)
        if inter_name is None:
            slope = base_coef
            # variance is just base_var
            var = base_var
        else:
            inter_coef = params[inter_name]
            slope = base_coef + inter_coef
            # variance of sum = var(base) + var(inter) + 2*cov(base,inter)
            var_inter = cov.loc[inter_name, inter_name]
            cov_base_inter = cov.loc['age_z', inter_name]
            var = base_var + var_inter + 2.0 * cov_base_inter

        se = np.sqrt(var) if var >= 0 else np.nan
        z = slope / se if se and not np.isnan(se) else np.nan
        p = 2.0 * (1.0 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_lower = slope - norm.ppf(0.975) * se if not np.isnan(se) else np.nan
        ci_upper = slope + norm.ppf(0.975) * se if not np.isnan(se) else np.nan

        rows.append({
            'culture': cat,
            'slope_logodds_per_SD_age': slope,
            'se': se,
            'z': z,
            'p_value': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    result_df = pd.DataFrame(rows).set_index('culture')

    # Short interpretation summary
    alpha = 0.05
    sig_pos = result_df[(result_df['p_value'] < alpha) & (result_df['slope_logodds_per_SD_age'] > 0)].index.tolist()
    sig_neg = result_df[(result_df['p_value'] < alpha) & (result_df['slope_logodds_per_SD_age'] < 0)].index.tolist()

    desc_lines = []
    desc_lines.append("Per-culture estimated age slopes (effect of 1 SD increase in age on log-odds of choosing the majority).")
    desc_lines.append(f"Baseline (reference) culture used by the model: '{baseline}'. For the baseline, the displayed slope is the 'age_z' coefficient; for other cultures it is age_z + interaction_term.")
    if len(sig_pos) == 0 and len(sig_neg) == 0:
        desc_lines.append("No culture shows a statistically significant age-related change in majority preference at alpha=0.05.")
    else:
        if sig_pos:
            desc_lines.append("Cultures with significant positive age effects (increased reliance on majority with age): " + ", ".join(sig_pos))
        if sig_neg:
            desc_lines.append("Cultures with significant negative age effects (decreased reliance on majority with age): " + ", ".join(sig_neg))
    desc_lines.append(f"n_demo (number used to fit this model) = {model_output.get('n_demo', 'unknown')}")

    description = " ".join(desc_lines)

    return {"object": result_df, "description": description}