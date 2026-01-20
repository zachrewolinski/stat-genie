def extract_final_answer(model_output):
    """
    Extract per-site age slopes (developmental change in reliance on majority) from a fitted
    statsmodels GLMResults (optionally with cluster-robust covariances).
    
    Returns a dictionary with:
      - "object": pandas.DataFrame with rows for each Site and columns:
          slope (log-odds per year), se, z, pvalue, ci_lower, ci_upper
      - "description": brief interpretation of the numbers.
    
    Notes:
      - The model formula expected: 'MajorityChoice ~ Age_c * C(Site) + Gender_male + MajorityFirst'
      - The baseline/reference site's slope is the coefficient "Age_c".
      - Other sites' slopes are Age_c + Age_c:C(Site)[T.<site>] (and standard errors use covariance).
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy import stats

    res = model_output

    # Extract parameter estimates, covariance matrix (robust if provided), and names
    try:
        params = res.params.copy()
        cov = res.cov_params().copy()
        pvalues = res.pvalues.copy()
        bse = res.bse.copy()
    except Exception as e:
        raise RuntimeError(f"Could not extract params/cov from model_output: {e}")

    # Try to get site categories from the original model data if available
    site_levels = None
    try:
        df = res.model.data.frame  # pandas DataFrame used to fit model
        if 'Site' in df.columns:
            site_col = df['Site']
            # If categorical, get the categories in their order; otherwise use unique values in order seen
            if hasattr(site_col, 'cat') and getattr(site_col, 'dtype').name.startswith('category'):
                site_levels = list(site_col.cat.categories)
            else:
                # preserve order of appearance
                site_levels = list(dict.fromkeys(list(site_col)))
    except Exception:
        site_levels = None

    # If not found from data, attempt to infer site names from coefficient names
    if site_levels is None:
        # look for C(Site)[T.<level>] patterns in param names
        pattern = re.compile(r'C\(Site\)\[T\.([^]]+)\]')
        inferred = []
        for name in params.index:
            m = pattern.search(name)
            if m:
                inferred.append(m.group(1))
        # we may also infer interaction terms Age_c:C(Site)[T.<level>]
        pattern2 = re.compile(r'Age_c:C\(Site\)\[T\.([^]]+)\]')
        for name in params.index:
            m = pattern2.search(name)
            if m and m.group(1) not in inferred:
                inferred.append(m.group(1))
        if inferred:
            # We cannot know the reference level from params alone reliably; assume the first seen is reference
            site_levels = inferred.copy()
        else:
            # give up and set a single unnamed site (only Age_c present)
            site_levels = []

    # Determine reference site (the baseline that has no C(Site)[T.<level>] term).
    # If we have categories from data, statsmodels/patsy uses the first category as the reference.
    reference = None
    if site_levels:
        reference = site_levels[0]

    # Identify the base Age_c coefficient
    if 'Age_c' not in params.index:
        raise RuntimeError("Model does not contain an 'Age_c' coefficient; cannot extract age effect.")
    beta_age = float(params['Age_c'])

    # Find interaction coefficient names and map to site
    interaction_map = {}  # site -> param name
    for name in params.index:
        # match 'Age_c:C(Site)[T.<site>]' exactly
        if name.startswith('Age_c:C(Site)[T.'):
            # extract site between '[T.' and ']'
            m = re.match(r'Age_c:C\(Site\)\[T\.([^]]+)\]', name)
            if m:
                site = m.group(1)
                interaction_map[site] = name

    # Build results per site
    rows = []
    if site_levels:
        # if we have explicit site list, include them; otherwise fall back to any sites found in interactions
        sites_to_report = site_levels
        # If interactions are present but the site list is missing the interaction sites, include them too
        for s in interaction_map:
            if s not in sites_to_report:
                sites_to_report.append(s)
    else:
        # No site-level info: report only reference slope (Age_c)
        sites_to_report = []

    # If no explicit sites detected, still report the baseline slope
    if not sites_to_report:
        # only baseline
        var_age = cov.loc['Age_c', 'Age_c'] if ('Age_c' in cov.index and 'Age_c' in cov.columns) else float(bse['Age_c'])**2
        se_age = np.sqrt(var_age)
        z = beta_age / se_age if se_age > 0 else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
        ci_low, ci_high = beta_age - 1.96 * se_age, beta_age + 1.96 * se_age
        rows.append({'Site': '(pooled)', 'slope_logodds_per_year': beta_age,
                     'se': se_age, 'z': z, 'pvalue': p,
                     'ci_lower': ci_low, 'ci_upper': ci_high})
    else:
        for i, site in enumerate(sites_to_report):
            if i == 0:
                # reference site: slope = Age_c
                slope = beta_age
                # variance = var(Age_c)
                if 'Age_c' in cov.index and 'Age_c' in cov.columns:
                    var = cov.loc['Age_c', 'Age_c']
                else:
                    var = float(bse['Age_c'])**2
            else:
                # other site: slope = Age_c + interaction
                inter_name = interaction_map.get(site, None)
                if inter_name is None:
                    # no interaction term found for this site: treat as equal to reference
                    slope = beta_age
                    if 'Age_c' in cov.index and 'Age_c' in cov.columns:
                        var = cov.loc['Age_c', 'Age_c']
                    else:
                        var = float(bse['Age_c'])**2
                else:
                    slope = beta_age + float(params[inter_name])
                    # variance = var(Age_c) + var(interaction) + 2*cov(Age_c, interaction)
                    if ('Age_c' in cov.index and 'Age_c' in cov.columns
                        and inter_name in cov.index and inter_name in cov.columns):
                        var = (cov.loc['Age_c', 'Age_c']
                               + cov.loc[inter_name, inter_name]
                               + 2.0 * cov.loc['Age_c', inter_name])
                    else:
                        # fallback to sum of bse^2 (ignores covariance)
                        var_age = float(bse['Age_c'])**2 if 'Age_c' in bse.index else np.nan
                        var_inter = float(bse[inter_name])**2 if inter_name in bse.index else np.nan
                        if not np.isnan(var_age) and not np.isnan(var_inter):
                            var = var_age + var_inter
                        else:
                            var = np.nan

            se = np.sqrt(var) if (var is not None and not np.isnan(var) and var >= 0) else np.nan
            z = slope / se if (se is not None and not np.isnan(se) and se > 0) else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            ci_low, ci_high = (slope - 1.96 * se, slope + 1.96 * se) if not np.isnan(se) else (np.nan, np.nan)

            rows.append({'Site': site, 'slope_logodds_per_year': slope,
                         'se': se, 'z': z, 'pvalue': p,
                         'ci_lower': ci_low, 'ci_upper': ci_high})

    results_df = pd.DataFrame(rows).set_index('Site')

    # Provide a human-readable description
    desc_lines = []
    desc_lines.append("Per-site age slopes (log-odds change in choosing the majority per additional year).")
    desc_lines.append("Positive slope => greater reliance on majority with age; negative => less reliance with age.")
    desc_lines.append("P-values and 95% CIs are computed using the model's covariance matrix (cluster-robust if provided).")
    description = " ".join(desc_lines)

    return {"object": results_df, "description": description}