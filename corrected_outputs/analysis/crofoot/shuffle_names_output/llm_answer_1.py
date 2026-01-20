def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, z-stats, p-values and 95% CIs for:
      - the main effect of RelGroupSize
      - interaction terms RelGroupSize:C(Location)[T.*]
    and computes the combined effect of RelGroupSize at each Location level
    (i.e., the slope of RelGroupSize within each location = main RelGroupSize coef
    + interaction coef for that location, if present).

    Returns:
      {
        "object": pandas.DataFrame with rows for:
                    - "RelGroupSize (reference location)" (base slope)
                    - "RelGroupSize at <Level>" for each non-reference Location level
                  columns: term, estimate, se, z, pvalue, ci_lower, ci_upper
        "description": human-readable explanation of what these numbers mean
      }
    """
    import re
    import pandas as pd
    import numpy as np
    from scipy import stats

    res = model_output  # expected to be a statsmodels results object (possibly robustcov)
    # Basic parameter objects
    params = getattr(res, 'params', None)
    bse = getattr(res, 'bse', None)
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        # fallback: try attribute directly
        cov = getattr(res, 'cov_params', None)
        if callable(cov):
            cov = cov()

    if params is None or bse is None:
        raise ValueError("Provided model_output does not have accessible params or bse attributes.")

    # Convert to Series/DataFrame for easier indexing
    params = pd.Series(params)
    bse = pd.Series(bse)

    # Find main RelGroupSize term name (should be exactly 'RelGroupSize')
    if 'RelGroupSize' not in params.index:
        # Try to find any parameter that includes 'RelGroupSize' but not interaction prefix
        candidates = [n for n in params.index if re.fullmatch(r'RelGroupSize', n) or re.fullmatch(r'.*RelGroupSize$', n)]
        if not candidates:
            raise ValueError("Could not find a parameter named 'RelGroupSize' in the model output.")
        rel_name = candidates[0]
    else:
        rel_name = 'RelGroupSize'

    # Identify interaction terms of the form RelGroupSize:C(Location)[T.<Level>] (or similar naming)
    interaction_pattern = re.compile(r'^RelGroupSize:.*Location.*\[T\.?(?P<lvl>.+)\]$')
    # Some versions may name it like 'RelGroupSize:C(Location)[T.Away]' or 'RelGroupSize:C(Location)[T.Away]'
    interaction_terms = {}
    for name in params.index:
        m = interaction_pattern.match(name)
        if m:
            lvl = m.group('lvl')
            interaction_terms[lvl] = name

    # Identify location levels present by also checking C(Location) main effects if needed
    # Determine reference level by checking which levels have interactions; reference is the one without an interaction term
    # But we cannot reliably get all factor levels from params alone for the reference unless from model data.
    # We'll present the reference as "reference_location" and include explicit rows for detected levels.
    rows = []

    # Base (reference) slope of RelGroupSize
    base_coef = float(params[rel_name])
    base_var = float(cov.loc[rel_name, rel_name]) if cov is not None and rel_name in cov.index else float(bse[rel_name])**2
    base_se = float(np.sqrt(base_var))
    base_z = base_coef / base_se if base_se > 0 else np.nan
    base_p = 2 * stats.norm.sf(abs(base_z)) if not np.isnan(base_z) else np.nan
    ci_lo = base_coef - 1.96 * base_se
    ci_hi = base_coef + 1.96 * base_se

    rows.append({
        'term': f'RelGroupSize (reference location)',
        'estimate': base_coef,
        'se': base_se,
        'z': base_z,
        'pvalue': base_p,
        'ci_lower': ci_lo,
        'ci_upper': ci_hi
    })

    # For each detected interaction level, compute combined effect (base + interaction)
    for lvl, it_name in interaction_terms.items():
        inter_coef = float(params[it_name])
        # construct variance of sum: var(base)+var(inter)+2*cov(base,inter)
        if cov is not None and rel_name in cov.index and it_name in cov.index:
            var_sum = float(cov.loc[rel_name, rel_name] + cov.loc[it_name, it_name] + 2 * cov.loc[rel_name, it_name])
        else:
            # fallback: use bse (assume zero covariance) - conservative but not ideal
            var_sum = (float(bse[rel_name])**2) + (float(bse[it_name])**2)
        sum_coef = base_coef + inter_coef
        sum_se = float(np.sqrt(var_sum)) if var_sum >= 0 else np.nan
        sum_z = sum_coef / sum_se if sum_se > 0 else np.nan
        sum_p = 2 * stats.norm.sf(abs(sum_z)) if not np.isnan(sum_z) else np.nan
        sum_ci_lo = sum_coef - 1.96 * sum_se
        sum_ci_hi = sum_coef + 1.96 * sum_se

        rows.append({
            'term': f'RelGroupSize at Location={lvl}',
            'estimate': sum_coef,
            'se': sum_se,
            'z': sum_z,
            'pvalue': sum_p,
            'ci_lower': sum_ci_lo,
            'ci_upper': sum_ci_hi
        })

    # Also include the raw parameter rows for main location dummies and interactions (if user wants them)
    # Filter relevant raw params to show in output for completeness
    raw_rows = []
    for name in params.index:
        if ('RelGroupSize' in name) or ('C(Location)' in name) or ('Location' in name):
            coef = float(params[name])
            se_val = float(bse[name]) if name in bse.index else np.nan
            z_val = coef / se_val if se_val > 0 else np.nan
            p_val = 2 * stats.norm.sf(abs(z_val)) if not np.isnan(z_val) else np.nan
            raw_rows.append({
                'param': name,
                'estimate': coef,
                'se': se_val,
                'z': z_val,
                'pvalue': p_val,
                'ci_lower': coef - 1.96 * se_val if not np.isnan(se_val) else np.nan,
                'ci_upper': coef + 1.96 * se_val if not np.isnan(se_val) else np.nan
            })

    result_df = pd.DataFrame(rows).set_index('term')
    raw_params_df = pd.DataFrame(raw_rows).set_index('param')

    description_lines = [
        "This output reports the estimated log-odds slope of RelGroupSize for the reference location",
        "and the combined slope (RelGroupSize main effect + interaction) for each detected Location level.",
        "Columns are: estimate (log-odds), se (standard error), z (Wald z-statistic), pvalue (two-sided),",
        "ci_lower and ci_upper (approximate 95% CIs using normal approximation).",
        "",
        "Interpretation notes:",
        "- A positive estimate means that a larger relative group size increases the log-odds of the focal group winning.",
        "- To assess statistical significance, check pvalue (e.g., p < 0.05).",
        "- Differences between location levels are indicated by the interaction terms; a significant interaction",
        "  means the slope of RelGroupSize differs between that level and the reference location.",
        "",
        "Returned objects:",
        "- 'object' is a dictionary with two DataFrames: 'slopes_by_location' and 'raw_params'.",
        "  'slopes_by_location' gives the estimated RelGroupSize slope in log-odds units for the reference and each detected location.",
        "  'raw_params' shows the raw model parameters related to RelGroupSize and Location for transparency."
    ]

    return {
        "object": {
            "slopes_by_location": result_df,
            "raw_params": raw_params_df
        },
        "description": "\n".join(description_lines)
    }