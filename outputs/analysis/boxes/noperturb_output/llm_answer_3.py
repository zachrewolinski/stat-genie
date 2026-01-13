def extract_final_answer(model_output):
    """
    Extracts per-culture age effects (coefficients, SE, z, p, 95% CI, odds-ratio and OR CI)
    from a statsmodels GLMResultsWrapper fitted with the formula:
        MajorityChoice ~ age_c * C(culture) + is_male + majority_first

    Returns:
      {
        "object": list_of_dicts_per_culture,
        "description": brief_explanation_string
      }

    Each dict in list_of_dicts_per_culture contains:
      - culture: culture level name (string)
      - age_coef: estimated log-odds coefficient for age in that culture
      - se: standard error (delta-method when interaction applies)
      - z: z-statistic (coef / se)
      - p: two-sided p-value for the age effect in that culture
      - ci_lower, ci_upper: 95% Wald confidence interval for the age coefficient
      - odds_ratio: exp(age_coef)
      - or_ci_lower, or_ci_upper: 95% CI for the odds ratio
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # Basic checks
    if model_output is None:
        return {
            "object": None,
            "description": "No model_output provided."
        }

    # Extract coefficients, covariance matrix, etc.
    params = model_output.params
    cov = model_output.cov_params()
    param_names = list(params.index)

    # Ensure 'age_c' is present
    if 'age_c' not in param_names:
        return {
            "object": None,
            "description": "The model does not contain a main effect parameter named 'age_c'."
        }

    # Try to obtain full list of culture values from the original data if available
    cultures_in_data = None
    try:
        df = model_output.model.data.frame
        if 'culture' in df.columns:
            # preserve the observed unique values in the data (order as observed)
            cultures_in_data = list(pd.unique(df['culture']))
    except Exception:
        cultures_in_data = None

    # Find interaction parameter names that involve age_c and culture
    # Typical names: 'age_c:C(culture)[T.X]' or 'age_c:C(culture)[T.X]'
    interaction_params = [n for n in param_names if (n != 'age_c') and ('age_c' in n)]

    # Extract culture levels from interaction parameter names using regex
    extracted_levels = []
    interaction_map = {}  # map culture_level -> interaction_param_name
    prog = re.compile(r'\[T\.(.*)\]')  # captures level inside [T.level]
    for name in interaction_params:
        m = prog.search(name)
        if m:
            level = m.group(1)
        else:
            # fallback: try splitting by ':' and taking last token
            parts = name.split(':')[-1]
            # try to extract after '[' and ']' if present
            m2 = re.search(r'\[(.*)\]', parts)
            if m2:
                level = m2.group(1).replace('T.', '')
            else:
                # As a last resort, use the whole param name
                level = parts
        extracted_levels.append(level)
        interaction_map[level] = name

    extracted_levels = list(dict.fromkeys(extracted_levels))  # unique, preserve order

    # Determine baseline culture: the one not present among extracted_levels (treatment coding baseline)
    baseline = None
    if cultures_in_data is not None:
        # choose the first culture in observed unique list that is NOT in extracted_levels
        for c in cultures_in_data:
            if c not in extracted_levels:
                baseline = c
                break
        if baseline is None and len(cultures_in_data) > 0:
            # fallback to first observed
            baseline = cultures_in_data[0]
    else:
        # if we don't have original data, infer baseline as 'baseline' (unknown)
        # but we can still report effects for extracted levels and a 'baseline' entry
        baseline = 'baseline (unnamed)'

    # Build list of cultures to report: baseline + all extracted_levels (avoid duplicates)
    cultures_to_report = [baseline] + [c for c in extracted_levels if c != baseline]

    results_list = []
    for cult in cultures_to_report:
        if cult == baseline:
            # age effect is the main age_c coefficient
            coef = params['age_c']
            var = cov.loc['age_c', 'age_c']
            interaction_name = None
        else:
            # age effect = age_c + age_c:C(culture)[T.cult]
            if cult not in interaction_map:
                # If interaction param not found, skip or report NA
                results_list.append({
                    'culture': cult,
                    'age_coef': np.nan,
                    'se': np.nan,
                    'z': np.nan,
                    'p': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'odds_ratio': np.nan,
                    'or_ci_lower': np.nan,
                    'or_ci_upper': np.nan,
                    'note': 'interaction parameter not found for this culture'
                })
                continue
            interaction_name = interaction_map[cult]
            coef = params['age_c'] + params[interaction_name]
            # delta-method variance: var(age_c) + var(interaction) + 2*cov(age_c, interaction)
            var_age = cov.loc['age_c', 'age_c']
            var_inter = cov.loc[interaction_name, interaction_name]
            cov_term = cov.loc['age_c', interaction_name]
            var = var_age + var_inter + 2 * cov_term

        se = np.sqrt(var) if var >= 0 else np.nan
        if not np.isfinite(se) or se == 0:
            z = np.nan
            p = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
        else:
            z = coef / se
            p = 2 * (1 - norm.cdf(abs(z)))
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

        # Odds ratio and CI
        try:
            odds_ratio = float(np.exp(coef))
            or_ci_lower = float(np.exp(ci_lower))
            or_ci_upper = float(np.exp(ci_upper))
        except Exception:
            odds_ratio = np.nan
            or_ci_lower = np.nan
            or_ci_upper = np.nan

        results_list.append({
            'culture': cult,
            'age_coef': float(coef) if np.isfinite(coef) else np.nan,
            'se': float(se) if np.isfinite(se) else np.nan,
            'z': float(z) if np.isfinite(z) else np.nan,
            'p': float(p) if np.isfinite(p) else np.nan,
            'ci_lower': float(ci_lower) if np.isfinite(ci_lower) else np.nan,
            'ci_upper': float(ci_upper) if np.isfinite(ci_upper) else np.nan,
            'odds_ratio': odds_ratio,
            'or_ci_lower': or_ci_lower,
            'or_ci_upper': or_ci_upper,
            'interaction_param': interaction_name
        })

    description = (
        "This output reports, for each cultural site, the estimated effect of age (age_c) on the log-odds "
        "of choosing the majority option, along with SE, z, two-sided p-value, 95% Wald confidence interval, "
        "and the corresponding odds ratio (with 95% CI). The baseline culture is the reference level used by "
        "the model (no 'C(culture)[T.X]' term). For non-baseline cultures, the age effect is the sum of the "
        "main 'age_c' coefficient and the culture-specific age-by-culture interaction parameter (delta-method "
        "is used to compute SE and CI)."
    )

    return {
        "object": results_list,
        "description": description
    }