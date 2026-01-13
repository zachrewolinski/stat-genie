def extract_final_answer(model_output):
    """
    Extracts age-related coefficients and culture-specific linear age effects
    from a fitted statsmodels GLM/RegressionResults object.

    Returns a dictionary with:
      - "object": a dict containing:
          * age_c: coef, se, z, p, 95% CI
          * age_c_sq: coef, se, z, p, 95% CI
          * per_culture_linear_age_effects: for each non-reference culture
              - coef (age_c + interaction), se, z, p, 95% CI
            and for the reference (baseline) culture:
              - coef (age_c only), se, z, p, 95% CI
          * raw_params: full params Series (for inspection)
      - "description": short explanation of what these numbers mean
    """
    import re
    import math
    import numpy as np
    import pandas as pd

    # Helper: two-sided p-value from z using math.erfc for portability
    def z_to_p(z):
        return math.erfc(abs(z) / math.sqrt(2))

    # Try to get parameter estimates, cov matrix, pvalues, conf_int
    try:
        params = model_output.params  # pandas Series
    except Exception as e:
        raise ValueError("Could not extract params from model_output: %s" % e)

    try:
        cov = model_output.cov_params()  # DataFrame
    except Exception as e:
        # fallback: try to get unscaled cov or raise
        raise ValueError("Could not extract covariance matrix from model_output: %s" % e)

    try:
        pvalues = model_output.pvalues
    except Exception:
        # compute p-values from params and bse if available
        if hasattr(model_output, 'bse'):
            bse = model_output.bse
            zvals = params / bse
            pvalues = zvals.apply(z_to_p)
        else:
            pvalues = pd.Series(index=params.index, data=[np.nan]*len(params))

    # Confidence intervals (if available)
    try:
        ci_df = model_output.conf_int()
        # conf_int returns DataFrame with [lower, upper]
    except Exception:
        ci_df = None

    # Ensure age_c and age_c_sq exist
    if 'age_c' not in params.index:
        raise ValueError("Model does not contain an 'age_c' coefficient in params.")

    if 'age_c_sq' not in params.index:
        raise ValueError("Model does not contain an 'age_c_sq' coefficient in params.")

    # Base age effects
    age_coef = float(params['age_c'])
    age_se = float(np.sqrt(cov.loc['age_c', 'age_c']))
    age_z = age_coef / age_se if age_se != 0 else np.nan
    age_p = float(pvalues.get('age_c', z_to_p(age_z)))
    if ci_df is not None and 'age_c' in ci_df.index:
        age_ci = tuple(ci_df.loc['age_c'].values)
    else:
        age_ci = (age_coef - 1.96 * age_se, age_coef + 1.96 * age_se)

    age_sq_coef = float(params['age_c_sq'])
    age_sq_se = float(np.sqrt(cov.loc['age_c_sq', 'age_c_sq']))
    age_sq_z = age_sq_coef / age_sq_se if age_sq_se != 0 else np.nan
    age_sq_p = float(pvalues.get('age_c_sq', z_to_p(age_sq_z)))
    if ci_df is not None and 'age_c_sq' in ci_df.index:
        age_sq_ci = tuple(ci_df.loc['age_c_sq'].values)
    else:
        age_sq_ci = (age_sq_coef - 1.96 * age_sq_se, age_sq_coef + 1.96 * age_sq_se)

    # Identify culture levels from parameter names:
    # Look for patterns like "C(culture_cat)[T.site]" and interactions with age_c
    param_index = list(params.index.astype(str))

    # Find interaction params where age_c interacts with culture
    inter_patterns = [
        re.compile(r'age_c:C\(culture_cat\)\[T\.(.*?)\]'),
        re.compile(r'C\(culture_cat\)\[T\.(.*?)\]:age_c')
    ]
    interaction_map = {}  # culture_label -> param_name
    for pname in param_index:
        for pat in inter_patterns:
            m = pat.search(pname)
            if m:
                lab = m.group(1)
                interaction_map[lab] = pname
                break

    # Find cultural fixed-effect main terms (to detect which levels are present)
    main_culture_pattern = re.compile(r'C\(culture_cat\)\[T\.(.*?)\]')
    culture_levels = set()
    for pname in param_index:
        m = main_culture_pattern.search(pname)
        if m:
            culture_levels.add(m.group(1))

    # The baseline (reference) culture is not present in C(...) terms.
    # We'll assemble list of cultures for which we can report combined linear age effect:
    # - 'reference' (baseline): uses age_c only
    # - any cultures in interaction_map: combined = age_c + interaction_param
    per_culture = {}

    # Baseline culture (label as 'reference' because we don't know its name)
    baseline_label = 'reference (model baseline)'
    per_culture[baseline_label] = {}
    per_culture[baseline_label]['coef'] = age_coef
    per_culture[baseline_label]['se'] = age_se
    per_culture[baseline_label]['z'] = age_z
    per_culture[baseline_label]['p'] = age_p
    per_culture[baseline_label]['95_CI'] = age_ci

    # For each detected culture interaction, compute combined coef and SE using cov matrix
    for lab, pname in interaction_map.items():
        inter_coef = float(params[pname])
        # combined coef = age_coef + inter_coef
        comb_coef = age_coef + inter_coef

        # var(comb) = var(age_c) + var(inter) + 2*cov(age_c, inter)
        try:
            var_age = cov.loc['age_c', 'age_c']
            var_inter = cov.loc[pname, pname]
            cov_ai = cov.loc['age_c', pname]
            comb_var = var_age + var_inter + 2.0 * cov_ai
            comb_se = float(np.sqrt(comb_var)) if comb_var >= 0 else float(np.nan)
        except Exception:
            # If cov elements not available, fall back to NaN
            comb_se = float(np.nan)

        comb_z = comb_coef / comb_se if (comb_se and not np.isnan(comb_se)) else np.nan
        comb_p = float(z_to_p(comb_z)) if not np.isnan(comb_z) else float(np.nan)

        if ci_df is not None and 'age_c' in ci_df.index and pname in ci_df.index:
            # We don't have direct CI for the sum; compute via se
            comb_ci = (comb_coef - 1.96 * comb_se, comb_coef + 1.96 * comb_se)
        else:
            comb_ci = (comb_coef - 1.96 * comb_se, comb_coef + 1.96 * comb_se) if not np.isnan(comb_se) else (np.nan, np.nan)

        per_culture[lab] = {
            'interaction_param_name': pname,
            'interaction_coef': inter_coef,
            'coef': comb_coef,
            'se': comb_se,
            'z': comb_z,
            'p': comb_p,
            '95_CI': comb_ci
        }

    # Prepare output object
    output_object = {
        'age_c': {
            'coef': age_coef,
            'se': age_se,
            'z': age_z,
            'p': age_p,
            '95_CI': age_ci
        },
        'age_c_sq': {
            'coef': age_sq_coef,
            'se': age_sq_se,
            'z': age_sq_z,
            'p': age_sq_p,
            '95_CI': age_sq_ci
        },
        'per_culture_linear_age_effects': per_culture,
        'raw_params': params,
        'raw_cov': cov
    }

    description_lines = [
        "Extracted statistics relevant to how reliance on the majority develops with age:",
        "- 'age_c' is the model coefficient for centered age (linear effect). It represents the change in log-odds of choosing the majority per unit increase in age (centered) for the baseline culture.",
        "- 'age_c_sq' is the quadratic age term (shared across cultures in this model); its sign indicates acceleration (+) or deceleration (-) of the age effect.",
        "- 'per_culture_linear_age_effects' gives the combined linear age effect for each culture:",
        "    * For the model baseline (reference) culture, the linear effect is simply the 'age_c' coefficient.",
        "    * For other cultures, the linear effect = age_c + age_c:C(culture_cat)[T.<level>] (we report coef, SE, z, two-sided p, and 95% CI).",
        "- All reported coefficients are on the log-odds scale (logit). To convert to change in probability, compute predicted probabilities at representative ages.",
        "- Use the p-values / 95% CIs to assess whether age effects (overall and culture-specific) are statistically different from zero; differences between cultures are in the interaction coefficients and are reflected in the combined per-culture values reported here."
    ]
    description = " ".join(description_lines)

    return {"object": output_object, "description": description}