def extract_final_answer(model_output):
    """
    Extracts statistics from a fitted statsmodels GLMResultsWrapper for the model:
      ChoseMajority ~ Age_c * C(Site) + Gender + MajorityShownFirst

    Returns a dictionary with:
      - "object": dict containing numeric results (main Age effect, site-specific age slopes,
                  joint test of Age:Site interactions)
      - "description": short interpretation of those results in context

    The returned "object" contains:
      - main_age: {coef, se, z, p, ci_low, ci_high, odds_ratio, or_ci_low, or_ci_high}
      - site_slopes: dict keyed by site name (baseline site included) with same stats for the
                     age slope within that site (i.e., log-odds change per year)
      - interaction_wald_test: joint test (Wald) of all Age_c:C(Site) interaction coefficients = 0
                              (statistic, df_num, df_denom, pvalue)
    """
    import re
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    res = model_output  # statsmodels GLMResultsWrapper

    # Parameter names and basic tables
    params = res.params
    param_names = list(params.index)
    bse = None
    try:
        bse = res.bse
    except Exception:
        # fallback: compute from cov_params if available
        cov = res.cov_params()
        bse = pd.Series(np.sqrt(np.diag(cov)), index=param_names)

    pvalues = None
    try:
        pvalues = res.pvalues
    except Exception:
        pvalues = pd.Series([np.nan] * len(param_names), index=param_names)

    # confidence intervals
    try:
        conf = res.conf_int()
        # conf may be DataFrame with index = param_names
        conf_is_df = isinstance(conf, (pd.DataFrame, pd.Series))
    except Exception:
        conf = None
        conf_is_df = False

    # covariance matrix (for computing SE of sums)
    cov_params = res.cov_params()
    cov_is_df = isinstance(cov_params, pd.DataFrame)

    # Helper to get conf interval for a param
    name_to_index = {n: i for i, n in enumerate(param_names)}

    def get_conf(param):
        if conf is None:
            return (np.nan, np.nan)
        if conf_is_df:
            try:
                row = conf.loc[param]
                return (float(row[0]), float(row[1]))
            except Exception:
                # try positional
                idx = name_to_index[param]
                row = conf[idx]
                return (float(row[0]), float(row[1]))
        else:
            idx = name_to_index[param]
            return (float(conf[idx, 0]), float(conf[idx, 1]))

    # Find main Age_c param name (should be exactly 'Age_c')
    age_param = None
    if 'Age_c' in param_names:
        age_param = 'Age_c'
    else:
        # fallback: the param that equals 'Age' or startswith 'Age_c' and not containing C(Site)
        candidates = [n for n in param_names if ('Age' in n and 'C(Site)' not in n)]
        age_param = candidates[0] if candidates else None

    if age_param is None:
        raise ValueError("Could not find Age main-effect parameter name in the model parameters.")

    # Identify interaction parameter names that represent Age_c:C(Site)[T.<level>]
    interaction_names = [n for n in param_names if ('Age' in n and 'C(Site)' in n)]
    # Also allow the reversed order (C(Site)...:Age_c)
    if not interaction_names:
        interaction_names = [n for n in param_names if ('C(Site)' in n and 'Age' in n)]

    # Try to get all site levels from the original data if available
    site_levels = None
    reference_site = None
    try:
        # statsmodels stores the original frame often at res.model.data.frame
        df = res.model.data.frame
        if 'Site' in df.columns:
            site_levels = list(pd.Categorical(df['Site']).categories)
            # Determine reference level by checking which C(Site)[T.<level>] appear in params
            nonref_levels_in_params = []
            for name in interaction_names:
                m = re.search(r'C\(Site\)\[T\.(.+?)\]', name)
                if m:
                    nonref_levels_in_params.append(m.group(1))
                else:
                    m2 = re.search(r'\[T\.(.+?)\]', name)
                    if m2:
                        nonref_levels_in_params.append(m2.group(1))
            # reference is the one in site_levels not in nonref_levels_in_params (if any)
            reference_candidates = [lvl for lvl in site_levels if lvl not in nonref_levels_in_params]
            reference_site = reference_candidates[0] if reference_candidates else f"(reference)"
    except Exception:
        # fallback: infer non-reference levels from interaction names, leave reference unknown
        site_levels = None
        reference_site = "(reference)"

    # If site_levels is None, build a list of non-reference levels extracted from param names.
    nonref_levels = []
    for name in interaction_names:
        m = re.search(r'C\(Site\)\[T\.(.+?)\]', name)
        if not m:
            m = re.search(r'\[T\.(.+?)\]', name)
        if m:
            lvl = m.group(1)
        else:
            # as last resort, try to extract text between C(Site) and end
            parts = re.split(r':|\\:', name)
            lvl = name
        nonref_levels.append(lvl)
    nonref_levels = list(dict.fromkeys(nonref_levels))  # unique preserving order

    # Compute main age stats
    main_coef = float(params.loc[age_param])
    main_se = float(bse.loc[age_param]) if age_param in bse.index else float(bse[name_to_index[age_param]])
    main_z = main_coef / main_se if main_se != 0 else np.nan
    main_p = float(pvalues.loc[age_param]) if age_param in pvalues.index else float(2 * (1 - norm.cdf(abs(main_z))))
    main_ci_low, main_ci_high = get_conf(age_param)
    main_or = float(np.exp(main_coef))
    or_ci_low = float(np.exp(main_ci_low)) if not np.isnan(main_ci_low) else np.nan
    or_ci_high = float(np.exp(main_ci_high)) if not np.isnan(main_ci_high) else np.nan

    main_age_result = {
        'param_name': age_param,
        'coef_log_odds': main_coef,
        'se': main_se,
        'z': main_z,
        'p_value': main_p,
        'ci_95_log_odds': [main_ci_low, main_ci_high],
        'odds_ratio_per_year': main_or,
        'ci_95_odds_ratio': [or_ci_low, or_ci_high],
        'interpretation': ("Positive coef => higher log-odds of choosing majority as age increases; "
                           "negative => lower reliance with age.")
    }

    # Prepare site-specific slopes: baseline (reference) + each non-reference (Age + interaction)
    site_slopes = {}

    # Baseline/reference site slope is the Age_c main effect
    site_slopes[reference_site] = {
        'param_combination': f"{age_param} (baseline slope)",
        'coef_log_odds': main_coef,
        'se': main_se,
        'z': main_z,
        'p_value': main_p,
        'ci_95_log_odds': [main_ci_low, main_ci_high],
        'odds_ratio_per_year': main_or,
        'ci_95_odds_ratio': [or_ci_low, or_ci_high]
    }

    # For each interaction, compute slope = Age_c + interaction_coeff
    for inter_name in interaction_names:
        # Extract site level label
        m = re.search(r'C\(Site\)\[T\.(.+?)\]', inter_name)
        if not m:
            m = re.search(r'\[T\.(.+?)\]', inter_name)
        site_label = m.group(1) if m else inter_name

        inter_coef = float(params.loc[inter_name])
        # variance and covariance handling
        try:
            if cov_is_df:
                var_age = float(cov_params.loc[age_param, age_param])
                var_inter = float(cov_params.loc[inter_name, inter_name])
                cov_ai = float(cov_params.loc[age_param, inter_name])
            else:
                idx_a = name_to_index[age_param]
                idx_i = name_to_index[inter_name]
                var_age = float(cov_params[idx_a, idx_a])
                var_inter = float(cov_params[idx_i, idx_i])
                cov_ai = float(cov_params[idx_a, idx_i])
        except Exception:
            # fallback to sum of variances without covariance (less accurate)
            var_age = main_se ** 2
            var_inter = float(bse.loc[inter_name]) ** 2 if inter_name in bse.index else np.nan
            cov_ai = 0.0

        slope_coef = main_coef + inter_coef
        slope_var = var_age + var_inter + 2.0 * cov_ai
        slope_se = float(np.sqrt(slope_var)) if slope_var >= 0 else float(np.nan)
        slope_z = slope_coef / slope_se if slope_se != 0 else np.nan
        # p-value: use normal approx since GLM z
        slope_p = float(2 * (1 - norm.cdf(abs(slope_z)))) if not np.isnan(slope_z) else np.nan

        # Confidence interval on log-odds
        if not np.isnan(slope_se):
            slope_ci_low = slope_coef - 1.96 * slope_se
            slope_ci_high = slope_coef + 1.96 * slope_se
        else:
            slope_ci_low, slope_ci_high = (np.nan, np.nan)

        slope_or = float(np.exp(slope_coef)) if not np.isnan(slope_coef) else np.nan
        or_low = float(np.exp(slope_ci_low)) if not np.isnan(slope_ci_low) else np.nan
        or_high = float(np.exp(slope_ci_high)) if not np.isnan(slope_ci_high) else np.nan

        site_slopes[site_label] = {
            'param_combination': f"{age_param} + ({inter_name})",
            'coef_log_odds': slope_coef,
            'se': slope_se,
            'z': slope_z,
            'p_value': slope_p,
            'ci_95_log_odds': [slope_ci_low, slope_ci_high],
            'odds_ratio_per_year': slope_or,
            'ci_95_odds_ratio': [or_low, or_high]
        }

    # Joint Wald test: test that all interaction coefficients are zero
    interaction_test_result = None
    if interaction_names:
        # Build restriction matrix R such that R * beta = 0 selects the interaction params
        k_params = len(param_names)
        m = len(interaction_names)
        R = np.zeros((m, k_params))
        for i, name in enumerate(interaction_names):
            idx = name_to_index[name]
            R[i, idx] = 1.0
        try:
            wt = res.wald_test(R)
            # wt may be a statsmodels object; extract statistic and pvalue
            stat = float(wt.statistic) if hasattr(wt, 'statistic') else float(np.nan)
            pval = float(wt.pvalue) if hasattr(wt, 'pvalue') else (float(wt.pf) if hasattr(wt, 'pf') else np.nan)
            # df info if present
            df_num = int(wt.df_num) if hasattr(wt, 'df_num') else m
            df_denom = int(wt.df_denom) if hasattr(wt, 'df_denom') else 0
            interaction_test_result = {
                'test_type': 'Wald test (all Age_c:C(Site) coefficients = 0)',
                'statistic': stat,
                'df_num': df_num,
                'df_denom': df_denom,
                'p_value': pval
            }
        except Exception:
            interaction_test_result = {
                'test_type': 'Wald test attempted but failed',
                'error': 'Could not compute joint Wald test with res.wald_test(R).'
            }
    else:
        interaction_test_result = {
            'test_type': 'No Age:Site interaction parameters found',
            'note': 'Model does not include Age x Site interactions (or different naming).'
        }

    # Short descriptive interpretation
    # Determine significance summary for main age and joint interaction
    main_sig = (main_p < 0.05) if (not np.isnan(main_p)) else False
    joint_sig = False
    if interaction_test_result and ('p_value' in interaction_test_result) and (not np.isnan(interaction_test_result['p_value'])):
        joint_sig = interaction_test_result['p_value'] < 0.05

    description_lines = [
        "Extracted results relevant to whether children's reliance on the majority changes with age across sites.",
        f"Main effect of Age (across the baseline/reference site '{reference_site}'): "
        f"coef(log-odds) = {main_coef:.4f}, se = {main_se:.4f}, z = {main_z:.3f}, p = {main_p:.4g}.",
        ("Interpretation: a positive coefficient means the odds of choosing the majority increase with age; "
         "a negative coefficient means they decrease with age."),
        f"Site-specific age slopes (log-odds change per year) are provided for the reference site and each non-reference site.",
    ]
    if interaction_test_result and ('p_value' in interaction_test_result):
        description_lines.append(
            f"Joint test of Age x Site interactions: statistic = {interaction_test_result.get('statistic', np.nan):.4g}, "
            f"p = {interaction_test_result.get('p_value', np.nan):.4g}. "
            + ("This indicates the developmental trajectory differs across sites (jointly significant)."
               if joint_sig else "No evidence that trajectories differ across sites (joint test not significant).")
        )
    else:
        description_lines.append("Could not compute a joint test of Age x Site interactions.")

    description = " ".join(description_lines)

    result_object = {
        'main_age': main_age_result,
        'site_slopes': site_slopes,
        'interaction_wald_test': interaction_test_result
    }

    return {
        "object": result_object,
        "description": description
    }