def extract_final_answer(model_output):
    """
    Extracts age-by-culture effects from a fitted statsmodels Logit (BinaryResultsWrapper).
    Returns a dictionary with:
      - "object": dict containing per-culture age slopes (log-odds), SE, z, p, odds ratios and 95% CIs,
                  the baseline age coefficient, and a likelihood-ratio test for the interaction.
      - "description": human-readable explanation of the extracted statistics.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    import statsmodels.formula.api as smf

    res = model_output  # BinaryResultsWrapper

    # Basic parameter tables
    params = res.params
    pvalues = res.pvalues
    cov = res.cov_params()
    conf = res.conf_int()  # 2.5% and 97.5% for coefficients (log-odds scale)

    # Try to get original dataframe used to fit the model
    try:
        df = res.model.data.frame.copy()
    except Exception:
        # fallback: try to access model.data.orig_endog / exog not used here
        df = None

    # Identify all culture levels if possible
    if df is not None and 'culture' in df.columns:
        all_levels = list(pd.Categorical(df['culture']).categories)
    else:
        # If we cannot get original df, infer levels from parameter names:
        # find all interaction parameter names and extract level names.
        all_levels = []
        for name in params.index:
            if name.startswith('C(culture)[T.'):
                # extract between T. and ]
                try:
                    level = name.split('C(culture)[T.')[1].split(']')[0]
                    all_levels.append(level)
                except Exception:
                    pass
        # we can't know reference level; leave all_levels possibly incomplete

    # Find interaction parameter names for age_c by culture
    interaction_terms = [n for n in params.index if n.startswith('age_c:C(culture)')]
    # Extract level names present in interactions
    interaction_levels = []
    for t in interaction_terms:
        # pattern is age_c:C(culture)[T.<level>]
        try:
            level = t.split('age_c:C(culture)[T.')[1].split(']')[0]
            interaction_levels.append(level)
        except Exception:
            continue

    # Determine reference level (a level in all_levels not in interaction_levels)
    reference_level = None
    if all_levels:
        ref_candidates = [lv for lv in all_levels if lv not in interaction_levels]
        if len(ref_candidates) == 1:
            reference_level = ref_candidates[0]
        elif len(ref_candidates) > 1:
            # If multiple candidates, choose the first as a best guess
            reference_level = ref_candidates[0]
        else:
            # if none found (maybe only one level), pick the first known level
            reference_level = all_levels[0]
    else:
        # If we could not infer all levels, infer reference as "reference" placeholder
        reference_level = None

    # Get main age coefficient (this is slope for reference level)
    if 'age_c' not in params.index:
        raise ValueError("Model does not contain 'age_c' main effect parameter. Check model specification.")
    age_main = float(params['age_c'])

    # Prepare per-culture slope estimates
    age_slopes = {}
    # Create mapping of interaction term name for quick lookup
    interaction_map = {}
    for t in interaction_terms:
        try:
            level = t.split('age_c:C(culture)[T.')[1].split(']')[0]
            interaction_map[level] = t
        except Exception:
            continue

    # Use covariance matrix to compute SE for linear combinations
    cov_matrix = cov

    # For each known level, compute slope = age_main + interaction(if exists)
    levels_to_report = all_levels if all_levels else (interaction_levels[:] if interaction_levels else [reference_level])
    # Deduplicate and drop None
    levels_to_report = [lv for i, lv in enumerate(levels_to_report) if lv is not None and lv not in levels_to_report[:i]]

    for lvl in levels_to_report:
        inter_name = interaction_map.get(lvl, None)
        if inter_name:
            coef_inter = float(params.get(inter_name, 0.0))
        else:
            coef_inter = 0.0
        slope = age_main + coef_inter  # log-odds change in majority choice per centered year for this culture

        # compute SE using Var(a + b) = Var(a) + Var(b) + 2Cov(a,b)
        var_a = cov_matrix.loc['age_c', 'age_c']
        var_b = cov_matrix.loc[inter_name, inter_name] if inter_name is not None and inter_name in cov_matrix.index else 0.0
        cov_ab = cov_matrix.loc['age_c', inter_name] if inter_name is not None and inter_name in cov_matrix.index else 0.0
        var_sum = var_a + var_b + 2.0 * cov_ab
        se = float(np.sqrt(var_sum)) if var_sum >= 0 else float(np.nan)

        # z and p
        z = slope / se if se and not np.isnan(se) else float('nan')
        p = float(2 * (1 - norm.cdf(abs(z)))) if not np.isnan(z) else float('nan')

        # odds ratio and 95% CI on OR scale
        or_est = float(np.exp(slope))
        # CI for slope: slope +/- 1.96*se
        if not np.isnan(se):
            lower_log = slope - 1.96 * se
            upper_log = slope + 1.96 * se
            ci_lower_or = float(np.exp(lower_log))
            ci_upper_or = float(np.exp(upper_log))
        else:
            ci_lower_or = ci_upper_or = float('nan')

        age_slopes[lvl] = {
            'slope_log_odds': float(slope),
            'se': float(se),
            'z': float(z) if not np.isnan(z) else None,
            'p_value': float(p) if not np.isnan(p) else None,
            'odds_ratio': float(or_est),
            'or_95ci': [ci_lower_or, ci_upper_or],
            'note': 'slope is effect of centered age (age_c) on log-odds of choosing majority in this culture'
        }

    # Also provide the main age coefficient (reference)
    main_entry = {
        'parameter_name': 'age_c',
        'coef_log_odds': float(age_main),
        'p_value': float(pvalues['age_c']) if 'age_c' in pvalues.index else None,
        'conf_int_log_odds': [float(conf.loc['age_c', 0]), float(conf.loc['age_c', 1])] if 'age_c' in conf.index else None,
        'odds_ratio': float(np.exp(age_main)),
        'odds_ratio_95ci': [float(np.exp(conf.loc['age_c', 0])), float(np.exp(conf.loc['age_c', 1]))] if 'age_c' in conf.index else None,
        'reference_level': reference_level
    }

    # Test whether the interaction (age_c * C(culture)) is jointly significant.
    # We'll fit a reduced model without the interaction and do a LR test (if original df available).
    interaction_lr_test = None
    try:
        if df is not None:
            # Build reduced formula: remove the interaction, keep main effects
            # The original formula from the prompt was: 'MajorityChoice ~ age_c * C(culture) + is_male + majority_first'
            # Reduced formula: MajorityChoice ~ age_c + C(culture) + is_male + majority_first
            reduced_formula = 'MajorityChoice ~ age_c + C(culture) + is_male + majority_first'
            reduced = smf.logit(formula=reduced_formula, data=df).fit(disp=False)
            lr_stat, lr_pvalue, lr_df = res.compare_lr_test(reduced)
            interaction_lr_test = {
                'lr_stat': float(lr_stat),
                'p_value': float(lr_pvalue),
                'df_diff': int(lr_df),
                'interpretation': 'Small p-value indicates the age-by-culture interaction improves model fit (i.e., developmental slopes differ across cultures).'
            }
        else:
            # If df is not available, attempt a Wald test for joint zero of interaction terms
            if len(interaction_terms) > 0:
                # Build restriction matrix via string like "term1 = 0, term2 = 0"
                restr = ', '.join([f"{t} = 0" for t in interaction_terms])
                wt = res.wald_test(restr)
                # wt is a WaldTestResults object; extract stat and p-value if available
                try:
                    stat = float(wt.statistic)
                    pval = float(wt.pvalue)
                    df_diff = int(wt.df_denom) if hasattr(wt, 'df_denom') else len(interaction_terms)
                except Exception:
                    stat = pval = None
                    df_diff = len(interaction_terms)
                interaction_lr_test = {
                    'wald_stat': stat,
                    'p_value': pval,
                    'df': df_diff,
                    'note': 'Wald test of joint zero for age-by-culture interaction terms'
                }
            else:
                interaction_lr_test = {
                    'note': 'No interaction terms present (only one culture).'
                }
    except Exception as e:
        interaction_lr_test = {'error': str(e), 'note': 'Failed to compute LR/Wald test for interaction.'}

    result_object = {
        'main_age_coefficient': main_entry,
        'age_slopes_by_culture': age_slopes,
        'interaction_test': interaction_lr_test,
        'model_params': {k: float(v) for k, v in params.items()},
        'model_pvalues': {k: float(v) for k, v in pvalues.items()}
    }

    description_lines = [
        "Extracted per-culture developmental slopes (effect of centered age on log-odds of choosing the majority).",
        "For each culture: reported slope (log-odds per unit age_c), its SE, z, two-sided p-value, odds ratio, and 95% CI for the odds ratio.",
        "Also performed a likelihood-ratio test (or Wald test when LR not available) examining whether the age-by-culture interaction improves model fit.",
        "Interpretation: significant positive slope means older children in that culture are more likely to choose the majority; differences in slopes across cultures are indicated by a significant interaction test."
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}