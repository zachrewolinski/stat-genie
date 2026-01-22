def extract_final_answer(model_output):
    """
    Extracts age-related effects (linear and quadratic) and culture moderation from the
    binary logistic model predicting majority choice (majority vs not), and returns:
      - Age linear coefficient, SE, p-value, 95% CI
      - Age quadratic coefficient, SE, p-value, 95% CI
      - Wald test for joint significance of Age_c and all Age_c x Culture interaction terms
        (tests whether the linear age slope differs across cultures)
      - Per-culture linear-age slopes (Age_c + Age_c x Culture_i where present) with SE, z, p
    
    Returns a dict with keys "object" (a dict of numeric results) and "description" (textual
    interpretation of what the numbers mean).
    """
    import numpy as np
    from scipy import stats

    logit_res = model_output['logit_majority_result']
    exog_cols = model_output.get('exog_columns', None)

    params = logit_res.params          # pandas Series indexed by exog names
    bse = logit_res.bse
    pvals = logit_res.pvalues
    ci_df = logit_res.conf_int()      # DataFrame with two columns (lower, upper)
    cov = logit_res.cov_params()      # covariance matrix of parameter estimates

    results = {}

    # 1) Linear age effect (Age_c)
    if 'Age_c' in params.index:
        age_coef = float(params['Age_c'])
        age_se = float(bse['Age_c'])
        age_p = float(pvals['Age_c'])
        age_ci = [float(ci_df.loc['Age_c', 0]), float(ci_df.loc['Age_c', 1])]
        results['age_linear'] = {
            'coef': age_coef,
            'se': age_se,
            'p': age_p,
            '95%_ci': age_ci,
            'interpretation': "Change in log-odds of choosing the majority per unit increase in mean-centered age "
                              "for the reference culture (culture_1)."
        }
    else:
        results['age_linear'] = None

    # 2) Quadratic age effect (Age_c_sq)
    if 'Age_c_sq' in params.index:
        age2_coef = float(params['Age_c_sq'])
        age2_se = float(bse['Age_c_sq'])
        age2_p = float(pvals['Age_c_sq'])
        age2_ci = [float(ci_df.loc['Age_c_sq', 0]), float(ci_df.loc['Age_c_sq', 1])]
        results['age_quadratic'] = {
            'coef': age2_coef,
            'se': age2_se,
            'p': age2_p,
            '95%_ci': age2_ci,
            'interpretation': "Quadratic (age^2) term in log-odds of choosing majority; indicates non-linear development."
        }
    else:
        results['age_quadratic'] = None

    # 3) Wald test: joint test that linear age slope (Age_c) and all Age_c x Culture interaction terms are zero.
    #    This tests whether the linear age effect differs across cultures (if interactions jointly non-zero).
    # Build list of parameter names to test: 'Age_c' plus any columns ending with '_x_Age_c'
    param_names = list(params.index)
    interaction_names = [name for name in param_names if name.endswith('_x_Age_c')]
    wald_params = ['Age_c'] + interaction_names if 'Age_c' in param_names else interaction_names

    if len(wald_params) > 0:
        # Build R matrix with one row per restriction (we want them all = 0), i.e., identity selecting those params
        k_params = len(param_names)
        name_to_idx = {name: idx for idx, name in enumerate(param_names)}
        R = np.zeros((len(wald_params), k_params))
        for i, pname in enumerate(wald_params):
            idx = name_to_idx[pname]
            R[i, idx] = 1.0
        # Perform Wald test; statsmodels returns an object with .statistic and .pvalue
        wald_res = logit_res.wald_test(R)
        # wald_res may be an object or a tuple; try to extract statistic and pvalue robustly
        try:
            w_stat = float(wald_res.statistic)
            w_p = float(wald_res.pvalue)
            w_df = getattr(wald_res, 'df_denom', None) or getattr(wald_res, 'df', None)
        except Exception:
            # Fall back to attributes for older/newer statsmodels
            w_stat = float(wald_res.stat)
            w_p = float(wald_res.pval)
            w_df = None
        results['wald_test_age_and_interactions'] = {
            'tested_parameters': wald_params,
            'chi2_statistic': w_stat,
            'p': w_p,
            'df': w_df,
            'interpretation': "Joint test that the linear age slope (Age_c) and all Age_c x Culture interactions equal zero. "
                              "A small p-value indicates the linear age relationship (and/or its modulation by culture) is significant."
        }
    else:
        results['wald_test_age_and_interactions'] = None

    # 4) Per-culture linear-age slopes (for cultures 1..8). For culture_i:
    #    slope_i = coef(Age_c) + coef(culture_i_x_Age_c) [if interaction exists]
    #    SE computed by delta method: var = var(Age_c) + var(interaction) + 2*cov(Age_c, interaction)
    slopes = {}
    # assume up to 8 sites as in the model description
    for i in range(1, 9):
        inter_name = f'culture_{i}_x_Age_c'
        base_name = 'Age_c'
        if base_name not in params.index:
            # Cannot compute slopes without base Age_c
            slopes[i] = None
            continue
        base_coef = float(params[base_name])
        inter_coef = float(params[inter_name]) if inter_name in params.index else 0.0
        slope = base_coef + inter_coef

        # compute variance
        var_base = float(cov.loc[base_name, base_name])
        if inter_name in params.index:
            var_inter = float(cov.loc[inter_name, inter_name])
            cov_base_inter = float(cov.loc[base_name, inter_name])
            var_slope = var_base + var_inter + 2.0 * cov_base_inter
        else:
            var_slope = var_base

        se_slope = float(np.sqrt(var_slope)) if var_slope >= 0 else float(np.nan)
        z = slope / se_slope if se_slope > 0 else float('nan')
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z))) if se_slope > 0 else float('nan')

        slopes[i] = {
            'slope_log_odds_per_age_unit': slope,
            'se': se_slope,
            'z': z,
            'p': p,
            'components': {
                'Age_c_coef': base_coef,
                inter_name: inter_coef if inter_name in params.index else None
            },
            'interpretation': "Positive slope => increasing log-odds (and therefore probability) of choosing majority with increasing age."
        }

    results['slopes_by_culture'] = slopes

    # Package a concise textual description
    description_lines = []
    description_lines.append(
        "Extracted statistics come from the binary logistic regression predicting whether the child chose the majority option."
    )
    if results['age_linear'] is not None:
        description_lines.append(
            f"Reference culture (culture_1) linear age effect: coef={results['age_linear']['coef']:.4f}, "
            f"SE={results['age_linear']['se']:.4f}, p={results['age_linear']['p']:.4g}."
        )
    if results['age_quadratic'] is not None:
        description_lines.append(
            f"Quadratic age effect (Age_c_sq): coef={results['age_quadratic']['coef']:.4f}, "
            f"SE={results['age_quadratic']['se']:.4f}, p={results['age_quadratic']['p']:.4g}."
        )
    if results['wald_test_age_and_interactions'] is not None:
        w = results['wald_test_age_and_interactions']
        description_lines.append(
            f"Wald test for Age_c and Age_c x Culture interactions: chi2={w['chi2_statistic']:.3f}, p={w['p']:.4g}. "
            "This tests whether the linear age slope (and its moderation by culture) is jointly zero."
        )
    description_lines.append(
        "Per-culture slopes (log-odds change per unit mean-centered age) are provided in results['slopes_by_culture']."
    )

    description = " ".join(description_lines)

    return {
        "object": results,
        "description": description
    }