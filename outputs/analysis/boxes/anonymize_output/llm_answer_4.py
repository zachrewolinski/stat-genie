def extract_final_answer(model_output):
    """
    Extracts age-by-site interaction results from a fitted statsmodels GLMResultsWrapper.

    Returns a dict with:
      - "object": structured results including per-site age slopes (coefficient, SE, z, p, OR, 95% CI),
                  an overall Wald test of the Age_c x Site interaction, and the main Age_c coefficient.
      - "description": short interpretive text about what the numbers mean.

    This function does not require scipy; it uses math.erfc for normal tail p-values and
    relies on model_output having the usual statsmodels attributes (params, bse, cov_params, conf_int,
    model.data.frame, wald_test).
    """
    import numpy as np
    import math
    import pandas as pd

    def safe_exp(x):
        """Exponentiate x, returning inf for very large positive x (instead of raising OverflowError)."""
        try:
            return math.exp(x)
        except OverflowError:
            return float('inf') if x > 0 else 0.0

    res = model_output  # statsmodels GLMResultsWrapper

    # Basic extracted items
    params = res.params  # pandas Series
    bse = res.bse
    pvals = res.pvalues
    conf = res.conf_int()  # DataFrame with 0 and 1 columns
    cov = res.cov_params()  # DataFrame (covariance matrix)

    # Helper for two-sided normal p-value from z
    def z_pval(z):
        try:
            return math.erfc(abs(z) / math.sqrt(2.0))
        except Exception:
            return float('nan')

    # Check that Age_c main term exists
    if 'Age_c' not in params.index:
        raise ValueError("Model does not contain term 'Age_c' in parameters. Check formula/model.")

    beta_age = float(params['Age_c'])
    se_age = float(bse['Age_c'])
    z_age = beta_age / se_age if se_age > 0 else float('nan')
    p_age = z_pval(z_age)
    ci_age = (beta_age - 1.96 * se_age, beta_age + 1.96 * se_age)
    or_age = safe_exp(beta_age)
    or_age_ci = (safe_exp(ci_age[0]), safe_exp(ci_age[1]))

    # Identify interaction terms like 'Age_c:C(Site)[T.<level>]' (naming used by patsy/statsmodels)
    interaction_prefix = 'Age_c:C(Site)'
    interaction_terms = [name for name in params.index if name.startswith(interaction_prefix)]
    # Identify site main-effect dummies: 'C(Site)[T.<level>]'
    site_dummy_prefix = 'C(Site)'
    site_dummy_terms = [name for name in params.index if name.startswith(site_dummy_prefix) and ':' not in name]

    # Attempt to get site levels from the original DataFrame if available
    site_levels = None
    try:
        df = res.model.data.frame
        if 'Site' in df.columns:
            # Preserve the observed order of levels as they appear in the data (unique)
            site_levels = list(pd.Categorical(df['Site']).categories)
    except Exception:
        site_levels = None

    # If we couldn't get site levels from data.frame, infer from parameter names
    if site_levels is None:
        inferred = []
        # From site dummies
        for name in site_dummy_terms:
            # name looks like "C(Site)[T.<level>]"
            if name.startswith('C(Site)[T.'):
                level = name.split('C(Site)[T.')[1].rstrip(']')
                inferred.append(level)
        # Add reference level (the one not present among dummies) unknown order; we can only produce slopes
        # for reference (no interaction term) plus for those with interaction terms. We'll infer levels from interactions too.
        for name in interaction_terms:
            # name looks like "Age_c:C(Site)[T.<level>]"
            if 'C(Site)[T.' in name:
                level = name.split('C(Site)[T.')[1].rstrip(']')
                if level not in inferred:
                    inferred.append(level)
        # If we got any inferred levels, treat them as non-reference; reference unknown - create placeholder
        site_levels = inferred.copy()
        # We cannot reliably know the actual reference label if it was not encoded in params; label it as "<reference>"
        # and include it as a level for which the slope is just the main Age_c.
        if len(site_levels) == 0:
            site_levels = ['<reference>']
        else:
            # Insert a placeholder reference at start (since statsmodels omits one level)
            site_levels = ['<reference>'] + site_levels

    # For each site level compute slope = beta_Age + beta_interaction (if exists for that level),
    # standard error using covariance matrix (delta method), z, p, OR, CI.
    site_results = {}
    # Build a map from interaction term name to its coefficient index
    param_index = list(params.index)
    k_params = len(param_index)

    for lvl in site_levels:
        if lvl == '<reference>':
            inter_name = None
        else:
            inter_name = f'Age_c:C(Site)[T.{lvl}]'
            if inter_name not in params.index:
                # Interaction might be named with slightly different formatting; try alternative
                # e.g., older statsmodels might use 'C(Site)[T.<lvl>]:Age_c' order
                alt = f'C(Site)[T.{lvl}]:Age_c'
                if alt in params.index:
                    inter_name = alt
                else:
                    inter_name = None

        if inter_name is None:
            slope_beta = beta_age
            # variance is var(beta_age)
            try:
                var_slope = cov.loc['Age_c', 'Age_c'] if 'Age_c' in cov.index else (se_age ** 2)
            except Exception:
                var_slope = se_age ** 2
        else:
            slope_beta = beta_age + float(params[inter_name])
            # var = var(beta_age) + var(beta_inter) + 2*cov(beta_age, beta_inter)
            try:
                v_a = cov.loc['Age_c', 'Age_c']
                v_b = cov.loc[inter_name, inter_name]
                cov_ab = cov.loc['Age_c', inter_name]
                var_slope = v_a + v_b + 2.0 * cov_ab
            except Exception:
                # Fallback to NaN if covariance elements not found
                var_slope = float('nan')

        se_slope = math.sqrt(var_slope) if (isinstance(var_slope, (int, float)) and var_slope >= 0) else float('nan')
        z_slope = slope_beta / se_slope if se_slope > 0 else float('nan')
        p_slope = z_pval(z_slope)
        ci_low = slope_beta - 1.96 * se_slope if not math.isnan(se_slope) else float('nan')
        ci_high = slope_beta + 1.96 * se_slope if not math.isnan(se_slope) else float('nan')
        or_slope = safe_exp(slope_beta)
        or_ci = (safe_exp(ci_low), safe_exp(ci_high))

        site_results[lvl] = {
            'slope_coef': float(slope_beta),
            'slope_se': float(se_slope),
            'slope_z': float(z_slope),
            'slope_p': float(p_slope),
            'slope_95ci': (float(ci_low), float(ci_high)),
            'odds_ratio_per_year': float(or_slope),
            'odds_ratio_95ci': (float(or_ci[0]), float(or_ci[1])),
            'interaction_term_used': inter_name
        }

    # Overall Wald test for joint significance of all Age_c:C(Site) interaction terms.
    # Build restriction matrix R such that R * params = 0 tests the interactions = 0.
    # We'll identify indices of interaction parameters in params.index.
    interaction_indices = [
        param_index.index(name)
        for name in params.index
        if name.startswith(interaction_prefix) or (':Age_c' in name and 'C(Site)' in name)
    ]
    interaction_test = None
    if len(interaction_indices) == 0:
        interaction_test = {
            'message': 'No Age_c x Site interaction terms found in the model parameters; cannot perform joint test.'
        }
    else:
        # Build R matrix: each row selects one interaction coefficient
        m = len(interaction_indices)
        R = np.zeros((m, k_params))
        for i, idx in enumerate(interaction_indices):
            R[i, idx] = 1.0
        try:
            wt = res.wald_test(R)
            # Extract statistic and pvalue robustly
            stat_arr = np.atleast_1d(getattr(wt, 'statistic', np.array([np.nan])))
            stat = float(stat_arr.ravel()[0]) if stat_arr.size > 0 else float('nan')
            pval_wald = None
            if hasattr(wt, 'pvalue'):
                try:
                    pval_wald = float(wt.pvalue)
                except Exception:
                    # wt.pvalue might be array-like
                    try:
                        pval_wald = float(np.atleast_1d(wt.pvalue)[0])
                    except Exception:
                        pval_wald = None
            df = int(m)
            interaction_test = {'chi2': stat, 'df': df, 'p_value': pval_wald}
        except Exception as e:
            interaction_test = {'message': f'Wald test failed: {e}'}

    # Pack a compact object result
    output_object = {
        'main_Age_c': {
            'coef': float(beta_age),
            'se': float(se_age),
            'z': float(z_age),
            'p': float(p_age),
            '95ci': (float(ci_age[0]), float(ci_age[1])),
            'odds_ratio_per_year': float(or_age),
            'odds_ratio_95ci': (float(or_age_ci[0]), float(or_age_ci[1]))
        },
        'site_slopes': site_results,
        'interaction_test': interaction_test,
        # also include raw parameter table for reference (subset)
        'params_table': {name: float(params[name]) for name in params.index},
        'pvalues': {name: float(pvals[name]) for name in pvals.index}
    }

    # Description: brief interpretation guidance
    description_lines = []
    description_lines.append("For each Site, 'slope_coef' is the logistic regression coefficient for Age_c (change in log-odds per year).")
    description_lines.append("Positive slope_coef => increasing reliance on the majority with age; negative => decreasing reliance.")
    description_lines.append("Odds ratio interprets the multiplicative change in odds of choosing majority per 1-year increase in age.")
    description_lines.append("The 'interaction_test' gives a joint Wald test for whether the Age_c x Site interaction terms are all zero (i.e., whether age slopes differ across sites).")
    description = " ".join(description_lines)

    return {"object": output_object, "description": description}