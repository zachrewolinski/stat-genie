def extract_final_answer(model_output):
    """
    Extract relevant statistics from the model output produced by the modeling function.

    Returns a dictionary with:
      - "object": a dict containing:
          - 'coef_table': pandas.DataFrame with coef, clustered SE, z, p, OR, OR 95% CI for each parameter
          - 'marginal_rel_size': pandas.DataFrame with marginal effect of rel_size_z when focal_home=0 and focal_home=1
          - 'clustered_cov': the clustered covariance matrix (numpy array) if available
      - "description": a brief interpretation of the key results (which effects are statistically significant,
                       direction, and what that means for the research question)
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    # Basic checks / retrieve pieces
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict produced by the modeling function.")
    glm_fit = model_output.get('glm_fit', None)
    glm_clustered = model_output.get('glm_clustered', None)

    if glm_fit is None or glm_clustered is None:
        raise ValueError("model_output must contain 'glm_fit' and 'glm_clustered' keys.")

    params = glm_fit.params  # pandas Series
    # clustered SE: may be a pandas Series or numpy array
    se_cluster = glm_clustered.get('se', None)
    cov_cluster = glm_clustered.get('cov', None)

    if se_cluster is None:
        raise ValueError("clustered standard errors not found in model_output['glm_clustered']['se'].")

    # Convert se to pandas Series aligned with params
    if isinstance(se_cluster, np.ndarray):
        se = pd.Series(se_cluster, index=params.index)
    else:
        se = pd.Series(se_cluster)  # if it's already a Series or similar
        # ensure index alignment
        try:
            se = se.reindex(params.index)
        except Exception:
            se = pd.Series(se.values, index=params.index)

    # Compute Wald z-statistics and two-sided p-values using normal approximation
    z_stats = params / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))

    # Confidence intervals on log-odds scale, then exponentiate to OR scale
    z_crit = norm.ppf(0.975)
    ci_low_log = params - z_crit * se
    ci_high_log = params + z_crit * se

    or_vals = np.exp(params)
    or_ci_low = np.exp(ci_low_log)
    or_ci_high = np.exp(ci_high_log)

    # Build coefficient table
    coef_table = pd.DataFrame({
        'coef_logodds': params,
        'se_cluster': se,
        'z': z_stats,
        'p_value': p_values,
        'OR': or_vals,
        'OR_2.5%': or_ci_low,
        'OR_97.5%': or_ci_high
    })

    # Compute marginal effect of rel_size_z when focal_home = 0 and focal_home = 1
    # rel_size effect when focal_home=0 is coef(rel_size_z)
    # when focal_home=1 is coef(rel_size_z) + coef(rel_size_z:focal_home)
    marginal_rows = []
    base_name = 'rel_size_z'
    interact_name = 'rel_size_z:focal_home'
    if base_name not in params.index:
        raise KeyError(f"Expected parameter '{base_name}' not found in model params.")
    coef_base = params.loc[base_name]
    se_base = se.loc[base_name]

    # case focal_home = 0
    me0_coef = coef_base
    me0_se = se_base
    me0_z = me0_coef / me0_se
    me0_p = 2 * (1 - norm.cdf(abs(me0_z)))
    me0_or = np.exp(me0_coef)
    me0_or_ci_low = np.exp(me0_coef - z_crit * me0_se)
    me0_or_ci_high = np.exp(me0_coef + z_crit * me0_se)
    marginal_rows.append({
        'focal_home': 0,
        'coef_logodds': me0_coef,
        'se': me0_se,
        'z': me0_z,
        'p_value': me0_p,
        'OR': me0_or,
        'OR_2.5%': me0_or_ci_low,
        'OR_97.5%': me0_or_ci_high
    })

    # case focal_home = 1 (sum of base and interaction). Need covariance to compute SE.
    if interact_name in params.index:
        coef_inter = params.loc[interact_name]
        # default if covariance not available: compute SE by naive sum of variances (no cov) - but prefer cov
        if cov_cluster is None:
            # attempt to get covariance from glm_fit.cov_params() as fallback (not clustered)
            try:
                cov_est = glm_fit.cov_params()
            except Exception:
                cov_est = None
        else:
            cov_est = cov_cluster

        if cov_est is None:
            # fallback: treat cov = 0
            var_sum = se_base ** 2 + se.loc[interact_name] ** 2
        else:
            # cov_est may be numpy array or DataFrame
            # ensure we can access cov(a,b)
            try:
                # if cov_est is numpy array, map indices
                if isinstance(cov_est, np.ndarray):
                    # get index positions
                    idx_map = {name: i for i, name in enumerate(params.index)}
                    i = idx_map[base_name]
                    j = idx_map[interact_name]
                    var_sum = cov_est[i, i] + cov_est[j, j] + 2 * cov_est[i, j]
                else:
                    # assume pandas DataFrame-like
                    var_sum = cov_est.loc[base_name, base_name] + cov_est.loc[interact_name, interact_name] + \
                              2 * cov_est.loc[base_name, interact_name]
            except Exception:
                # last resort: sum variances ignoring covariance
                var_sum = se_base ** 2 + se.loc[interact_name] ** 2

        me1_coef = coef_base + coef_inter
        me1_se = np.sqrt(var_sum)
        me1_z = me1_coef / me1_se if me1_se > 0 else np.nan
        me1_p = 2 * (1 - norm.cdf(abs(me1_z))) if me1_se > 0 else np.nan
        me1_or = np.exp(me1_coef)
        me1_or_ci_low = np.exp(me1_coef - z_crit * me1_se)
        me1_or_ci_high = np.exp(me1_coef + z_crit * me1_se)

        marginal_rows.append({
            'focal_home': 1,
            'coef_logodds': me1_coef,
            'se': me1_se,
            'z': me1_z,
            'p_value': me1_p,
            'OR': me1_or,
            'OR_2.5%': me1_or_ci_low,
            'OR_97.5%': me1_or_ci_high
        })
    else:
        # no interaction term present
        marginal_rows.append({
            'focal_home': 1,
            'coef_logodds': np.nan,
            'se': np.nan,
            'z': np.nan,
            'p_value': np.nan,
            'OR': np.nan,
            'OR_2.5%': np.nan,
            'OR_97.5%': np.nan
        })

    marginal_rel_size = pd.DataFrame(marginal_rows).set_index('focal_home')

    # Determine which primary predictors are statistically significant at alpha=0.05
    primary_vars = ['rel_size_z', 'dist_diff_z', interact_name, 'focal_home']
    sig = {}
    for v in primary_vars:
        if v in coef_table.index:
            sig[v] = float(coef_table.loc[v, 'p_value']) < 0.05
        else:
            sig[v] = None

    # Build a concise description
    sig_list = [v for v, is_sig in sig.items() if is_sig]
    if len(sig_list) == 0:
        significance_summary = "No primary predictors (rel_size_z, dist_diff_z, focal_home, or their interaction) are statistically significant at alpha = 0.05."
    else:
        significance_summary = "Significant predictors at alpha = 0.05: " + ", ".join(sig_list) + "."

    # Directional interpretation for rel_size and dist_diff (based on OR)
    def interpret_effect(name):
        if name not in coef_table.index:
            return f"Parameter {name} not in model."
        orv = coef_table.loc[name, 'OR']
        p = coef_table.loc[name, 'p_value']
        direction = "increase" if orv > 1 else "decrease" if orv < 1 else "no change"
        return f"{name}: OR={orv:.3f}, p={p:.3f} -> {direction} in odds of focal group winning per 1 SD increase in {name}."

    interp_rel = interpret_effect('rel_size_z') if 'rel_size_z' in coef_table.index else ""
    interp_dist = interpret_effect('dist_diff_z') if 'dist_diff_z' in coef_table.index else ""

    description_lines = [
        "Extracted clustered (dyad-level) standard errors, Wald z-tests, two-sided p-values, odds ratios (OR) and 95% Wald CIs for all model parameters.",
        significance_summary,
        "Interpretation highlights:",
        interp_rel,
        interp_dist,
        "Additionally computed marginal effect of relative group size (rel_size_z) when focal_home=0 (away) and focal_home=1 (home) using the clustered covariance matrix (if available)."
    ]
    description = " ".join([ln for ln in description_lines if ln])

    # Package object
    obj = {
        'coef_table': coef_table,
        'marginal_rel_size': marginal_rel_size,
        'clustered_cov': cov_cluster
    }

    return {
        "object": obj,
        "description": description
    }