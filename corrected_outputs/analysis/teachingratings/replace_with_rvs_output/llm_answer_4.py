def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs for the beauty terms
    and computes marginal (partial) effect of standardized beauty (beauty_z)
    on evaluation for males and females at selected beauty levels (-1, 0, +1).
    
    Returns:
      {
        "object": {
           "coefficients": { "beauty_z": {...}, "beauty_sq": {...}, "beauty_gender": {...} },
           "marginal_effects": {
               "male": {
                   beauty_value: {"estimate": ..., "se": ..., "z": ..., "p": ..., "95ci":(...,...)},
                   ...
               },
               "female": { same structure ... }
           },
           "notes": "..."
        },
        "description": "Brief interpretation of what the extracted numbers mean."
      }
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # required parameter names
    params_needed = ['beauty_z', 'beauty_sq', 'beauty_gender']
    # check presence
    param_index = set(res.params.index)
    missing = [p for p in params_needed if p not in param_index]
    if missing:
        raise ValueError("Model output is missing expected parameter(s): " + ", ".join(missing))

    # Extract basic stats for the three parameters
    coeffs = {}
    conf_int = res.conf_int(alpha=0.05)  # DataFrame with 0 and 1 columns
    cov = res.cov_params()               # covariance matrix (DataFrame)
    for p in params_needed:
        est = float(res.params[p])
        se = float(res.bse[p])
        z = est / se if se != 0 else np.nan
        pval = 2 * (1 - stats.norm.cdf(abs(z)))  # large-sample (cluster-robust) approx
        ci_low, ci_high = float(conf_int.loc[p, 0]), float(conf_int.loc[p, 1])
        coeffs[p] = {
            "estimate": est,
            "se": se,
            "z": z,
            "p_value": pval,
            "95ci": (ci_low, ci_high)
        }

    # Function to compute marginal effect of beauty_z given beauty_z_value and gender_female (0/1)
    # Marginal derivative: d Eval / d beauty_z = beta_beauty + 2 * beta_beauty_sq * beauty_z_value + beta_beauty_gender * gender
    def marginal_effect(beauty_val, gender_val):
        # weights vector for the full parameter vector (only three params nonzero)
        # Build an array aligned to res.params.index
        weights = np.zeros(len(res.params))
        idx_map = {name: i for i, name in enumerate(res.params.index)}
        # set weights
        weights[idx_map['beauty_z']] = 1.0
        weights[idx_map['beauty_sq']] = 2.0 * beauty_val
        weights[idx_map['beauty_gender']] = float(gender_val)
        # point estimate
        est = float(np.dot(weights, res.params.values))
        # variance via delta method
        # cov is DataFrame; convert to ndarray
        cov_mat = cov.values
        var = float(weights @ cov_mat @ weights)
        se = np.sqrt(var) if var >= 0 else np.nan
        z = est / se if se != 0 else np.nan
        pval = 2 * (1 - stats.norm.cdf(abs(z)))
        # 95% CI
        crit = stats.norm.ppf(0.975)
        ci_low = est - crit * se
        ci_high = est + crit * se
        return {"estimate": est, "se": se, "z": z, "p_value": pval, "95ci": (ci_low, ci_high)}

    # Compute marginal effects at beauty_z = -1, 0, +1 (these are in standardized units)
    beauty_points = [-1.0, 0.0, 1.0]
    marg_effects = {"male": {}, "female": {}}
    for b in beauty_points:
        marg_effects["male"][b] = marginal_effect(b, 0)
        marg_effects["female"][b] = marginal_effect(b, 1)

    # Assemble object to return
    result_object = {
        "coefficients": coeffs,
        "marginal_effects": marg_effects,
        "marginal_formula": "d(eval)/d(beauty_z) = beta_beauty + 2 * beta_beauty_sq * beauty_z + beta_beauty_gender * gender_female",
        "note_on_inference": (
            "P-values and CIs are computed using the model's cluster-robust covariance matrix; "
            "z-tests (normal approximation) are used for significance as is common with clustered SEs."
        )
    }

    # Short description for users
    description = (
        "The returned object contains the estimated coefficients, standard errors, z-statistics, p-values, "
        "and 95% confidence intervals for the beauty linear term (beauty_z), the quadratic term (beauty_sq), "
        "and the beauty-by-female interaction (beauty_gender). It also reports the marginal effect of a one-SD "
        "change in beauty (beauty_z = +/-1 and 0) on the evaluation score for males and females, with SEs and CIs. "
        "Use the marginal effects to answer whether beauty meaningfully affects evaluations and whether that effect "
        "differs by gender."
    )

    return {"object": result_object, "description": description}