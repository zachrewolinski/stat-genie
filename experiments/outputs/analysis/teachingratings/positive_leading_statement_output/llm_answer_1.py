def extract_final_answer(model_output):
    """
    Extract coefficients, robust SEs, p-values, confidence intervals for beauty-related terms,
    and compute marginal (partial) effect of beauty on eval for males and females at
    beauty_z = -1, 0, +1 (with standard errors, CIs, p-values).
    
    Returns a dict with keys:
      - "object": dict containing 'coefficients', 'marginal_effects', and 'conclusion'
      - "description": brief explanation of what was returned and how to interpret it
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Basic parameter objects (works for statsmodels RegressionResultsWrapper)
    params = res.params
    try:
        bse = res.bse
        pvals = res.pvalues
        conf = res.conf_int()
        cov = res.cov_params()  # should reflect clustering/robust cov used in fit
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract parameters from model_output: {e}"
        }

    # Variables of interest
    beauty_terms = ['beauty_z', 'beauty_z_sq', 'beauty_gender_interaction']

    # Extract coef, se, pval, confidence interval for each beauty-related term if present
    coeffs = {}
    for term in beauty_terms:
        if term in params.index:
            coef = float(params[term])
            se = float(bse[term]) if term in bse.index else None
            p = float(pvals[term]) if term in pvals.index else None
            # conf may be DataFrame or ndarray; use .loc if DataFrame
            try:
                ci_low, ci_high = float(conf.loc[term, 0]), float(conf.loc[term, 1])
            except Exception:
                # fallback if conf_int returned as ndarray without index
                ci = np.asarray(conf)
                idx = list(params.index).index(term)
                ci_low, ci_high = float(ci[idx, 0]), float(ci[idx, 1])
            coeffs[term] = {
                'coef': coef,
                'se': se,
                'pval': p,
                'ci_lower': ci_low,
                'ci_upper': ci_high
            }

    # If beauty_z not in model, nothing further to compute
    if 'beauty_z' not in coeffs:
        return {
            "object": {
                "coefficients": coeffs,
                "marginal_effects": {},
                "conclusion": "Model does not include 'beauty_z' term; cannot assess effect of beauty."
            },
            "description": "The model output did not contain the 'beauty_z' coefficient. "
                           "No beauty effect could be extracted."
        }

    # Prepare covariance matrix and index mapping for linear-combination variance calculation
    cov_mat = np.asarray(cov)  # cov_params may be DataFrame or ndarray
    param_names = list(params.index)
    name_to_idx = {n: i for i, n in enumerate(param_names)}

    # Degrees of freedom for t-distribution approximations (may be large)
    df_resid = None
    try:
        df_resid = int(res.df_resid)
    except Exception:
        df_resid = None

    # Function to compute marginal effect (derivative of eval wrt beauty_z)
    # derivative = beta_beauty_z + 2*beta_beauty_z_sq * beauty_val + beta_beauty_gender_interaction * gender_female
    def marginal_effect(beauty_val, gender_female):
        # build linear combination vector a so that effect = a' * params
        a = np.zeros(len(param_names))
        # derivative w.r.t beauty_z multiplies the coef for beauty_z
        if 'beauty_z' in name_to_idx:
            a[name_to_idx['beauty_z']] = 1.0
        # derivative contribution from quadratic term: 2 * beauty_val * beta_beauty_z_sq
        if 'beauty_z_sq' in name_to_idx:
            a[name_to_idx['beauty_z_sq']] = 2.0 * beauty_val
        # contribution from interaction term: beta * gender_female
        if 'beauty_gender_interaction' in name_to_idx:
            a[name_to_idx['beauty_gender_interaction']] = 1.0 * gender_female

        # compute effect value
        effect = float(np.dot(a, np.asarray(params)))
        # compute variance via delta method: var = a' cov a
        var = float(np.dot(a, np.dot(cov_mat, a)))
        se = float(np.sqrt(var)) if var >= 0 else float('nan')

        # t-statistic and p-value (use t with df_resid if available, else normal approx)
        if se == 0 or np.isnan(se):
            tstat = float('nan')
            pval = float('nan')
        else:
            tstat = effect / se
            if df_resid is not None and df_resid > 0:
                pval = float(2.0 * stats.t.sf(abs(tstat), df_resid))
                crit = float(stats.t.ppf(0.975, df_resid))
            else:
                pval = float(2.0 * stats.norm.sf(abs(tstat)))
                crit = float(stats.norm.ppf(0.975))
        ci_lower = effect - crit * se if (se == se and 'crit' in locals()) else float('nan')
        ci_upper = effect + crit * se if (se == se and 'crit' in locals()) else float('nan')

        return {
            'beauty_val': float(beauty_val),
            'gender_female': int(gender_female),
            'effect': effect,
            'se': se,
            'tstat': tstat,
            'pval': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    # Compute marginal effects at beauty_z = -1, 0, +1 for male (0) and female (1)
    beauty_points = [-1.0, 0.0, 1.0]
    marg_effects = {'male': {}, 'female': {}}
    for b in beauty_points:
        marg_effects['male'][str(b)] = marginal_effect(b, 0)
        marg_effects['female'][str(b)] = marginal_effect(b, 1)

    # Formulate a concise conclusion about statistical significance of beauty-related terms
    significant_terms = []
    for term, info in coeffs.items():
        if info.get('pval') is not None and info['pval'] < 0.05:
            significant_terms.append(term)
    # also check marginal effects at mean (beauty=0) for male/female
    if marg_effects['male']['0.0']['pval'] is not None and marg_effects['male']['0.0']['pval'] < 0.05:
        significant_terms.append('marginal_male_at_mean')
    if marg_effects['female']['0.0']['pval'] is not None and marg_effects['female']['0.0']['pval'] < 0.05:
        significant_terms.append('marginal_female_at_mean')

    if len(significant_terms) > 0:
        conclusion = ("There is evidence that beauty affects teaching evaluations: "
                      "significant beauty-related terms found: " + ", ".join(significant_terms) + ".")
    else:
        conclusion = ("No clear evidence that beauty has a statistically significant effect on evaluations "
                      "(no beauty-related coefficient or marginal effect significant at p < 0.05).")

    result_object = {
        'coefficients': coeffs,
        'marginal_effects': marg_effects,
        'conclusion': conclusion
    }

    description = (
        "Returned the estimated coefficients (coef, robust SE, p-value, 95% CI) for beauty-related terms "
        "('beauty_z', 'beauty_z_sq', 'beauty_gender_interaction') when present. Also computed the marginal "
        "effect of a one-unit increase in standardized beauty (d eval / d beauty_z) for males and females at "
        "beauty_z = -1, 0, +1, with SE, t-stat, p-value, and 95% CI (delta-method using the model covariance). "
        "Use 'conclusion' to see a brief yes/no-style interpretation based on p < 0.05."
    )

    return {"object": result_object, "description": description}