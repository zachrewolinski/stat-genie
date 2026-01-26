def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, and
    incidence-rate-ratio (IRR) for the main effect of PlayerDark and its interaction
    with meanIAT from a fitted statsmodels results object (GLMResultsWrapper).
    Also computes the marginal (combined) effect of PlayerDark at the sample mean
    of meanIAT if the original data frame is available on the model object.

    Returns:
      {
        "object": { ... detailed numeric results ... },
        "description": "Interpretation of results in the context of the question"
      }

    The function is defensive: it tries to handle both plain results and results
    returned by get_robustcov_results (cluster-robust).
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Helper to safely get attributes
    def safe_get(attr_name, default=None):
        return getattr(res, attr_name, default)

    # Try to obtain coefficients, bse, pvalues, conf_int, cov_params
    try:
        params = safe_get('params')
        bse = safe_get('bse')
        pvalues = safe_get('pvalues')
        # conf_int may be a method or attribute
        conf_int = None
        if callable(safe_get('conf_int')):
            conf_int = res.conf_int()
        else:
            conf_int = safe_get('conf_int', None)
        # covariance matrix (should reflect robust cov if robust results were returned)
        try:
            cov = res.cov_params()
        except Exception:
            # some wrappers may expose 'normalized_cov_params' or 'cov_params_default'
            cov = None
            if safe_get('cov_params') is not None:
                try:
                    cov = res.cov_params
                except Exception:
                    cov = None
    except Exception as e:
        raise RuntimeError("Unable to extract model results attributes: " + str(e))

    # Prepare output structure
    results = {}
    terms = ['PlayerDark', 'PlayerDark:meanIAT']

    # Check presence
    for term in terms:
        if params is None or term not in params.index:
            results[term] = {
                'present': False,
                'note': f"Term '{term}' not found in model results."
            }
            continue

        coef = float(params[term])
        se = float(bse[term]) if (bse is not None and term in bse.index) else None
        pval = float(pvalues[term]) if (pvalues is not None and term in pvalues.index) else None
        ci_low, ci_upp = (None, None)
        if conf_int is not None and term in conf_int.index:
            ci_low, ci_upp = float(conf_int.loc[term, 0]), float(conf_int.loc[term, 1])

        irr = float(np.exp(coef))
        irr_ci = (None, None)
        if ci_low is not None:
            irr_ci = (float(np.exp(ci_low)), float(np.exp(ci_upp)))

        results[term] = {
            'present': True,
            'coef_log_rate': coef,
            'se': se,
            'p_value': pval,
            'conf_int_log_rate': (ci_low, ci_upp),
            'IRR': irr,
            'IRR_conf_int': irr_ci
        }

    # Attempt to compute marginal effect of PlayerDark at mean(meanIAT) in the model data (if available)
    marginal = {'computed': False}
    meanIAT_val = None
    try:
        # model.data.frame is available when statsmodels has the original DataFrame
        df = None
        md = safe_get('model')
        if md is not None:
            data_obj = getattr(md, 'data', None)
            if data_obj is not None:
                # Many statsmodels versions store the DataFrame in data.frame
                df = getattr(data_obj, 'frame', None)
                # fallback: some store the orig_exog or design_info; try to reconstruct
                if df is None:
                    # sometimes data.orig_exog is a numpy array; try to get column names via exog_names
                    try:
                        exog_names = getattr(md, 'exog_names', None)
                        exog = getattr(md, 'exog', None)
                        if exog is not None and exog_names is not None:
                            import pandas as _pd
                            df = _pd.DataFrame(exog, columns=exog_names)
                    except Exception:
                        df = None

        if df is not None and 'meanIAT' in df.columns:
            meanIAT_val = float(df['meanIAT'].mean())
            # beta_PlayerDark + beta_interaction * meanIAT
            if results.get('PlayerDark', {}).get('present') and results.get('PlayerDark:meanIAT', {}).get('present'):
                b1 = results['PlayerDark']['coef_log_rate']
                b3 = results['PlayerDark:meanIAT']['coef_log_rate']
                marg_coef = b1 + b3 * meanIAT_val

                # Compute standard error via delta method if covariance matrix available
                marg_se = None
                marg_p = None
                marg_ci = (None, None)
                if cov is not None:
                    # Ensure cov is a DataFrame-like with .loc
                    try:
                        cov_mat = cov
                        # If cov is a function or method object (unlikely), try calling
                        if callable(cov_mat):
                            cov_mat = cov_mat()
                        # Extract variances and covariance
                        var_b1 = float(cov_mat.loc['PlayerDark', 'PlayerDark'])
                        var_b3 = float(cov_mat.loc['PlayerDark:meanIAT', 'PlayerDark:meanIAT'])
                        cov_b1b3 = float(cov_mat.loc['PlayerDark', 'PlayerDark:meanIAT'])
                        var_marg = var_b1 + (meanIAT_val**2) * var_b3 + 2 * meanIAT_val * cov_b1b3
                        marg_se = float(np.sqrt(max(var_marg, 0.0)))
                        z = marg_coef / marg_se if marg_se and marg_se > 0 else None
                        if z is not None:
                            marg_p = float(2 * (1 - stats.norm.cdf(abs(z))))
                            marg_ci = (float(marg_coef - 1.96 * marg_se), float(marg_coef + 1.96 * marg_se))
                    except Exception:
                        marg_se = None
                        marg_p = None
                        marg_ci = (None, None)

                marg_irr = float(np.exp(marg_coef))
                marg_irr_ci = (None, None)
                if marg_ci[0] is not None:
                    marg_irr_ci = (float(np.exp(marg_ci[0])), float(np.exp(marg_ci[1])))

                marginal = {
                    'computed': True,
                    'meanIAT_used': meanIAT_val,
                    'marginal_log_rate_coef': marg_coef,
                    'marginal_se': marg_se,
                    'marginal_p_value': marg_p,
                    'marginal_conf_int_log_rate': marg_ci,
                    'marginal_IRR': marg_irr,
                    'marginal_IRR_conf_int': marg_irr_ci
                }
    except Exception:
        marginal = {'computed': False, 'note': 'Failed to compute marginal effect at meanIAT.'}

    output_object = {
        'terms': results,
        'marginal_at_meanIAT': marginal
    }

    # Prepare short description explaining the meaning of the key numbers
    # We do not draw a binary yes/no decision here; we summarize effect direction and significance.
    desc_lines = []
    desc_lines.append("Extracted coefficients are log rate ratios from a Negative Binomial GLM (offset = log_games).")
    if results.get('PlayerDark', {}).get('present'):
        r = results['PlayerDark']
        desc_lines.append(
            f"PlayerDark main effect (log-rate) = {r['coef_log_rate']:.4f}, SE = {r['se']}, "
            f"p = {r['p_value']}, IRR = {r['IRR']:.4f} (CI {r['IRR_conf_int'][0]}, {r['IRR_conf_int'][1]}). "
            "This coefficient is the log rate-ratio comparing Dark vs Light when meanIAT = 0."
        )
    else:
        desc_lines.append("PlayerDark main effect not found in model output.")

    if results.get('PlayerDark:meanIAT', {}).get('present'):
        r = results['PlayerDark:meanIAT']
        desc_lines.append(
            f"Interaction PlayerDark:meanIAT (log-rate per unit meanIAT) = {r['coef_log_rate']:.4f}, "
            f"SE = {r['se']}, p = {r['p_value']}. This indicates how the Dark vs Light difference "
            "changes with referee-country mean implicit bias (meanIAT)."
        )
    else:
        desc_lines.append("Interaction term PlayerDark:meanIAT not found in model output.")

    if marginal.get('computed'):
        desc_lines.append(
            f"At the sample mean of meanIAT = {marginal['meanIAT_used']:.4f}, the estimated "
            f"log-rate difference (Dark vs Light) = {marginal['marginal_log_rate_coef']:.4f}, "
            f"IRR = {marginal['marginal_IRR']:.4f} with 95% CI {marginal['marginal_IRR_conf_int']} "
            f"and p = {marginal['marginal_p_value']} (if computed). IRR > 1 means dark-skinned players "
            "have a higher red-card rate at that meanIAT level; IRR < 1 means lower."
        )
    else:
        desc_lines.append("Marginal effect at mean(meanIAT) could not be computed from the model object.")

    description = " ".join(desc_lines)

    return {
        "object": output_object,
        "description": description
    }