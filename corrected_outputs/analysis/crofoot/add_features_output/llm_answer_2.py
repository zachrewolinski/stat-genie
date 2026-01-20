def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals, odds ratios,
    and marginal effect of relative group size when the contest is closer vs farther
    from the focal group's center from a fitted statsmodels logistic result.

    Returns a dict with:
      - "object": a dict containing numeric results for:
          * 'params' : dict of parameter -> {coef, se, pvalue, OR, CI_95}
          * 'size_effect_when_far' : effect of size_log_ratio when CloserToFocal==0
          * 'size_effect_when_closer' : effect of size_log_ratio when CloserToFocal==1
            (these include coef, se, z, p, OR, CI_95)
          * 'notes' : any warnings about missing terms
      - "description": short explanation of the meaning of the reported quantities.
    """
    import numpy as np
    from math import exp, sqrt
    from scipy.stats import norm

    res = model_output

    # Basic parameter table
    try:
        params = res.params
        bse = res.bse
        pvalues = res.pvalues
        ci = res.conf_int()  # DataFrame or ndarray with [lower, upper]
        cov = res.cov_params()
        names = list(params.index)
    except Exception as e:
        raise ValueError(f"Provided model_output does not have expected attributes: {e}")

    # Helper to find parameter name matching substrings
    def find_param_name(substr, exclude_colon=False):
        for n in names:
            if substr in n and (not exclude_colon or ':' not in n):
                return n
        return None

    notes = []

    # Identify main terms and interaction
    name_size = find_param_name('size_log_ratio', exclude_colon=True)
    name_closer = find_param_name('CloserToFocal', exclude_colon=True)
    # interaction: any param that contains both substrings (likely with ':')
    name_inter = None
    for n in names:
        if ('size_log_ratio' in n) and ('CloserToFocal' in n) and (':' in n or '.' in n):
            name_inter = n
            break

    results_obj = {'params': {}}

    # Populate base params if found
    for n in names:
        coef = float(params[n])
        se = float(bse[n]) if n in bse.index else float(np.nan)
        p = float(pvalues[n]) if n in pvalues.index else float(np.nan)
        # find CI row
        try:
            row = ci.loc[n]
            ci_lower, ci_upper = float(row[0]), float(row[1])
        except Exception:
            # ci may be ndarray with same ordering as names
            try:
                idx = names.index(n)
                ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
            except Exception:
                ci_lower, ci_upper = float(np.nan), float(np.nan)

        or_val = exp(coef)
        or_ci = (exp(ci_lower), exp(ci_upper))

        results_obj['params'][n] = {
            'coef': coef,
            'se': se,
            'pvalue': p,
            'OR': or_val,
            'CI_95': (ci_lower, ci_upper),
            'OR_CI_95': or_ci
        }

    # Compute marginal effect of size_log_ratio when CloserToFocal == 0 (far) and ==1 (closer)
    if name_size is None:
        raise ValueError("Could not find a parameter matching 'size_log_ratio' in model_output.params")

    # Effect when far (CloserToFocal == 0): just the size coefficient
    beta_far = float(params[name_size])
    se_far = float(bse[name_size]) if name_size in bse.index else float(np.nan)
    z_far = beta_far / se_far if se_far and not np.isnan(se_far) else float(np.nan)
    p_far = 2 * (1 - norm.cdf(abs(z_far))) if not np.isnan(z_far) else float(np.nan)
    ci_far = results_obj['params'][name_size]['CI_95']
    or_far = exp(beta_far)
    or_ci_far = results_obj['params'][name_size]['OR_CI_95']

    results_obj['size_effect_when_far'] = {
        'coef': beta_far,
        'se': se_far,
        'z': z_far,
        'pvalue': p_far,
        'CI_95': ci_far,
        'OR': or_far,
        'OR_CI_95': or_ci_far
    }

    # Effect when closer (CloserToFocal == 1): size coef + interaction coef (if interaction exists)
    if name_inter is None:
        # no interaction term present: effect is same as far
        results_obj['size_effect_when_closer'] = results_obj['size_effect_when_far'].copy()
        notes.append("No interaction term found; effect of size does not vary by CloserToFocal in the model.")
    else:
        beta_inter = float(params[name_inter])
        # Sum of betas
        beta_closer = beta_far + beta_inter

        # Var(beta_sum) = Var(beta_far) + Var(beta_inter) + 2*Cov(beta_far, beta_inter)
        try:
            var_far = float(cov.loc[name_size, name_size])
            var_inter = float(cov.loc[name_inter, name_inter])
            covar = float(cov.loc[name_size, name_inter])
            var_sum = var_far + var_inter + 2.0 * covar
            se_sum = sqrt(var_sum) if var_sum >= 0 else float(np.nan)
        except Exception:
            # fallback to NaN if covariance not available
            se_sum = float(np.nan)

        z_closer = beta_closer / se_sum if se_sum and not np.isnan(se_sum) else float(np.nan)
        p_closer = 2 * (1 - norm.cdf(abs(z_closer))) if not np.isnan(z_closer) else float(np.nan)

        # CI for beta_closer
        if not np.isnan(se_sum):
            ci_lower = beta_closer - 1.96 * se_sum
            ci_upper = beta_closer + 1.96 * se_sum
        else:
            ci_lower, ci_upper = (float(np.nan), float(np.nan))

        or_closer = exp(beta_closer)
        or_ci_closer = (exp(ci_lower), exp(ci_upper)) if (not np.isnan(ci_lower) and not np.isnan(ci_upper)) else (float(np.nan), float(np.nan))

        results_obj['size_effect_when_closer'] = {
            'coef': beta_closer,
            'se': se_sum,
            'z': z_closer,
            'pvalue': p_closer,
            'CI_95': (ci_lower, ci_upper),
            'OR': or_closer,
            'OR_CI_95': or_ci_closer,
            'components': {
                'size_coef': {'name': name_size, 'coef': beta_far},
                'interaction_coef': {'name': name_inter, 'coef': beta_inter}
            }
        }

    # Include identified names for clarity
    results_obj['identified_terms'] = {
        'size_term_name': name_size,
        'closer_term_name': name_closer,
        'interaction_term_name': name_inter
    }
    if name_closer is None:
        notes.append("Could not find a main effect parameter for 'CloserToFocal' (check how variable was encoded).")

    results_obj['notes'] = notes

    description = (
        "Returned objects are logistic regression parameter estimates and derived quantities.\n"
        "- For each model parameter: coefficient (log-odds), SE, p-value, 95% CI, odds-ratio (OR) and OR 95% CI.\n"
        "- 'size_effect_when_far' is the effect of relative group size (size_log_ratio) on the log-odds of the focal group winning when the contest location is NOT closer to the focal group's center (CloserToFocal==0).\n"
        "- 'size_effect_when_closer' is the corresponding effect when the contest location IS closer to the focal group's center (CloserToFocal==1); if the model includes an interaction, this equals size coef + interaction coef, with SE/CIs computed using the covariance matrix (delta method).\n"
        "- Interpret coefficients: positive coef => higher log-odds (and OR>1) that the focal group wins as the predictor increases. Use p-values/CI to assess statistical evidence.\n"
        "Use the returned numbers to determine whether relative group size and contest location (and their interaction) significantly influence winning probability."
    )

    return {"object": results_obj, "description": description}