def extract_final_answer(model_output):
    """
    Extracts and interprets the Reader View effect from a fitted statsmodels results object
    that includes an interaction between reader_view and dyslexia_bin.

    Returns a dictionary with:
      - "object": dict of extracted numeric results (coefficients, SEs, p-values, CIs,
                  marginal effects for non-dyslexic and dyslexic readers, and percent-change
                  interpretation on the original scale).
      - "description": short plain-language interpretation of whether Reader View improves
                       reading speed for readers with dyslexia.
    """
    import math
    import numpy as np
    import pandas as pd

    # Helper: normal CDF (avoids SciPy dependency)
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    # Extract parameter series and names, robust to numpy.ndarray / dict / Series
    try:
        raw_params = model_output.params
    except Exception as e:
        raise ValueError("model_output must expose .params (a pandas Series or array-like).") from e

    # Build a pandas Series 'params' with an index of parameter names
    if isinstance(raw_params, pd.Series):
        params = raw_params.copy()
    elif isinstance(raw_params, dict):
        params = pd.Series(raw_params)
    else:
        # array-like (e.g., numpy ndarray)
        arr = np.asarray(raw_params)
        # Try to find parameter names from common attributes
        names = None
        if hasattr(model_output, 'param_names'):
            try:
                names = list(model_output.param_names)
            except Exception:
                names = None
        if names is None and hasattr(model_output, 'model') and hasattr(model_output.model, 'exog_names'):
            try:
                names = list(model_output.model.exog_names)
            except Exception:
                names = None
        if names is None:
            # Fallback: generate generic names
            names = [f"param_{i}" for i in range(len(arr))]
        params = pd.Series(arr, index=[str(n) for n in names])

    # Now create a list of parameter names as strings
    try:
        names = [str(n) for n in params.index]
    except Exception:
        names = [str(i) for i in range(len(params))]

    # Robust way to find parameter names relevant to our question
    def find_name_containing(*parts, exclude_colon=False):
        for n in names:
            if all(p in n for p in parts):
                if exclude_colon and (':' in n):
                    continue
                return n
        return None

    # Main effects / interaction names (try common variants)
    reader_name = find_name_containing('reader_view', exclude_colon=True) or find_name_containing('reader_view')
    dyslexia_name = find_name_containing('dyslexia') or find_name_containing('dyslexia_bin')
    interaction_name = None
    # look for something containing both
    for n in names:
        if 'reader_view' in n and ('dyslex' in n or 'dyslexia' in n or 'dyslexia_bin' in n):
            interaction_name = n
            break
    # If we still don't have an interaction name, try the colon pattern
    if interaction_name is None:
        # find any name containing both reader_view and dyslexia parts with a colon
        interaction_name = next((n for n in names if ':' in n and 'reader_view' in n and ('dyslex' in n or 'dyslexia' in n or 'dyslexia_bin' in n)), None)
        if interaction_name is None:
            # try any colon-containing name that includes both substrings in either order
            for n in names:
                if ':' in n:
                    left, right = n.split(':', 1)
                    if ('reader_view' in left or 'reader_view' in right) and ('dyslex' in left or 'dyslex' in right or 'dyslexia_bin' in left or 'dyslexia_bin' in right):
                        interaction_name = n
                        break

    # Validate we found required terms
    if reader_name is None:
        raise ValueError(f"Could not locate a parameter corresponding to 'reader_view' in model params: {names}")
    if interaction_name is None:
        raise ValueError(f"Could not locate an interaction parameter between reader_view and dyslexia in model params: {names}")
    if dyslexia_name is None:
        # It's possible dyslexia main effect is not in model (e.g., if absorbed), but interaction still exists.
        # We'll proceed but warn in the description later.
        dyslexia_name = None

    # Get covariance matrix (as DataFrame for convenient indexing)
    cov = None
    try:
        cov_mat = model_output.cov_params()
    except Exception:
        # fallback: try attribute
        cov_mat = getattr(model_output, 'cov_params_default', None)
    if cov_mat is None:
        raise ValueError("Could not obtain covariance matrix from model_output via .cov_params().")

    if isinstance(cov_mat, np.ndarray):
        cov = pd.DataFrame(cov_mat, index=names, columns=names)
    else:
        # assume it's already a DataFrame-like
        cov = pd.DataFrame(cov_mat)
        # ensure index/columns match param names; if not, reindex using params.index
        try:
            cov = cov.reindex(index=params.index, columns=params.index)
        except Exception:
            # last resort: force numeric alignment if shapes match
            try:
                arr_cov = np.asarray(cov_mat)
                if arr_cov.shape[0] == len(names) and arr_cov.shape[1] == len(names):
                    cov = pd.DataFrame(arr_cov, index=names, columns=names)
            except Exception:
                pass

    # Helper to get scalar param value and its se/pvalue/ci using model_output where available
    def _get_param_info(name):
        if name not in params.index:
            raise KeyError(f"Parameter name '{name}' not found in params: {list(params.index)}")
        val = float(params[name])
        # bse may not reflect clustered cov; but model_output.bse should be clustered if provided
        try:
            se = float(model_output.bse[name])
        except Exception:
            # fallback to sqrt(diag(cov))
            se = float(math.sqrt(max(0.0, cov.loc[name, name])))
        # pvalue from model_output if available
        pval = None
        try:
            if hasattr(model_output, 'pvalues') and name in model_output.pvalues.index:
                pval = float(model_output.pvalues[name])
        except Exception:
            pval = None
        # 95% CI from model_output.conf_int() when available
        try:
            ci_df = model_output.conf_int()
            if name in ci_df.index:
                ci = ci_df.loc[name].astype(float).tolist()
            else:
                ci = [val - 1.96 * se, val + 1.96 * se]
        except Exception:
            # approximate via normal approx
            ci = [val - 1.96 * se, val + 1.96 * se]
        return {'name': name, 'coef': val, 'se': se, 'pvalue': pval, 'ci_95': ci}

    info_reader = _get_param_info(reader_name)
    info_inter = _get_param_info(interaction_name)
    info_dys = _get_param_info(dyslexia_name) if dyslexia_name is not None else None

    # Marginal effect of Reader View for non-dyslexic readers (dyslexia_bin = 0)
    beta_nd = info_reader['coef']
    # SE: sqrt(var(reader))
    var_reader = float(cov.loc[reader_name, reader_name])
    se_nd = math.sqrt(max(0.0, var_reader))
    z_nd = beta_nd / se_nd if se_nd > 0 else float('nan')
    p_nd = 2.0 * (1.0 - _norm_cdf(abs(z_nd))) if se_nd > 0 else None
    ci_nd = [beta_nd - 1.96 * se_nd, beta_nd + 1.96 * se_nd]

    # Marginal effect of Reader View for dyslexic readers (dyslexia_bin = 1)
    beta_int = info_inter['coef']
    beta_d = beta_nd + beta_int
    # Var(beta_nd + beta_int) = Var(reader) + Var(interaction) + 2*Cov(reader, interaction)
    cov_rr = float(cov.loc[reader_name, reader_name])
    cov_ii = float(cov.loc[interaction_name, interaction_name])
    cov_ri = float(cov.loc[reader_name, interaction_name])
    var_d = cov_rr + cov_ii + 2.0 * cov_ri
    se_d = math.sqrt(max(0.0, var_d))
    z_d = beta_d / se_d if se_d > 0 else float('nan')
    p_d = 2.0 * (1.0 - _norm_cdf(abs(z_d))) if se_d > 0 else None
    ci_d = [beta_d - 1.96 * se_d, beta_d + 1.96 * se_d]

    # Convert log-scale effects to percent changes: (exp(beta)-1)*100
    pct_nd = (math.exp(beta_nd) - 1.0) * 100.0
    pct_nd_ci = [(math.exp(ci_nd[0]) - 1.0) * 100.0, (math.exp(ci_nd[1]) - 1.0) * 100.0]

    pct_d = (math.exp(beta_d) - 1.0) * 100.0
    pct_d_ci = [(math.exp(ci_d[0]) - 1.0) * 100.0, (math.exp(ci_d[1]) - 1.0) * 100.0]

    # Prepare return object
    result_object = {
        'reader_param': {
            'name': reader_name,
            'coef_log': info_reader['coef'],
            'se_log': info_reader['se'],
            'pvalue': info_reader['pvalue'],
            'ci_log_95': info_reader['ci_95']
        },
        'interaction_param': {
            'name': interaction_name,
            'coef_log': info_inter['coef'],
            'se_log': info_inter['se'],
            'pvalue': info_inter['pvalue'],
            'ci_log_95': info_inter['ci_95']
        },
        'dyslexia_param': (info_dys if info_dys is not None else None),
        'marginal_effects': {
            'non_dyslexic': {
                'coef_log': beta_nd,
                'se_log': se_nd,
                't_or_z': z_nd,
                'pvalue': p_nd,
                'ci_log_95': ci_nd,
                'percent_change': pct_nd,
                'percent_ci_95': pct_nd_ci
            },
            'dyslexic': {
                'coef_log': beta_d,
                'se_log': se_d,
                't_or_z': z_d,
                'pvalue': p_d,
                'ci_log_95': ci_d,
                'percent_change': pct_d,
                'percent_ci_95': pct_d_ci
            }
        },
        'notes': (
            "CIs and p-values for linear combinations were computed using the covariance matrix "
            "from model_output and a normal approximation (95% CI ≈ coef ± 1.96*SE). "
            "If you prefer t-based intervals or exact cluster-inference, compute those with "
            "the appropriate degrees of freedom or test routines."
        )
    }

    # Short interpretation / conclusion regarding the task question
    # We'll label 'improves' if the effect for dyslexic readers is positive and statistically significant at alpha=0.05
    conclusion = "inconclusive"
    p_thresh = result_object['marginal_effects']['dyslexic']['pvalue']
    coef_d = result_object['marginal_effects']['dyslexic']['coef_log']
    if p_thresh is not None:
        if (p_thresh < 0.05) and (coef_d > 0):
            conclusion = "yes"
        elif (p_thresh < 0.05) and (coef_d < 0):
            conclusion = "no_decrease"  # significant decrease
        else:
            conclusion = "no_evidence"

    # Friendly description
    if conclusion == "yes":
        description = (
            "Reader View appears to improve reading speed for readers with dyslexia: the "
            "estimated effect (log-scale) for dyslexic readers is {:.4f} (SE={:.4f}), "
            "corresponding to a {:.2f}% increase in speed (95% CI [{:.2f}%, {:.2f}%]), "
            "p = {:.3g} (two-sided, normal approximation)."
        ).format(coef_d, se_d, pct_d, pct_d_ci[0], pct_d_ci[1], p_d)
    elif conclusion == "no_decrease":
        description = (
            "Reader View significantly changed reading speed for readers with dyslexia, "
            "but the effect was a decrease. Estimated log-effect = {:.4f} (SE={:.4f}), "
            "corresponding to a {:.2f}% change (95% CI [{:.2f}%, {:.2f}%]), p = {:.3g}."
        ).format(coef_d, se_d, pct_d, pct_d_ci[0], pct_d_ci[1], p_d)
    elif conclusion == "no_evidence":
        description = (
            "There is no statistically significant evidence that Reader View changes reading speed "
            "for readers with dyslexia (estimated log-effect = {:.4f}, SE = {:.4f}, p = {:.3g}). "
            "Point estimate corresponds to a {:.2f}% change (95% CI [{:.2f}%, {:.2f}%])."
        ).format(coef_d, se_d, p_d, pct_d, pct_d_ci[0], pct_d_ci[1])
    else:
        description = (
            "Unable to determine a clear conclusion from the model output. "
            "Please inspect the returned 'object' for coefficients, SEs and CIs."
        )

    # Return structure required by the task
    return {
        "object": result_object,
        "description": description
    }