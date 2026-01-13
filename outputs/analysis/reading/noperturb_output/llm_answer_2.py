def extract_final_answer(model_output):
    """
    Extract statistics relevant to the effect of Reader View on reading speed
    for dyslexic readers from a fitted statsmodels OLS results object.

    Returns a dictionary with keys:
      - "object": nested dict with coefficients, SEs, CIs, p-values, percent changes,
                  and a computed marginal effect of Reader View for dyslexic readers.
      - "description": brief natural-language summary interpreting the results
                       (whether Reader View improves reading speed for dyslexic readers).

    Notes:
      - Expects model_output to be a statsmodels RegressionResults (possibly
        from get_robustcov_results). The model formula should include the terms
        "reader_view" and the interaction between reader_view and dyslexia_bin
        (something containing both "reader_view" and "dyslexia_bin" in its name).
      - The dependent variable is log_reading_speed, so reported effect sizes
        are converted to percent change: (exp(beta)-1)*100.
    """
    import numpy as np
    from math import exp
    from scipy import stats

    res = model_output

    # Try to obtain primary result components; allow for numpy arrays or pandas objects
    try:
        raw_params = getattr(res, 'params', None)
        raw_pvalues = getattr(res, 'pvalues', None)
        raw_bse = getattr(res, 'bse', None)
        raw_conf = None
        try:
            raw_conf = res.conf_int()
        except Exception:
            raw_conf = None
        raw_cov = None
        try:
            raw_cov = res.cov_params()
        except Exception:
            raw_cov = None
    except Exception as e:
        raise ValueError("Provided model_output does not look like a statsmodels results object: " + str(e))

    if raw_params is None:
        raise ValueError("model_output has no 'params' attribute")

    # Determine parameter names robustly
    param_names = None
    # If params is a pandas Series-like with index
    try:
        if hasattr(raw_params, 'index'):
            param_names = [str(x) for x in raw_params.index]
    except Exception:
        param_names = None

    # other fallbacks
    if param_names is None:
        if hasattr(res, 'param_names'):
            try:
                param_names = list(res.param_names)
            except Exception:
                param_names = None

    if param_names is None and hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
        try:
            param_names = list(res.model.exog_names)
        except Exception:
            param_names = None

    # Final fallback: if params is array-like, create generic names
    if param_names is None:
        try:
            arr = np.asarray(raw_params)
            param_names = [f'param_{i}' for i in range(arr.shape[0])]
        except Exception:
            raise ValueError("Could not determine parameter names from model_output")

    # Build mapping name -> index for lookups
    name_to_index = {name: idx for idx, name in enumerate(param_names)}

    # Helper to extract scalar arrays/dicts into name->value maps
    def build_map(raw_obj):
        if raw_obj is None:
            return {}
        # If object has index (pandas Series)
        try:
            if hasattr(raw_obj, 'index'):
                return {str(k): float(v) for k, v in zip(raw_obj.index, raw_obj.values)}
        except Exception:
            pass
        # If it's dict-like
        try:
            if isinstance(raw_obj, dict):
                return {str(k): float(v) for k, v in raw_obj.items()}
        except Exception:
            pass
        # If it's array-like, map by position using param_names
        try:
            arr = np.asarray(raw_obj)
            if arr.ndim == 1 and arr.shape[0] == len(param_names):
                return {name: float(arr[idx]) for idx, name in enumerate(param_names)}
        except Exception:
            pass
        # otherwise return empty
        return {}

    params_map = build_map(raw_params)
    pvalues_map = build_map(raw_pvalues)
    bse_map = build_map(raw_bse)

    # Handle confidence intervals: could be DataFrame/array with shape (n_params, 2)
    conf_map = {}
    if raw_conf is not None:
        try:
            # pandas DataFrame with index
            if hasattr(raw_conf, 'index'):
                for idx, row in zip(raw_conf.index, raw_conf.values):
                    conf_map[str(idx)] = (float(row[0]), float(row[1]))
            else:
                arr = np.asarray(raw_conf)
                if arr.ndim == 2 and arr.shape[0] == len(param_names) and arr.shape[1] >= 2:
                    for i, name in enumerate(param_names):
                        conf_map[name] = (float(arr[i, 0]), float(arr[i, 1]))
        except Exception:
            conf_map = {}

    # Covariance: handle DataFrame or ndarray
    cov_matrix = raw_cov  # could be None
    # find main reader term and interaction robustly
    def find_main_reader_term():
        if 'reader_view' in param_names:
            return 'reader_view'
        for nm in param_names:
            if 'reader_view' in nm and not (':' in nm or '*' in nm):
                return nm
        return None

    def find_interaction_term():
        for nm in param_names:
            if ('reader_view' in nm) and ('dyslexia_bin' in nm):
                return nm
        return None

    rname = find_main_reader_term()
    iname = find_interaction_term()
    # find dyslexia main effect name if present (prefer non-interaction)
    dname = None
    for nm in param_names:
        if 'dyslexia_bin' in nm and nm != iname:
            # exclude interactions with reader_view
            if not (('reader_view' in nm) and ('dyslexia_bin' in nm)):
                dname = nm
                break
    if dname is None and 'dyslexia_bin' in param_names:
        dname = 'dyslexia_bin'

    # Helper to get conf row
    def get_conf(name):
        if name is None:
            return (np.nan, np.nan)
        if name in conf_map:
            return conf_map[name]
        # fallback: if cov or other structures exist but conf_map missing, try to call conf_int again per-param
        return (np.nan, np.nan)

    # Helper to get term stats
    def get_term_stats(name):
        if name is None or name not in param_names:
            return None
        coef = float(params_map.get(name, np.nan))
        se = float(bse_map.get(name, np.nan)) if name in bse_map else float(np.nan)
        p = float(pvalues_map.get(name, np.nan)) if name in pvalues_map else float(np.nan)
        ci_low, ci_high = get_conf(name)
        # If conf_map lacked entries but raw_conf is ndarray, try index access
        if np.isnan(ci_low) or np.isnan(ci_high):
            try:
                if raw_conf is not None:
                    arr = np.asarray(raw_conf)
                    idx = name_to_index[name]
                    if arr.ndim == 2 and arr.shape[0] == len(param_names):
                        ci_low = float(arr[idx, 0])
                        ci_high = float(arr[idx, 1])
            except Exception:
                pass
        pct = (np.exp(coef) - 1.0) * 100.0 if not np.isnan(coef) else np.nan
        pct_ci_low = (np.exp(ci_low) - 1.0) * 100.0 if not np.isnan(ci_low) else np.nan
        pct_ci_high = (np.exp(ci_high) - 1.0) * 100.0 if not np.isnan(ci_high) else np.nan

        return {
            'term': name,
            'coef_log': coef,
            'se': se,
            'p_value': p,
            '95%_CI_log': [ci_low, ci_high],
            'pct_change': pct,
            '95%_CI_pct': [pct_ci_low, pct_ci_high],
        }

    reader_stats = get_term_stats(rname)
    interaction_stats = get_term_stats(iname)
    dyslexia_stats = get_term_stats(dname)

    if reader_stats is None:
        raise ValueError("Could not find a main 'reader_view' term in the model parameters. Check model formula/term names.")

    # Compute marginal effect of Reader View for dyslexic readers:
    beta_r = reader_stats['coef_log']
    if interaction_stats is not None:
        beta_int = interaction_stats['coef_log']
        beta_comb = beta_r + beta_int

        # compute variance of sum using covariance matrix if available
        se_comb = np.nan
        try:
            if cov_matrix is not None:
                # DataFrame-like
                if hasattr(cov_matrix, 'loc'):
                    var_r = float(cov_matrix.loc[rname, rname])
                    var_int = float(cov_matrix.loc[iname, iname])
                    cov_r_int = float(cov_matrix.loc[rname, iname])
                else:
                    arr = np.asarray(cov_matrix)
                    idx_r = name_to_index[rname]
                    idx_int = name_to_index[iname]
                    var_r = float(arr[idx_r, idx_r])
                    var_int = float(arr[idx_int, idx_int])
                    cov_r_int = float(arr[idx_r, idx_int])
                var_comb = var_r + var_int + 2.0 * cov_r_int
                se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else np.nan
        except Exception:
            se_comb = np.nan

        if np.isnan(se_comb):
            # fallback: approximate using sqrt(se_r^2 + se_int^2)
            se_r = reader_stats.get('se', np.nan)
            se_int = interaction_stats.get('se', np.nan)
            if not np.isnan(se_r) and not np.isnan(se_int):
                se_comb = float(np.sqrt(se_r ** 2 + se_int ** 2))
            else:
                se_comb = np.nan

        # CI on log scale
        df_resid = getattr(res, 'df_resid', None)
        try:
            if df_resid is not None and np.isfinite(df_resid) and df_resid > 0:
                tcrit = float(stats.t.ppf(0.975, df_resid))
            else:
                tcrit = float(stats.norm.ppf(0.975))
        except Exception:
            tcrit = float(stats.norm.ppf(0.975))

        ci_low_log = beta_comb - tcrit * se_comb if not np.isnan(se_comb) else np.nan
        ci_high_log = beta_comb + tcrit * se_comb if not np.isnan(se_comb) else np.nan
        pct_comb = (np.exp(beta_comb) - 1.0) * 100.0 if not np.isnan(beta_comb) else np.nan
        pct_ci_low = (np.exp(ci_low_log) - 1.0) * 100.0 if not np.isnan(ci_low_log) else np.nan
        pct_ci_high = (np.exp(ci_high_log) - 1.0) * 100.0 if not np.isnan(ci_high_log) else np.nan

        # p-value for combined effect
        try:
            t_stat = beta_comb / se_comb
            if df_resid is not None and np.isfinite(df_resid) and df_resid > 0:
                p_comb = 2.0 * float(stats.t.sf(abs(t_stat), df_resid))
            else:
                p_comb = 2.0 * float(stats.norm.sf(abs(t_stat)))
        except Exception:
            p_comb = float(np.nan)

        combined_stats = {
            'coef_log': beta_comb,
            'se': se_comb,
            't_or_z': (beta_comb / se_comb) if (se_comb not in (None, 0) and np.isfinite(se_comb)) else float(np.nan),
            'p_value': p_comb,
            '95%_CI_log': [ci_low_log, ci_high_log],
            'pct_change': pct_comb,
            '95%_CI_pct': [pct_ci_low, pct_ci_high],
            'note': 'Effect of Reader View (reader_view) when dyslexia_bin = 1 (dyslexic readers).'
        }
    else:
        # No interaction term: marginal effect is same as main effect
        se_r = reader_stats.get('se', np.nan)
        t_or_z = (beta_r / se_r) if (se_r not in (None, 0) and np.isfinite(se_r)) else float(np.nan)
        combined_stats = {
            'coef_log': beta_r,
            'se': se_r,
            't_or_z': t_or_z,
            'p_value': reader_stats.get('p_value', np.nan),
            '95%_CI_log': reader_stats.get('95%_CI_log', [np.nan, np.nan]),
            'pct_change': reader_stats.get('pct_change', np.nan),
            '95%_CI_pct': reader_stats.get('95%_CI_pct', [np.nan, np.nan]),
            'note': 'No reader_view:dyslexia interaction present; effect for dyslexic readers equals the main reader_view effect.'
        }

    # Construct a concise conclusion about whether Reader View "improves" reading speed for dyslexic readers.
    concl = {}
    coef_for_dys = combined_stats.get('coef_log', np.nan)
    p_for_dys = combined_stats.get('p_value', np.nan)
    pct_for_dys = combined_stats.get('pct_change', np.nan)
    ci_pct = combined_stats.get('95%_CI_pct', [np.nan, np.nan])
    try:
        if np.isnan(p_for_dys):
            conclusion_text = ("Could not compute a p-value for the marginal Reader View effect in dyslexic readers. "
                               "See extracted coefficients and CIs for inspection.")
        else:
            if (p_for_dys < 0.05) and (coef_for_dys > 0):
                conclusion_text = (f"Yes — Reader View is associated with a statistically significant increase in reading speed "
                                   f"for dyslexic readers (estimated change = {pct_for_dys:.1f}% ; 95% CI = [{ci_pct[0]:.1f}%, {ci_pct[1]:.1f}%], p = {p_for_dys:.3g}).")
            elif (p_for_dys < 0.05) and (coef_for_dys < 0):
                conclusion_text = (f"No — Reader View is associated with a statistically significant decrease in reading speed "
                                   f"for dyslexic readers (estimated change = {pct_for_dys:.1f}% ; 95% CI = [{ci_pct[0]:.1f}%, {ci_pct[1]:.1f}%], p = {p_for_dys:.3g}).")
            else:
                conclusion_text = (f"No — there is no statistically significant effect of Reader View on reading speed for dyslexic readers "
                                   f"(estimated change = {pct_for_dys:.1f}% ; 95% CI = [{ci_pct[0]:.1f}%, {ci_pct[1]:.1f}%], p = {p_for_dys:.3g}).")
    except Exception:
        conclusion_text = "Could not generate conclusion text due to missing statistics."

    # Build the object to return
    out = {
        'reader_view_term': reader_stats,
        'interaction_term': interaction_stats,
        'dyslexia_term': dyslexia_stats,
        'reader_view_effect_for_dyslexic': combined_stats,
        'conclusion': conclusion_text
    }

    description = ("This output gives coefficients, standard errors, p-values, and 95% confidence intervals "
                   "for the reader_view main effect and the reader_view:dyslexia interaction (if present), "
                   "and computes the marginal effect of Reader View for dyslexic readers (coef on log scale, "
                   "converted to percent change). The 'conclusion' field states whether the effect is "
                   "statistically significant at alpha = 0.05 and the direction/size of the effect.")

    return {'object': out, 'description': description}