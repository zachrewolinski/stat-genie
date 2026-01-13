def extract_final_answer(model_output):
    """
    Extracts statistics related to the ReaderView effect from a fitted statsmodels-like results object.
    Returns a dictionary with keys:
      - "object": dictionary containing numeric results for:
          * ReaderView effect for non-dyslexic readers (main effect)
          * Interaction term (ReaderView:Dyslexia) if present
          * Combined ReaderView effect for dyslexic readers (main + interaction) if interaction present
        Each entry includes coefficient, SE, t, p, 95% CI on log-WPM scale and the equivalent
        percent change in WPM (exp(coef)-1) with CI translated to percent change.
      - "description": A short human-readable interpretation of the reported numbers.
    The function is robust to statsmodels results objects where params, pvalues, tvalues,
    conf_int, or cov_params may be numpy arrays rather than pandas objects.
    """
    import numpy as np
    from math import erf, sqrt, isnan

    res = model_output

    # Try to fetch commonly used attributes, but accept that they might be numpy arrays
    try:
        params_raw = res.params
    except Exception as e:
        raise ValueError(f"Model output does not have 'params': {e}")

    # Helper to turn various container types into numpy arrays
    def to_1d_array(x):
        if x is None:
            return None
        try:
            arr = np.asarray(x)
            # flatten column/row vectors
            if arr.ndim > 1:
                arr = arr.ravel()
            return arr
        except Exception:
            return None

    params_arr = to_1d_array(params_raw)

    # Build parameter names list from several possible locations
    names = None
    if hasattr(params_raw, "index"):
        # pandas Series
        names = [str(n) for n in params_raw.index]
    elif hasattr(res, "param_names"):
        try:
            names = [str(n) for n in res.param_names]
        except Exception:
            names = None
    elif hasattr(res, "model") and hasattr(res.model, "exog_names"):
        try:
            names = [str(n) for n in res.model.exog_names]
        except Exception:
            names = None

    if names is None:
        # fallback: create generic names if length is known
        if params_arr is not None:
            names = [f"param_{i}" for i in range(params_arr.size)]
        else:
            raise ValueError("Could not determine parameter names from model output.")

    # Ensure consistent lengths
    n_params = len(names)
    if params_arr is None:
        # try to build from res.params iteratively
        try:
            params_arr = np.array([float(res.params[n]) for n in names])
        except Exception:
            raise ValueError("Could not coerce params to numeric array.")
    if params_arr.size != n_params:
        # Try to reshape or truncate/pad if needed
        params_arr = params_arr.ravel()
        if params_arr.size < n_params:
            # pad with nans
            params_arr = np.concatenate([params_arr, np.full(n_params - params_arr.size, np.nan)])
        else:
            params_arr = params_arr[:n_params]

    # pvalues and tvalues
    pvalues_arr = None
    tvalues_arr = None
    try:
        pvalues_arr = to_1d_array(res.pvalues)
    except Exception:
        pvalues_arr = None
    try:
        tvalues_arr = to_1d_array(res.tvalues)
    except Exception:
        tvalues_arr = None

    # If pvalues/tvalues are None or wrong length, try to build by name lookup
    if pvalues_arr is None or pvalues_arr.size != n_params:
        # attempt to build from res.pvalues indexing by name
        p_tmp = []
        if hasattr(res, "pvalues"):
            try:
                for nm in names:
                    p_tmp.append(float(res.pvalues[nm]))
                pvalues_arr = np.array(p_tmp)
            except Exception:
                pvalues_arr = np.full(n_params, np.nan)
        else:
            pvalues_arr = np.full(n_params, np.nan)

    if tvalues_arr is None or tvalues_arr.size != n_params:
        t_tmp = []
        if hasattr(res, "tvalues"):
            try:
                for nm in names:
                    t_tmp.append(float(res.tvalues[nm]))
                tvalues_arr = np.array(t_tmp)
            except Exception:
                tvalues_arr = np.full(n_params, np.nan)
        else:
            tvalues_arr = np.full(n_params, np.nan)

    # conf_int: could be DataFrame, ndarray, or method
    conf_int_arr = None
    try:
        ci_raw = res.conf_int()
        conf_int_arr = np.asarray(ci_raw)
        # Expect shape (n_params, 2)
        if conf_int_arr.ndim == 1 and conf_int_arr.size == 2 * n_params:
            conf_int_arr = conf_int_arr.reshape((n_params, 2))
    except Exception:
        conf_int_arr = None

    # cov params: could be DataFrame or ndarray
    cov_arr = None
    try:
        cov_raw = res.cov_params()
        cov_arr = np.asarray(cov_raw)
        if cov_arr.ndim != 2:
            cov_arr = None
    except Exception:
        cov_arr = None

    # bse fallback
    bse_arr = None
    try:
        bse_arr = to_1d_array(res.bse)
        if bse_arr is not None and bse_arr.size != n_params:
            bse_arr = None
    except Exception:
        bse_arr = None

    # Helper to find a parameter name by exact match or fuzzy match
    def find_name(target):
        # exact match first
        for n in names:
            if n == target:
                return n
        # substring match (prefer no ':' included)
        for n in names:
            if target in n and ':' not in n:
                return n
        # any substring match
        for n in names:
            if target in n:
                return n
        return None

    # Primary names we expect
    name_r = find_name('ReaderView')
    name_d = find_name('Dyslexia')

    # Find interaction term (could be 'ReaderView:Dyslexia' or similar)
    name_inter = None
    for n in names:
        if ('ReaderView' in n) and ('Dyslexia' in n) and n != name_r and n != name_d:
            name_inter = n
            break

    if name_r is None:
        # try any param containing 'ReaderView'
        matches = [n for n in names if 'ReaderView' in n]
        if matches:
            name_r = matches[0]
    if name_d is None:
        matches = [n for n in names if 'Dyslexia' in n]
        if matches:
            name_d = matches[0]

    if name_r is None:
        raise ValueError("Could not find a parameter for 'ReaderView' in model parameters: "
                         f"available params = {names}")

    # mapping name -> index
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Utility to build result dict for a given param name
    def param_summary(param_name):
        if param_name not in name_to_idx:
            raise KeyError(f"Parameter name '{param_name}' not found among {names}")
        idx = name_to_idx[param_name]
        coef = float(params_arr[idx]) if not isnan(params_arr[idx]) else float('nan')

        # SE: prefer cov matrix diagonal, then bse array, then try res.bse[name]
        se = None
        if cov_arr is not None and cov_arr.shape[0] == n_params and cov_arr.shape[1] == n_params:
            se = float(np.sqrt(max(cov_arr[idx, idx], 0.0)))
        elif bse_arr is not None:
            se = float(bse_arr[idx])
        else:
            # try attribute lookup
            try:
                se = float(res.bse[param_name])
            except Exception:
                se = float('nan')

        # t-value
        if tvalues_arr is not None and idx < tvalues_arr.size and not isnan(tvalues_arr[idx]):
            t = float(tvalues_arr[idx])
        else:
            t = float(coef / se) if se != 0 and not isnan(se) else float('nan')

        # p-value
        if pvalues_arr is not None and idx < pvalues_arr.size and not isnan(pvalues_arr[idx]):
            p = float(pvalues_arr[idx])
        else:
            # two-sided normal approximation from t
            p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) if not isnan(t) else float('nan')

        # confidence interval
        if conf_int_arr is not None and conf_int_arr.shape[0] == n_params and conf_int_arr.shape[1] >= 2:
            ci_low, ci_high = float(conf_int_arr[idx, 0]), float(conf_int_arr[idx, 1])
        else:
            ci_low, ci_high = (coef - 1.96 * se, coef + 1.96 * se) if not isnan(se) else (float('nan'), float('nan'))

        # Translate log-WPM coefficient to multiplicative change in WPM
        wpm_multiplier = float(np.exp(coef)) if not isnan(coef) else float('nan')
        pct_change = float(wpm_multiplier - 1.0) if not isnan(wpm_multiplier) else float('nan')
        pct_ci_low = float(np.exp(ci_low) - 1.0) if not isnan(ci_low) else float('nan')
        pct_ci_high = float(np.exp(ci_high) - 1.0) if not isnan(ci_high) else float('nan')

        return {
            "param_name": param_name,
            "coef_logWPM": coef,
            "se_logWPM": se,
            "t_value": t,
            "p_value": p,
            "ci_logWPM": [ci_low, ci_high],
            "wpm_multiplier": wpm_multiplier,  # multiplicative factor on WPM
            "wpm_pct_change": pct_change,      # fractional change (e.g., 0.10 = +10%)
            "wpm_pct_ci": [pct_ci_low, pct_ci_high]
        }

    results = {}

    # Main effect: ReaderView coefficient (this is the effect when Dyslexia == 0)
    results['ReaderView_for_non_dyslexic'] = param_summary(name_r)

    # Interaction term summary if present
    if name_inter is not None:
        results['Interaction_ReaderView_x_Dyslexia'] = param_summary(name_inter)

        # Compute combined effect for dyslexic readers: ReaderView + Interaction
        expr = f"{name_r} + {name_inter}"
        combined_coef = None
        combined_se = None
        combined_t = None
        combined_p = None
        combined_ci_low = None
        combined_ci_high = None

        # Try to use res.t_test if available and the expression uses known parameter names
        used_ttest = False
        try:
            if hasattr(res, "t_test"):
                tt = res.t_test(expr)
                # ContrastResults effect and sd can be arrays/scalars
                eff = getattr(tt, "effect", None)
                sd = getattr(tt, "sd", None)
                tval = getattr(tt, "tvalue", None)
                pval = getattr(tt, "pvalue", None)
                ci = None
                try:
                    ci = tt.conf_int()
                    ci = np.asarray(ci)
                except Exception:
                    ci = None

                # Extract scalars
                if eff is not None:
                    eff = np.asarray(eff).ravel()
                    combined_coef = float(eff[0])
                if sd is not None:
                    sd = np.asarray(sd).ravel()
                    combined_se = float(sd[0])
                if tval is not None:
                    tval = np.asarray(tval).ravel()
                    combined_t = float(tval[0])
                if pval is not None:
                    pval = np.asarray(pval).ravel()
                    combined_p = float(pval[0]) if pval.size == 1 else float(pval.ravel()[0])
                if ci is not None and ci.ndim >= 2:
                    combined_ci_low, combined_ci_high = float(ci[0, 0]), float(ci[0, 1])

                used_ttest = True
        except Exception:
            used_ttest = False

        if not used_ttest or combined_coef is None or combined_se is None:
            # fallback to manual sum using covariance if available
            idx_r = name_to_idx[name_r]
            idx_i = name_to_idx[name_inter]
            coef_r = float(params_arr[idx_r])
            coef_int = float(params_arr[idx_i])
            combined_coef = coef_r + coef_int

            if cov_arr is not None and cov_arr.shape[0] == n_params and cov_arr.shape[1] == n_params:
                var_r = float(max(cov_arr[idx_r, idx_r], 0.0))
                var_int = float(max(cov_arr[idx_i, idx_i], 0.0))
                cov_ri = float(cov_arr[idx_r, idx_i])
            else:
                # try bse fallback (assume independence if no covariance)
                var_r = float(bse_arr[idx_r] ** 2) if bse_arr is not None else 0.0
                var_int = float(bse_arr[idx_i] ** 2) if bse_arr is not None else 0.0
                cov_ri = 0.0

            combined_se = float(np.sqrt(max(var_r + var_int + 2 * cov_ri, 0.0)))
            combined_t = combined_coef / combined_se if combined_se != 0 else float('nan')
            combined_p = 2 * (1 - 0.5 * (1 + erf(abs(combined_t) / sqrt(2)))) if not isnan(combined_t) else float('nan')
            combined_ci_low = combined_coef - 1.96 * combined_se
            combined_ci_high = combined_coef + 1.96 * combined_se

        pct_change = float(np.exp(combined_coef) - 1.0) if not isnan(combined_coef) else float('nan')
        pct_ci_low = float(np.exp(combined_ci_low) - 1.0) if not isnan(combined_ci_low) else float('nan')
        pct_ci_high = float(np.exp(combined_ci_high) - 1.0) if not isnan(combined_ci_high) else float('nan')

        results['ReaderView_for_dyslexic'] = {
            "combined_coef_logWPM": combined_coef,
            "combined_se_logWPM": combined_se,
            "combined_t_value": combined_t,
            "combined_p_value": combined_p,
            "combined_ci_logWPM": [combined_ci_low, combined_ci_high],
            "combined_wpm_multiplier": float(np.exp(combined_coef)) if not isnan(combined_coef) else float('nan'),
            "combined_wpm_pct_change": pct_change,
            "combined_wpm_pct_ci": [pct_ci_low, pct_ci_high],
            "expression_used": expr
        }

    else:
        results['Interaction_ReaderView_x_Dyslexia'] = None
        # Without an interaction term, the ReaderView effect is the same for both groups;
        # replicate the non-dyslexic result for dyslexic for clarity.
        results['ReaderView_for_dyslexic'] = results['ReaderView_for_non_dyslexic']

    # Build a concise description
    if name_inter is not None:
        descr = (
            f"Model reports the ReaderView main effect ({name_r}) as the effect for non-dyslexic readers. "
            "The interaction term shows how that effect differs for readers with dyslexia. "
            "We report coefficients on the log(WPM) scale and translate them to percent change in WPM "
            "(exp(coef)-1). 'ReaderView_for_non_dyslexic' is the effect when Dyslexia=0; "
            "'ReaderView_for_dyslexic' is the combined effect (main + interaction) when Dyslexia=1."
        )
    else:
        descr = (
            f"Model reports a single ReaderView effect ({name_r}) (no interaction term detected). "
            "That coefficient applies to all readers; we provide log-WPM coefficient, SE, t, p, 95% CI, "
            "and the equivalent multiplicative/percent change in WPM (exp(coef)-1)."
        )

    return {"object": results, "description": descr}