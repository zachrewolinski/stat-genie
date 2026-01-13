def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View on log WPM for:
      - non-dyslexic readers (dyslexia_bin = 0)
      - dyslexic readers (dyslexia_bin = 1)
    Also extracts the interaction coefficient and provides percent-change
    interpretations (exp(coef)-1) with 95% CIs and p-values.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results
      - "description": a short textual interpretation (including a yes/no
                       statement about whether Reader View improves reading
                       speed for individuals with dyslexia at alpha=0.05)
    The function is robust to several common shapes of model_output (statsmodels
    results, dict-like params, numpy arrays with model.exog_names, etc.).
    """
    import numpy as np
    from math import exp, erf, sqrt
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = model_output

    # Helper: safe retrieval of parameter names
    param_index = None
    params_raw = None
    try:
        params_raw = getattr(res, "params", None)
    except Exception:
        params_raw = None

    # Try a series of fallbacks to obtain parameter names
    if params_raw is not None:
        # pandas-like Series
        if hasattr(params_raw, "index"):
            try:
                param_index = list(params_raw.index)
            except Exception:
                param_index = None
        # dict-like
        if param_index is None and isinstance(params_raw, dict):
            param_index = list(params_raw.keys())
        # array-like params, try to get names from model.exog_names or param_names
        if param_index is None and isinstance(params_raw, (list, tuple, np.ndarray)):
            # try model.exog_names
            model_obj = getattr(res, "model", None)
            if model_obj is not None and hasattr(model_obj, "exog_names"):
                try:
                    param_index = list(model_obj.exog_names)
                except Exception:
                    param_index = None
            # try res.param_names
            if param_index is None and hasattr(res, "param_names"):
                try:
                    param_index = list(getattr(res, "param_names"))
                except Exception:
                    param_index = None
            # otherwise create generic names so we can index by position
            if param_index is None:
                try:
                    n = len(params_raw)
                    param_index = [f"param_{i}" for i in range(n)]
                except Exception:
                    param_index = None

    # If still None, try other attributes on res
    if param_index is None:
        # try res.model.data.param_names (some interfaces)
        try:
            model_obj = getattr(res, "model", None)
            if model_obj is not None and hasattr(model_obj, "data") and hasattr(model_obj.data, "param_names"):
                param_index = list(model_obj.data.param_names)
        except Exception:
            param_index = None

    # Final fallback: if we still have nothing, raise a clear error
    if param_index is None:
        raise ValueError(
            "Unable to read parameter names from model_output. "
            "Tried res.params.index, dict/res.params, res.model.exog_names, and other fallbacks."
        )

    # Build a mapping name -> coef value (float)
    params_map = {}
    if params_raw is not None:
        # If params_raw is dict-like, use it directly where possible
        if isinstance(params_raw, dict):
            for k, v in params_raw.items():
                params_map[str(k)] = float(v)
        elif hasattr(params_raw, "index"):
            # pandas Series-like
            try:
                for k in param_index:
                    params_map[str(k)] = float(params_raw[k])
            except Exception:
                # fallback iterate values
                try:
                    vals = list(params_raw)
                    for name, val in zip(param_index, vals):
                        params_map[str(name)] = float(val)
                except Exception:
                    params_map = {}
        else:
            # array-like params_raw with param_index defined
            try:
                vals = list(params_raw)
                for name, val in zip(param_index, vals):
                    params_map[str(name)] = float(val)
            except Exception:
                params_map = {}

    # Retrieve covariance matrix
    cov_raw = None
    try:
        if hasattr(res, "cov_params"):
            cov_raw = res.cov_params()
        else:
            # some objects store cov as attribute
            cov_raw = getattr(res, "cov", None)
    except Exception:
        cov_raw = None

    # Helper to get covariance element (i,j) by parameter names
    def get_cov(name_i, name_j):
        # If cov_raw is a pandas DataFrame with loc
        if cov_raw is None:
            return None
        try:
            if hasattr(cov_raw, "loc"):
                return float(cov_raw.loc[name_i, name_j])
        except Exception:
            pass
        # If cov_raw is a numpy array, try to map names to indices via param_index
        try:
            if isinstance(cov_raw, (np.ndarray, list, tuple)):
                cov_arr = np.asarray(cov_raw)
                idx_i = param_index.index(name_i)
                idx_j = param_index.index(name_j)
                return float(cov_arr[idx_i, idx_j])
        except Exception:
            pass
        # If cov_raw supports keys like dict of dicts
        try:
            if isinstance(cov_raw, dict):
                return float(cov_raw[name_i][name_j])
        except Exception:
            pass
        # Give up
        return None

    # Heuristics to find parameter names
    # Prefer exact matches, then substring matches
    def find_name(candidates):
        # candidates: list of strings to search for (all must be present in name)
        for name in param_index:
            low = name.lower()
            ok = True
            for c in candidates:
                if c.lower() not in low:
                    ok = False
                    break
            if ok:
                return name
        return None

    # Main reader_view term
    main_name = None
    # try exact 'reader_view'
    if "reader_view" in param_index:
        main_name = "reader_view"
    else:
        # try plausible matches
        main_name = find_name(["reader", "view"])
        if main_name is None:
            # try 'reader' alone
            main_name = find_name(["reader"])
    if main_name is None:
        raise ValueError(f"Could not find main 'reader_view' coefficient among params: {param_index}")

    # Interaction term: prefer names containing both reader_view and dyslexia_bin (or dyslex)
    inter_name = None
    inter_name = find_name(["reader", "dyslexia"])
    if inter_name is None:
        # try common separators
        for sep in (":", "*", "."):
            cand = f"reader_view{sep}dyslexia_bin"
            if cand in param_index:
                inter_name = cand
                break
    # If still None, look for any param containing reader_view and ':' or '*'
    if inter_name is None:
        for n in param_index:
            low = n.lower()
            if ("reader" in low and ("dyslex" in low)) or (("reader" in low) and (":" in n or "*" in n)):
                inter_name = n
                break
    # Leave inter_name as None if not found (we'll treat it as zero effect)

    # Dyslexia main effect
    dys_name = None
    if "dyslexia_bin" in param_index:
        dys_name = "dyslexia_bin"
    else:
        dys_name = find_name(["dyslex"])

    # Helper to get coefficient by name (returns float or None)
    def get_coef(name):
        if name is None:
            return None
        if name in params_map:
            return params_map[name]
        # try alternate keys (string conversions)
        for k in params_map.keys():
            if k == name:
                return params_map[k]
        # not found
        return None

    beta_r = get_coef(main_name)
    if beta_r is None:
        # as a last resort, try position 0 if only one param_index matches something
        raise ValueError(f"Could not retrieve coefficient for main term '{main_name}' from model_output.")

    beta_r = float(beta_r)

    beta_int = 0.0
    if inter_name is not None:
        got = get_coef(inter_name)
        if got is not None:
            beta_int = float(got)
        else:
            # interaction name indicated but value not found: treat as zero
            beta_int = 0.0
            inter_name = None
    else:
        beta_int = 0.0

    # Retrieve variances and covariances
    # var_r
    var_r = None
    try:
        vr = get_cov(main_name, main_name)
        if vr is not None:
            var_r = float(vr)
    except Exception:
        var_r = None
    # var_int
    var_int = None
    if inter_name is not None:
        try:
            vi = get_cov(inter_name, inter_name)
            if vi is not None:
                var_int = float(vi)
        except Exception:
            var_int = None
    # cov between r and int
    cov_r_int = None
    if inter_name is not None:
        try:
            cr = get_cov(main_name, inter_name)
            if cr is not None:
                cov_r_int = float(cr)
        except Exception:
            cov_r_int = None

    # If any of these are None, try to extract cov_raw as array and fallback to zeros where appropriate
    if var_r is None:
        # try diagonal from cov_raw by locating index
        try:
            if cov_raw is not None:
                if hasattr(cov_raw, "loc"):
                    var_r = float(cov_raw.loc[main_name, main_name])
                else:
                    cov_arr = np.asarray(cov_raw)
                    idx = param_index.index(main_name)
                    var_r = float(cov_arr[idx, idx])
        except Exception:
            var_r = None
    if var_r is None:
        # As a last resort, set to nan to avoid zero-division silently
        var_r = float("nan")

    if inter_name is None:
        # No interaction present: set interaction variance and covariance to zero
        var_int = 0.0
        cov_r_int = 0.0
    else:
        if var_int is None:
            try:
                if cov_raw is not None:
                    if hasattr(cov_raw, "loc"):
                        var_int = float(cov_raw.loc[inter_name, inter_name])
                    else:
                        cov_arr = np.asarray(cov_raw)
                        idx = param_index.index(inter_name)
                        var_int = float(cov_arr[idx, idx])
            except Exception:
                var_int = float("nan")
        if cov_r_int is None:
            try:
                if cov_raw is not None:
                    if hasattr(cov_raw, "loc"):
                        cov_r_int = float(cov_raw.loc[main_name, inter_name])
                    else:
                        cov_arr = np.asarray(cov_raw)
                        idx_r = param_index.index(main_name)
                        idx_i = param_index.index(inter_name)
                        cov_r_int = float(cov_arr[idx_r, idx_i])
            except Exception:
                cov_r_int = float("nan")

    # Effects:
    # - effect_non_dys (dyslexia_bin = 0) = beta_r
    # - effect_dys (dyslexia_bin = 1) = beta_r + beta_int
    effect_non_dys = float(beta_r)
    se_non_dys = float(np.sqrt(var_r)) if (var_r is not None and not np.isnan(var_r)) else float("nan")

    effect_dys = float(beta_r + beta_int)
    # compute se_dys = sqrt(var_r + var_int + 2*cov_r_int)
    try:
        se_dys_val = var_r + var_int + 2.0 * cov_r_int
        se_dys = float(np.sqrt(se_dys_val)) if (se_dys_val is not None and not np.isnan(se_dys_val)) else float("nan")
    except Exception:
        se_dys = float("nan")

    # Interaction stats
    se_inter = float(np.sqrt(var_int)) if (var_int is not None and not np.isnan(var_int)) else float("nan")

    # Degrees of freedom for t-based CIs/p-values (fallback to large-sample normal if not present)
    try:
        df_resid = float(res.df_resid)
        use_t = True
    except Exception:
        df_resid = None
        use_t = False

    def two_sided_p_from_t(tval, df):
        if tval is None or (isinstance(tval, float) and np.isnan(tval)):
            return float("nan")
        if stats is not None and df is not None:
            return float(2.0 * stats.t.sf(abs(tval), df))
        elif stats is not None:
            return float(2.0 * stats.norm.sf(abs(tval)))
        else:
            # normal approximation using erf
            z = abs(tval)
            # normal tail: 0.5*(1 - erf(z/sqrt(2)))
            tail = 0.5 * (1.0 - erf(z / sqrt(2.0)))
            return float(2.0 * tail)

    def ci_for_estimate(est, se, df, alpha=0.05):
        if est is None or se is None or np.isnan(se):
            return (None, None)
        if stats is not None and df is not None:
            tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
        elif stats is not None:
            tcrit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        else:
            tcrit = 1.96
        return (est - tcrit * se, est + tcrit * se)

    # Compute t/p/CI for non-dys and dys effects
    t_non = effect_non_dys / se_non_dys if se_non_dys > 0 and not np.isnan(se_non_dys) else float("nan")
    p_non = two_sided_p_from_t(t_non, df_resid if use_t else None)
    ci_non = ci_for_estimate(effect_non_dys, se_non_dys, df_resid if use_t else None)

    t_dys = effect_dys / se_dys if se_dys > 0 and not np.isnan(se_dys) else float("nan")
    p_dys = two_sided_p_from_t(t_dys, df_resid if use_t else None)
    ci_dys = ci_for_estimate(effect_dys, se_dys, df_resid if use_t else None)

    # Interaction stats
    t_inter = beta_int / se_inter if se_inter > 0 and not np.isnan(se_inter) else float("nan")
    p_inter = two_sided_p_from_t(t_inter, df_resid if use_t else None)
    ci_inter = ci_for_estimate(beta_int, se_inter, df_resid if use_t else None)

    # Percent-change interpretation (from log outcome) — guard against None
    def pct_from_log(est):
        if est is None or (isinstance(est, float) and np.isnan(est)):
            return float("nan")
        return (exp(est) - 1.0) * 100.0

    pct_non = pct_from_log(effect_non_dys)
    pct_non_ci = (pct_from_log(ci_non[0]) if ci_non[0] is not None else float("nan"),
                  pct_from_log(ci_non[1]) if ci_non[1] is not None else float("nan"))

    pct_dys = pct_from_log(effect_dys)
    pct_dys_ci = (pct_from_log(ci_dys[0]) if ci_dys[0] is not None else float("nan"),
                  pct_from_log(ci_dys[1]) if ci_dys[1] is not None else float("nan"))

    # Decide whether Reader View improves reading speed for dyslexic readers at alpha=0.05
    alpha = 0.05
    improves_for_dys = False
    try:
        improves_for_dys = (not np.isnan(p_dys)) and (p_dys < alpha) and (effect_dys > 0)
    except Exception:
        improves_for_dys = False

    # Build return object
    out = {
        "reader_view_main_param_name": main_name,
        "interaction_param_name": inter_name,
        "dyslexia_param_name": dys_name,
        "non_dys": {
            "coef_log_wpm": effect_non_dys,
            "se": se_non_dys,
            "t": t_non,
            "p_value": p_non,
            "95ci_log_wpm": ci_non,
            "percent_change_wpm": pct_non,
            "percent_change_95ci": pct_non_ci,
        },
        "dyslexic": {
            "coef_log_wpm": effect_dys,
            "se": se_dys,
            "t": t_dys,
            "p_value": p_dys,
            "95ci_log_wpm": ci_dys,
            "percent_change_wpm": pct_dys,
            "percent_change_95ci": pct_dys_ci,
        },
        "interaction": {
            "coef_log_wpm": float(beta_int),
            "se": se_inter,
            "t": t_inter,
            "p_value": p_inter,
            "95ci_log_wpm": ci_inter,
        },
        "model_df_resid": df_resid,
        "alpha": alpha,
        "improves_for_dyslexic_readers_at_alpha": bool(improves_for_dys),
    }

    # Helper formatting for description (handle None/NaN gracefully)
    def fmt_val(x, digits=4):
        try:
            if x is None:
                return "NA"
            if isinstance(x, float) and np.isnan(x):
                return "NA"
            fmt = f"{{:.{digits}f}}"
            return fmt.format(float(x))
        except Exception:
            return str(x)

    def fmt_p(x):
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return "NA"
            if x < 0.001:
                return f"{x:.2e}"
            return f"{x:.3g}"
        except Exception:
            return str(x)

    if improves_for_dys:
        verdict = "YES"
        reason = (
            f"At alpha = {alpha}, Reader View is associated with a statistically significant "
            f"increase in reading speed for readers with dyslexia (log-WPM coef = {fmt_val(effect_dys)}, "
            f"95% CI [{fmt_val(ci_dys[0])}, {fmt_val(ci_dys[1])}], p = {fmt_p(p_dys)}). "
            f"This corresponds to an estimated {fmt_val(pct_dys, 1)}% increase in WPM "
            f"(95% CI [{fmt_val(pct_dys_ci[0],1)}%, {fmt_val(pct_dys_ci[1],1)}%])."
        )
    else:
        verdict = "NO"
        reason = (
            f"At alpha = {alpha}, Reader View is NOT associated with a statistically significant "
            f"increase in reading speed for readers with dyslexia (log-WPM coef = {fmt_val(effect_dys)}, "
            f"95% CI [{fmt_val(ci_dys[0])}, {fmt_val(ci_dys[1])}], p = {fmt_p(p_dys)}). "
            f"The point estimate corresponds to a {fmt_val(pct_dys,1)}% change in WPM "
            f"(95% CI [{fmt_val(pct_dys_ci[0],1)}%, {fmt_val(pct_dys_ci[1],1)}%])."
        )

    description = (
        f"Extracted coefficients for Reader View effect (interaction model):\n"
        f"- Effect among non-dyslexic readers (dyslexia_bin=0): coef={fmt_val(effect_non_dys)}, se={fmt_val(se_non_dys)}, "
        f"95% CI=[{fmt_val(ci_non[0])}, {fmt_val(ci_non[1])}], p={fmt_p(p_non)}, approx {fmt_val(pct_non,1)}% change in WPM.\n"
        f"- Effect among dyslexic readers (dyslexia_bin=1): coef={fmt_val(effect_dys)}, se={fmt_val(se_dys)}, "
        f"95% CI=[{fmt_val(ci_dys[0])}, {fmt_val(ci_dys[1])}], p={fmt_p(p_dys)}, approx {fmt_val(pct_dys,1)}% change in WPM.\n"
        f"- Interaction term (reader_view x dyslexia_bin): coef={fmt_val(beta_int)}, se={fmt_val(se_inter)}, "
        f"95% CI=[{fmt_val(ci_inter[0])}, {fmt_val(ci_inter[1])}], p={fmt_p(p_inter)}.\n\n"
        f"Verdict: {verdict}. {reason}\n\n"
        f"Notes: All effects are on log(WPM). Percent changes computed as (exp(coef)-1)*100. "
        f"Standard errors and p-values reflect the covariance matrix in the provided model_output "
        f"(cluster-robust if supplied)."
    )

    return {"object": out, "description": description}