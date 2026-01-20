def extract_final_answer(model_output):
    """
    Extracts statistics relevant to the effect of Reader View on log_speed from a fitted statsmodels
    results object (possibly robustified via get_robustcov_results).

    Returns:
      {
        "object": {...},       # numeric results and marginal effects
        "description": "..."   # short interpretation in plain language
      }
    """
    import math
    import numpy as np

    # Try to import scipy.stats for more accurate p-values/conf intervals; fall back to normal approx if not available
    try:
        from scipy import stats
        has_scipy = True
    except Exception:
        stats = None
        has_scipy = False

    res = model_output

    # Safely access required attributes
    params = getattr(res, "params", None)
    bse = getattr(res, "bse", None)
    pvalues = getattr(res, "pvalues", None)
    try:
        ci_df = res.conf_int()
        has_conf_int = True
    except Exception:
        ci_df = None
        has_conf_int = False

    # Build a parameter name list and mapping to indices (works if params is ndarray or pandas Series)
    def get_param_names(res_obj, params_obj):
        # Prefer model.exog_names if available
        try:
            names = list(res_obj.model.exog_names)
            if names:
                return names
        except Exception:
            pass

        # If params is a pandas object with index
        if params_obj is not None:
            if hasattr(params_obj, "index"):
                try:
                    return list(params_obj.index)
                except Exception:
                    pass
            # If params is a dict-like
            try:
                keys = list(params_obj.keys())
                if keys:
                    return keys
            except Exception:
                pass
            # If params is an array-like, generate generic names
            try:
                length = len(params_obj)
                return [f"param_{i}" for i in range(length)]
            except Exception:
                pass
        # Fallback empty
        return []

    param_names = get_param_names(res, params)
    param_index = {name: idx for idx, name in enumerate(param_names)}

    # Helper: find parameter name robustly
    def find_param_key(substr, avoid=None):
        """
        Find a parameter key that contains substr. If avoid provided, ensure avoid not in key.
        Returns the first match or None.
        """
        for k in param_names:
            if substr in k and (avoid is None or avoid not in k):
                return k
        return None

    # Identify keys
    key_reader = find_param_key("reader_view", avoid="dyslexia")
    # Interaction might be 'reader_view:dyslexia_bin' or similar
    key_inter = None
    for k in param_names:
        if "reader_view" in k and "dyslexia" in k:
            key_inter = k
            break

    # Dyslexia main effect key (not strictly needed here, but attempt)
    key_dys = find_param_key("dyslexia_bin", avoid="reader_view")

    # Extract helper to safely get numeric values from series/arrays/dataframes
    def safe_get(series, key):
        if series is None or key is None:
            return None
        # pandas Series / DataFrame access
        try:
            if hasattr(series, "loc") and key in getattr(series, "index", []):
                return float(series.loc[key])
        except Exception:
            pass
        # dict-like
        try:
            if isinstance(series, dict) and key in series:
                return float(series[key])
        except Exception:
            pass
        # array-like using param_index mapping
        try:
            idx = param_index.get(key)
            if idx is not None and idx < len(series):
                return float(series[idx])
        except Exception:
            pass
        return None

    def safe_get_ci(ci_obj, key, col):
        if ci_obj is None or key is None:
            return None
        # pandas DataFrame
        try:
            if hasattr(ci_obj, "loc") and key in getattr(ci_obj, "index", []):
                return float(ci_obj.loc[key, col])
        except Exception:
            pass
        # array-like: assume shape (n_params, 2) or (n_params, k)
        try:
            idx = param_index.get(key)
            if idx is not None and idx < len(ci_obj):
                return float(ci_obj[idx, col])
        except Exception:
            pass
        return None

    reader_coef = safe_get(params, key_reader)
    reader_se = safe_get(bse, key_reader)
    reader_p = safe_get(pvalues, key_reader)
    if has_conf_int:
        try:
            lower = safe_get_ci(ci_df, key_reader, 0)
            upper = safe_get_ci(ci_df, key_reader, 1)
            reader_ci = (lower, upper)
        except Exception:
            reader_ci = (None, None)
    else:
        reader_ci = (None, None)

    inter_coef = safe_get(params, key_inter)
    inter_se = safe_get(bse, key_inter)
    inter_p = safe_get(pvalues, key_inter)
    if has_conf_int and key_inter is not None:
        try:
            lower = safe_get_ci(ci_df, key_inter, 0)
            upper = safe_get_ci(ci_df, key_inter, 1)
            inter_ci = (lower, upper)
        except Exception:
            inter_ci = (None, None)
    else:
        inter_ci = (None, None)

    # Marginal effects:
    # For no-dyslexia: effect = reader_coef
    marginal_no = {"coef": reader_coef, "se": reader_se, "pval": reader_p, "ci": reader_ci}
    # For dyslexia=1: effect = reader_coef + inter_coef (if inter exists)
    coef_dys = None
    se_sum = None
    p_dys = None
    ci_lower = None
    ci_upper = None

    if reader_coef is not None and inter_coef is not None:
        coef_dys = reader_coef + inter_coef

        cov = None
        try:
            cov = res.cov_params()
        except Exception:
            cov = None

        if cov is not None:
            # Try DataFrame access first
            try:
                if hasattr(cov, "loc"):
                    if key_reader in getattr(cov, "index", []) and key_inter in getattr(cov, "index", []):
                        var_sum = float(cov.loc[key_reader, key_reader]) + float(cov.loc[key_inter, key_inter]) + 2.0 * float(cov.loc[key_reader, key_inter])
                        se_sum = math.sqrt(max(var_sum, 0.0))
                        # p-value and CI
                        df_resid = getattr(res, "df_resid", None)
                        if has_scipy and df_resid is not None:
                            tstat = coef_dys / se_sum if se_sum > 0 else float("nan")
                            p_dys = 2.0 * (1.0 - stats.t.cdf(abs(tstat), df=df_resid))
                            t_crit = stats.t.ppf(0.975, df=df_resid)
                            ci_lower = coef_dys - t_crit * se_sum
                            ci_upper = coef_dys + t_crit * se_sum
                        else:
                            zstat = coef_dys / se_sum if se_sum > 0 else float("nan")
                            if has_scipy:
                                p_dys = 2.0 * (1.0 - stats.norm.cdf(abs(zstat)))
                                z_crit = stats.norm.ppf(0.975)
                            else:
                                p_dys = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(zstat) / math.sqrt(2.0))))
                                z_crit = 1.959963984540054
                            ci_lower = coef_dys - z_crit * se_sum
                            ci_upper = coef_dys + z_crit * se_sum
                else:
                    # array-like covariance
                    idx_r = param_index.get(key_reader)
                    idx_i = param_index.get(key_inter)
                    if idx_r is not None and idx_i is not None and idx_r < cov.shape[0] and idx_i < cov.shape[0]:
                        var_sum = float(cov[idx_r, idx_r]) + float(cov[idx_i, idx_i]) + 2.0 * float(cov[idx_r, idx_i])
                        se_sum = math.sqrt(max(var_sum, 0.0))
                        df_resid = getattr(res, "df_resid", None)
                        if has_scipy and df_resid is not None:
                            tstat = coef_dys / se_sum if se_sum > 0 else float("nan")
                            p_dys = 2.0 * (1.0 - stats.t.cdf(abs(tstat), df=df_resid))
                            t_crit = stats.t.ppf(0.975, df=df_resid)
                            ci_lower = coef_dys - t_crit * se_sum
                            ci_upper = coef_dys + t_crit * se_sum
                        else:
                            zstat = coef_dys / se_sum if se_sum > 0 else float("nan")
                            if has_scipy:
                                p_dys = 2.0 * (1.0 - stats.norm.cdf(abs(zstat)))
                                z_crit = stats.norm.ppf(0.975)
                            else:
                                p_dys = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(zstat) / math.sqrt(2.0))))
                                z_crit = 1.959963984540054
                            ci_lower = coef_dys - z_crit * se_sum
                            ci_upper = coef_dys + z_crit * se_sum
            except Exception:
                # If any step fails, leave se_sum/p_dys/ci as None
                se_sum = None
                p_dys = None
                ci_lower = None
                ci_upper = None
    # else: coef_dys already None or set above

    marginal_dys = {"coef": coef_dys, "se": se_sum, "pval": p_dys, "ci": (ci_lower, ci_upper)}

    # Convert log-scale coefficients to percent change for interpretability
    def pct_change_from_log(coef):
        if coef is None:
            return None
        try:
            return (math.exp(coef) - 1.0) * 100.0
        except Exception:
            return None

    reader_pct = pct_change_from_log(reader_coef)
    dys_pct = pct_change_from_log(coef_dys)

    # Build result object
    result_object = {
        "param_names": {"reader_key": key_reader, "interaction_key": key_inter},
        "reader_view": {
            "coef_log": reader_coef,
            "se": reader_se,
            "pval": reader_p,
            "ci": reader_ci,
            "pct_change": reader_pct
        },
        "reader_view_by_dyslexia": {
            "coef_log": coef_dys,
            "se": se_sum,
            "pval": p_dys,
            "ci": (ci_lower, ci_upper),
            "pct_change": dys_pct
        },
        "notes": "Marginal effect for dyslexia = reader_view coef + interaction coef. SE/p/CI for that sum computed from cov_params() when available; otherwise reported as None."
    }

    # Compose a concise interpretation
    def sig_label(p):
        if p is None:
            return "p-value unavailable"
        try:
            p = float(p)
            if p < 0.001:
                return "p < 0.001"
            else:
                return f"p = {p:.3f}"
        except Exception:
            return "p-value unavailable"

    lines = []
    if reader_coef is not None:
        reader_se_str = f"{reader_se:.4f}" if reader_se is not None else "NA"
        lines.append(
            f"Main effect (reader_view) on log_speed: coef = {reader_coef:.4f}, se = {reader_se_str}, {sig_label(reader_p)}."
        )
        if reader_pct is not None:
            lines.append(f"  -> Interpreted as {reader_pct:.2f}% change in reading speed when Reader View is ON for readers without dyslexia (dyslexia=0).")
    else:
        lines.append("Main effect for reader_view not found in model parameters.")

    if coef_dys is not None:
        se_str = f"{se_sum:.4f}" if se_sum is not None else "NA"
        lines.append(
            f"Marginal effect for readers with dyslexia (reader_view + interaction): coef = {coef_dys:.4f}, se = {se_str}, {sig_label(p_dys)}."
        )
        if dys_pct is not None:
            lines.append(f"  -> Interpreted as {dys_pct:.2f}% change in reading speed when Reader View is ON for readers with dyslexia (dyslexia=1).")
    else:
        lines.append("Interaction term not present or insufficient information to compute marginal effect for dyslexia.")

    # Summarize whether Reader View "improves" reading speed:
    # improvement means positive percent change in raw speed; log coef >0 => increase
    conclusion = ""
    try:
        alpha = 0.05

        def is_significant(p):
            return (p is not None) and (p < alpha)

        main_sig = is_significant(reader_p)
        dys_sig = is_significant(p_dys)

        if reader_coef is not None:
            if reader_coef > 0:
                main_dir = "increase"
            elif reader_coef < 0:
                main_dir = "decrease"
            else:
                main_dir = "no change"
        else:
            main_dir = "unknown"

        if coef_dys is not None:
            if coef_dys > 0:
                dys_dir = "increase"
            elif coef_dys < 0:
                dys_dir = "decrease"
            else:
                dys_dir = "no change"
        else:
            dys_dir = "unknown"

        conclusion += "Summary conclusion: "
        # Overall (no-dyslexia)
        if reader_coef is not None:
            conclusion += f"For readers without dyslexia, Reader View is associated with a {main_dir} in reading speed"
            if reader_pct is not None:
                conclusion += f" (~{reader_pct:.2f}% change)"
            if main_sig:
                conclusion += " that is statistically significant."
            else:
                conclusion += " that is not statistically significant."
        else:
            conclusion += "Insufficient info about the main effect."

        # Dyslexia subgroup
        if coef_dys is not None:
            conclusion += " For readers with dyslexia, Reader View is associated with a "
            conclusion += f"{dys_dir} in reading speed"
            if dys_pct is not None:
                conclusion += f" (~{dys_pct:.2f}% change)"
            if dys_sig:
                conclusion += " that is statistically significant."
            else:
                conclusion += " that is not statistically significant."
        else:
            conclusion += " Could not determine the subgroup (dyslexia) marginal effect due to missing interaction or covariance info."

    except Exception:
        conclusion = "Could not form a full conclusion due to missing statistics."

    description = "\n".join(lines) + "\n\n" + conclusion

    return {"object": result_object, "description": description}