def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of Reader View for individuals with dyslexia
    from a fitted statsmodels-like results object.

    Returns:
      {
        "object": { ... },
        "description": "Text explanation..."
      }

    This function is robust to model_output.params and covariance matrices being
    numpy arrays, dicts, or pandas objects.
    """
    import numpy as np
    from math import sqrt
    from scipy import stats

    res = model_output

    # Obtain params object
    params_obj = getattr(res, "params", None)
    if params_obj is None:
        raise ValueError("model_output has no attribute 'params'")

    # Derive parameter names in a few fallback ways
    param_index = None
    if hasattr(params_obj, "index"):
        try:
            param_index = list(params_obj.index)
        except Exception:
            param_index = None

    if param_index is None:
        # statsmodels results often have model.exog_names
        if hasattr(res, "model") and hasattr(res.model, "exog_names"):
            try:
                param_index = list(res.model.exog_names)
            except Exception:
                param_index = None

    if param_index is None and hasattr(res, "param_names"):
        try:
            param_index = list(res.param_names)
        except Exception:
            param_index = None

    if param_index is None and isinstance(params_obj, dict):
        param_index = list(params_obj.keys())

    if param_index is None and isinstance(params_obj, (list, tuple, np.ndarray)):
        param_index = [f"param_{i}" for i in range(len(params_obj))]

    if param_index is None:
        raise ValueError("Could not read parameter names from model_output (no index, exog_names, param_names, or dict).")

    # Map names to indices for array-like access
    name_to_idx = {n: i for i, n in enumerate(param_index)}

    # Helpers to find parameter names robustly
    def find_main_reader_view_name(names):
        # Prefer an exact match; otherwise pick a name that contains 'reader_view' but not 'dyslexia'
        if "reader_view" in names:
            return "reader_view"
        for n in names:
            nl = n.lower()
            if "reader_view" in nl and "dyslexia" not in nl and ":" not in nl:
                return n
        for n in names:
            if "reader_view" in n.lower():
                return n
        return None

    def find_interaction_name(names):
        # Prefer names containing both 'reader_view' and 'dyslexia'
        for n in names:
            nl = n.lower()
            if "reader_view" in nl and "dyslexia" in nl:
                return n
        # Also consider patterns with ':' combining the two variable tokens
        for n in names:
            nl = n.lower()
            if ":" in n and "reader_view" in nl and "dyslexia" in nl:
                return n
        # No interaction found
        return None

    main_name = find_main_reader_view_name(param_index)
    inter_name = find_interaction_name(param_index)

    # Obtain covariance matrix (could be DataFrame or ndarray)
    cov = None
    try:
        cov = res.cov_params()
    except Exception:
        if hasattr(res, "normalized_cov_params"):
            cov = res.normalized_cov_params
        else:
            cov = None

    # Accessor helpers
    def get_param_value(name):
        if name is None:
            return None
        # If params_obj supports keyed access (like Series or dict)
        try:
            if hasattr(params_obj, "__getitem__") and not isinstance(params_obj, np.ndarray):
                return float(params_obj[name])
        except Exception:
            pass
        # fallback to positional
        idx = name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"Parameter name '{name}' not found among {param_index}")
        # handle numpy array or list-like
        try:
            return float(params_obj[idx])
        except Exception as e:
            raise ValueError(f"Could not extract parameter '{name}': {e}")

    def get_cov_entry(name1, name2):
        if name1 is None or name2 is None or cov is None:
            return None
        # If cov is a pandas-like DataFrame with .loc
        if hasattr(cov, "loc"):
            try:
                return float(cov.loc[name1, name2])
            except Exception:
                # try reversed or fallback
                try:
                    return float(cov.loc[name2, name1])
                except Exception:
                    return None
        # If cov is ndarray or matrix
        if isinstance(cov, (list, tuple)):
            cov_arr = np.asarray(cov)
        elif isinstance(cov, np.ndarray):
            cov_arr = cov
        else:
            # unknown type
            try:
                cov_arr = np.asarray(cov)
            except Exception:
                return None
        idx1 = name_to_idx.get(name1)
        idx2 = name_to_idx.get(name2)
        if idx1 is None or idx2 is None:
            return None
        try:
            return float(cov_arr[idx1, idx2])
        except Exception:
            return None

    def safe_get_coef(name):
        if name is None:
            return None
        try:
            return float(get_param_value(name))
        except Exception:
            return None

    def safe_get_var(name):
        if name is None:
            return None
        return get_cov_entry(name, name)

    def safe_get_se(name):
        v = safe_get_var(name)
        if v is None:
            return None
        try:
            return float(np.sqrt(v))
        except Exception:
            return None

    def safe_get_p(name):
        if name is None:
            return None
        # If res.pvalues exists, try to extract
        pvals_obj = getattr(res, "pvalues", None)
        if pvals_obj is not None:
            try:
                if hasattr(pvals_obj, "__getitem__") and not isinstance(pvals_obj, np.ndarray):
                    if name in getattr(pvals_obj, "index", list(pvals_obj.keys()) if isinstance(pvals_obj, dict) else []):
                        return float(pvals_obj[name])
                    # if name not in index, still attempt keyed access
                    try:
                        return float(pvals_obj[name])
                    except Exception:
                        pass
                # if ndarray, use positional
                if isinstance(pvals_obj, np.ndarray):
                    idx = name_to_idx.get(name)
                    if idx is not None and idx < len(pvals_obj):
                        return float(pvals_obj[idx])
            except Exception:
                pass
        # otherwise compute from coef and se
        try:
            coef = get_param_value(name)
        except Exception:
            return None
        se = safe_get_se(name)
        if se is None or se == 0:
            return None
        tval = coef / se
        df = getattr(res, "df_resid", None)
        if df is None or (isinstance(df, float) and np.isnan(df)) or (isinstance(df, (int, float)) and df <= 0):
            p = 2 * (1 - stats.norm.cdf(abs(tval)))
        else:
            p = 2 * (1 - stats.t.cdf(abs(tval), df))
        return float(p)

    coef_main = safe_get_coef(main_name)
    se_main = safe_get_se(main_name)
    p_main = safe_get_p(main_name)

    coef_inter = safe_get_coef(inter_name)
    se_inter = safe_get_se(inter_name)
    p_inter = safe_get_p(inter_name)

    # Compute combined effect for dyslexic individuals: beta_reader_view + beta_interaction
    if coef_main is None:
        raise ValueError("Could not find a parameter for the main effect of reader_view in the model output.")
    if inter_name is None or coef_inter is None:
        coef_combined = coef_main
        var_combined = safe_get_var(main_name)
    else:
        coef_combined = coef_main + coef_inter
        var_main = safe_get_var(main_name) or 0.0
        var_inter = safe_get_var(inter_name) or 0.0
        cov12 = get_cov_entry(main_name, inter_name)
        if cov12 is None:
            # try reversed or zero
            cov12 = 0.0
        var_combined = var_main + var_inter + 2.0 * cov12

    se_combined = None
    if var_combined is not None:
        try:
            se_combined = float(np.sqrt(var_combined))
        except Exception:
            se_combined = None

    # Compute p-value and 95% CI for combined effect
    if se_combined is None or se_combined == 0:
        p_combined = None
        ci_lower = None
        ci_upper = None
    else:
        tval = coef_combined / se_combined
        df = getattr(res, "df_resid", None)
        if df is None or (isinstance(df, float) and np.isnan(df)) or (isinstance(df, (int, float)) and df <= 0):
            p_combined = float(2 * (1 - stats.norm.cdf(abs(tval))))
            crit = stats.norm.ppf(0.975)
        else:
            p_combined = float(2 * (1 - stats.t.cdf(abs(tval), df)))
            crit = float(stats.t.ppf(0.975, df))
        ci_lower = float(coef_combined - crit * se_combined)
        ci_upper = float(coef_combined + crit * se_combined)

    # Convert effect on log_speed to approximate percent change in speed:
    try:
        pct_change = (np.exp(coef_combined) - 1.0) * 100.0
    except Exception:
        pct_change = None

    # Decision: is effect statistically significant at alpha=0.05?
    sig = None
    if p_combined is not None:
        sig = (p_combined < 0.05)

    # Prepare return object
    out_obj = {
        "coef_reader_view": float(coef_main) if coef_main is not None else None,
        "se_reader_view": float(se_main) if se_main is not None else None,
        "p_reader_view": float(p_main) if p_main is not None else None,
        "coef_interaction_reader_view_dyslexia": float(coef_inter) if coef_inter is not None else None,
        "se_interaction_reader_view_dyslexia": float(se_inter) if se_inter is not None else None,
        "p_interaction_reader_view_dyslexia": float(p_inter) if p_inter is not None else None,
        "coef_reader_view_dyslexic": float(coef_combined),
        "se_reader_view_dyslexic": float(se_combined) if se_combined is not None else None,
        "p_reader_view_dyslexic": float(p_combined) if p_combined is not None else None,
        "ci_95_reader_view_dyslexic": [ci_lower, ci_upper],
        "pct_change_speed_dyslexic": float(pct_change) if pct_change is not None else None,
        "significant_at_0.05_for_dyslexic": bool(sig) if sig is not None else None,
        "used_param_names": {"main": main_name, "interaction": inter_name},
    }

    # Description text
    if out_obj["p_reader_view_dyslexic"] is None or out_obj["ci_95_reader_view_dyslexic"][0] is None:
        desc = (
            "Could not compute p-value/CI for the combined effect of Reader View for dyslexic readers "
            "(missing standard errors, covariance entries, or degrees of freedom). Returned coefficient and available stats."
        )
    else:
        direction = "increase" if out_obj["coef_reader_view_dyslexic"] > 0 else "decrease"
        desc = (
            f"The estimated effect of turning Reader View ON for readers with dyslexia (dyslexia_bin=1) "
            f"is {out_obj['coef_reader_view_dyslexic']:.4f} on log(speed+1), "
            f"which corresponds to an approximate {out_obj['pct_change_speed_dyslexic']:.2f}% {direction} "
            f"in reading speed (exp(coef)-1). The 95% CI is [{out_obj['ci_95_reader_view_dyslexic'][0]:.4f}, "
            f"{out_obj['ci_95_reader_view_dyslexic'][1]:.4f}]. The two-sided p-value is "
            f"{out_obj['p_reader_view_dyslexic']:.4f}. "
        )
        if out_obj["significant_at_0.05_for_dyslexic"]:
            desc += "This effect is statistically significant at alpha=0.05."
        else:
            desc += "This effect is not statistically significant at alpha=0.05."
        desc += " The returned fields also include the separate main effect and interaction coefficients (and their SE/p-values if available)."

    return {"object": out_obj, "description": desc}