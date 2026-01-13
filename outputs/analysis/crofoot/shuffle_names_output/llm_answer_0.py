def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, odds ratios, and
    marginal effect of RelGroupSize when LocFocal = 0 vs 1 from a fitted
    statsmodels Logit (or similar) results object or from a dictionary-like
    representation.

    Returns: dict with keys:
      - "object": dict containing numeric results (coefficients, p-values,
                  95% CIs, odds ratios, marginal effects and p-values)
      - "description": short plain-language interpretation of the key results

    This function is robust to a few different shapes for model_output:
      - a statsmodels results object with attributes/methods: params, pvalues,
        bse, conf_int(), cov_params()
      - a dict-like object with keys: "params", "pvalues", "bse", "conf_int",
        "cov" (or "cov_params")
      - simple mappings (e.g., params as dict, arrays, or pandas Series)
    If only partial information is available the function will include what it
    can and will not raise unless absolutely nothing meaningful can be inferred.
    """
    import numpy as np
    import pandas as pd
    import math

    # Helpers to flexibly get things from object or mapping
    def _has_attr(o, name):
        return hasattr(o, name)

    def _is_mapping(o):
        return hasattr(o, "get") and callable(getattr(o, "get"))

    def _maybe_call(obj):
        try:
            return obj() if callable(obj) else obj
        except Exception:
            # If calling fails, return the object itself
            return obj

    def _get_raw(name, alt_names=()):
        # Try attribute
        if _has_attr(model_output, name):
            return _maybe_call(getattr(model_output, name))
        # Try mapping-style
        if _is_mapping(model_output) and name in model_output:
            val = model_output[name]
            return _maybe_call(val)
        # Try alternative keys in mapping
        if _is_mapping(model_output):
            for an in alt_names:
                if an in model_output:
                    return _maybe_call(model_output[an])
        # Not found
        return None

    # Retrieve raw pieces
    raw_params = _get_raw("params")
    raw_pvalues = _get_raw("pvalues")
    raw_bse = _get_raw("bse", alt_names=("std_err", "stderr"))
    raw_conf_int = _get_raw("conf_int", alt_names=("confint", "conf_int_"))
    # cov could be a method cov_params or mapping key 'cov'/'cov_params'
    raw_cov = None
    if _has_attr(model_output, "cov_params"):
        raw_cov = _maybe_call(getattr(model_output, "cov_params"))
    if raw_cov is None:
        raw_cov = _get_raw("cov", alt_names=("cov_params", "covariance"))

    # Build pandas Series/DataFrames where possible
    def _to_series(x, name_hint=None):
        if x is None:
            return pd.Series(dtype=float)
        if isinstance(x, pd.Series):
            return x.astype(float)
        if isinstance(x, pd.DataFrame):
            # If a DataFrame provided where a Series expected, try to take a column
            if x.shape[1] == 1:
                s = x.iloc[:, 0]
                return s.astype(float)
            # else cannot convert directly
            raise ValueError(f"Cannot convert DataFrame to Series for {name_hint}")
        if isinstance(x, dict):
            try:
                return pd.Series(x).astype(float)
            except Exception:
                return pd.Series({k: float(v) for k, v in x.items()})
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            # try to get index from conf_int or other hints later
            # create numeric index strings for now
            idx = [str(i) for i in range(len(arr))]
            return pd.Series(arr, index=idx)
        # fallback: try to construct series from iterables
        try:
            return pd.Series(x).astype(float)
        except Exception:
            return pd.Series(dtype=float)

    def _to_confint(x, index_hint=None):
        if x is None:
            return pd.DataFrame(columns=[0, 1], dtype=float)
        if isinstance(x, pd.DataFrame):
            # assume columns [0,1] or ['lower','upper']
            if x.shape[1] >= 2:
                df = x.iloc[:, :2].copy()
                df.columns = [0, 1]
                if index_hint is not None and len(df.index) != len(index_hint):
                    # try to set if lengths match
                    if len(index_hint) == len(df.index):
                        df.index = index_hint
                return df.astype(float)
            else:
                raise ValueError("conf_int DataFrame has fewer than 2 columns")
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                df = pd.DataFrame(arr[:, :2], index=index_hint)
                df.columns = [0, 1]
                return df
        if isinstance(x, dict):
            # dict of name -> (low, high)
            try:
                df = pd.DataFrame.from_dict(x, orient="index")
                if df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = [0, 1]
                    return df.astype(float)
            except Exception:
                pass
        # fallback empty
        return pd.DataFrame(columns=[0, 1], dtype=float)

    def _to_cov(x, index_hint=None):
        if x is None:
            return pd.DataFrame(dtype=float)
        if isinstance(x, pd.DataFrame):
            return x.astype(float)
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                if index_hint is not None and len(index_hint) == arr.shape[0]:
                    return pd.DataFrame(arr, index=index_hint, columns=index_hint)
                else:
                    idx = [str(i) for i in range(arr.shape[0])]
                    return pd.DataFrame(arr, index=idx, columns=idx)
        if isinstance(x, dict):
            # dict of dicts or mapping to rows
            try:
                df = pd.DataFrame.from_dict(x)
                return df.astype(float)
            except Exception:
                pass
        return pd.DataFrame(dtype=float)

    # Build param series first to get an index hint
    params = _to_series(raw_params, name_hint="params")
    # Try to use conf_int index if params had generic numeric index
    conf_int = _to_confint(raw_conf_int, index_hint=(list(params.index) if not params.empty else None))
    # If conf_int has index and params is empty, derive params index from conf_int
    if params.empty and not conf_int.empty:
        params = pd.Series(index=list(conf_int.index), dtype=float)

    pvalues = _to_series(raw_pvalues, name_hint="pvalues")
    bse = _to_series(raw_bse, name_hint="bse")
    cov = _to_cov(raw_cov, index_hint=(list(params.index) if not params.empty else None))

    # If we still have empty params but pvalues or bse have indices, try to take them
    if params.empty:
        if not pvalues.empty:
            params = pd.Series(index=list(pvalues.index), dtype=float)
        elif not bse.empty:
            params = pd.Series(index=list(bse.index), dtype=float)
        elif not cov.empty:
            params = pd.Series(index=list(cov.index), dtype=float)
        else:
            # Nothing to work with: return a safe, informative response (do not raise)
            description = "No parameter information could be extracted from model_output."
            return {"object": {"coefficients": {}, "odds_ratios": {}, "marginal_effects": {}},
                    "description": description}

    # If params had values but pvalues/bse missing values, align them to params index with NaN
    pvalues = pvalues.reindex(params.index).astype(float)
    bse = bse.reindex(params.index).astype(float)
    # Align conf_int index and cov index
    if not conf_int.empty:
        conf_int = conf_int.reindex(params.index)
    else:
        # create NaN conf_int
        conf_int = pd.DataFrame(index=params.index, columns=[0, 1], dtype=float)

    if not cov.empty:
        cov = cov.reindex(index=params.index, columns=params.index)
    else:
        # create NaN covariance matrix (diagonal from bse^2 when possible)
        cov = pd.DataFrame(np.nan, index=params.index, columns=params.index)
        if not bse.isnull().all():
            for i in params.index:
                try:
                    cov.loc[i, i] = float(bse.get(i, np.nan)) ** 2
                except Exception:
                    cov.loc[i, i] = np.nan

    param_names = list(params.index)

    # Identify parameter names robustly (handles slightly different naming)
    rel_candidates = [n for n in param_names if "RelGroupSize" in str(n) and "LocFocal" not in str(n)]
    loc_candidates = [n for n in param_names if "LocFocal" in str(n) and "RelGroupSize" not in str(n)]
    inter_candidates = [n for n in param_names if "RelGroupSize" in str(n) and "LocFocal" in str(n)]

    rel_name = rel_candidates[0] if rel_candidates else None
    loc_name = loc_candidates[0] if loc_candidates else None
    inter_name = inter_candidates[0] if inter_candidates else None

    # Prepare output structure
    out = {"coefficients": {}, "odds_ratios": {}, "marginal_effects": {}}

    # Utility to extract coef/pval/ci/odds ratio for a parameter name
    def _extract_param(name):
        if name is None or name not in params.index:
            return None
        try:
            coef = float(params[name]) if not pd.isna(params[name]) else np.nan
        except Exception:
            coef = np.nan
        try:
            se = float(bse[name]) if name in bse.index and not pd.isna(bse[name]) else np.nan
        except Exception:
            se = np.nan
        try:
            p = float(pvalues[name]) if name in pvalues.index and not pd.isna(pvalues[name]) else np.nan
        except Exception:
            p = np.nan
        try:
            ci_row = conf_int.loc[name]
            ci_low, ci_high = float(ci_row.iloc[0]), float(ci_row.iloc[1])
        except Exception:
            # fallback to coef +/- 1.96*se if possible
            if not np.isnan(coef) and not np.isnan(se):
                ci_low, ci_high = coef - 1.96 * se, coef + 1.96 * se
            else:
                ci_low, ci_high = np.nan, np.nan
        try:
            or_ = float(np.exp(coef)) if not np.isnan(coef) else np.nan
            or_ci = (float(np.exp(ci_low)) if not np.isnan(ci_low) else np.nan,
                     float(np.exp(ci_high)) if not np.isnan(ci_high) else np.nan)
        except Exception:
            or_, or_ci = np.nan, (np.nan, np.nan)
        return {"name": name, "coef": coef, "se": se, "pvalue": p,
                "ci95": (ci_low, ci_high), "odds_ratio": or_, "odds_ratio_ci95": or_ci}

    # Extract for main terms and interaction (if present)
    rel_res = _extract_param(rel_name)
    loc_res = _extract_param(loc_name)
    inter_res = _extract_param(inter_name)

    if rel_res:
        out["coefficients"]["RelGroupSize"] = rel_res
    else:
        out["coefficients"]["RelGroupSize"] = "parameter not found in model"

    if loc_res:
        out["coefficients"]["LocFocal"] = loc_res
    else:
        out["coefficients"]["LocFocal"] = "parameter not found in model"

    if inter_res:
        out["coefficients"]["RelGroupSize:LocFocal_interaction"] = inter_res
    else:
        out["coefficients"]["RelGroupSize:LocFocal_interaction"] = "interaction parameter not found in model"

    # Compute marginal effect of RelGroupSize when LocFocal = 0 and = 1.
    # Coef when LocFocal=0 is beta_rel. When LocFocal=1 it's beta_rel + beta_inter.
    try:
        import scipy.stats as st  # optional but preferred
        norm_cdf = lambda x: st.norm.cdf(x)
    except Exception:
        # fallback to math.erf-based approx of normal cdf if scipy not available
        norm_cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _marginal_effect(rel_name, inter_name):
        if rel_name is None or rel_name not in params.index:
            return None
        try:
            beta_rel = float(params[rel_name]) if not pd.isna(params[rel_name]) else np.nan
        except Exception:
            beta_rel = np.nan
        try:
            var_rel = float(cov.loc[rel_name, rel_name]) if rel_name in cov.index and rel_name in cov.columns and not pd.isna(cov.loc[rel_name, rel_name]) else np.nan
        except Exception:
            var_rel = np.nan

        # LocFocal = 0
        se_rel0 = float(np.sqrt(var_rel)) if not np.isnan(var_rel) and var_rel >= 0 else np.nan
        z0 = beta_rel / se_rel0 if se_rel0 and not np.isnan(se_rel0) else np.nan
        p_rel0 = 2 * (1 - norm_cdf(abs(z0))) if not np.isnan(z0) else np.nan
        or_rel0 = float(np.exp(beta_rel)) if not np.isnan(beta_rel) else np.nan
        ci_rel0 = (float(np.exp(beta_rel - 1.96 * se_rel0)), float(np.exp(beta_rel + 1.96 * se_rel0))) if not np.isnan(se_rel0) and not np.isnan(beta_rel) else (None, None)

        # LocFocal = 1
        if inter_name is not None and inter_name in params.index:
            try:
                beta_int = float(params[inter_name]) if not pd.isna(params[inter_name]) else np.nan
            except Exception:
                beta_int = np.nan
            try:
                var_int = float(cov.loc[inter_name, inter_name]) if inter_name in cov.index and inter_name in cov.columns and not pd.isna(cov.loc[inter_name, inter_name]) else np.nan
            except Exception:
                var_int = np.nan
            try:
                cov_rel_int = float(cov.loc[rel_name, inter_name]) if rel_name in cov.index and inter_name in cov.columns and not pd.isna(cov.loc[rel_name, inter_name]) else np.nan
            except Exception:
                cov_rel_int = np.nan

            if np.isnan(beta_rel) and not np.isnan(beta_int):
                beta_rel1 = beta_int
            elif not np.isnan(beta_rel) and not np.isnan(beta_int):
                beta_rel1 = beta_rel + beta_int
            else:
                beta_rel1 = np.nan

            if not np.isnan(var_rel) and not np.isnan(var_int) and not np.isnan(cov_rel_int):
                var_rel1 = var_rel + var_int + 2 * cov_rel_int
            else:
                var_rel1 = np.nan

            se_rel1 = float(np.sqrt(var_rel1)) if not np.isnan(var_rel1) and var_rel1 >= 0 else np.nan
            z1 = beta_rel1 / se_rel1 if se_rel1 and not np.isnan(se_rel1) else np.nan
            p_rel1 = 2 * (1 - norm_cdf(abs(z1))) if not np.isnan(z1) else np.nan
            or_rel1 = float(np.exp(beta_rel1)) if not np.isnan(beta_rel1) else np.nan
            ci_rel1 = (float(np.exp(beta_rel1 - 1.96 * se_rel1)), float(np.exp(beta_rel1 + 1.96 * se_rel1))) if not np.isnan(se_rel1) and not np.isnan(beta_rel1) else (None, None)
        else:
            beta_rel1 = None
            se_rel1 = None
            p_rel1 = None
            or_rel1 = None
            ci_rel1 = (None, None)

        return {
            "Rel_at_LocFocal_0": {
                "coef": beta_rel,
                "se": se_rel0,
                "pvalue": p_rel0,
                "odds_ratio_per_unit_rel": or_rel0,
                "odds_ratio_CI95": ci_rel0
            },
            "Rel_at_LocFocal_1": {
                "coef": beta_rel1,
                "se": se_rel1,
                "pvalue": p_rel1,
                "odds_ratio_per_unit_rel": or_rel1,
                "odds_ratio_CI95": ci_rel1
            }
        }

    marg = _marginal_effect(rel_name, inter_name)
    out["marginal_effects"] = marg

    # Prepare short human-readable description
    def _sig_label(p):
        try:
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return "unknown"
            if p < 0.001:
                return "p < 0.001"
            if p < 0.01:
                return "p < 0.01"
            if p < 0.05:
                return "p < 0.05"
            return f"p = {p:.3f}"
        except Exception:
            return "unknown"

    desc_parts = []
    if isinstance(out["coefficients"]["RelGroupSize"], dict):
        rr = out["coefficients"]["RelGroupSize"]
        desc_parts.append(
            f"RelGroupSize: coef={rr['coef']:.3f} (OR={rr['odds_ratio']:.3f}), {_sig_label(rr['pvalue'])}"
        )
    else:
        desc_parts.append("RelGroupSize: parameter not found")

    if isinstance(out["coefficients"]["LocFocal"], dict):
        lf = out["coefficients"]["LocFocal"]
        desc_parts.append(
            f"LocFocal (home advantage): coef={lf['coef']:.3f} (OR={lf['odds_ratio']:.3f}), {_sig_label(lf['pvalue'])}"
        )
    else:
        desc_parts.append("LocFocal: parameter not found")

    if isinstance(out["coefficients"]["RelGroupSize:LocFocal_interaction"], dict):
        it = out["coefficients"]["RelGroupSize:LocFocal_interaction"]
        desc_parts.append(
            f"Interaction RelGroupSize:LocFocal: coef={it['coef']:.3f} (OR={it['odds_ratio']:.3f}), {_sig_label(it['pvalue'])}"
        )
    else:
        desc_parts.append("Interaction: not found")

    # Include marginal summaries
    if marg is not None:
        r0 = marg.get("Rel_at_LocFocal_0")
        if r0 is not None:
            desc_parts.append(
                f"Marginal effect of RelGroupSize when away (LocFocal=0): coef={r0['coef']:.3f}, OR={r0['odds_ratio_per_unit_rel']:.3f}, {_sig_label(r0['pvalue'])}"
            )
        r1 = marg.get("Rel_at_LocFocal_1")
        if r1 is not None and r1["coef"] is not None:
            # r1["coef"] might be None if interaction missing
            desc_parts.append(
                f"Marginal effect of RelGroupSize when home (LocFocal=1): coef={r1['coef']:.3f}, OR={r1['odds_ratio_per_unit_rel']:.3f}, {_sig_label(r1['pvalue'])}"
            )

    description = " | ".join(desc_parts)

    return {"object": out, "description": description}