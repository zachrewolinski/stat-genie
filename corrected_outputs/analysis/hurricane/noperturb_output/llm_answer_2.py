def extract_final_answer(model_output):
    """
    Extracts key statistics for the name-femininity predictors from a statsmodels
    OLS results-like object and provides a short interpretation relative to the
    hypothesis:
      "Hurricanes with more feminine names are perceived as less threatening
       and hence lead to fewer precautionary measures by the general public."

    The function is defensive: it handles cases where attributes like params,
    bse, tvalues, pvalues, and conf_int are numpy arrays or pandas Series/DataFrames,
    and it attempts to discover parameter names from common places (e.g. model.exog_names).
    If the requested parameter name is not present, it returns None for that parameter.

    Returns a dict:
      - "object": dict with extracted numeric results for 'masfem_c' and 'gender_female' (or None)
      - "description": string explaining/examining the estimates in plain language
    """
    import numpy as np

    res = model_output

    # Helper: obtain param array-like and names list in a robust way
    def _as_array(obj):
        try:
            return np.asarray(obj)
        except Exception:
            return None

    # Get parameter values array
    params_arr = None
    param_names = None
    if hasattr(res, "params"):
        params = getattr(res, "params")
        # If it's a pandas Series, extract values and index
        try:
            # pandas Series has .values and .index
            params_arr = _as_array(params.values if hasattr(params, "values") else params)
            if hasattr(params, "index"):
                param_names = list(params.index)
        except Exception:
            params_arr = _as_array(params)
    else:
        # fallback if model_output itself is an array or dict
        if isinstance(res, (list, tuple, np.ndarray)):
            params_arr = _as_array(res)
        elif isinstance(res, dict) and "params" in res:
            params_arr = _as_array(res["params"])
            if "param_names" in res:
                param_names = list(res["param_names"])

    # Try to discover names from model metadata if not found yet
    if param_names is None and hasattr(res, "model"):
        mod = getattr(res, "model")
        if hasattr(mod, "exog_names"):
            try:
                param_names = list(mod.exog_names)
            except Exception:
                param_names = None
        elif hasattr(mod, "data") and hasattr(mod.data, "param_names"):
            try:
                param_names = list(mod.data.param_names)
            except Exception:
                param_names = None

    # Additional fallback: res may have attribute param_names
    if param_names is None and hasattr(res, "param_names"):
        try:
            param_names = list(getattr(res, "param_names"))
        except Exception:
            param_names = None

    # Helper to get an index for a given parameter name
    def _find_index(name):
        if param_names is None:
            return None
        try:
            return param_names.index(name)
        except ValueError:
            return None

    # Helper to extract confidence intervals robustly
    def _get_conf_int(name_idx_or_name):
        # Prefer using res.conf_int() if available
        if hasattr(res, "conf_int"):
            try:
                ci = res.conf_int()
                # If DataFrame-like with index:
                if hasattr(ci, "loc"):
                    if isinstance(name_idx_or_name, str):
                        lower, upper = ci.loc[name_idx_or_name].astype(float)
                        return float(lower), float(upper)
                    else:
                        lower, upper = ci.iloc[name_idx_or_name].astype(float)
                        return float(lower), float(upper)
                else:
                    # ndarray-like, shape (k,2)
                    ci_arr = _as_array(ci)
                    if ci_arr is not None and ci_arr.ndim == 2:
                        idx = name_idx_or_name if isinstance(name_idx_or_name, int) else None
                        if idx is not None and 0 <= idx < ci_arr.shape[0]:
                            return float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            except Exception:
                pass
        # If no conf_int method, attempt to look for attribute conf_int_ or fallback to NaNs
        return (np.nan, np.nan)

    # Helper to safely extract statistic from res given a name
    def get_stats(name):
        # Determine index of parameter
        idx = _find_index(name)
        if idx is None:
            # Parameter name not available in discovered names
            return None

        # Extract coefficient
        try:
            if hasattr(res, "params"):
                p = getattr(res, "params")
                coef = float(p[name]) if hasattr(p, "get") or hasattr(p, "loc") or hasattr(p, "index") else float(np.asarray(p)[idx])
            else:
                coef = float(params_arr[idx])
        except Exception:
            # Try direct array access if previous failed
            try:
                coef = float(params_arr[idx])
            except Exception:
                return None

        # Standard error
        se = None
        if hasattr(res, "bse"):
            try:
                b = getattr(res, "bse")
                se = float(b[name]) if hasattr(b, "get") or hasattr(b, "loc") or hasattr(b, "index") else float(np.asarray(b)[idx])
            except Exception:
                try:
                    se = float(np.asarray(res.bse)[idx])
                except Exception:
                    se = np.nan
        else:
            se = np.nan

        # t-value
        tval = None
        if hasattr(res, "tvalues"):
            try:
                t = getattr(res, "tvalues")
                tval = float(t[name]) if hasattr(t, "get") or hasattr(t, "loc") or hasattr(t, "index") else float(np.asarray(t)[idx])
            except Exception:
                try:
                    tval = float(np.asarray(res.tvalues)[idx])
                except Exception:
                    tval = np.nan
        else:
            # fallback: compute t from coef/se if possible
            try:
                tval = float(coef / se) if (se is not None and not np.isnan(se) and se != 0) else np.nan
            except Exception:
                tval = np.nan

        # p-value
        pval = None
        if hasattr(res, "pvalues"):
            try:
                pv = getattr(res, "pvalues")
                pval = float(pv[name]) if hasattr(pv, "get") or hasattr(pv, "loc") or hasattr(pv, "index") else float(np.asarray(pv)[idx])
            except Exception:
                try:
                    pval = float(np.asarray(res.pvalues)[idx])
                except Exception:
                    pval = np.nan
        else:
            pval = np.nan

        # Confidence interval
        ci_lower, ci_upper = _get_conf_int(name if isinstance(name, str) else idx)

        # Percent change in raw deaths for one-unit increase in predictor:
        # Using exact transformation: (exp(coef) - 1) * 100
        try:
            pct_change = (np.exp(coef) - 1.0) * 100.0
        except Exception:
            pct_change = np.nan
        try:
            pct_ci_lower = (np.exp(ci_lower) - 1.0) * 100.0 if (ci_lower is not None and not np.isnan(ci_lower)) else np.nan
            pct_ci_upper = (np.exp(ci_upper) - 1.0) * 100.0 if (ci_upper is not None and not np.isnan(ci_upper)) else np.nan
        except Exception:
            pct_ci_lower, pct_ci_upper = np.nan, np.nan

        return {
            "coef": float(coef) if not np.isnan(coef) else None,
            "se": float(se) if not np.isnan(se) else None,
            "t": float(tval) if not np.isnan(tval) else None,
            "p_value": float(pval) if not np.isnan(pval) else None,
            "ci_95": [float(ci_lower) if not np.isnan(ci_lower) else None,
                      float(ci_upper) if not np.isnan(ci_upper) else None],
            "percent_change_in_deaths": float(pct_change) if not np.isnan(pct_change) else None,
            "percent_change_ci_95": [
                float(pct_ci_lower) if not np.isnan(pct_ci_lower) else None,
                float(pct_ci_upper) if not np.isnan(pct_ci_upper) else None,
            ],
            "significant_at_0.05": bool((not np.isnan(pval)) and (pval < 0.05)) if (pval is not None and not np.isnan(pval)) else None
        }

    masfem_stats = get_stats("masfem_c")
    gender_stats = get_stats("gender_female")

    # Build interpretation text
    lines = []
    lines.append("Extracted estimates and interpretation for name-related predictors:")
    if masfem_stats is not None:
        coef = masfem_stats["coef"]
        se = masfem_stats["se"]
        t = masfem_stats["t"]
        p = masfem_stats["p_value"]
        ci0, ci1 = masfem_stats["ci_95"]
        pct = masfem_stats["percent_change_in_deaths"]
        pcl, pcu = masfem_stats["percent_change_ci_95"]

        lines.append(
            f"masfem_c: coef = {coef:.4f}, SE = {se:.4f}, t = {t:.2f}, p = {p:.3f}, 95% CI = [{ci0:.4f}, {ci1:.4f}]."
            if (coef is not None and se is not None and t is not None and p is not None and ci0 is not None and ci1 is not None)
            else "masfem_c: Estimates found but incomplete numeric details."
        )
        if pct is not None and pcl is not None and pcu is not None:
            lines.append(
                f"Interpretation: A one-unit increase in femininity (masfem_c) is associated with {pct:.2f}% change in expected total deaths (exact transformation: (exp(beta)-1)*100). 95% CI for percent change: [{pcl:.2f}%, {pcu:.2f}%]."
            )
        else:
            lines.append("Interpretation: Percent-change transformation or CI could not be fully computed.")

        sig = masfem_stats["significant_at_0.05"]
        if sig is True:
            if coef is not None and coef > 0:
                lines.append("Conclusion for masfem_c: Statistically significant positive effect — results are consistent with the hypothesis (more feminine names → higher fatalities).")
            else:
                lines.append("Conclusion for masfem_c: Statistically significant negative effect — results go against the hypothesis.")
        elif sig is False:
            lines.append("Conclusion for masfem_c: Not statistically significant at alpha=0.05 — no strong evidence for an effect of name femininity on fatalities.")
        else:
            lines.append("Conclusion for masfem_c: Significance could not be determined due to missing p-value.")
    else:
        lines.append("masfem_c: Not found in model output.")

    if gender_stats is not None:
        coef = gender_stats["coef"]
        se = gender_stats["se"]
        t = gender_stats["t"]
        p = gender_stats["p_value"]
        ci0, ci1 = gender_stats["ci_95"]
        pct = gender_stats["percent_change_in_deaths"]
        pcl, pcu = gender_stats["percent_change_ci_95"]

        lines.append(
            f"gender_female (female name indicator): coef = {coef:.4f}, SE = {se:.4f}, t = {t:.2f}, p = {p:.3f}, 95% CI = [{ci0:.4f}, {ci1:.4f}]."
            if (coef is not None and se is not None and t is not None and p is not None and ci0 is not None and ci1 is not None)
            else "gender_female: Estimates found but incomplete numeric details."
        )
        if pct is not None and pcl is not None and pcu is not None:
            lines.append(
                f"Interpretation: Being a female-named storm (vs male-named) is associated with {pct:.2f}% change in expected total deaths. 95% CI: [{pcl:.2f}%, {pcu:.2f}%]."
            )
        else:
            lines.append("Interpretation: Percent-change transformation or CI could not be fully computed for gender_female.")

        sig = gender_stats["significant_at_0.05"]
        if sig is True:
            if coef is not None and coef > 0:
                lines.append("Conclusion for gender_female: Statistically significant positive effect — female names linked to higher fatalities (consistent with hypothesis).")
            else:
                lines.append("Conclusion for gender_female: Statistically significant negative effect — female names linked to lower fatalities (contradicts hypothesis).")
        elif sig is False:
            lines.append("Conclusion for gender_female: Not statistically significant at alpha=0.05 — no strong evidence for a difference by binary gender of name.")
        else:
            lines.append("Conclusion for gender_female: Significance could not be determined due to missing p-value.")
    else:
        lines.append("gender_female: Not found in model output.")

    description = " ".join(lines)

    result_object = {
        "masfem_c": masfem_stats,
        "gender_female": gender_stats
    }

    return {"object": result_object, "description": description}