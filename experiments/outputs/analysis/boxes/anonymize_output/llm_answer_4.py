def extract_final_answer(model_output):
    """
    Extracts age-related effects (linear and quadratic) from the two fitted models
    in model_output and returns numeric summaries plus a short interpretation.

    Returns a dict with keys:
      - "object": nested dict with extracted statistics for Age_c and Age_c2 for
                  both models (DemonstratedChosen and MajorityChosen).
      - "description": brief interpretation in plain language.

    The function is defensive: it will try to use the cluster-robust result if present,
    and fall back to the fitted model otherwise. If a model is missing or a term is
    missing, that is reported.
    """
    import numpy as np

    out = {"object": {}, "description": ""}

    def get_result_obj(model_part):
        # Prefer cluster-robust result if available, otherwise fitted model
        if not isinstance(model_part, dict):
            return None
        return model_part.get("cluster_robust_model") or model_part.get("fitted_model")

    def _get_name_list(res_obj):
        # Try likely attributes that list parameter names
        if res_obj is None:
            return None
        if hasattr(res_obj, "param_names"):
            try:
                return list(res_obj.param_names)
            except Exception:
                pass
        if hasattr(res_obj, "model") and hasattr(res_obj.model, "exog_names"):
            try:
                return list(res_obj.model.exog_names)
            except Exception:
                pass
        # some objects may have k_vars/k_constant, but without names we can't map
        return None

    def _fetch_attr_by_term(res_obj, attr_name, term):
        """
        Attempts to fetch a single value for 'term' from attribute attr_name of res_obj.
        Handles pandas Series/DataFrame, dicts, numpy arrays (if names available), and callables.
        Returns (value_or_none, raw_attr_obj_or_none)
        """
        if res_obj is None:
            return None, None
        attr = getattr(res_obj, attr_name, None)
        if attr is None:
            return None, None
        # If callable (like conf_int), try calling it (no args)
        try:
            maybe = attr() if callable(attr) else attr
        except Exception:
            # If calling fails, fall back to attribute object itself
            maybe = attr

        # If it's a pandas-like object with .loc, prefer label-based extraction
        try:
            if hasattr(maybe, "loc"):
                try:
                    val = maybe.loc[term]
                    # If val is a Series (e.g., conf_int row), return it as-is
                    return val if not getattr(val, "__len__", lambda: None) is None and np.shape(val) == () else val, maybe
                except Exception:
                    pass
        except Exception:
            pass

        # If it's a dict-like, try key access
        try:
            if isinstance(maybe, dict):
                if term in maybe:
                    return maybe[term], maybe
        except Exception:
            pass

        # If it's list/tuple/ndarray, try mapping term -> index using names
        try:
            if isinstance(maybe, (list, tuple, np.ndarray)):
                names = _get_name_list(res_obj)
                if names and term in names:
                    idx = names.index(term)
                    val = maybe[idx]
                    return val, maybe
        except Exception:
            pass

        # If nothing matched, return None
        return None, maybe

    def summarize_term(res_obj, term):
        # Try to extract coef, se, pvalue, conf_int; fallback if attributes absent.
        if res_obj is None:
            return {"error": "no model object"}
        # coef
        coef_raw, _ = _fetch_attr_by_term(res_obj, "params", term)
        if coef_raw is None:
            return {"error": f"term '{term}' not in model params"}
        try:
            coef = float(coef_raw)
        except Exception:
            # If coef_raw is array-like, try first element
            try:
                coef = float(np.asarray(coef_raw).item())
            except Exception:
                return {"error": f"could not coerce coefficient for term '{term}' to float"}
        # bse
        bse_raw, _ = _fetch_attr_by_term(res_obj, "bse", term)
        bse = None
        if bse_raw is not None:
            try:
                bse = float(bse_raw)
            except Exception:
                try:
                    bse = float(np.asarray(bse_raw).item())
                except Exception:
                    bse = None
        # pvalue
        pval_raw, _ = _fetch_attr_by_term(res_obj, "pvalues", term)
        pval = None
        if pval_raw is not None:
            try:
                pval = float(pval_raw)
            except Exception:
                try:
                    pval = float(np.asarray(pval_raw).item())
                except Exception:
                    pval = None
        # conf_int
        lower = upper = None
        ci_obj = None
        # try conf_int attribute/method
        ci_raw, ci_obj = _fetch_attr_by_term(res_obj, "conf_int", term)
        # conf_int often returns a 2-column array/DF row; handle accordingly
        if ci_raw is not None:
            try:
                arr = np.asarray(ci_raw)
                if arr.size == 2:
                    lower = float(arr[0])
                    upper = float(arr[1])
                else:
                    # If ci_raw is a row with two columns but with shape (2,) or (1,2)
                    if arr.ndim == 1 and arr.size >= 2:
                        lower = float(arr[0])
                        upper = float(arr[1])
            except Exception:
                # fall through to try other extraction below
                ci_raw = None
        if ci_raw is None:
            # As fallback: if bse available, approximate CI using Normal approx
            if bse is not None:
                lower = coef - 1.96 * bse
                upper = coef + 1.96 * bse
            else:
                lower = upper = None
        # odds ratios
        try:
            or_coef = float(np.exp(coef)) if coef is not None else None
            or_lower = float(np.exp(lower)) if lower is not None else None
            or_upper = float(np.exp(upper)) if upper is not None else None
        except Exception:
            or_coef = or_lower = or_upper = None

        return {
            "coef": coef,
            "std_err": bse,
            "p_value": pval,
            "ci95": [lower, upper],
            "odds_ratio": or_coef,
            "odds_ratio_ci95": [or_lower, or_upper],
            "significant_p_lt_0.05": (pval is not None and pval < 0.05)
        }

    # 1) DemonstratedChosen model
    dem_part = model_output.get("demonstrated_model")
    dem_res = get_result_obj(dem_part)
    dem_age_c = summarize_term(dem_res, "Age_c")
    dem_age_c2 = summarize_term(dem_res, "Age_c2")
    out["object"]["DemonstratedChosen"] = {
        "Age_c": dem_age_c,
        "Age_c2": dem_age_c2
    }

    # 2) MajorityChosen model
    maj_part = model_output.get("majoritypref_model")
    if isinstance(maj_part, dict) and maj_part.get("error"):
        out["object"]["MajorityChosen"] = {"error": maj_part.get("error"), "n_demonstrated": maj_part.get("n")}
    else:
        maj_res = get_result_obj(maj_part)
        maj_age_c = summarize_term(maj_res, "Age_c")
        maj_age_c2 = summarize_term(maj_res, "Age_c2")
        # also include sample size if present
        n_dem = None
        if isinstance(maj_part, dict):
            n_dem = maj_part.get("n_demonstrated") or maj_part.get("n")
        out["object"]["MajorityChosen"] = {
            "Age_c": maj_age_c,
            "Age_c2": maj_age_c2,
            "n_demonstrated": n_dem
        }

    # Build a concise description/interpretation based on extracted stats
    desc_lines = []
    # Interpreting DemonstratedChosen
    dac = out["object"]["DemonstratedChosen"]
    # Check if both terms produced errors
    a1 = dac.get("Age_c")
    a2 = dac.get("Age_c2")
    if (isinstance(a1, dict) and a1.get("error")) and (isinstance(a2, dict) and a2.get("error")):
        desc_lines.append("DemonstratedChosen: model or age terms not available.")
    else:
        if isinstance(a1, dict) and a1.get("error"):
            desc_lines.append("DemonstratedChosen: Age_c not available in model.")
        elif isinstance(a2, dict) and a2.get("error"):
            desc_lines.append("DemonstratedChosen: Age_c2 not available in model.")
        else:
            # Both present (or at least one usable)
            sig1 = isinstance(a1, dict) and a1.get("significant_p_lt_0.05", False)
            sig2 = isinstance(a2, dict) and a2.get("significant_p_lt_0.05", False)
            if not sig1 and not sig2:
                desc_lines.append(
                    "DemonstratedChosen: No evidence that age (linear or quadratic) predicts children's reliance on social information "
                    "(both Age_c and Age_c2 have p >= 0.05 or missing p-values)."
                )
            else:
                parts = []
                if sig1:
                    coef = a1.get("coef")
                    pval = a1.get("p_value")
                    dir1 = "higher" if coef > 0 else "lower"
                    coef_str = f"{coef:.3f}" if coef is not None else "NA"
                    pstr = f"{pval:.3f}" if pval is not None else "NA"
                    parts.append(f"linear age effect: older children are {dir1} in odds (coef={coef_str}, p={pstr})")
                if sig2:
                    coef2 = a2.get("coef")
                    pval2 = a2.get("p_value")
                    coef2_str = f"{coef2:.3f}" if coef2 is not None else "NA"
                    p2str = f"{pval2:.3f}" if pval2 is not None else "NA"
                    quad_dir = "U-shaped (higher at age extremes)" if coef2 > 0 else "inverted-U (higher near the mean age)"
                    parts.append(f"quadratic age effect: {quad_dir} (coef={coef2_str}, p={p2str})")
                desc_lines.append("DemonstratedChosen: " + "; ".join(parts))

    # Interpreting MajorityChosen
    mobj = out["object"]["MajorityChosen"]
    if isinstance(mobj, dict) and mobj.get("error"):
        desc_lines.append(f"MajorityChosen: {mobj.get('error')} (n_demonstrated={mobj.get('n')})")
    else:
        a1 = mobj.get("Age_c")
        a2 = mobj.get("Age_c2")
        if (isinstance(a1, dict) and a1.get("error")) and (isinstance(a2, dict) and a2.get("error")):
            desc_lines.append("MajorityChosen: model or age terms not available.")
        else:
            if isinstance(a1, dict) and a1.get("error"):
                desc_lines.append("MajorityChosen: Age_c not available in model.")
            elif isinstance(a2, dict) and a2.get("error"):
                desc_lines.append("MajorityChosen: Age_c2 not available in model.")
            else:
                sig_lin = isinstance(a1, dict) and a1.get("significant_p_lt_0.05", False)
                sig_quad = isinstance(a2, dict) and a2.get("significant_p_lt_0.05", False)
                if not sig_lin and not sig_quad:
                    desc_lines.append(
                        "MajorityChosen: No evidence that age (linear or quadratic) predicts majority preference (both Age_c and Age_c2 non-significant or missing p-values)."
                    )
                else:
                    parts = []
                    if sig_lin:
                        coef = a1.get("coef")
                        pval = a1.get("p_value")
                        dir1 = "increase" if coef > 0 else "decrease"
                        coef_str = f"{coef:.3f}" if coef is not None else "NA"
                        pstr = f"{pval:.3f}" if pval is not None else "NA"
                        parts.append(f"linear effect: {dir1} in odds per year (coef={coef_str}, p={pstr})")
                    if sig_quad:
                        coef2 = a2.get("coef")
                        pval2 = a2.get("p_value")
                        coef2_str = f"{coef2:.3f}" if coef2 is not None else "NA"
                        p2str = f"{pval2:.3f}" if pval2 is not None else "NA"
                        if coef2 is not None and coef2 > 0:
                            quad_interp = ("a convex (U-shaped) relation: preference is lower near the mean age and higher at younger "
                                           "and older ages (coef={:.3f}, p={:.3f})").format(coef2, pval2 if pval2 is not None else float('nan'))
                        else:
                            quad_interp = ("a concave (inverted-U) relation: preference peaks near the mean age and is lower at younger "
                                           "and older ages (coef={:.3f}, p={:.3f})").format(coef2 if coef2 is not None else float('nan'),
                                                                                          pval2 if pval2 is not None else float('nan'))
                        parts.append("quadratic effect: " + quad_interp)
                    n_demonstrated = mobj.get("n_demonstrated")
                    parts_joined = "; ".join(parts)
                    parts_joined += (f" (n_demonstrated={n_demonstrated})" if n_demonstrated is not None else "")
                    desc_lines.append("MajorityChosen: " + parts_joined)

    out["description"] = " ".join(desc_lines)
    return out

# Example usage:
# final = extract_final_answer(model_output)
# print(final["description"])