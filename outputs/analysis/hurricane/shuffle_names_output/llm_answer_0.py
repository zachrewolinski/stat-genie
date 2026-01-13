def extract_final_answer(model_output):
    """
    Extracts statistics for the 'name_z' predictor from a fitted statsmodels GLMResultsWrapper
    (Negative Binomial or similar) and provides a brief interpretation relative to the hypothesis:
      "Hurricanes with more feminine names are perceived as less threatening and hence lead to fewer precautionary measures..."
    
    Returns:
      dict with keys:
        - "object": dict with numeric results (coef, se, z, pvalue, conf_int, IRR, IRR_conf_int, nobs, model_family, supports_hypothesis)
        - "description": brief interpretation of those results in the context of the task
    """
    import numpy as np
    import warnings

    res = model_output

    def _as_array(x):
        # Safely convert a statsmodels object (Series/ndarray/list) to numpy array
        try:
            return np.asarray(x)
        except Exception:
            return None

    def _get_names():
        # Attempt to get parameter names from several possible attributes
        names = None
        try:
            if hasattr(res, "params") and hasattr(res.params, "index"):
                names = list(res.params.index)
                return names
        except Exception:
            pass
        # Common statsmodels model attribute
        try:
            if hasattr(res, "model") and hasattr(res.model, "exog_names"):
                names = list(res.model.exog_names)
                return names
        except Exception:
            pass
        # Alternate attribute
        try:
            if hasattr(res, "param_names"):
                names = list(res.param_names)
                return names
        except Exception:
            pass
        return None

    def _format_num(x, fmt="{:.4f}"):
        try:
            if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
                return "NA"
            return fmt.format(x)
        except Exception:
            return str(x)

    # Prepare default empty response if something is missing
    obj = {}
    description = ""

    try:
        raw_params = res.params
        raw_bse = res.bse if hasattr(res, "bse") else None
        raw_pvalues = res.pvalues if hasattr(res, "pvalues") else None
        # conf_int may be a method or attribute
        try:
            ci_raw = res.conf_int()
        except Exception:
            ci_raw = None
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract results from the model_output object: {e}"
        }

    names = _get_names()

    # Convert params, bse, pvalues to dicts keyed by name if possible
    params_arr = _as_array(raw_params)
    bse_arr = _as_array(raw_bse) if raw_bse is not None else None
    pvalues_arr = _as_array(raw_pvalues) if raw_pvalues is not None else None

    params = {}
    bse = {}
    pvalues = {}

    if names is not None and params_arr is not None and len(names) == params_arr.shape[0]:
        for n, v in zip(names, params_arr):
            params[n] = v
        if bse_arr is not None and bse_arr.shape[0] == len(names):
            for n, v in zip(names, bse_arr):
                bse[n] = v
        if pvalues_arr is not None and pvalues_arr.shape[0] == len(names):
            for n, v in zip(names, pvalues_arr):
                pvalues[n] = v
    else:
        # If we can't map names, try to use pandas-like indexing if available
        try:
            # raw_params might be a pandas Series
            for n in getattr(raw_params, "index", []):
                params[n] = float(raw_params[n])
        except Exception:
            pass
        # Fallback: if params_arr is 1D numeric and names unknown, map by integer index names
        if not params and params_arr is not None:
            for i, v in enumerate(params_arr):
                params[str(i)] = v
            if bse_arr is not None and bse_arr.shape[0] == params_arr.shape[0]:
                for i, v in enumerate(bse_arr):
                    bse[str(i)] = v
            if pvalues_arr is not None and pvalues_arr.shape[0] == params_arr.shape[0]:
                for i, v in enumerate(pvalues_arr):
                    pvalues[str(i)] = v

    # Check presence of 'name_z' parameter
    if 'name_z' not in params:
        return {
            "object": None,
            "description": "The model does not contain a parameter named 'name_z'."
        }

    # Extract numeric values (cast to float or None)
    def _safe_float(d, key):
        try:
            v = d.get(key, None)
            if v is None:
                return None
            vf = float(np.asarray(v).tolist())
            if np.isnan(vf) or np.isinf(vf):
                return None
            return vf
        except Exception:
            return None

    coef = _safe_float(params, 'name_z')
    se = _safe_float(bse, 'name_z')
    pval = _safe_float(pvalues, 'name_z')

    # Confidence interval extraction
    ci_lower = ci_upper = None
    try:
        # ci_raw might be a DataFrame-like with .loc or a numpy array
        if ci_raw is not None:
            # If ci_raw has .loc and index
            if hasattr(ci_raw, "loc"):
                try:
                    row = ci_raw.loc['name_z']
                    ci_lower = float(np.asarray(row.iloc[0]).tolist()) if hasattr(row, "iloc") else float(np.asarray(row[0]).tolist())
                    ci_upper = float(np.asarray(row.iloc[1]).tolist()) if hasattr(row, "iloc") else float(np.asarray(row[1]).tolist())
                except Exception:
                    # maybe index numeric or different; try to match by position using names
                    if names is not None and 'name_z' in names:
                        pos = names.index('name_z')
                        arr = _as_array(ci_raw)
                        if arr is not None and arr.ndim == 2 and arr.shape[0] > pos:
                            ci_lower = float(arr[pos, 0])
                            ci_upper = float(arr[pos, 1])
            else:
                # ci_raw might be ndarray
                arr = _as_array(ci_raw)
                if arr is not None:
                    if arr.ndim == 2:
                        if names is not None and 'name_z' in names:
                            pos = names.index('name_z')
                            if pos < arr.shape[0]:
                                ci_lower = float(arr[pos, 0])
                                ci_upper = float(arr[pos, 1])
                        else:
                            # If no names, assume same order as params and find first matching coef value
                            if params_arr is not None:
                                # find index where params_arr equals coef (approx)
                                idx = None
                                for i, v in enumerate(params_arr):
                                    try:
                                        if np.isclose(float(v), float(coef), equal_nan=False):
                                            idx = i
                                            break
                                    except Exception:
                                        continue
                                if idx is None:
                                    # fallback to 0 if shapes match
                                    if arr.shape[0] == params_arr.shape[0]:
                                        idx = 0
                                if idx is not None and idx < arr.shape[0]:
                                    ci_lower = float(arr[idx, 0])
                                    ci_upper = float(arr[idx, 1])
    except Exception:
        ci_lower = ci_upper = None

    # Compute z statistic
    z_stat = None
    try:
        if coef is not None and se is not None and se != 0:
            z_stat = float(coef / se)
    except Exception:
        z_stat = None

    # IRR for log-link count models
    irr = irr_ci_lower = irr_ci_upper = None
    try:
        if coef is not None:
            irr = float(np.exp(coef))
        if ci_lower is not None:
            irr_ci_lower = float(np.exp(ci_lower))
        if ci_upper is not None:
            irr_ci_upper = float(np.exp(ci_upper))
    except Exception:
        irr = irr_ci_lower = irr_ci_upper = None

    # nobs if available
    nobs = None
    try:
        if hasattr(res, "nobs"):
            nobs_attr = res.nobs
            # nobs can be array-like or float
            if isinstance(nobs_attr, (list, tuple, np.ndarray)):
                # take first element
                try:
                    nobs = int(np.asarray(nobs_attr).sum()) if np.asarray(nobs_attr).size > 1 else int(np.asarray(nobs_attr).item())
                except Exception:
                    nobs = None
            else:
                try:
                    nobs = int(nobs_attr)
                except Exception:
                    nobs = None
    except Exception:
        nobs = None

    model_family = None
    try:
        model_family = getattr(getattr(res, "model", None), "family", None)
        if model_family is not None:
            model_family = str(model_family)
    except Exception:
        model_family = None

    obj = {
        "coef_name_z": coef,
        "se_name_z": se,
        "z_stat_name_z": z_stat,
        "pvalue_name_z": pval,
        "ci95_name_z": [ci_lower, ci_upper],
        "irr_name_z": irr,
        "irr95_name_z": [irr_ci_lower, irr_ci_upper],
        "nobs": nobs,
        "model_family": model_family
    }

    # Interpretation relative to hypothesis
    if pval is None:
        significance_text = "p-value unavailable; cannot assess statistical significance."
    else:
        alpha = 0.05
        sig = pval < alpha
        significance_text = (
            f"The coefficient is {'statistically significant' if sig else 'not statistically significant'} "
            f"at alpha={alpha} (p = {pval:.4g})."
        )

    if coef is None:
        direction_text = "unknown"
    else:
        direction_text = "negative" if coef < 0 else ("zero" if coef == 0 else "positive")

    if irr is not None:
        pct_change = (irr - 1) * 100.0
        pct_text = f"An increase of 1 SD in name femininity is associated with a {pct_change:.2f}% change in the expected death count (IRR = {irr:.3f})."
    else:
        pct_text = "IRR unavailable."

    desc_coef = _format_num(coef, "{:.4f}") if coef is not None else "NA"
    desc_se = _format_num(se, "{:.4f}") if se is not None else "NA"
    desc_z = _format_num(z_stat, "{:.3f}") if z_stat is not None else "NA"
    desc_ci_lower = _format_num(ci_lower, "{:.4f}") if ci_lower is not None else "NA"
    desc_ci_upper = _format_num(ci_upper, "{:.4f}") if ci_upper is not None else "NA"
    desc_p = _format_num(pval, "{:.4g}") if pval is not None else "NA"

    description = (
        f"Extracted statistics for predictor 'name_z': coefficient = {desc_coef}, SE = {desc_se}, z = {desc_z}, "
        f"95% CI = [{desc_ci_lower}, {desc_ci_upper}], p-value = {desc_p}. {significance_text} "
        f"The coefficient is {direction_text}, which means that "
        + (
            "more feminine names are associated with fewer deaths"
            if (coef is not None and coef < 0)
            else ("more feminine names are associated with more deaths" if (coef is not None and coef > 0) else "no directional effect detected")
        )
        + f". {pct_text} Number of observations used: {nobs if nobs is not None else 'NA'}."
    )

    # Also add an explicit boolean on whether results support the hypothesis:
    supports_hypothesis = None
    if pval is not None and coef is not None:
        # supports if coef negative AND statistically significant at 0.05
        supports_hypothesis = (coef < 0) and (pval < 0.05)
    obj["supports_hypothesis"] = supports_hypothesis

    return {
        "object": obj,
        "description": description
    }