def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, t-statistic, p-value, 95% CI and a short
    interpretation for the StudentTeacherRatio coefficient from a fitted statsmodels OLSResults
    object.

    Returns a dict with keys:
      - "object": a dict of numeric results and a boolean 'significant' flag
      - "description": a short plain-language interpretation in the context of the task
    """
    # Basic validation
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params)")

    param_name = "StudentTeacherRatio"
    params = model_output.params

    # Try to determine parameter names (robust to params being a pandas Series or numpy array)
    param_names = None
    if hasattr(params, "index"):
        # pandas Series or similar
        try:
            param_names = list(params.index)
        except Exception:
            param_names = None

    # statsmodels results usually expose model.exog_names
    if param_names is None and hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
        try:
            param_names = list(model_output.model.exog_names)
        except Exception:
            param_names = None

    # If still None, but params is an ndarray, just use positional access
    is_array_like_params = not hasattr(params, "index") and hasattr(params, "__len__")

    # Helper to fetch a value by name robustly
    def _get_by_name(container, name, names_list):
        # container can be Series, ndarray, list, dict, etc.
        if container is None:
            return None
        # dict-like
        if isinstance(container, dict):
            return container.get(name, None)
        # pandas-like with index
        if hasattr(container, "index"):
            try:
                return container[name]
            except Exception:
                return None
        # list/ndarray-like: need names_list
        if names_list is not None:
            try:
                idx = names_list.index(name)
                return container[idx]
            except Exception:
                return None
        # fallback: cannot find by name
        return None

    # Check param exists
    if param_names is not None:
        if param_name not in param_names:
            raise ValueError(f"Parameter '{param_name}' not found in model output. Available params: {param_names}")
    else:
        # We don't have names; if params is array-like we at least check length
        if is_array_like_params:
            # we cannot verify existence by name; proceed but will error if retrieval fails
            pass
        else:
            raise ValueError("Could not determine parameter names from model output to locate the requested parameter.")

    # Extract coefficient
    coef_raw = _get_by_name(params, param_name, param_names)
    if coef_raw is None:
        # try positional 0 if only one param (fallback)
        try:
            coef_raw = params[0]
        except Exception:
            raise ValueError(f"Could not extract coefficient for '{param_name}'.")
    coef = float(coef_raw)

    # standard error, t-value, p-value (robust to series/ndarray/dict)
    bse = getattr(model_output, "bse", None)
    tvalues = getattr(model_output, "tvalues", None)
    pvalues = getattr(model_output, "pvalues", None)

    se_raw = _get_by_name(bse, param_name, param_names)
    if se_raw is None and hasattr(bse, "__len__") and is_array_like_params:
        # try positional
        try:
            # use same index as coefficient
            if param_names is not None:
                idx = param_names.index(param_name)
                se_raw = bse[idx]
            else:
                se_raw = bse[0]
        except Exception:
            se_raw = None
    se = float(se_raw) if se_raw is not None else None

    tstat_raw = _get_by_name(tvalues, param_name, param_names)
    if tstat_raw is None and hasattr(tvalues, "__len__") and is_array_like_params:
        try:
            if param_names is not None:
                idx = param_names.index(param_name)
                tstat_raw = tvalues[idx]
            else:
                tstat_raw = tvalues[0]
        except Exception:
            tstat_raw = None
    tstat = float(tstat_raw) if tstat_raw is not None else None

    pvalue_raw = _get_by_name(pvalues, param_name, param_names)
    if pvalue_raw is None and hasattr(pvalues, "__len__") and is_array_like_params:
        try:
            if param_names is not None:
                idx = param_names.index(param_name)
                pvalue_raw = pvalues[idx]
            else:
                pvalue_raw = pvalues[0]
        except Exception:
            pvalue_raw = None
    pvalue = float(pvalue_raw) if pvalue_raw is not None else None

    # 95% confidence interval
    ci_lower, ci_upper = None, None
    if hasattr(model_output, "conf_int"):
        try:
            ci_result = model_output.conf_int(alpha=0.05)
            # ci_result may be a DataFrame (with index) or ndarray
            if hasattr(ci_result, "loc"):
                # DataFrame-like
                try:
                    ci_row = ci_result.loc[param_name]
                    ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
                except Exception:
                    # try positional if names known
                    if param_names is not None:
                        try:
                            idx = param_names.index(param_name)
                            row = ci_result.iloc[idx]
                            ci_lower, ci_upper = float(row[0]), float(row[1])
                        except Exception:
                            ci_lower, ci_upper = None, None
                    else:
                        ci_lower, ci_upper = None, None
            else:
                # ndarray-like: assume shape (k_params, 2)
                try:
                    if param_names is not None:
                        idx = param_names.index(param_name)
                        ci_lower, ci_upper = float(ci_result[idx, 0]), float(ci_result[idx, 1])
                    else:
                        ci_lower, ci_upper = float(ci_result[0, 0]), float(ci_result[0, 1])
                except Exception:
                    ci_lower, ci_upper = None, None
        except Exception:
            ci_lower, ci_upper = None, None

    # sample size if available
    n_obs = None
    if hasattr(model_output, "nobs"):
        try:
            # statsmodels stores nobs as float in some versions
            n_obs = int(model_output.nobs)
        except Exception:
            n_obs = None

    # Interpretation rules: use 5% significance threshold
    alpha = 0.05
    significant = (pvalue is not None) and (pvalue < alpha)

    # Build human-readable pieces safely (avoid formatting None with numeric formats)
    def fmt(x, fmt_spec=".4f"):
        return format(x, fmt_spec) if (x is not None) else "NA"

    # Direction interpretation:
    # Note: StudentTeacherRatio = students per teacher. Lower values = smaller class sizes.
    if significant:
        if coef < 0:
            direction = ("Statistically significant negative coefficient. "
                         "Higher student-teacher ratio (more students per teacher) is associated with LOWER AvgTestScore; "
                         "equivalently, a LOWER student-teacher ratio (fewer students per teacher / smaller classes) "
                         "is associated with HIGHER AvgTestScore.")
            conclusion = "Yes — evidence that lower student-teacher ratio is associated with higher academic performance (at the 5% level)."
        else:
            direction = ("Statistically significant positive coefficient. "
                         "Higher student-teacher ratio is associated with HIGHER AvgTestScore; "
                         "equivalently, a LOWER student-teacher ratio would be associated with LOWER AvgTestScore.")
            conclusion = "No — evidence that lower student-teacher ratio is associated with lower academic performance (at the 5% level)."
    else:
        # not significant
        if coef < 0:
            direction = ("Coefficient is negative (suggesting smaller classes → higher scores) but NOT statistically significant "
                         f"(p = {fmt(pvalue, '.3g')}).")
        else:
            direction = ("Coefficient is positive (suggesting smaller classes → lower scores) but NOT statistically significant "
                         f"(p = {fmt(pvalue, '.3g')}).")
        conclusion = ("No statistically significant association at the 5% level. The point estimate's direction is reported above, "
                      "but we cannot reject the null of no effect.")

    # Put numeric results into the "object" field (machine-friendly)
    result_object = {
        "parameter": param_name,
        "coef": coef,
        "std_error": se,
        "t_stat": tstat,
        "p_value": pvalue,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "n_obs": n_obs,
        "significant_at_0_05": significant
    }

    # Human-readable description
    description = (
        f"StudentTeacherRatio coefficient = {fmt(coef)}, SE = {fmt(se)}, "
        f"t = {fmt(tstat, '.3f') if tstat is not None else 'NA'}, p = {fmt(pvalue, '.3g')}, "
        f"95% CI = [{fmt(ci_lower)}, {fmt(ci_upper)}] (if available). "
        f"{direction} {conclusion}"
    )

    return {"object": result_object, "description": description}