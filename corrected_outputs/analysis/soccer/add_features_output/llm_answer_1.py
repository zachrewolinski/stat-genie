def extract_final_answer(model_output):
    """
    Extracts the estimated effect of dark_skin on red card counts from the model_output
    produced by the provided modeling function.

    Returns a dictionary with:
      - "object": a dict with numeric results (coef, clustered SE, z, p-value, IRR, IRR 95% CI, n_obs)
      - "description": short interpretation answering whether dark-skinned players are more likely
                       than light-skinned players to receive red cards (based on the model).
    The function is defensive and robust to a variety of container types (pandas Series/DataFrame,
    numpy arrays) for params, bse, and confidence intervals.
    """
    import numpy as np
    from scipy import stats as sstats
    import pandas as pd

    out = {"object": None, "description": None}

    # Defensive extraction of objects that should be present
    clustered = model_output.get('clustered_results', None)
    fit = model_output.get('fit', None)
    irr_series = model_output.get('irr', None)
    conf_int_irr_df = model_output.get('conf_int_irr', None)

    def _get_names_from_result(res):
        # Try various places to extract parameter names
        if res is None:
            return None
        # statsmodels result: res.params has index or res.model.exog_names
        try:
            params = getattr(res, 'params', None)
            if isinstance(params, (pd.Series, pd.DataFrame)) and hasattr(params, 'index'):
                return list(params.index)
        except Exception:
            pass
        try:
            if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                return list(res.model.exog_names)
        except Exception:
            pass
        # Fall back to model_output keys that might list names
        for key in ('param_names', 'exog_names', 'names'):
            val = model_output.get(key)
            if val is not None:
                return list(val)
        return None

    def _get_value_by_name(container, name, res_obj=None):
        """
        Extract a numeric value for 'name' from container which may be:
        - pandas Series/DataFrame (with index)
        - dict-like
        - numpy array (in which case try to find index via res_obj or model_output)
        """
        if container is None:
            return None
        # If pandas Series/DataFrame
        try:
            if isinstance(container, pd.Series):
                if name in container.index:
                    return float(container.loc[name])
            if isinstance(container, pd.DataFrame):
                # In case container is DataFrame and we want a row or column
                if name in container.index:
                    # return first numeric value if ambiguous
                    row = container.loc[name]
                    # If it's a Series, return the first value
                    if isinstance(row, pd.Series):
                        return float(row.iloc[0])
                    else:
                        return float(row)
        except Exception:
            pass
        # If dict-like
        try:
            if isinstance(container, dict):
                if name in container:
                    return float(container[name])
        except Exception:
            pass
        # If numpy array or other array-like: try to find index from res_obj or model_output
        try:
            arr = np.asarray(container)
            names = _get_names_from_result(res_obj)
            if names is None:
                names = _get_names_from_result(container)
            if names is not None and name in names:
                idx = names.index(name)
                return float(arr[idx])
        except Exception:
            pass
        return None

    def _get_confint_by_name(confint_obj, name, res_obj=None):
        """
        Extract confidence interval tuple (lower, upper) for a parameter name.
        confint_obj may be a DataFrame, ndarray, or object providing conf_int()
        If confint_obj is None but res_obj has conf_int(), try that.
        """
        if confint_obj is None and res_obj is not None:
            try:
                confint_obj = res_obj.conf_int()
            except Exception:
                confint_obj = None

        if confint_obj is None:
            return None

        # pandas DataFrame or ndarray
        try:
            if isinstance(confint_obj, pd.DataFrame):
                if name in confint_obj.index:
                    row = confint_obj.loc[name]
                    return (float(row.iloc[0]), float(row.iloc[1]))
            if isinstance(confint_obj, pd.Series):
                # unlikely shape, but handle
                if name in confint_obj.index:
                    val = confint_obj.loc[name]
                    if hasattr(val, '__iter__'):
                        return (float(val[0]), float(val[1]))
            # numpy array
            arr = np.asarray(confint_obj)
            if arr.ndim == 2:
                # try to resolve row by name using res_obj's names
                names = _get_names_from_result(res_obj)
                if names is None:
                    names = _get_names_from_result(confint_obj)
                if names is not None and name in names:
                    idx = names.index(name)
                    return (float(arr[idx, 0]), float(arr[idx, 1]))
                # If shape matches 2 and only one row, maybe it's already the row
                if arr.shape[0] == 2 and arr.shape[1] == 1:
                    # not typical; skip
                    pass
        except Exception:
            pass
        return None

    # Get coefficient (log scale) for dark_skin and clustered SE and clustered confint
    coef = None
    bse = None
    conf_int_log = None

    # Prefer clustered results if available
    if clustered is not None:
        # Try to extract params
        coef = _get_value_by_name(getattr(clustered, 'params', None), 'dark_skin', res_obj=clustered)
        # Try to extract bse (cluster-robust se may be in .bse or .bs or as attribute)
        bse_candidates = [getattr(clustered, 'bse', None), getattr(clustered, 'bs', None), getattr(clustered, 'std_err', None)]
        for cand in bse_candidates:
            if cand is None:
                continue
            bse_val = _get_value_by_name(cand, 'dark_skin', res_obj=clustered)
            if bse_val is not None:
                bse = bse_val
                break
        # Try clustered conf_int
        conf_int_log = _get_confint_by_name(None, 'dark_skin', res_obj=clustered)

    # Fall back to non-clustered fit if needed
    if coef is None and fit is not None:
        coef = _get_value_by_name(getattr(fit, 'params', None), 'dark_skin', res_obj=fit)
        if bse is None:
            bse = _get_value_by_name(getattr(fit, 'bse', None), 'dark_skin', res_obj=fit)
        if conf_int_log is None:
            conf_int_log = _get_confint_by_name(None, 'dark_skin', res_obj=fit)

    # Compute z and p-value using bse (cluster-robust if available)
    z = None
    p_value = None
    try:
        if coef is not None and bse is not None and bse > 0:
            z = coef / bse
            p_value = 2.0 * (1.0 - sstats.norm.cdf(abs(z)))
    except Exception:
        z = None
        p_value = None

    # Incident Rate Ratio (IRR) and CI on IRR scale
    irr = None
    irr_ci = (None, None)
    # Try provided irr series first
    if irr_series is not None:
        irr_val = _get_value_by_name(irr_series, 'dark_skin', res_obj=None)
        if irr_val is not None:
            irr = irr_val

    # Try conf_int_irr_df if provided
    if conf_int_irr_df is not None:
        try:
            # conf_int_irr_df may be DataFrame with columns 'IRR_ci_lower' and 'IRR_ci_upper' or similar
            if isinstance(conf_int_irr_df, pd.DataFrame) and 'dark_skin' in conf_int_irr_df.index:
                row = conf_int_irr_df.loc['dark_skin']
                lower = None
                upper = None
                # attempt by column names
                for col in ['IRR_ci_lower', 'IRR_ci_upper', 'lower', 'upper', 0, 1]:
                    pass
                try:
                    lower = float(row.get('IRR_ci_lower', row.iloc[0]))
                except Exception:
                    try:
                        lower = float(row.iloc[0])
                    except Exception:
                        lower = None
                try:
                    upper = float(row.get('IRR_ci_upper', row.iloc[1]))
                except Exception:
                    try:
                        upper = float(row.iloc[1])
                    except Exception:
                        upper = None
                irr_ci = (lower, upper)
        except Exception:
            irr_ci = (None, None)

    # If conf_int_irr_df not provided or failed, exponentiate log-scale CI if present
    if (irr_ci[0] is None or irr_ci[1] is None):
        if conf_int_log is not None and conf_int_log[0] is not None and conf_int_log[1] is not None:
            try:
                irr_ci = (float(np.exp(conf_int_log[0])), float(np.exp(conf_int_log[1])))
            except Exception:
                irr_ci = (None, None)

    # If irr still none, compute from coef if available
    if irr is None and coef is not None:
        try:
            irr = float(np.exp(coef))
        except Exception:
            irr = None

    # Number of observations (if available)
    n_obs = None
    if fit is not None:
        try:
            n_obs = int(getattr(fit, 'nobs'))
        except Exception:
            try:
                n_obs = int(fit.model.endog.shape[0])
            except Exception:
                n_obs = None
    elif clustered is not None:
        try:
            n_obs = int(getattr(clustered, 'nobs'))
        except Exception:
            try:
                n_obs = int(clustered.model.endog.shape[0])
            except Exception:
                n_obs = None

    # Build the numeric result object
    numeric_result = {
        "coef_log_scale": coef,
        "clustered_se": bse,
        "z_value": z,
        "p_value": p_value,
        "IRR": irr,
        "IRR_95_CI": irr_ci,
        "n_obs": n_obs
    }

    # Short interpretation
    def _fmt(x, fmt_str="{:.3f}"):
        try:
            if x is None:
                return "NA"
            return fmt_str.format(x)
        except Exception:
            return str(x)

    if numeric_result["p_value"] is not None:
        sig = numeric_result["p_value"] < 0.05
    else:
        sig = None

    if numeric_result["IRR"] is not None and sig is True:
        direction = "higher" if numeric_result["IRR"] > 1.0 else "lower"
        desc = (
            f"Estimated IRR for dark_skin = {_fmt(numeric_result['IRR'])} "
            f"(95% CI [{_fmt(numeric_result['IRR_95_CI'][0])}, {_fmt(numeric_result['IRR_95_CI'][1])}]). "
            f"Cluster-robust p = {_fmt(numeric_result['p_value'])}. "
            f"This indicates that players classified as dark-skinned have a {_fmt(numeric_result['IRR'], '{:.2f}')}x "
            f"{direction} rate of receiving red cards compared to light-skinned players, "
            f"and the effect is statistically significant at the 0.05 level."
        )
    elif numeric_result["IRR"] is not None and sig is False:
        direction = "higher" if numeric_result["IRR"] > 1.0 else "lower"
        desc = (
            f"Estimated IRR for dark_skin = {_fmt(numeric_result['IRR'])} "
            f"(95% CI [{_fmt(numeric_result['IRR_95_CI'][0])}, {_fmt(numeric_result['IRR_95_CI'][1])}]). "
            f"Cluster-robust p = {_fmt(numeric_result['p_value'])} (not < 0.05). "
            f"This suggests a {direction} rate of red cards for dark-skinned players compared to light-skinned players, "
            f"but the association is not statistically significant at the 0.05 level."
        )
    else:
        desc = (
            "Could not compute a complete inferential summary for dark_skin (some statistics missing). "
            "Available numeric outputs are returned in the 'object' field."
        )

    out["object"] = numeric_result
    out["description"] = desc

    return out