def extract_final_answer(model_output):
    """
    Extracts statistics for the key predictor 'MasFem_z' from the provided model_output.
    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, p-value, 95% CI, model used, nobs)
      - "description": plain-language interpretation relative to the hypothesis

    model_output is expected to be a dict with keys 'neg_binomial' and/or 'ols_log'.
    Preference is given to the count model ('neg_binomial') if present; otherwise OLS on log(Fatalities+1) is used.
    """
    import numpy as np
    import pandas as pd

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys like 'neg_binomial' and 'ols_log'.")

    nb_res = model_output.get('neg_binomial', None)
    ols_res = model_output.get('ols_log', None)

    # Choose model to extract from: prefer negative-binomial if present, else OLS
    if nb_res is not None:
        res = nb_res
        model_used = 'neg_binomial'
    elif ols_res is not None:
        res = ols_res
        model_used = 'ols_log'
    else:
        return {
            "object": None,
            "description": "No fitted model available in model_output (both 'neg_binomial' and 'ols_log' are None)."
        }

    # Name of the predictor we care about
    key = 'MasFem_z'

    # Get raw params
    params_raw = getattr(res, 'params', None)
    if params_raw is None:
        return {
            "object": None,
            "description": f"The chosen model ('{model_used}') does not expose 'params'."
        }

    # Determine parameter names and index for the key
    param_names = None
    param_index = None

    # If params is a pandas Series/DataFrame-like with an index
    if hasattr(params_raw, 'index'):
        try:
            param_names = list(params_raw.index)
        except Exception:
            param_names = None

    # Try common locations for parameter names if not found above
    if param_names is None:
        # 1) result may expose param_names
        pn = getattr(res, 'param_names', None)
        if pn is not None:
            try:
                param_names = list(pn)
            except Exception:
                param_names = None

    if param_names is None:
        # 2) model may expose data.param_names or exog_names
        model = getattr(res, 'model', None)
        if model is not None:
            data = getattr(model, 'data', None)
            if data is not None:
                dpn = getattr(data, 'param_names', None)
                if dpn is not None:
                    try:
                        param_names = list(dpn)
                    except Exception:
                        param_names = None
            if param_names is None:
                exn = getattr(model, 'exog_names', None)
                if exn is not None:
                    try:
                        param_names = list(exn)
                    except Exception:
                        param_names = None

    # 3) as a last resort, if params_raw is a dict-like
    if param_names is None and isinstance(params_raw, dict):
        try:
            param_names = list(params_raw.keys())
        except Exception:
            param_names = None

    # If still None and params_raw is an ndarray, we can't map names -> index without model info
    if param_names is None and isinstance(params_raw, (list, tuple, np.ndarray)):
        # Try to infer length only; we cannot find key by name
        param_names = None

    # Now determine if key is present and its index
    if param_names is not None:
        if key not in param_names:
            return {
                "object": None,
                "description": f"The chosen model ('{model_used}') does not contain a parameter named '{key}'."
            }
        param_index = int(param_names.index(key))
    else:
        # param_names unavailable; if params_raw is a pandas-like Series handled above already,
        # but if it's a numpy array and we don't have names, we cannot find 'MasFem_z'
        return {
            "object": None,
            "description": f"Could not determine parameter names for the chosen model ('{model_used}'), so cannot locate '{key}'."
        }

    # Helper to extract a scalar from potentially different container types
    def _extract_by_name_or_index(container, name, index):
        if container is None:
            return None
        # pandas Series / DataFrame with .loc or .at
        try:
            if hasattr(container, 'loc') and name in container.index:
                return float(container.loc[name])
        except Exception:
            pass
        try:
            # mapping/dict-like
            if isinstance(container, dict) and name in container:
                return float(container[name])
        except Exception:
            pass
        # ndarray/list-like: use index
        try:
            if isinstance(container, (list, tuple, np.ndarray)):
                return float(container[int(index)])
        except Exception:
            pass
        # fallback: try attribute by name on the container
        try:
            val = getattr(container, name)
            return float(val)
        except Exception:
            pass
        return None

    # Extract statistics
    coef = _extract_by_name_or_index(params_raw, key, param_index)
    if coef is None:
        # If params_raw supports positional indexing but name lookup didn't work, try position
        try:
            if isinstance(params_raw, (list, tuple, np.ndarray)):
                coef = float(params_raw[param_index])
        except Exception:
            coef = None

    # Standard error and p-value
    bse_raw = getattr(res, 'bse', None)
    pval_raw = getattr(res, 'pvalues', None)

    bse = _extract_by_name_or_index(bse_raw, key, param_index)
    pval = _extract_by_name_or_index(pval_raw, key, param_index)

    # Confidence interval (default 95%)
    ci_low, ci_high = None, None
    try:
        ci = res.conf_int(alpha=0.05)
        if isinstance(ci, pd.DataFrame):
            if key in ci.index:
                ci_low = float(ci.loc[key, 0])
                ci_high = float(ci.loc[key, 1])
            else:
                # try by position
                if param_index is not None and 0 <= param_index < ci.shape[0]:
                    ci_low = float(ci.iloc[param_index, 0])
                    ci_high = float(ci.iloc[param_index, 1])
        elif isinstance(ci, (list, tuple, np.ndarray)):
            ci_arr = np.asarray(ci)
            if ci_arr.ndim == 2 and param_index is not None and 0 <= param_index < ci_arr.shape[0]:
                ci_low = float(ci_arr[param_index, 0])
                ci_high = float(ci_arr[param_index, 1])
    except Exception:
        ci_low, ci_high = None, None

    # Number of observations if available
    nobs = None
    try:
        if hasattr(res, 'nobs'):
            nobs_tmp = getattr(res, 'nobs')
            if nobs_tmp is not None:
                nobs = int(nobs_tmp)
    except Exception:
        nobs = None
    if nobs is None:
        model = getattr(res, 'model', None)
        if model is not None:
            try:
                nobs_tmp = getattr(model, 'nobs', None)
                if nobs_tmp is None and hasattr(model, 'data') and getattr(model.data, 'nobs', None) is not None:
                    nobs_tmp = getattr(model.data, 'nobs')
                if nobs_tmp is not None:
                    nobs = int(nobs_tmp)
            except Exception:
                nobs = None
    if nobs is None:
        nobs = 0

    # Round for readability
    def _r(x):
        return None if x is None else float(np.round(x, 4))

    coef_r = _r(coef)
    bse_r = _r(bse)
    pval_r = _r(pval)
    ci_low_r = _r(ci_low)
    ci_high_r = _r(ci_high)

    # Interpretation relative to hypothesis:
    # Hypothesis: More feminine names => perceived as less threatening => fewer precautionary measures => fewer fatalities
    # That implies a negative association between MasFem_z and Fatalities (or log(Fatalities+1)).
    if pval is None:
        significance_text = "statistical significance could not be determined (p-value unavailable)."
        supports = None
    else:
        alpha = 0.05
        try:
            significant = (pval < alpha)
        except Exception:
            significant = False
        if coef is None:
            supports = None
            significance_text = "Coefficient unavailable, cannot assess statistical support."
        else:
            if coef < 0 and significant:
                supports = True
                significance_text = f"statistically significant (p = {pval_r} < {alpha})."
            elif coef < 0 and not significant:
                supports = False
                significance_text = f"negative point estimate but not statistically significant (p = {pval_r} ≥ {alpha})."
            elif coef > 0 and significant:
                supports = False
                significance_text = f"statistically significant in the opposite direction (positive coefficient, p = {pval_r} < {alpha})."
            else:
                supports = False
                significance_text = f"positive point estimate (or non-significant) so it does not support the hypothesized negative association (p = {pval_r})."

    # Build the returned object
    obj = {
        "model_used": model_used,
        "parameter": key,
        "coef": coef_r,
        "std_error": bse_r,
        "p_value": pval_r,
        "ci_95_lower": ci_low_r,
        "ci_95_upper": ci_high_r,
        "nobs": nobs,
        "supports_hypothesis": supports
    }

    # Compose description
    if model_used == 'ols_log':
        # For log(Fatalities+1), coef is approx change in log outcome per 1 SD increase in femininity.
        percent_text = ""
        if coef is not None:
            try:
                approx_pct = np.round((np.expm1(coef) * 100), 2)
                percent_text = f"Approx multiplicative change in (Fatalities+1): exp(coef)-1 = {approx_pct}%.\n"
            except Exception:
                percent_text = ""
        desc = (
            f"Used OLS on log(Fatalities+1). The coefficient for '{key}' = {coef_r} "
            f"(SE = {bse_r}, p = {pval_r}), 95% CI = [{ci_low_r}, {ci_high_r}], n = {nobs}.\n"
            f"{percent_text}"
            f"Interpretation: A {'negative' if (coef is not None and coef < 0) else 'positive'} coefficient means that higher femininity is "
            f"{'associated with fewer' if (coef is not None and coef < 0) else 'associated with more'} fatalities (on the log scale). "
            f"Assessment: {significance_text}"
        )
    else:
        # For count model, coefficients are on the log-count scale (multiplicative effects)
        exp_text = ""
        if coef is not None:
            try:
                exp_coef = np.round(np.exp(coef), 4)
                exp_text = f" exp(coef) = {exp_coef}."
            except Exception:
                exp_text = ""
        desc = (
            f"Used negative-binomial (or Poisson fallback) count model. The coefficient for '{key}' = {coef_r} "
            f"(SE = {bse_r}, p = {pval_r}), 95% CI = [{ci_low_r}, {ci_high_r}], n = {nobs}.\n"
            f"Interpretation: Exponentiating the coefficient gives the multiplicative change in expected fatalities "
            f"per 1 SD increase in name femininity:{exp_text} "
            f"Assessment: {significance_text}"
        )

    return {
        "object": obj,
        "description": desc
    }