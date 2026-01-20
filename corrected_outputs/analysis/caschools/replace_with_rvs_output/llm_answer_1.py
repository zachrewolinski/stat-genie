def extract_final_answer(model_output):
    """
    Extracts the coefficient, robust standard error, t-statistic, p-value, and 95% CI
    for the StudentTeacherRatio variable from a fitted statsmodels results object
    (expected to be the object returned by get_robustcov_results or a similar wrapper).
    Returns a dictionary with keys "object" and "description".
    """
    import numpy as np
    import pandas as pd

    res = model_output
    var = 'StudentTeacherRatio'

    # Helper: convert various param-like objects into a pandas Series with names
    def to_series(obj, names_guess=None):
        if obj is None:
            return None
        if isinstance(obj, pd.Series):
            return obj
        if isinstance(obj, dict):
            return pd.Series(obj)
        if isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj)
            # determine names: prefer provided guess, then try model.exog_names, then param_names
            names = None
            if names_guess is not None:
                names = names_guess
            else:
                model = getattr(res, 'model', None)
                if model is not None and hasattr(model, 'exog_names'):
                    names = list(model.exog_names)
                elif hasattr(res, 'param_names'):
                    try:
                        names = list(res.param_names)
                    except Exception:
                        names = None
            if names is None or len(names) != arr.shape[0]:
                # fallback to generic names
                names = [f'param_{i}' for i in range(arr.shape[0])]
            return pd.Series(arr, index=names)
        # unknown type: try to coerce to Series
        try:
            return pd.Series(obj)
        except Exception:
            return None

    # Prepare safe accessor for series-like objects
    def safe_get_from_series(series_like, key):
        if series_like is None:
            return np.nan
        # if it's already a Series or convertible, index check will work
        try:
            return series_like.loc[key]
        except Exception:
            try:
                return series_like[key]
            except Exception:
                # If key not found, return nan
                return np.nan

    # Get params as Series
    params_raw = getattr(res, 'params', None)
    params = to_series(params_raw)

    if params is None or var not in params.index:
        # Try alternate ways to find parameter names (in case params was ndarray without names)
        alt_names = None
        model = getattr(res, 'model', None)
        if model is not None and hasattr(model, 'exog_names'):
            alt_names = list(model.exog_names)
        elif hasattr(res, 'param_names'):
            try:
                alt_names = list(res.param_names)
            except Exception:
                alt_names = None

        available = list(params.index) if params is not None else alt_names if alt_names is not None else 'None'
        return {
            "object": None,
            "description": f"Variable '{var}' not found in the model output parameters. Available parameters: {available}"
        }

    # Extract coefficient
    coef = float(safe_get_from_series(params, var))

    # Standard error
    bse_raw = getattr(res, 'bse', None)
    bse_ser = to_series(bse_raw)
    bse = np.nan
    if bse_ser is not None and var in bse_ser.index:
        try:
            bse = float(safe_get_from_series(bse_ser, var))
        except Exception:
            bse = np.nan
    else:
        # try cov_params
        try:
            cov = None
            # cov_params might be a method or attribute
            cov_attr = getattr(res, 'cov_params', None)
            if callable(cov_attr):
                cov = cov_attr()
            else:
                cov = cov_attr
            cov_df = to_series(cov)  # if cov is array, to_series will create 1D, so avoid; better to handle DataFrame/ndarray
            # Prefer DataFrame access
            if isinstance(cov, pd.DataFrame):
                if var in cov.index and var in cov.columns:
                    bse = float(np.sqrt(np.abs(cov.loc[var, var])))
                else:
                    bse = np.nan
            else:
                # if cov is ndarray, try to get index of var in params.index
                try:
                    cov_arr = np.asarray(cov)
                    idx = list(params.index).index(var)
                    bse = float(np.sqrt(np.abs(cov_arr[idx, idx])))
                except Exception:
                    bse = np.nan
        except Exception:
            bse = np.nan

    # t-statistic
    tval_raw = getattr(res, 'tvalues', None)
    tval_ser = to_series(tval_raw)
    if tval_ser is not None and var in tval_ser.index:
        try:
            tval = float(safe_get_from_series(tval_ser, var))
        except Exception:
            tval = (coef / bse) if (not np.isnan(bse) and bse != 0) else np.nan
    else:
        tval = (coef / bse) if (not np.isnan(bse) and bse != 0) else np.nan

    # p-value
    pval_raw = getattr(res, 'pvalues', None)
    pval_ser = to_series(pval_raw)
    if pval_ser is not None and var in pval_ser.index:
        try:
            pval = float(safe_get_from_series(pval_ser, var))
        except Exception:
            pval = np.nan
    else:
        pval = np.nan

    # Confidence interval
    try:
        ci_obj = getattr(res, 'conf_int', None)
        if callable(ci_obj):
            ci_df = ci_obj(alpha=0.05)
        else:
            ci_df = ci_obj
        # ci_df may be DataFrame or ndarray
        if isinstance(ci_df, pd.DataFrame):
            ci_lower = float(ci_df.loc[var, 0])
            ci_upper = float(ci_df.loc[var, 1])
        else:
            # assume ndarray with rows matching params.index
            ci_arr = np.asarray(ci_df)
            idx = list(params.index).index(var)
            ci_lower = float(ci_arr[idx, 0])
            ci_upper = float(ci_arr[idx, 1])
    except Exception:
        if not np.isnan(bse):
            ci_lower = coef - 1.96 * bse
            ci_upper = coef + 1.96 * bse
        else:
            ci_lower = ci_upper = np.nan

    # Determine significance and direction
    alpha = 0.05
    significant = False
    if not np.isnan(pval):
        significant = (pval < alpha)

    if coef < 0:
        direction = "negative"
        conclusion_text = ("Lower student-teacher ratio (fewer students per teacher) is associated with higher average scores"
                           if significant else
                           "Point estimate indicates lower student-teacher ratio is associated with higher average scores, but this effect is not statistically significant at alpha=0.05")
    elif coef > 0:
        direction = "positive"
        conclusion_text = ("Lower student-teacher ratio (fewer students per teacher) is associated with lower average scores"
                           if significant else
                           "Point estimate indicates lower student-teacher ratio is associated with lower average scores, but this effect is not statistically significant at alpha=0.05")
    else:
        direction = "zero"
        conclusion_text = "Estimated effect is (approximately) zero."

    # Normalize numeric values for output
    try:
        coef_rounded = round(coef, 4)
    except Exception:
        coef_rounded = None
    try:
        std_err_out = round(float(bse), 4) if not np.isnan(bse) else None
    except Exception:
        std_err_out = None
    try:
        t_stat_out = round(float(tval), 4) if not np.isnan(tval) else None
    except Exception:
        t_stat_out = None
    try:
        p_value_out = round(float(pval), 4) if not np.isnan(pval) else None
    except Exception:
        p_value_out = None
    try:
        ci_lower_out = round(float(ci_lower), 4) if not np.isnan(ci_lower) else None
        ci_upper_out = round(float(ci_upper), 4) if not np.isnan(ci_upper) else None
    except Exception:
        ci_lower_out = ci_upper_out = None

    # n_obs
    n_obs = None
    try:
        n_obs_attr = getattr(res, 'nobs', None)
        if n_obs_attr is not None:
            n_obs = int(n_obs_attr)
    except Exception:
        n_obs = None

    numeric_output = {
        "variable": var,
        "coef": coef_rounded,
        "std_err": std_err_out,
        "t_stat": t_stat_out,
        "p_value": p_value_out,
        "95%_CI": (ci_lower_out, ci_upper_out),
        "significant_at_0.05": bool(significant),
        "direction": direction,
        "n_obs": n_obs
    }

    # Description: concise interpretation in context
    coef_display = numeric_output['coef'] if numeric_output['coef'] is not None else coef
    std_display = numeric_output['std_err']
    t_display = numeric_output['t_stat']
    p_display = numeric_output['p_value']
    ci_display = numeric_output['95%_CI']

    description = (
        f"Coefficient on StudentTeacherRatio = {coef_display}. "
        f"Interpretation: a one-unit increase in student-teacher ratio is associated with a "
        f"{'increase' if coef>0 else 'decrease' if coef<0 else 'no change'} of {abs(coef_display) if coef_display is not None else abs(coef)} points in AvgScore, "
        f"holding controls and county fixed effects constant. "
        f"Robust SE = {std_display}, t = {t_display}, p = {p_display}, "
        f"95% CI = {ci_display}. "
        f"{'This effect is statistically significant at the 0.05 level.' if numeric_output['significant_at_0.05'] else 'This effect is NOT statistically significant at the 0.05 level.'} "
        f"Conclusion: {conclusion_text}."
    )

    return {"object": numeric_output, "description": description}