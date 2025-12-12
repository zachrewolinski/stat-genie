def extract_final_answer(model_output):
    """
    Extract statistics about the effect of StudentTeacherRatio on AvgScore from a fitted
    statsmodels OLSResults (or compatible) object.

    Returns a dict with keys:
      - "object": a dict with numeric results:
            {
              "coefficient": float,
              "std_error": float,
              "t_stat": float,
              "p_value": float,
              "ci_lower": float,
              "ci_upper": float,
              "standardized_coefficient": float or None,
              "significant_at_0.05": bool
            }
      - "description": a short human-readable interpretation of the coefficient in context.
    """
    import numpy as np

    # Check that model_output looks like a statsmodels results object
    if model_output is None:
        raise ValueError("model_output is None")

    # Determine the parameter name/index for StudentTeacherRatio
    param_name = 'StudentTeacherRatio'
    # Try to get parameter values by name first
    params = getattr(model_output, 'params', None)
    pvalues = getattr(model_output, 'pvalues', None)
    bse = getattr(model_output, 'bse', None)
    tvalues = getattr(model_output, 'tvalues', None)

    if params is None:
        raise ValueError("The provided model_output does not have 'params' attribute.")

    # helper to extract by name or by index
    def _get_by_name_or_index(obj, name, fallback_index):
        try:
            # handles pandas Series with labels
            return obj[name]
        except Exception:
            # handle numpy array-like
            try:
                return obj[fallback_index]
            except Exception:
                raise KeyError(f"Could not extract '{name}' from the model output.")

    # find index of StudentTeacherRatio in model exog names
    exog_names = None
    try:
        exog_names = list(model_output.model.exog_names)
    except Exception:
        # fallback: try results.param index if it's an Index
        try:
            exog_names = list(params.index)
        except Exception:
            exog_names = None

    if exog_names is not None and param_name in exog_names:
        idx = exog_names.index(param_name)
    else:
        # if we cannot find by name, try to see if params has the name
        try:
            _ = params[param_name]
            idx = list(params.index).index(param_name)
        except Exception:
            raise KeyError(f"Variable '{param_name}' not found among model parameters. "
                           f"Available parameters: {exog_names if exog_names is not None else list(params.index)}")

    # Extract coefficient, std error, t-stat, p-value
    coef = float(_get_by_name_or_index(params, param_name, idx))
    se = float(_get_by_name_or_index(bse, param_name, idx)) if bse is not None else None
    tstat = float(_get_by_name_or_index(tvalues, param_name, idx)) if tvalues is not None else None
    pval = float(_get_by_name_or_index(pvalues, param_name, idx)) if pvalues is not None else None

    # Confidence interval (uses the model's conf_int if available)
    try:
        ci = model_output.conf_int(alpha=0.05)
        # conf_int may be a DataFrame/ndarray
        if hasattr(ci, 'loc'):
            ci_row = ci.loc[param_name].values
        else:
            ci_row = ci[idx]
        ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Attempt to compute a standardized coefficient (beta) if data are accessible
    std_coef = None
    try:
        # model.endog and model.exog are numpy arrays in the same order as params/exog_names
        y = model_output.model.endog
        X = model_output.model.exog
        # find index in exog that corresponds to StudentTeacherRatio
        # (exog includes intercept if present)
        x_col = X[:, idx]
        sd_y = np.std(y, ddof=1)
        sd_x = np.std(x_col, ddof=1)
        if sd_y > 0:
            std_coef = float(coef * (sd_x / sd_y))
        else:
            std_coef = None
    except Exception:
        std_coef = None

    significant = (pval is not None) and (pval < 0.05)

    result_object = {
        "coefficient": coef,
        "std_error": se,
        "t_stat": tstat,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standardized_coefficient": std_coef,
        "significant_at_0.05": significant
    }

    # Build interpretation string
    direction = "positive" if coef > 0 else ("negative" if coef < 0 else "zero")
    significance_text = ("statistically significant (p < 0.05)" if significant
                         else "not statistically significant at the 0.05 level")
    ci_text = (f"95% CI [{ci_lower:.3f}, {ci_upper:.3f}]" if (ci_lower is not None and ci_upper is not None)
               else "95% CI not available")

    std_text = (f"Standardized coefficient = {std_coef:.3f}. "
                if std_coef is not None else "")

    description = (
        f"The estimated effect of StudentTeacherRatio on AvgScore is {coef:.3f} (SE = {se:.3f}). "
        f"This is {direction} and {significance_text}. {ci_text}. "
        f"{std_text}"
        f"Interpretation: holding controls constant, a one-unit increase in StudentTeacherRatio "
        f"(one more student per teacher) is associated with a {coef:.3f} point "
        f"change in AvgScore on average. "
    )

    return {"object": result_object, "description": description}