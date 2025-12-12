def extract_final_answer(model_output):
    """
    Extracts the coefficient, p-value, 95% CI, standardized effect, sample size, and a short conclusion
    about whether a lower student-teacher ratio is associated with higher academic performance.

    Returns:
      {
        "object": {  # numeric results you can programmatically use
          "coef": float,                # coefficient on StudentTeacherRatio_clipped
          "pvalue": float,              # two-sided p-value
          "ci_2.5": float,              # lower bound of 95% CI
          "ci_97.5": float,             # upper bound of 95% CI
          "std_effect": float or None,  # standardized coefficient (change in SD of AvgScore per 1 SD of ratio)
          "n_obs": int,                 # number of observations used in the regression
          "significant_at_0.05": bool,  # whether pvalue < 0.05
          "direction": "negative|positive|zero"  # sign of coefficient
        },
        "description": str  # plain-language interpretation in context
      }
    """
    var = 'StudentTeacherRatio_clipped'
    results = model_output

    # helper formatters
    def _fmt_num(x, precision=4):
        if x is None:
            return "NA"
        try:
            return f"{x:.{precision}g}"
        except Exception:
            return str(x)

    # Try to obtain parameter names robustly
    params_obj = None
    try:
        params_obj = results.params
    except Exception as e:
        raise ValueError("Provided model_output does not look like a fitted statsmodels results object.") from e

    # Determine parameter names depending on type
    param_names = None
    if hasattr(params_obj, "index"):
        try:
            param_names = list(params_obj.index)
        except Exception:
            param_names = None
    if param_names is None:
        # try model.exog_names
        param_names = getattr(getattr(results, "model", None), "exog_names", None)
        if param_names is None and hasattr(results, "model"):
            # fallback: try to infer from k_params if available
            k = getattr(results, "k_params", None)
            if isinstance(k, int):
                param_names = [f"param_{i}" for i in range(k)]
    if param_names is None:
        # last resort: if params_obj is an ndarray, create index positions
        if hasattr(params_obj, "__len__"):
            try:
                param_names = [str(i) for i in range(len(params_obj))]
            except Exception:
                param_names = []

    if var not in param_names:
        raise KeyError(f"Variable '{var}' not found among model parameters. Available params: {list(param_names)}")

    # Extract coefficient
    try:
        if hasattr(params_obj, "__getitem__") and not hasattr(params_obj, "index"):
            # likely ndarray
            idx = list(param_names).index(var)
            coef = float(params_obj[idx])
        else:
            coef = float(params_obj.loc[var])
    except Exception:
        # last resort attempt
        try:
            coef = float(params_obj[var])
        except Exception as e:
            raise RuntimeError(f"Could not extract coefficient for '{var}'.") from e

    # Extract p-value
    pvalue = None
    try:
        p_obj = results.pvalues
        if hasattr(p_obj, "loc") and var in getattr(p_obj, "index", []):
            pvalue = float(p_obj.loc[var])
        elif hasattr(p_obj, "__getitem__"):
            # handle ndarray or list-like by matching param_names
            try:
                idx = list(param_names).index(var)
                pvalue = float(p_obj[idx])
            except Exception:
                # try direct access
                pvalue = float(p_obj[var])
        else:
            pvalue = float(p_obj)
    except Exception:
        pvalue = None

    # Extract confidence interval
    ci_low, ci_high = None, None
    try:
        ci_obj = results.conf_int()
        # DataFrame-like with index
        if hasattr(ci_obj, "loc") and var in getattr(ci_obj, "index", []):
            row = ci_obj.loc[var]
            ci_low, ci_high = float(row.iloc[0]), float(row.iloc[1])
        else:
            # ndarray-like: find index of var in param_names
            if hasattr(ci_obj, "__len__") and len(ci_obj) == len(param_names):
                idx = list(param_names).index(var)
                row = ci_obj[idx]
                ci_low, ci_high = float(row[0]), float(row[1])
    except Exception:
        ci_low, ci_high = None, None

    # n_obs
    n_obs = None
    try:
        n_obs_val = getattr(results, "nobs", None)
        if n_obs_val is not None:
            try:
                n_obs = int(n_obs_val)
            except Exception:
                # sometimes nobs is array-like
                n_obs = int(n_obs_val[0])
        else:
            # try model.data.endog length
            if hasattr(results, "model") and hasattr(results.model, "data"):
                data_obj = results.model.data
                endog = getattr(data_obj, "endog", None)
                if endog is not None:
                    try:
                        n_obs = int(len(endog))
                    except Exception:
                        n_obs = None
    except Exception:
        n_obs = None

    # Attempt to compute standardized effect (beta): coef * (sd_x / sd_y)
    std_effect = None
    df = None
    try:
        if hasattr(results, "model") and hasattr(results.model, "data"):
            data_obj = results.model.data
            # statsmodels formula API stores the DataFrame as data.frame in many versions
            if hasattr(data_obj, "frame") and getattr(data_obj, "frame") is not None:
                df = data_obj.frame
            else:
                # fallback to endog/exog
                endog = getattr(data_obj, "endog", None)
                exog = getattr(data_obj, "exog", None)
                exog_names = getattr(results.model, "exog_names", None)
                if (endog is not None) and (exog is not None) and (exog_names is not None):
                    import pandas as _pd, numpy as _np
                    y = _pd.Series(endog) if not isinstance(endog, _pd.Series) else endog
                    exog_df = _pd.DataFrame(exog, columns=exog_names)
                    if var in exog_df.columns and hasattr(y, "std"):
                        sd_x = float(exog_df[var].std(ddof=1))
                        sd_y = float(y.std(ddof=1))
                        if sd_x > 0 and sd_y > 0:
                            std_effect = float(coef * (sd_x / sd_y))
    except Exception:
        std_effect = None

    # If df was found earlier (data_obj.frame), compute standardized effect using it
    if std_effect is None:
        try:
            if df is not None and var in df.columns and 'AvgScore' in df.columns:
                sd_x = float(df[var].std(ddof=1))
                sd_y = float(df['AvgScore'].std(ddof=1))
                if sd_x > 0 and sd_y > 0:
                    std_effect = float(coef * (sd_x / sd_y))
        except Exception:
            std_effect = None

    significant = (pvalue is not None) and (pvalue < 0.05)
    if coef < 0:
        direction = "negative"
        if significant:
            conclusion = ("Yes — the estimated coefficient on StudentTeacherRatio_clipped is negative "
                          f"({_fmt_num(coef)}, 95% CI [{_fmt_num(ci_low)}, {_fmt_num(ci_high)}], p = {_fmt_num(pvalue,3)}). "
                          "This implies that a lower student–teacher ratio (fewer students per teacher) is associated "
                          "with higher average academic performance, and the association is statistically significant at α=0.05.")
        else:
            conclusion = ("Weak/no strong evidence — the estimated coefficient on StudentTeacherRatio_clipped is negative "
                          f"({_fmt_num(coef)}, 95% CI [{_fmt_num(ci_low)}, {_fmt_num(ci_high)}], p = {_fmt_num(pvalue,3)}). "
                          "The sign suggests lower student–teacher ratios are associated with higher performance, "
                          "but the association is not statistically significant at α=0.05.")
    elif coef > 0:
        direction = "positive"
        if significant:
            conclusion = ("No — the estimated coefficient on StudentTeacherRatio_clipped is positive "
                          f"({ _fmt_num(coef) }, 95% CI [{_fmt_num(ci_low)}, {_fmt_num(ci_high)}], p = {_fmt_num(pvalue,3)}). "
                          "This implies that higher student–teacher ratios (more students per teacher) are associated "
                          "with higher average academic performance, and the association is statistically significant at α=0.05. "
                          "This is opposite to the hypothesis that lower ratios improve performance.")
        else:
            conclusion = ("Weak/no strong evidence — the estimated coefficient on StudentTeacherRatio_clipped is positive "
                          f"({ _fmt_num(coef) }, 95% CI [{_fmt_num(ci_low)}, {_fmt_num(ci_high)}], p = {_fmt_num(pvalue,3)}). "
                          "The sign suggests higher ratios are associated with better performance, but the association is not statistically significant at α=0.05.")
    else:
        direction = "zero"
        conclusion = ("No effect — the estimated coefficient is essentially zero "
                      f"({_fmt_num(coef)}), p = {_fmt_num(pvalue,3)}.")

    # Assemble object to return
    obj = {
        "coef": coef,
        "pvalue": pvalue,
        "ci_2.5": ci_low,
        "ci_97.5": ci_high,
        "std_effect": std_effect,
        "n_obs": n_obs,
        "significant_at_0.05": significant,
        "direction": direction
    }

    description = (
        "Extracted results for StudentTeacherRatio_clipped from the fitted model. "
        "Interpretation: the coefficient gives the change in AvgScore for a one-unit increase in student–teacher ratio "
        "(students per teacher). A negative coefficient means that lower ratios (fewer students per teacher) are associated with higher AvgScore. "
        f"Summary conclusion: {conclusion} "
        + (f"Standardized effect (change in SD of AvgScore per 1 SD of ratio): {_fmt_num(std_effect)}." if std_effect is not None else "Standardized effect could not be computed from the model object.")
    )

    return {"object": obj, "description": description}