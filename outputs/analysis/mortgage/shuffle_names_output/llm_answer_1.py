def extract_final_answer(model_output):
    """
    Extracts the estimated effect of 'Female' from a fitted model result object.
    Returns a dictionary with keys:
      - "object": a dict with numeric results (coef, std_err, p_value, 95% CI on log-odds,
                  odds ratio and its 95% CI, booleans for whether effect was estimated and
                  whether it is statistically significant at alpha=0.05).
      - "description": a short human-readable interpretation of the result in context.
    The function is defensive and handles:
      - statsmodels-like results (params, bse, pvalues, conf_int or cov_params)
      - the custom ResultWrapper/ResultCompat used when 'Female' was constant (contains NaN placeholder)
      - other wrappers exposing params/pvalues/bse or conf_int
    """
    import math
    import numpy as np
    import pandas as pd

    def is_nan(x):
        try:
            return pd.isna(x)
        except Exception:
            try:
                return math.isnan(x)
            except Exception:
                return False

    def to_float_or_none(x):
        if x is None:
            return None
        if is_nan(x):
            return None
        try:
            return float(x)
        except Exception:
            return None

    # Try to get a result-like object that exposes params, pvalues, bse, conf_int or cov_params
    res = model_output

    # If the wrapper explicitly signals female was constant, note that but still try to read placeholders
    female_constant_flag = False
    try:
        female_constant_flag = bool(getattr(res, "female_constant", False))
    except Exception:
        female_constant_flag = False

    # Try to access params/pvalues/bse
    params = None
    pvalues = None
    bse = None
    conf_int_df = None

    # Helper to safely get attribute
    def safe_attr(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            return None

    params = safe_attr(res, "params")
    pvalues = safe_attr(res, "pvalues")
    bse = safe_attr(res, "bse")

    # If conf_int method exists, try it
    conf_int = None
    try:
        conf_int = res.conf_int()
        conf_int_df = conf_int
    except Exception:
        conf_int_df = None

    # If params not found directly, try to unwrap common wrappers
    if params is None:
        # try ._results, ._base, .model or .results
        for attr in ("_results", "_base", "results", "model", "base_result", "_wrapped"):
            candidate = safe_attr(res, attr)
            if candidate is not None:
                params = safe_attr(candidate, "params") or params
                pvalues = safe_attr(candidate, "pvalues") or pvalues
                bse = safe_attr(candidate, "bse") or bse
                if conf_int_df is None:
                    try:
                        conf_int_df = candidate.conf_int()
                    except Exception:
                        pass
                if params is not None:
                    break

    # Ensure params is something indexable (prefer pandas Series)
    param_index = None
    try:
        param_index = list(params.index) if params is not None and hasattr(params, "index") else None
    except Exception:
        param_index = None

    # Locate the parameter name for female (prefer exact 'Female', fallback case-insensitive contains)
    female_param_name = None
    if param_index is not None:
        if "Female" in param_index:
            female_param_name = "Female"
        else:
            # try case-insensitive match or substring match
            lowered = [str(n).lower() for n in param_index]
            for i, name in enumerate(lowered):
                if "female" == name or "female" in name:
                    female_param_name = list(param_index)[i]
                    break

    # If we still don't have params or no Female parameter, construct a safe response
    if params is None or female_param_name is None:
        # If wrapper explicitly said female was constant, produce that message
        if female_constant_flag:
            description = ("The model did not estimate an effect for 'Female' because 'Female' "
                           "was constant (no variation) in the input data. No coefficient, "
                           "standard error, or p-value are available.")
            obj = {
                "coef": None,
                "std_err": None,
                "p_value": None,
                "ci_lower": None,
                "ci_upper": None,
                "odds_ratio": None,
                "odds_ratio_ci_lower": None,
                "odds_ratio_ci_upper": None,
                "estimated": False,
                "significant": None,
                "note": "Female was constant / not estimable"
            }
            return {"object": obj, "description": description}
        else:
            description = ("Could not find an estimated parameter named 'Female' in the provided "
                           "model output. The model output must expose a parameter named 'Female' "
                           "(or similar) to extract the effect.")
            obj = {
                "coef": None,
                "std_err": None,
                "p_value": None,
                "ci_lower": None,
                "ci_upper": None,
                "odds_ratio": None,
                "odds_ratio_ci_lower": None,
                "odds_ratio_ci_upper": None,
                "estimated": False,
                "significant": None,
                "note": "Female parameter not found"
            }
            return {"object": obj, "description": description}

    # Extract numeric values for the female parameter if available
    raw_coef = None
    raw_p = None
    raw_bse = None
    try:
        raw_coef = params.loc[female_param_name]
    except Exception:
        try:
            raw_coef = params[female_param_name]
        except Exception:
            raw_coef = None

    if pvalues is not None:
        try:
            raw_p = pvalues.loc[female_param_name]
        except Exception:
            try:
                raw_p = pvalues[female_param_name]
            except Exception:
                raw_p = None

    if bse is not None:
        try:
            raw_bse = bse.loc[female_param_name]
        except Exception:
            try:
                raw_bse = bse[female_param_name]
            except Exception:
                raw_bse = None

    coef = to_float_or_none(raw_coef)
    pval = to_float_or_none(raw_p)
    std_err = to_float_or_none(raw_bse)

    # Determine 95% CI on log-odds scale
    ci_lower = None
    ci_upper = None
    # Prefer conf_int_df if available
    if conf_int_df is not None:
        try:
            # conf_int_df may be a DataFrame with rows indexed by param names
            if hasattr(conf_int_df, "loc"):
                if female_param_name in conf_int_df.index:
                    row = conf_int_df.loc[female_param_name]
                    # some conf_int return 2-column arrays
                    if len(row) >= 2:
                        ci_lower = to_float_or_none(row[0])
                        ci_upper = to_float_or_none(row[1])
            else:
                # conf_int_df may be a numpy array if no index; fall back to analytic method
                pass
        except Exception:
            pass

    # If conf_int not available, compute from std_err if possible
    if (ci_lower is None or ci_upper is None) and coef is not None and std_err is not None:
        z = 1.96
        ci_lower = coef - z * std_err
        ci_upper = coef + z * std_err

    # Compute odds ratio and its CI (if coef present)
    odds_ratio = None
    or_ci_lower = None
    or_ci_upper = None
    if coef is not None:
        try:
            odds_ratio = float(np.exp(coef))
        except Exception:
            odds_ratio = None
    if ci_lower is not None and ci_upper is not None:
        try:
            or_ci_lower = float(np.exp(ci_lower))
            or_ci_upper = float(np.exp(ci_upper))
        except Exception:
            or_ci_lower = None
            or_ci_upper = None

    # Determine estimated and significance
    estimated = coef is not None
    significant = None
    if pval is not None:
        significant = bool(pval < 0.05)

    # If the coefficient is None but female_constant_flag True, mark accordingly
    if female_constant_flag and not estimated:
        description = ("'Female' was constant in the data and thus its effect was not estimable. "
                       "No numeric estimate available.")
    else:
        # Compose a concise interpretation
        if not estimated:
            description = ("No numeric estimate for 'Female' could be extracted from the model output.")
        else:
            # Interpret direction: positive coef -> higher log-odds (and thus higher probability) for females
            direction = "higher" if coef > 0 else ("lower" if coef < 0 else "no difference")
            sign_text = "statistically significant" if significant else "not statistically significant"
            # Note that coefficient is on log-odds scale (logistic regression)
            description = (f"Estimated effect of being female on mortgage approval (logistic regression): "
                           f"coefficient (log-odds) = {coef:.6g}. This implies an odds ratio = "
                           f"{odds_ratio:.6g} (95% CI [{or_ci_lower:.6g}, {or_ci_upper:.6g}] if available). "
                           f"The effect direction is {direction}. The p-value = "
                           f"{pval:.4g} ({sign_text}).")
            # If CI or odds ratio not available, simplify message
            if or_ci_lower is None or or_ci_upper is None:
                description = (f"Estimated effect of being female on mortgage approval (log-odds) = {coef:.6g}; "
                               f"p-value = {pval:.4g}. (Odds ratio or CI not available.)")

    # Build object dict (JSON-serializable friendly)
    obj = {
        "coef": coef,
        "std_err": std_err,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_lower": or_ci_lower,
        "odds_ratio_ci_upper": or_ci_upper,
        "estimated": bool(estimated),
        "significant": (bool(significant) if significant is not None else None),
        "param_name": female_param_name
    }

    # If female was explicitly constant, include that note
    if female_constant_flag:
        obj["note"] = "Female was constant in the input; placeholder NaN values may be present."

    return {"object": obj, "description": description}