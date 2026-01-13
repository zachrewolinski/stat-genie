def extract_final_answer(model_output):
    """
    Extract statistics for the 'Female' coefficient from a fitted statsmodels results object
    (e.g., GLMResultsWrapper or robustcov results).
    Returns a dict with:
      - "object": dict of numeric results (coef, se, pvalue, 95% CI, odds ratio and its 95% CI, nobs)
      - "description": human-readable interpretation of the Female effect on mortgage approval

    Raises informative errors if expected attributes or the 'Female' parameter are missing.
    """
    import math
    import numpy as np

    # Helper to access attributes robustly
    def _get_attr(obj, name):
        if hasattr(obj, name):
            return getattr(obj, name)
        raise AttributeError(f"Model output has no attribute '{name}'")

    # Extract parameter vector, std errors, p-values, conf int, nobs
    try:
        params = _get_attr(model_output, "params")
        bse = _get_attr(model_output, "bse")
        pvalues = _get_attr(model_output, "pvalues")
        conf = _get_attr(model_output, "conf_int")() if callable(getattr(model_output, "conf_int", None)) else _get_attr(model_output, "conf_int")
    except AttributeError:
        # conf_int might be a method without parentheses in some wrappers; try calling if callable
        try:
            params = model_output.params
            bse = model_output.bse
            pvalues = model_output.pvalues
            conf = model_output.conf_int()
        except Exception as e:
            raise ValueError(f"Could not extract standard regression attributes from model_output: {e}")

    # Ensure params is indexable by name
    try:
        param_index = list(params.index)
    except Exception:
        raise ValueError("model_output.params does not appear to be a pandas Series with an index of parameter names.")

    # Find the parameter name for Female (allow case variants)
    female_key = None
    for name in param_index:
        if str(name).lower() == "female":
            female_key = name
            break
    if female_key is None:
        raise KeyError("Could not find a parameter named 'Female' (case-insensitive) in model_output.params")

    # Extract numeric values (handle conf possibly being array or DataFrame)
    coef = float(params[female_key])
    se = float(bse[female_key]) if female_key in bse.index else float(np.nan)
    pval = float(pvalues[female_key]) if female_key in pvalues.index else float(np.nan)

    # Confidence interval extraction
    try:
        # conf may be DataFrame-like with same index as params
        if hasattr(conf, "loc"):
            ci_lower = float(conf.loc[female_key, 0])
            ci_upper = float(conf.loc[female_key, 1])
        else:
            # conf may be numpy array with same row order as params
            idx = param_index.index(female_key)
            ci_lower = float(conf[idx, 0])
            ci_upper = float(conf[idx, 1])
    except Exception:
        # As a fallback, compute Wald CI from coef +/- 1.96*se if se is available
        if not math.isnan(se):
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            ci_lower = float("nan")
            ci_upper = float("nan")

    # Odds ratio and its CI
    try:
        or_point = float(math.exp(coef))
        or_ci_lower = float(math.exp(ci_lower)) if not math.isnan(ci_lower) else float("nan")
        or_ci_upper = float(math.exp(ci_upper)) if not math.isnan(ci_upper) else float("nan")
    except Exception:
        or_point = float("nan")
        or_ci_lower = float("nan")
        or_ci_upper = float("nan")

    # Number of observations if available
    nobs = getattr(model_output, "nobs", None)
    try:
        if nobs is not None:
            nobs = int(nobs)
    except Exception:
        nobs = None

    # Determine statistical significance at conventional levels
    significance = None
    if not math.isnan(pval):
        if pval < 0.01:
            significance = "p < 0.01"
        elif pval < 0.05:
            significance = "p < 0.05"
        elif pval < 0.1:
            significance = "p < 0.1"
        else:
            significance = f"p = {pval:.3f}"

    # Build the descriptive interpretation
    sign_text = "higher" if coef > 0 else "lower" if coef < 0 else "no difference in"
    pct_change = (or_point - 1.0) * 100 if not math.isnan(or_point) else float("nan")
    description_lines = []
    description_lines.append(f"Coefficient estimate for Female (female = 1 vs male = 0): {coef:.4f}")
    description_lines.append(f"Standard error: {se:.4f}, p-value: {pval:.4g} ({significance})")
    description_lines.append(f"95% CI for log-odds: [{ci_lower:.4f}, {ci_upper:.4f}]")
    description_lines.append(f"Odds ratio (exp(coef)): {or_point:.4f}")
    description_lines.append(f"95% CI for odds ratio: [{or_ci_lower:.4f}, {or_ci_upper:.4f}]")
    description_lines.append(f"Interpretation: Holding included controls constant, being female is associated with {sign_text} odds of mortgage approval.")
    if not math.isnan(pct_change):
        description_lines.append(f"Specifically, the odds change by approximately {pct_change:.1f}% (OR = {or_point:.3f}).")
    if significance is not None:
        if pval < 0.05:
            description_lines.append("This effect is statistically significant at the 5% level.")
        else:
            description_lines.append("This effect is not statistically significant at the 5% level.")
    if nobs is not None:
        description_lines.append(f"Number of observations used in the model: {nobs}")

    description = " ".join(description_lines)

    result_object = {
        "coef": coef,
        "std_err": se,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "odds_ratio": or_point,
        "odds_ratio_ci_lower": or_ci_lower,
        "odds_ratio_ci_upper": or_ci_upper,
        "nobs": nobs,
    }

    return {"object": result_object, "description": description}