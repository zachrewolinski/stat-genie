def extract_final_answer(model_output):
    """
    Extracts the effect of the 'female' indicator from a fitted logistic model result (or the RobustResultsLike wrapper).
    Returns a dict with keys "object" and "description".
    - "object" is a dictionary with numeric results (coef, se, pvalue, 95% CI, odds ratio and its CI, significance flag).
    - "description" is a short plain-language interpretation of the effect of being female on mortgage approval.
    """
    import numpy as np
    import pandas as pd
    from math import exp

    # Helper to safely get attribute (Series or array) from the result object
    def safe_get(attr_name):
        if hasattr(model_output, attr_name):
            return getattr(model_output, attr_name)
        return None

    params = safe_get('params')
    bse = safe_get('bse')
    pvalues = safe_get('pvalues')

    # conf_int may be a method or attribute
    conf_int = None
    if hasattr(model_output, 'conf_int'):
        try:
            conf_int = model_output.conf_int()
        except TypeError:
            # conf_int might be a property or raise unexpectedly
            try:
                conf_int = getattr(model_output, 'conf_int')
            except Exception:
                conf_int = None

    # Normalize params/bse/pvalues to pandas.Series if possible
    def to_series(x):
        if x is None:
            return None
        if isinstance(x, pd.Series):
            return x
        try:
            # If x is array-like and the model provides index names use them
            if hasattr(model_output, 'params') and isinstance(model_output.params, pd.Series):
                idx = model_output.params.index
                return pd.Series(np.asarray(x), index=idx)
        except Exception:
            pass
        try:
            return pd.Series(x)
        except Exception:
            return None

    params_s = to_series(params)
    bse_s = to_series(bse)
    pvalues_s = to_series(pvalues)

    # Check presence of 'female' in parameters
    if params_s is None or 'female' not in params_s.index:
        return {
            "object": None,
            "description": "The fitted model output does not contain a parameter named 'female'; cannot extract effect."
        }

    coef = float(params_s.loc['female'])
    se = float(bse_s.loc['female']) if (bse_s is not None and 'female' in bse_s.index) else None
    pval = float(pvalues_s.loc['female']) if (pvalues_s is not None and 'female' in pvalues_s.index) else None

    # Extract confidence interval for female
    ci_lower = ci_upper = None
    if conf_int is not None:
        # conf_int might be DataFrame, ndarray, or Series-like
        try:
            if isinstance(conf_int, pd.DataFrame):
                if 'female' in conf_int.index:
                    row = conf_int.loc['female']
                    # Some conf_int DataFrames have columns [0,1] or ['2.5%','97.5%']
                    ci_lower, ci_upper = float(row.iloc[0]), float(row.iloc[1])
                else:
                    # try positional matching based on param index
                    if params_s is not None:
                        pos = list(params_s.index).index('female')
                        ci_lower, ci_upper = float(conf_int.iloc[pos, 0]), float(conf_int.iloc[pos, 1])
            else:
                # try numpy array-like
                arr = np.asarray(conf_int)
                if arr.ndim == 2:
                    if params_s is not None and hasattr(params_s, "index"):
                        pos = list(params_s.index).index('female')
                        ci_lower, ci_upper = float(arr[pos, 0]), float(arr[pos, 1])
                    else:
                        # fallback: assume female is at same position as in params_s
                        pos = list(params_s.index).index('female') if params_s is not None else 0
                        ci_lower, ci_upper = float(arr[pos, 0]), float(arr[pos, 1])
        except Exception:
            ci_lower = ci_upper = None

    # Compute odds ratio and CI (if CI exists)
    odds_ratio = float(np.exp(coef))
    or_ci = None
    if ci_lower is not None and ci_upper is not None:
        or_ci = (float(np.exp(ci_lower)), float(np.exp(ci_upper)))

    significant = (pval is not None) and (pval < 0.05)

    # Prepare numeric object to return
    result_obj = {
        "coef_female_log_odds": coef,
        "se_female": se,
        "pvalue_female": pval,
        "ci95_female_log_odds": (ci_lower, ci_upper) if (ci_lower is not None and ci_upper is not None) else None,
        "odds_ratio_female": odds_ratio,
        "ci95_odds_ratio_female": or_ci,
        "significant_at_0.05": bool(significant)
    }

    # Build human-readable description
    # Interpret percent change in odds
    pct_change = (odds_ratio - 1.0) * 100.0
    pct_str = f"{pct_change:.1f}%"
    if pct_change >= 0:
        effect_dir = f"increase of {pct_str} in the odds"
    else:
        effect_dir = f"decrease of {abs(pct_change):.1f}% in the odds"

    # Compose the description with available stats
    desc_parts = []
    desc_parts.append("Estimated effect of being female on mortgage approval (log-odds): "
                      f"{coef:.4f}")
    if se is not None:
        desc_parts.append(f"(SE = {se:.4f})")
    if pval is not None:
        desc_parts.append(f"p = {pval:.3g}")
    if (ci_lower is not None and ci_upper is not None):
        desc_parts.append(f"95% CI (log-odds) = [{ci_lower:.4f}, {ci_upper:.4f}]")
    desc_parts = [" ".join(desc_parts)]

    desc_parts.append(f"In odds-ratio terms, being female is associated with an {effect_dir} of mortgage approval "
                      f"(OR = {odds_ratio:.3f}" +
                      (f", 95% CI = [{or_ci[0]:.3f}, {or_ci[1]:.3f}]" if or_ci is not None else "") +
                      ").")

    if significant:
        desc_parts.append("This effect is statistically significant at the 0.05 level.")
    else:
        desc_parts.append("This effect is NOT statistically significant at the 0.05 level.")

    description = " ".join(desc_parts)

    return {"object": result_obj, "description": description}