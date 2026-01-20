def extract_final_answer(model_output):
    """
    Extracts the effect of the 'IsHuman' predictor from a fitted statsmodels GEE results object.
    Returns a dictionary with keys:
      - "object": dict with numeric extracted statistics
      - "description": human-readable interpretation of the IsHuman effect
    
    Expected input: a statsmodels GEEResultsWrapper (or similar) from model.fit().
    """
    import numpy as np
    res = model_output

    # Ensure the results object has parameter information
    if not hasattr(res, "params"):
        raise ValueError("The provided model_output does not contain .params. "
                         "Provide a fitted statsmodels results object (e.g., GEEResultsWrapper).")

    # Find the parameter name corresponding to IsHuman (robust to slight naming differences)
    param_index = list(res.params.index)
    if "IsHuman" in param_index:
        coef_name = "IsHuman"
    else:
        # fallback: any parameter that starts with 'IsHuman'
        matches = [n for n in param_index if n.startswith("IsHuman")]
        if len(matches) == 0:
            raise ValueError(f"Could not find an 'IsHuman' parameter in model parameters: {param_index}")
        coef_name = matches[0]

    # Extract core statistics
    coef = float(res.params[coef_name])
    se = float(res.bse[coef_name]) if hasattr(res, "bse") else None
    pval = float(res.pvalues[coef_name]) if hasattr(res, "pvalues") else None

    z = float(coef / se) if (se is not None and se != 0) else None

    # 95% CI on log-odds scale (if available)
    try:
        ci_df = res.conf_int()
        # conf_int may return a DataFrame or ndarray-like; handle both
        if hasattr(ci_df, "loc"):
            ci_low, ci_high = float(ci_df.loc[coef_name].iloc[0]), float(ci_df.loc[coef_name].iloc[1])
        else:
            # assume same ordering as params
            idx = param_index.index(coef_name)
            ci_low, ci_high = float(ci_df[idx, 0]), float(ci_df[idx, 1])
        ci = [ci_low, ci_high]
    except Exception:
        ci = None

    # Odds ratio and its CI (exponentiate log-odds scale)
    odds_ratio = float(np.exp(coef))
    or_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))] if ci is not None else None

    # Formulate a concise interpretation regarding whether humans have higher AMTL
    alpha = 0.05
    if pval is None:
        interpretation = "Could not determine statistical significance: p-value is not available."
    else:
        if pval < alpha:
            if coef > 0:
                interpretation = (
                    "Yes — being human is associated with significantly higher odds of antemortem tooth loss. "
                    f"(coef = {coef:.4f} on log-odds scale; OR = {odds_ratio:.3f}; "
                    f"95% CI OR = {or_ci[0]:.3f}–{or_ci[1]:.3f}; p = {pval:.3g})"
                )
            else:
                interpretation = (
                    "Being human is associated with significantly lower odds of antemortem tooth loss. "
                    f"(coef = {coef:.4f}; OR = {odds_ratio:.3f}; "
                    f"95% CI OR = {or_ci[0]:.3f}–{or_ci[1]:.3f}; p = {pval:.3g})"
                )
        else:
            if coef > 0:
                interpretation = (
                    "No statistically significant evidence that humans have higher AMTL; "
                    "there is a non-significant positive association (trend) favoring higher AMTL in humans. "
                    f"(coef = {coef:.4f}; OR = {odds_ratio:.3f}; p = {pval:.3g})"
                )
            else:
                interpretation = (
                    "No statistically significant evidence that humans have higher AMTL; "
                    "there is a non-significant trend toward lower AMTL in humans. "
                    f"(coef = {coef:.4f}; OR = {odds_ratio:.3f}; p = {pval:.3g})"
                )

    # Prepare the returned object with numeric values for downstream programmatic use
    returned_object = {
        "coef_name": coef_name,
        "coef_log_odds": coef,
        "se": se,
        "z": z,
        "p_value": pval,
        "ci_log_odds_95": ci,
        "odds_ratio": odds_ratio,
        "odds_ratio_ci_95": or_ci,
    }

    description = (
        "Extracted statistics for the IsHuman predictor from the fitted GEE model. "
        + interpretation
        + " Returned fields: coef (log-odds), se, z, p_value, 95% CI on the log-odds scale, odds_ratio and its 95% CI."
    )

    return {"object": returned_object, "description": description}