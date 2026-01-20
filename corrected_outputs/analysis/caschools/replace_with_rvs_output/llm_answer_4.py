def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted statsmodels OLS result.
    Returns a dict with keys:
      - "object": a dict of numeric results (coef, se, t, p, 95% CI, significance, standardized coef if available)
      - "description": human-readable interpretation of the effect in context
    
    Expects model_output to be a statsmodels RegressionResultsWrapper (as returned by smf.ols(...).fit()).
    """
    import numpy as np

    res = model_output
    var = 'StudentTeacherRatio'

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    if var not in res.params.index:
        raise ValueError(f"Variable '{var}' not found in the model parameters. Available params: {list(res.params.index)}")

    # Extract core statistics
    coef = float(res.params[var])
    # Some attributes (bse, tvalues, pvalues) should exist in typical statsmodels results
    se = float(res.bse[var]) if (hasattr(res, "bse") and var in res.bse.index) else None
    t_stat = float(res.tvalues[var]) if (hasattr(res, "tvalues") and var in res.tvalues.index) else None
    p_value = float(res.pvalues[var]) if (hasattr(res, "pvalues") and var in res.pvalues.index) else None

    # 95% confidence interval
    try:
        ci = res.conf_int(alpha=0.05).loc[var].tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower = ci_upper = None

    # Significance at alpha=0.05 (if p-value available)
    significant = None
    if p_value is not None:
        significant = (p_value < 0.05)

    # Direction: interpret sign in context (StudentTeacherRatio = students per teacher).
    # A negative coefficient means higher ratio (more students per teacher) is associated with lower test scores,
    # so a lower ratio (fewer students per teacher = smaller classes) is associated with higher test scores.
    if coef < 0:
        direction = "negative (lower student-teacher ratio -> higher AvgTestScore)"
    elif coef > 0:
        direction = "positive (lower student-teacher ratio -> lower AvgTestScore)"
    else:
        direction = "zero (no association)"

    # Attempt to compute a standardized (beta) coefficient if the original DataFrame is available.
    std_coef = None
    try:
        # Try to get the DataFrame used in fitting
        df = None
        if hasattr(res.model, "data"):
            # statsmodels stores data in different attributes; try common ones
            if hasattr(res.model.data, "frame") and res.model.data.frame is not None:
                df = res.model.data.frame
            elif hasattr(res.model.data, "orig_exog") and hasattr(res.model.data, "orig_endog"):
                # build minimal df if possible
                import pandas as _pd
                exog = res.model.data.orig_exog
                endog = res.model.data.orig_endog
                # only proceed if exog is a DataFrame with named columns
                if hasattr(exog, "columns"):
                    df = _pd.DataFrame(exog, columns=exog.columns)
                    # attach endog if name available
                    if hasattr(res.model, "endog_names"):
                        df[res.model.endog_names] = endog
        # If DataFrame available and contains needed columns, compute standardized coef
        if df is not None and 'AvgTestScore' in df.columns and var in df.columns:
            sd_x = float(df[var].std(ddof=0))
            sd_y = float(df['AvgTestScore'].std(ddof=0))
            if sd_y != 0:
                std_coef = float(coef * (sd_x / sd_y))
    except Exception:
        std_coef = None

    # Build the object to return
    object_dict = {
        "variable": var,
        "coef": coef,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_at_0.05": significant,
        "direction_text": direction,
        "standardized_coef": std_coef
    }

    # Build a concise description interpreting the result in the context of the question.
    # Note: interpretation uses the sign and significance above.
    sign_word = "statistically significant" if significant else "not statistically significant" if significant is not None else "significance unknown"
    desc_lines = [
        f"Estimated effect of StudentTeacherRatio on AvgTestScore: coefficient = {coef:.4f}",
    ]
    if se is not None:
        desc_lines.append(f"(SE = {se:.4f})")
    if ci_lower is not None and ci_upper is not None:
        desc_lines.append(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    if p_value is not None:
        desc_lines.append(f"p-value = {p_value:.3g} -> {sign_word}")
    desc_lines.append(f"Interpretation: {direction}.")
    if std_coef is not None:
        desc_lines.append(f"Standardized effect (beta) ≈ {std_coef:.4f} (if original data frame available).")

    description = " ".join(desc_lines)

    return {"object": object_dict, "description": description}