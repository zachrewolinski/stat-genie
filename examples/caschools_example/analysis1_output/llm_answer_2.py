def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a statsmodels
    RegressionResultsWrapper and returns an interpretable summary.

    Returns:
      {
        "object": {
            "coef": float,
            "std_error": float,
            "t_value": float,
            "p_value": float,
            "ci_lower": float,
            "ci_upper": float,
            "std_effect": float  # standardized (beta) coefficient
        },
        "description": str  # short interpretation about whether lower ratio -> higher performance
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output

    target = 'StudentTeacherRatio'

    # Basic checks
    if not hasattr(res, 'params'):
        return {
            "object": None,
            "description": "The provided model_output does not appear to be a fitted statsmodels results object (missing .params)."
        }

    params = res.params
    if target not in params.index:
        return {
            "object": None,
            "description": f"The model does not contain a parameter named '{target}'. Available params: {list(params.index)}"
        }

    # Extract coefficient, se, t, p
    coef = float(params[target])
    # Some res objects store bse/pvalues as Series; handle accordingly
    try:
        se = float(res.bse[target])
    except Exception:
        se = float(np.nan)
    try:
        tval = float(res.tvalues[target])
    except Exception:
        tval = float(np.nan)
    try:
        pval = float(res.pvalues[target])
    except Exception:
        pval = float(np.nan)

    # Confidence interval (build DataFrame for safe indexing)
    try:
        ci_array = res.conf_int(alpha=0.05)
        ci_df = pd.DataFrame(ci_array, index=res.params.index, columns=['ci_lower', 'ci_upper'])
        ci_lower = float(ci_df.loc[target, 'ci_lower'])
        ci_upper = float(ci_df.loc[target, 'ci_upper'])
    except Exception:
        ci_lower = ci_upper = float(np.nan)

    # Standardized effect (beta): coef * (sd(X) / sd(Y))
    std_effect = float(np.nan)
    try:
        exog = np.asarray(res.model.exog)
        endog = np.asarray(res.model.endog)
        exog_names = list(res.model.exog_names)
        if target in exog_names:
            idx = exog_names.index(target)
            x_col = exog[:, idx]
            # Use sample standard deviation (ddof=1)
            sd_x = np.std(x_col, ddof=1)
            sd_y = np.std(endog, ddof=1)
            if sd_y != 0:
                std_effect = float(coef * (sd_x / sd_y))
    except Exception:
        std_effect = float(np.nan)

    # Interpretation / conclusion regarding the question:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    # Note: StudentTeacherRatio is defined as students / teachers. A negative coef implies
    # that higher ratio -> lower scores, so lower ratio -> higher scores.
    conclusion = ""
    sig_level = 0.05
    if np.isfinite(pval):
        if pval < sig_level:
            if coef < 0:
                conclusion = (
                    "Yes — the coefficient on StudentTeacherRatio is negative and statistically significant "
                    f"(coef = {coef:.4f}, p = {pval:.3g}). This implies that a lower student-teacher ratio "
                    "(fewer students per teacher) is associated with higher average academic performance. "
                    f"The 95% CI for the effect is [{ci_lower:.4f}, {ci_upper:.4f}]. "
                    f"Standardized effect (beta) ≈ {std_effect:.4f}."
                )
            else:
                conclusion = (
                    "No — the coefficient on StudentTeacherRatio is positive and statistically significant "
                    f"(coef = {coef:.4f}, p = {pval:.3g}). This implies that a lower student-teacher ratio "
                    "is associated with lower average academic performance (opposite direction). "
                    f"The 95% CI is [{ci_lower:.4f}, {ci_upper:.4f}]. "
                    f"Standardized effect (beta) ≈ {std_effect:.4f}."
                )
        else:
            conclusion = (
                "No strong evidence — the coefficient on StudentTeacherRatio is not statistically significant "
                f"(coef = {coef:.4f}, p = {pval:.3g}). We cannot conclude that lower student-teacher ratio is "
                "associated with higher academic performance based on this model. "
                f"The 95% CI is [{ci_lower:.4f}, {ci_upper:.4f}] (contains zero). "
                f"Standardized effect (beta) ≈ {std_effect:.4f}."
            )
    else:
        conclusion = "Could not determine statistical significance (p-value unavailable)."

    extracted = {
        "coef": coef,
        "std_error": se,
        "t_value": tval,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_effect": std_effect,
        "note": "coef units = change in AvgScore per 1 additional student per teacher"
    }

    return {
        "object": extracted,
        "description": conclusion
    }