def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, 95% CI, sample size, and a short
    interpretation regarding whether a lower student-teacher ratio (fewer students per teacher)
    is associated with higher average academic performance.

    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, pvalue, ci_lower, ci_upper, nobs, significant)
      - "description": short textual interpretation in the context of the task
    """
    import numpy as np

    var = 'StudentsPerTeacher'

    # Basic checks
    if not hasattr(model_output, 'params'):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    params = model_output.params
    if var not in params.index:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    # Extract estimates
    coef = float(params.loc[var])
    # robust se should already be in model_output.bse when fit(..., cov_type=...)
    se = float(model_output.bse.loc[var]) if hasattr(model_output, 'bse') else float(np.nan)
    pval = float(model_output.pvalues.loc[var]) if hasattr(model_output, 'pvalues') else float(np.nan)

    # 95% confidence interval (try built-in, fall back to t-based if needed)
    try:
        ci = model_output.conf_int().loc[var].tolist()
    except Exception:
        # fallback: use t-critical and bse (requires df_resid)
        try:
            from scipy import stats
            df_resid = float(model_output.df_resid)
            tcrit = stats.t.ppf(1 - 0.025, df_resid)
            ci = [coef - tcrit * se, coef + tcrit * se]
        except Exception:
            ci = [float(np.nan), float(np.nan)]

    # sample size
    try:
        nobs = int(getattr(model_output, 'nobs', model_output.model.nobs))
    except Exception:
        nobs = None

    # Significance decision (conventional 0.05 level)
    significant = (pval < 0.05) if not np.isnan(pval) else False

    # Interpret direction relative to the research question:
    # - StudentsPerTeacher: higher value = more students per teacher.
    # A negative coefficient implies that fewer students per teacher (lower ratio) is associated with higher AvgScore.
    if significant:
        if coef < 0:
            decision_text = (
                "Yes — statistically significant negative association: a higher StudentsPerTeacher is associated "
                "with LOWER AvgScore, so a lower student-teacher ratio (fewer students per teacher) is associated "
                "with HIGHER academic performance."
            )
        else:
            decision_text = (
                "No — statistically significant positive association: a higher StudentsPerTeacher is associated "
                "with HIGHER AvgScore (opposite of the hypothesized direction)."
            )
    else:
        decision_text = (
            "No strong evidence of an association at conventional significance levels (p >= 0.05). "
            "The estimated effect is not statistically significant."
        )

    # Pack results
    result_object = {
        "coef": coef,
        "se": se,
        "pvalue": pval,
        "ci_lower": float(ci[0]) if ci and len(ci) >= 1 else None,
        "ci_upper": float(ci[1]) if ci and len(ci) >= 2 else None,
        "nobs": nobs,
        "significant": bool(significant),
        "variable": var
    }

    description = (
        f"StudentsPerTeacher coefficient = {coef:.6g}, SE = {se:.6g}, p-value = {pval:.6g}, "
        f"95% CI = [{result_object['ci_lower']:.6g}, {result_object['ci_upper']:.6g}]. "
        f"{decision_text}"
    )

    return {"object": result_object, "description": description}