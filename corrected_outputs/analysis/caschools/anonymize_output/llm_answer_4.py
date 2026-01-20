def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted
    statsmodels OLS results object (assumed to be returned by the provided model()).

    Returns:
      {
        "object": {
            "coef": float,               # unstandardized coefficient (change in AvgTestScore per 1 unit increase in StudentTeacherRatio)
            "std_err": float,            # robust (HC3) standard error used in the fit
            "t_value": float,
            "p_value": float,
            "ci_95_low": float,
            "ci_95_high": float,
            "std_beta": float,           # standardized coefficient (SD change in AvgTestScore per 1 SD change in StudentTeacherRatio)
            "n_obs": int,
            "significant_at_0_05": bool
        },
        "description": str  # short interpretation w.r.t. the question: "Is a lower student-teacher ratio associated with higher academic performance?"
      }
    """
    import numpy as np

    res = model_output

    # Check that this looks like a statsmodels results object
    if not hasattr(res, "params") or not hasattr(res, "model"):
        return {
            "object": None,
            "description": "Input does not appear to be a statsmodels results object with .params and .model attributes."
        }

    # Ensure the coefficient name exists
    try:
        exog_names = list(res.model.exog_names)
    except Exception:
        # fallback if exog_names not available
        exog_names = list(res.params.index)

    if 'StudentTeacherRatio' not in exog_names:
        return {
            "object": None,
            "description": "Model does not include a predictor named 'StudentTeacherRatio'."
        }

    # Safe access of statistics by name (works whether params is Series or ndarray-like)
    try:
        coef = float(res.params['StudentTeacherRatio'])
        std_err = float(res.bse['StudentTeacherRatio'])
        t_value = float(res.tvalues['StudentTeacherRatio'])
        p_value = float(res.pvalues['StudentTeacherRatio'])
    except Exception:
        # fallback to positional indexing
        idx = exog_names.index('StudentTeacherRatio')
        coef = float(np.asarray(res.params)[idx])
        std_err = float(np.asarray(res.bse)[idx])
        t_value = float(np.asarray(res.tvalues)[idx])
        p_value = float(np.asarray(res.pvalues)[idx])

    # 95% confidence interval
    try:
        ci = res.conf_int(alpha=0.05)
        # ci could be DataFrame or ndarray
        if hasattr(ci, "loc"):
            ci_row = ci.loc['StudentTeacherRatio']
            ci_low = float(ci_row.iloc[0])
            ci_high = float(ci_row.iloc[1])
        else:
            ci_arr = np.asarray(ci)
            idx = exog_names.index('StudentTeacherRatio')
            ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
    except Exception:
        ci_low, ci_high = None, None

    # Standardized coefficient (beta): coef * (sd_x / sd_y)
    std_beta = None
    try:
        y = np.asarray(res.model.endog)
        # find predictor column in model.exog
        idx = exog_names.index('StudentTeacherRatio')
        x = np.asarray(res.model.exog)[:, idx]
        sd_y = np.std(y, ddof=1)
        sd_x = np.std(x, ddof=1)
        if sd_y > 0:
            std_beta = float(coef * (sd_x / sd_y))
    except Exception:
        std_beta = None

    # Number of observations (if available)
    n_obs = None
    try:
        n_obs = int(res.nobs)
    except Exception:
        try:
            n_obs = int(res.model.endog.shape[0])
        except Exception:
            n_obs = None

    significant = (p_value < 0.05) if (p_value is not None) else None

    # Short interpretation relative to the question:
    # StudentTeacherRatio = students per teacher. A negative coef means higher ratio -> lower scores,
    # so LOWER ratio (fewer students per teacher) would be associated with HIGHER scores.
    if coef is not None:
        if coef < 0:
            direction_text = (
                "The estimated coefficient is negative: higher student-teacher ratios (more students per teacher) "
                "are associated with lower AvgTestScore. Therefore, a lower student-teacher ratio (fewer students per teacher) "
                "is associated with higher academic performance."
            )
        elif coef > 0:
            direction_text = (
                "The estimated coefficient is positive: higher student-teacher ratios (more students per teacher) "
                "are associated with higher AvgTestScore. Therefore, a lower student-teacher ratio would be associated with lower academic performance."
            )
        else:
            direction_text = "The estimated coefficient is (nearly) zero, indicating no association between ratio and AvgTestScore in this model."
    else:
        direction_text = "Could not determine direction because coefficient could not be extracted."

    significance_text = (
        f" The effect is statistically {'significant' if significant else 'not significant'} at the 0.05 level (p = {p_value:.3g})."
        if p_value is not None else ""
    )

    desc = (
        f"Estimate for 'StudentTeacherRatio': coefficient={coef:.4g}, SE={std_err:.4g}, t={t_value:.3g}, p={p_value:.3g}, "
        f"95% CI=[{ci_low:.4g}, {ci_high:.4g}] (n={n_obs}). Standardized beta ≈ {std_beta:.4g}."
        + " " + direction_text + significance_text
    )

    # Build object to return
    result_obj = {
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "std_beta": std_beta,
        "n_obs": n_obs,
        "significant_at_0_05": significant
    }

    return {"object": result_obj, "description": desc}