def extract_final_answer(model_output):
    """
    Extract interpretable statistics from a fitted statsmodels GLM results object
    for a rate model where log(Hours) was used as an offset (i.e., coefficients
    are log rate ratios per hour).

    Returns a dictionary with:
      - "object": dict of numeric results (coefficients, p-values, CIs, rate ratios,
                  predicted rate per hour at mean covariates, dispersion, family).
      - "description": brief human-readable interpretation of the keys/values.

    The function is defensive and will try reasonable fallbacks if some attributes
    are missing.
    """
    import numpy as np

    res = model_output  # expected to be a statsmodels GLMResultsWrapper

    out = {}
    try:
        params = res.params.copy()
    except Exception:
        raise ValueError("model_output does not contain params.")

    # Basic tables: coefficients, std err, p-values, conf int
    try:
        bse = res.bse.copy()
    except Exception:
        bse = pd.Series([np.nan] * len(params), index=params.index)

    try:
        pvalues = res.pvalues.copy()
    except Exception:
        pvalues = pd.Series([np.nan] * len(params), index=params.index)

    try:
        ci = res.conf_int().copy()
    except Exception:
        # fallback: approximate using normal approximation if bse available
        z = 1.96
        ci = np.vstack([params - z * bse, params + z * bse]).T
        ci = pd.DataFrame(ci, index=params.index, columns=[0, 1])

    # Rate ratios and their CIs (exp of coef and CI)
    rr = np.exp(params)
    rr_ci_lower = np.exp(ci.iloc[:, 0])
    rr_ci_upper = np.exp(ci.iloc[:, 1])

    # Assemble per-variable summary with python native types
    coef_table = {}
    for name in params.index:
        coef_table[name] = {
            "coef": float(params[name]) if not np.isnan(params[name]) else None,
            "std_err": float(bse[name]) if name in bse.index and not np.isnan(bse[name]) else None,
            "p_value": float(pvalues[name]) if name in pvalues.index and not np.isnan(pvalues[name]) else None,
            "ci_95": (float(ci.loc[name, 0]), float(ci.loc[name, 1])) if name in ci.index else (None, None),
            "rate_ratio": float(rr[name]) if not np.isnan(rr[name]) else None,
            "rate_ratio_ci_95": (float(rr_ci_lower[name]), float(rr_ci_upper[name])) if name in rr.index else (None, None),
        }

    out['coefficients'] = coef_table

    # Family name
    try:
        fam = res.model.family
        family_name = fam.__class__.__name__
    except Exception:
        family_name = None
    out['family'] = family_name

    # Dispersion (deviance / df_resid) where informative (esp. for Poisson)
    try:
        deviance = float(res.deviance)
        df_resid = float(res.df_resid)
        dispersion = float(deviance / df_resid) if df_resid > 0 else None
    except Exception:
        deviance = None
        df_resid = None
        dispersion = None
    out['deviance'] = deviance
    out['df_resid'] = df_resid
    out['dispersion_deviance_over_df'] = dispersion

    # Predicted rate per hour for a "typical" group (mean of exog used at estimation)
    # model.exog is the design matrix used; its columns correspond to params.index order.
    try:
        exog = res.model.exog  # 2D array
        exog_names = res.model.exog_names
        mean_exog = np.asarray(exog).mean(axis=0)
        # Predicted log rate (per unit exposure) at mean covariates:
        pred_log_rate_at_mean = float(np.dot(mean_exog, params))
        # standard error of the linear predictor via covariance matrix
        cov = res.cov_params()
        # Ensure shapes align
        vec = mean_exog.reshape(-1, 1)
        se_log_rate = float(np.sqrt(float(np.dot(vec.T, np.dot(cov, vec)))))
        z = 1.96
        pred_log_ci = (pred_log_rate_at_mean - z * se_log_rate, pred_log_rate_at_mean + z * se_log_rate)
        pred_rate_at_mean = float(np.exp(pred_log_rate_at_mean))
        pred_rate_ci = (float(np.exp(pred_log_ci[0])), float(np.exp(pred_log_ci[1])))
        out['predicted_rate_per_hour_at_mean_covariates'] = {
            "log_rate": pred_log_rate_at_mean,
            "log_rate_se": se_log_rate,
            "rate_per_hour": pred_rate_at_mean,
            "rate_per_hour_95_ci": pred_rate_ci,
            "mean_exog_vector": {exog_names[i]: float(mean_exog[i]) for i in range(len(exog_names))}
        }
    except Exception:
        # If anything fails, set to None
        out['predicted_rate_per_hour_at_mean_covariates'] = None

    # Helpful reminders / interpretation notes
    description_lines = [
        "This extracts model coefficients from a GLM fit where log(Hours) was used as an offset.",
        "Coefficients are on the log(rate) scale: exp(coef) = multiplicative change in expected fish caught per hour.",
        "For each predictor the following are provided: coef, std_err, p_value, 95% CI (on log scale),",
        "and the rate ratio (exp(coef)) with its 95% CI.",
        "Also provided: model family, deviance-based dispersion (deviance / df_resid), and",
        "a predicted fish-catch rate per hour for an average observation (mean of design matrix) with 95% CI.",
        "Interpretation example: if coefficients['LiveBait']['rate_ratio'] == 1.50, then groups using live bait are expected",
        "to catch 1.50x as many fish per hour (i.e., 50% more) than groups not using live bait, all else equal.",
    ]
    description = " ".join(description_lines)

    return {"object": out, "description": description}