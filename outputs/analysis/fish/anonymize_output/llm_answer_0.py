def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (Poisson or NegBin).
    Returns a dictionary with:
      - "object": a dict of extracted numeric results (coefficients, p-values, rate ratios,
                  baseline & average rate-per-hour estimates, dispersion, etc.)
      - "description": a short human-readable interpretation of the main results.

    The function is robust to different statsmodels result shapes and will attempt to
    compute an average predicted rate-per-hour using the sample means of the fitted
    exogenous variables if those are available in the model object.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Basic parameter table
    try:
        params = pd.Series(res.params)
    except Exception:
        # fallback: try to get params as ndarray with names from model
        params = pd.Series(np.asarray(res.params))

    # Ensure we have names for params
    if hasattr(res, "model") and hasattr(res.model, "exog_names"):
        exog_names = list(res.model.exog_names)
        if len(exog_names) == len(params) and not np.array_equal(params.index.values, np.array(exog_names)):
            params.index = exog_names

    # Std errors, p-values, confint
    try:
        bse = pd.Series(res.bse, index=params.index)
    except Exception:
        bse = None
    try:
        pvalues = pd.Series(res.pvalues, index=params.index)
    except Exception:
        pvalues = None
    try:
        conf = res.conf_int()
        # conf_int returns DataFrame with columns [0,1]; convert to DataFrame with same index as params
        conf = pd.DataFrame(conf, index=params.index)
        conf.columns = ["2.5%", "97.5%"]
    except Exception:
        conf = None

    # Rate ratios (exp(coef)) and confidence intervals on rate ratio scale
    rr = np.exp(params)
    rr = pd.Series(rr, index=params.index)
    rr_ci = None
    if conf is not None:
        rr_ci = pd.DataFrame(np.exp(conf.values), index=params.index, columns=["2.5%", "97.5%"])

    # Model family name
    fam = None
    try:
        fam = res.model.family.__class__.__name__
    except Exception:
        fam = None

    # Attempt dispersion estimate (Pearson chi2 / df_resid)
    dispersion = None
    try:
        y = getattr(res.model, "endog", None)
        mu = getattr(res, "mu", None)
        if (y is not None) and (mu is not None):
            denom = np.where(mu > 0, mu, 1.0)
            pearson_chi2 = np.sum(((y - mu) ** 2) / denom)
            df_resid = res.df_resid if hasattr(res, "df_resid") else (len(y) - len(params))
            dispersion = float(pearson_chi2 / df_resid) if df_resid > 0 else None
    except Exception:
        dispersion = None

    # Baseline rate per hour: exp(intercept). Look for common intercept names.
    intercept_name = None
    for cand in ["const", "Intercept", "intercept"]:
        if cand in params.index:
            intercept_name = cand
            break
    baseline_rate_per_hour = None
    if intercept_name is not None:
        try:
            baseline_rate_per_hour = float(np.exp(params.loc[intercept_name]))
        except Exception:
            baseline_rate_per_hour = None

    # Attempt to compute average predicted rate-per-hour for a "typical" group using mean of exog
    avg_rate_per_hour = None
    avg_rate_ci = None
    try:
        # Get exog matrix (should match params order)
        exog = getattr(res.model, "exog", None)
        if exog is None and hasattr(res.model, "data") and hasattr(res.model.data, "frame"):
            # try to rebuild exog from the frame using exog_names
            df_frame = res.model.data.frame
            exog_names = res.model.exog_names
            exog = df_frame[exog_names].to_numpy()
        if exog is not None:
            exog = np.asarray(exog)
            # mean of each exogenous column
            exog_mean = np.mean(exog, axis=0)
            # linear predictor for rate per hour: X_mean @ params (note: offset removed in rate formulation)
            # Because model used offset=log(hours), the model parameterization implies:
            # log(mu) = log(hours) + X*beta  => mu/hour = exp(X*beta). So X*beta gives log(rate per hour).
            linear_mean = float(np.dot(exog_mean, params.values))
            avg_rate_per_hour = float(np.exp(linear_mean))
            # compute CI for linear predictor via covariance matrix (delta method)
            cov = res.cov_params()
            # ensure cov has correct index ordering
            cov = pd.DataFrame(cov, index=params.index, columns=params.index)
            exog_mean_vec = np.asarray(exog_mean).reshape(-1, 1)
            var_lin = float(np.dot(exog_mean_vec.T, np.dot(cov.values, exog_mean_vec)))
            se_lin = np.sqrt(max(var_lin, 0.0))
            z = 1.96
            lower_lin = linear_mean - z * se_lin
            upper_lin = linear_mean + z * se_lin
            avg_rate_ci = (float(np.exp(lower_lin)), float(np.exp(upper_lin)))
    except Exception:
        avg_rate_per_hour = None
        avg_rate_ci = None

    # Build coefficients summary dictionary
    coef_table = {}
    for name in params.index:
        coef_table[name] = {
            "coef": float(params.loc[name]),
            "std_err": float(bse.loc[name]) if bse is not None else None,
            "p_value": float(pvalues.loc[name]) if pvalues is not None else None,
            "ci_2.5%": float(conf.loc[name, "2.5%"]) if conf is not None else None,
            "ci_97.5%": float(conf.loc[name, "97.5%"]) if conf is not None else None,
            "rate_ratio": float(rr.loc[name]),
            "rr_ci_2.5%": float(rr_ci.loc[name, "2.5%"]) if rr_ci is not None else None,
            "rr_ci_97.5%": float(rr_ci.loc[name, "97.5%"]) if rr_ci is not None else None,
        }

    # Short human-readable description
    # We construct a concise interpretation focusing on:
    #  - model family
    #  - baseline (intercept) => fish per hour when predictors are zero
    #  - how to interpret coefficients (multiplicative on rate)
    #  - average predicted rate if available
    description_lines = []
    if fam:
        description_lines.append(f"Model family used: {fam}.")
    if dispersion is not None:
        description_lines.append(f"Estimated dispersion (Pearson chi2 / df_resid) = {dispersion:.3f}.")
    if baseline_rate_per_hour is not None:
        description_lines.append(
            f"Baseline rate (all predictors = 0) = {baseline_rate_per_hour:.3f} fish per hour "
            f"(intercept = {params.loc[intercept_name]:.3f})."
        )
    else:
        description_lines.append("Baseline rate per hour (intercept) not available / not found.")

    description_lines.append(
        "Coefficients are on the log-rate scale; exp(coef) gives the multiplicative change in fish-per-hour."
    )

    if avg_rate_per_hour is not None:
        description_lines.append(
            f"Using the sample means of the predictors, the predicted average rate is "
            f"{avg_rate_per_hour:.3f} fish per hour"
            + (f" (95% CI: {avg_rate_ci[0]:.3f}–{avg_rate_ci[1]:.3f})" if avg_rate_ci is not None else "")
            + "."
        )
    else:
        description_lines.append("Could not compute a data-driven average predicted rate-per-hour (exog not available).")

    description_lines.append(
        "For each predictor, the returned 'rate_ratio' indicates how many times the fish-per-hour rate "
        "is multiplied when that predictor increases by one unit (or from 0→1 for binaries)."
    )

    description = " ".join(description_lines)

    output = {
        "model_family": fam,
        "dispersion": dispersion,
        "baseline_rate_per_hour": baseline_rate_per_hour,
        "average_rate_per_hour": avg_rate_per_hour,
        "average_rate_per_hour_ci": avg_rate_ci,
        "coefficients": coef_table,
    }

    return {"object": output, "description": description}