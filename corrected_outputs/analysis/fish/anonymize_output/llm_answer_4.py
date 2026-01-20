def extract_final_answer(model_output):
    """
    Extracts coefficient estimates, inference stats, and per-hour rate estimates
    from a statsmodels GLMResultsWrapper (NegativeBinomial or Poisson) that was
    fit with offset = log(Hours).

    Returns a dictionary with keys:
      - "object": dict with numeric results (coefficients, p-values, CI, IRR, predicted rates)
      - "description": brief interpretation of the key results in context
    """
    import numpy as np

    res = model_output

    # Basic coefficient table
    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    try:
        ci = res.conf_int()
    except Exception:
        # fallback if conf_int not available
        ci = np.vstack([params - 1.96 * bse, params + 1.96 * bse]).T
        # convert to DataFrame-like structure with same index
        import pandas as _pd
        ci = _pd.DataFrame(ci, index=params.index, columns=[0, 1])

    # Exponentiated coefficients (incidence rate ratios) and their CIs
    irr = np.exp(params)
    irr_ci_low = np.exp(ci.iloc[:, 0])
    irr_ci_high = np.exp(ci.iloc[:, 1])

    # Prepare per-variable summary (convert to native Python types)
    var_summaries = {}
    for name in params.index:
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else None
        pval = float(pvalues[name]) if name in pvalues.index else None
        ci_low = float(ci.loc[name][0]) if name in ci.index else None
        ci_high = float(ci.loc[name][1]) if name in ci.index else None
        irr_val = float(irr[name])
        irr_low = float(irr_ci_low[name])
        irr_high = float(irr_ci_high[name])
        pct_change = (irr_val - 1.0) * 100.0  # percent change in rate per hour

        var_summaries[name] = {
            "coef_log_rate_per_hour": coef,
            "std_err": se,
            "p_value": pval,
            "conf_int_log_rate_per_hour": [ci_low, ci_high],
            "irr_rate_ratio_per_hour": irr_val,
            "irr_conf_int": [irr_low, irr_high],
            "percent_change_in_rate_per_hour": pct_change,
        }

    # Baseline rate per hour when predictors = 0 (intercept)
    intercept_name = None
    for candidate in ["const", "Const", "intercept", "Intercept"]:
        if candidate in params.index:
            intercept_name = candidate
            break

    baseline_rate_per_hour = None
    if intercept_name is not None:
        baseline_rate_per_hour = float(np.exp(params[intercept_name]))

    # Predicted per-hour rates for the original sample (if data available)
    # Use linear=True to get X*beta + offset, then subtract offset to get X*beta.
    try:
        linear_pred = res.predict(linear=True)
        offset = getattr(res.model, "offset", None)
        if offset is None:
            # If no offset attribute, try model.data or assume zero offset
            offset = getattr(res.model, "data", None)
            if offset is not None:
                # some models store offset in model.data.orig_endog etc.; safest is to assume no offset
                offset = None
        if offset is None:
            # No offset found; assume no offset (per-hour will be same as predicted mean)
            per_hour_log = linear_pred
        else:
            per_hour_log = np.asarray(linear_pred) - np.asarray(offset)
        per_hour_rates = np.exp(per_hour_log)
        mean_pred_rate_per_hour = float(np.mean(per_hour_rates))
        median_pred_rate_per_hour = float(np.median(per_hour_rates))
    except Exception:
        # fallback: use predicted counts and try to divide by Hours if available on model.data
        try:
            preds = res.predict()
            hours = getattr(res.model, "data").frame["Hours"]
            per_hour_rates = np.asarray(preds) / np.asarray(hours)
            mean_pred_rate_per_hour = float(np.mean(per_hour_rates))
            median_pred_rate_per_hour = float(np.median(per_hour_rates))
        except Exception:
            mean_pred_rate_per_hour = None
            median_pred_rate_per_hour = None
            per_hour_rates = None

    # Extract sample size if available
    try:
        n_obs = int(res.nobs)
    except Exception:
        n_obs = None

    # Construct the returned object
    out_object = {
        "variable_summaries": var_summaries,
        "baseline_rate_per_hour_when_predictors_zero": baseline_rate_per_hour,
        "mean_predicted_rate_per_hour_observed_sample": mean_pred_rate_per_hour,
        "median_predicted_rate_per_hour_observed_sample": median_pred_rate_per_hour,
        "n_observations": n_obs,
    }

    # Short description / interpretation
    description_lines = []
    description_lines.append(
        "Model is a count GLM with log(Hours) as offset, so coefficients are log changes in expected fish-caught rate per hour."
    )
    description_lines.append(
        "For each predictor, 'coef_log_rate_per_hour' is the estimated log change in fish-per-hour; "
        "'irr_rate_ratio_per_hour' = exp(coef) is the multiplicative change (rate ratio) in fish-per-hour."
    )
    description_lines.append(
        "Percent change gives (IRR - 1)*100. A p-value < 0.05 indicates a commonly used threshold for statistical significance."
    )
    if baseline_rate_per_hour is not None:
        description_lines.append(
            f"The intercept implies a baseline predicted rate of ~{baseline_rate_per_hour:.3g} fish per hour when all predictors = 0."
        )
    if mean_pred_rate_per_hour is not None:
        description_lines.append(
            f"Average predicted catch rate across the sample is ~{mean_pred_rate_per_hour:.3g} fish per hour (median {median_pred_rate_per_hour:.3g})."
        )
    description_lines.append(
        "Focus interpretation on 'LiveBait' and 'HasCamper': check their IRRs and p-values in variable_summaries to see whether using live bait and/or "
        "having a camper with the group is associated with higher or lower catch rates per hour."
    )

    description = " ".join(description_lines)

    return {"object": out_object, "description": description}