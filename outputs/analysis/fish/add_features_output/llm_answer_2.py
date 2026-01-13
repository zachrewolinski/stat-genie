def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels GLMResultsWrapper (Poisson or NB)
    that used a log(offset) for hours (exposure), so coefficients are log rate ratios
    and predicted counts divided by hours give fish-per-hour.

    Returns a dict with:
      - "object": dict containing summary table (list of term dicts), IRRs, CIs,
                  mean/median predicted fish/hour in the sample, sample size,
                  model family, and a focused summary for 'livebait' (if present).
      - "description": brief interpretation of the extracted outputs.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Extract basic pieces
    try:
        params = pd.Series(res.params)
    except Exception:
        params = pd.Series(np.asarray(res.params), index=getattr(res, 'params', None))

    try:
        pvalues = pd.Series(res.pvalues)
    except Exception:
        pvalues = pd.Series(np.full_like(params, np.nan), index=params.index)

    try:
        conf = res.conf_int()
        # conf is typically a DataFrame with columns [0,1] (lower, upper)
        conf_df = pd.DataFrame(conf.values, index=params.index, columns=['coef_ci_lower', 'coef_ci_upper'])
    except Exception:
        # Fallback: fill with NaNs
        conf_df = pd.DataFrame(index=params.index, columns=['coef_ci_lower', 'coef_ci_upper']).astype(float)

    # Incidence Rate Ratios (IRR) and their CIs
    irr = np.exp(params)
    irr = pd.Series(irr, index=params.index, name='IRR')
    irr_ci_lower = np.exp(conf_df['coef_ci_lower'])
    irr_ci_upper = np.exp(conf_df['coef_ci_upper'])
    irr_ci_df = pd.DataFrame({'IRR_ci_lower': irr_ci_lower, 'IRR_ci_upper': irr_ci_upper})

    # Build a summary table (one row per term)
    summary_df = pd.DataFrame({
        'term': params.index.astype(str),
        'coef': params.values.astype(float),
        'coef_ci_lower': conf_df['coef_ci_lower'].values.astype(float),
        'coef_ci_upper': conf_df['coef_ci_upper'].values.astype(float),
        'pvalue': pvalues.values.astype(float),
        'IRR': irr.values.astype(float),
        'IRR_ci_lower': irr_ci_df['IRR_ci_lower'].values.astype(float),
        'IRR_ci_upper': irr_ci_df['IRR_ci_upper'].values.astype(float),
    })

    # Predicted counts and conversion to rate per hour using the model offset = log(hours)
    try:
        fitted_counts = np.asarray(res.fittedvalues)  # expected counts per visit
    except Exception:
        # try predict with no args (uses model.exog)
        try:
            fitted_counts = np.asarray(res.predict())
        except Exception:
            fitted_counts = None

    # Try to retrieve offset (log(hours)). Multiple fallbacks:
    offset = None
    try:
        offset = getattr(res.model, 'offset', None)
    except Exception:
        offset = None

    if offset is None:
        # try to find exposure in original data frame if available
        try:
            df = res.model.data.frame
            if 'exposure_log' in df.columns:
                offset = np.asarray(df['exposure_log'])
            elif 'hours' in df.columns:
                offset = np.log(np.asarray(df['hours']))
        except Exception:
            offset = None

    # Compute hours and rates per hour
    if fitted_counts is None:
        mean_rate = None
        median_rate = None
        sd_rate = None
    else:
        if offset is not None:
            try:
                hours = np.exp(offset)
            except Exception:
                # if offset stored as column of strings, coerce
                hours = np.exp(np.asarray(offset, dtype=float))
        else:
            # if offset not available, assume exposure = 1 hour (rate == predicted count)
            hours = np.ones_like(fitted_counts)

        # Avoid division by zero
        hours = np.where(hours <= 0, np.nan, hours)
        rate_per_hour = fitted_counts / hours
        # remove NaNs for summaries
        finite_rates = rate_per_hour[np.isfinite(rate_per_hour)]
        if finite_rates.size > 0:
            mean_rate = float(np.mean(finite_rates))
            median_rate = float(np.median(finite_rates))
            sd_rate = float(np.std(finite_rates, ddof=1)) if finite_rates.size > 1 else float(0.0)
        else:
            mean_rate = None
            median_rate = None
            sd_rate = None

    # Focused info for 'livebait' if present
    livebait_info = None
    if 'livebait' in params.index:
        livebait_info = {
            'term': 'livebait',
            'coef': float(params['livebait']),
            'coef_ci_lower': float(conf_df.loc['livebait', 'coef_ci_lower']) if 'livebait' in conf_df.index else None,
            'coef_ci_upper': float(conf_df.loc['livebait', 'coef_ci_upper']) if 'livebait' in conf_df.index else None,
            'IRR': float(irr['livebait']),
            'IRR_ci_lower': float(irr_ci_df.loc['livebait', 'IRR_ci_lower']) if 'livebait' in irr_ci_df.index else None,
            'IRR_ci_upper': float(irr_ci_df.loc['livebait', 'IRR_ci_upper']) if 'livebait' in irr_ci_df.index else None,
            'pvalue': float(pvalues['livebait']) if 'livebait' in pvalues.index else None,
            'interpretation': (
                "IRR >1 means using live bait is associated with a higher fish catch rate "
                "(multiplicative effect on fish-per-hour)."
            )
        }

    # Assemble return object
    result_object = {
        'summary_table': summary_df.to_dict(orient='records'),
        'mean_predicted_fish_per_hour': mean_rate,
        'median_predicted_fish_per_hour': median_rate,
        'sd_predicted_fish_per_hour': sd_rate,
        'n_obs': int(len(res.model.endog)) if hasattr(res.model, 'endog') else None,
        'model_family': type(res.model.family).__name__ if hasattr(res.model, 'family') else None,
        'livebait_effect': livebait_info
    }

    # Short description interpreting outputs
    description_parts = []
    description_parts.append(
        "Returned coefficients are log rate ratios from a GLM with log(hours) as offset, "
        "so exp(coef) = IRR (incidence rate ratio) is the multiplicative effect on expected fish-per-hour."
    )
    description_parts.append(
        "The summary_table lists each model term with coefficient, 95% CI, p-value, and IRR with its CI."
    )
    if mean_rate is not None:
        description_parts.append(
            f"The model's mean predicted catch rate in the sample is {mean_rate:.3f} fish per hour "
            f"(median {median_rate:.3f}, sd {sd_rate:.3f})."
        )
    else:
        description_parts.append("Predicted fish/hour could not be computed from the model output.")
    if livebait_info is not None:
        description_parts.append(
            "If 'livebait' appears above, its IRR shows how many times higher (IRR>1) or lower (IRR<1) "
            "the expected fish-per-hour is when live bait is used, holding other covariates constant."
        )

    description = " ".join(description_parts)

    return {"object": result_object, "description": description}