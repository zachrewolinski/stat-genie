def extract_final_answer(model_output):
    """
    Extract interpretable results from a fitted statsmodels GLMResultsWrapper
    that modeled FishCaught with log(Hours) as an offset.

    Returns a dictionary with:
      - "object": a dict containing
          * baseline_rate_per_hour: exp(intercept) -> expected fish/hour when predictors = 0
          * coeff_table: dict keyed by coefficient name with values:
                { "coef", "pvalue", "rate_ratio", "rr_CI_lower", "rr_CI_upper" }
          * avg_predicted_rate_per_hour: average model-predicted fish/hour across the
                training sample (if Hours can be recovered), otherwise None
      - "description": short explanation of the numbers and how to interpret them.
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Extract basic statistics
    params = res.params.copy()
    conf = res.conf_int().copy()  # DataFrame with columns [0,1] = lower, upper on coef scale
    pvalues = res.pvalues.copy()

    # Compute rate ratios (exp(coef)) and CIs on rate-ratio scale
    rate_ratio = np.exp(params)
    try:
        ci_lower = np.exp(conf.iloc[:, 0])
        ci_upper = np.exp(conf.iloc[:, 1])
    except Exception:
        # defensive: if conf has different columns
        ci_lower = np.exp(conf.iloc[:, 0])
        ci_upper = np.exp(conf.iloc[:, 1])

    # Assemble a tidy table (convert to plain Python types where possible)
    table = {}
    for name in params.index:
        table[name] = {
            "coef": float(params[name]),
            "pvalue": float(pvalues.get(name, np.nan)),
            "rate_ratio": float(rate_ratio[name]),
            "rr_CI_lower": float(ci_lower[name]),
            "rr_CI_upper": float(ci_upper[name])
        }

    # Baseline rate per hour = exp(intercept). Intercept may be named 'Intercept' or 'const'
    intercept_name = None
    if 'Intercept' in params.index:
        intercept_name = 'Intercept'
    elif 'const' in [n.lower() for n in params.index]:
        # find exact name containing 'const' (case-insensitive)
        for n in params.index:
            if 'const' in n.lower():
                intercept_name = n
                break

    baseline_rate = None
    if intercept_name is not None:
        baseline_rate = float(np.exp(params[intercept_name]))

    # Try to compute average predicted fish/hour across training data
    avg_pred_rate = None
    try:
        fitted_counts = res.fittedvalues  # predicted expected counts (same units as FishCaught)
        hours = None

        # Prefer retrieving the original DataFrame if available
        df = None
        if hasattr(res.model, 'data') and getattr(res.model.data, 'frame', None) is not None:
            df = res.model.data.frame
        elif hasattr(res.model, 'data') and getattr(res.model.data, 'orig_exog', None) is not None:
            # fallback, though Hours likely not in orig_exog since Hours wasn't in formula
            df = res.model.data.orig_exog

        if df is not None and 'Hours' in df.columns:
            hours = pd.to_numeric(df['Hours'], errors='coerce').astype(float)
        else:
            # If the model stored the offset (log(Hours)), recover Hours via exp(offset)
            if hasattr(res.model, 'offset') and res.model.offset is not None:
                offset = np.asarray(res.model.offset, dtype=float)
                # offset was log(Hours) in model code, so exp gives Hours
                hours = np.exp(offset)

        if hours is not None:
            # Align lengths if necessary
            fitted_arr = np.asarray(fitted_counts, dtype=float)
            hours_arr = np.asarray(hours, dtype=float)
            if fitted_arr.shape[0] == hours_arr.shape[0]:
                avg_pred_rate = float((fitted_arr / hours_arr).mean())
    except Exception:
        avg_pred_rate = None

    result_object = {
        "baseline_rate_per_hour": baseline_rate,
        "coeff_table": table,
        "avg_predicted_rate_per_hour": avg_pred_rate
    }

    description = (
        "Results from a GLM for FishCaught with log(Hours) offset (so coefficients are on the log-rate scale).\n"
        "- 'baseline_rate_per_hour' = exp(intercept): expected fish/hour when all predictors = 0.\n"
        "- For each predictor, 'rate_ratio' = exp(coef) is the multiplicative change in fish/hour "
        "per one-unit increase in that predictor; 'rr_CI_lower' and 'rr_CI_upper' give a 95% CI for that ratio.\n"
        "- 'pvalue' is the coefficient p-value for testing coef=0 (on log-rate scale).\n"
        "- 'avg_predicted_rate_per_hour' is the mean model-predicted fish/hour across the training sample "
        "(computed from fitted expected counts divided by Hours) if Hours were recoverable, otherwise None."
    )

    return {"object": result_object, "description": description}