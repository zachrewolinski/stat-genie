def extract_final_answer(model_output):
    """
    Extract and summarize coefficient estimates from the provided model_output dict.

    Returns a dictionary with:
    - "object": a JSON-serializable dict containing:
        - model_used: which model was selected (NegativeBinomial or Poisson)
        - overdispersion: supplied overdispersion statistic
        - baseline_rate_per_hour: exp(intercept) = expected fish/hour when predictors = 0
        - baseline_rate_CI: 95% CI for the baseline rate (exp(conf_int(intercept)))
        - predictors: list of dicts for each term with:
            - term: name
            - coef_log: coefficient on the log(rate) scale
            - p_value: p-value for coef
            - rate_ratio: exp(coef) (multiplicative effect on fish-per-hour)
            - rate_ratio_CI: 95% CI for rate_ratio
            - significant: True if p_value < 0.05
    - "description": brief explanation of the model choice and how to interpret outputs.
    """
    import numpy as np

    # choose the model: prefer Negative Binomial when overdispersion > 1.5 and model present
    overdisp = model_output.get('overdispersion', np.nan)
    model = None
    model_name = None
    if (not np.isnan(overdisp)) and (overdisp > 1.5) and ('negbin_model' in model_output):
        model = model_output['negbin_model']
        model_name = 'NegativeBinomial'
    elif 'poisson_model' in model_output:
        model = model_output['poisson_model']
        model_name = 'Poisson'
    else:
        return {
            "object": None,
            "description": "No suitable fitted model found in model_output."
        }

    # Extract coefficient table, p-values, and confidence intervals
    params = model.params.copy()           # pandas Series
    pvalues = model.pvalues.copy()         # pandas Series
    try:
        conf = model.conf_int()            # pandas DataFrame with two columns (lower, upper)
    except Exception:
        # If conf_int fails for some reason, create NaN placeholders
        conf = None

    predictors = []
    for name in params.index:
        coef_log = float(params[name])
        pval = float(pvalues.get(name, np.nan))
        # extract CI; handle different conf_int column labels safely
        if conf is not None and name in conf.index:
            ci_vals = conf.loc[name].values
            ci_low_log = float(ci_vals[0])
            ci_high_log = float(ci_vals[1])
        else:
            ci_low_log = float("nan")
            ci_high_log = float("nan")

        rate_ratio = float(np.exp(coef_log)) if np.isfinite(coef_log) else None
        rr_low = float(np.exp(ci_low_log)) if np.isfinite(ci_low_log) else None
        rr_high = float(np.exp(ci_high_log)) if np.isfinite(ci_high_log) else None
        significant = (pval < 0.05) if (not np.isnan(pval)) else False

        predictors.append({
            "term": str(name),
            "coef_log": coef_log,
            "p_value": pval,
            "rate_ratio": rate_ratio,
            "rate_ratio_CI": [rr_low, rr_high],
            "significant": bool(significant)
        })

    # Baseline rate per hour (intercept). This is exp(const) and is the expected fish/hour
    # when numeric predictors are zero and categorical reference levels are used.
    if 'const' in params.index:
        intercept = float(params['const'])
        baseline_rate = float(np.exp(intercept))
        if conf is not None and 'const' in conf.index:
            ci0 = conf.loc['const'].values
            baseline_rate_CI = [float(np.exp(ci0[0])), float(np.exp(ci0[1]))]
        else:
            baseline_rate_CI = [None, None]
    else:
        intercept = None
        baseline_rate = None
        baseline_rate_CI = [None, None]

    output_object = {
        "model_used": model_name,
        "overdispersion": float(overdisp) if not np.isnan(overdisp) else None,
        "baseline_rate_per_hour": baseline_rate,
        "baseline_rate_CI": baseline_rate_CI,
        "predictors": predictors
    }

    description = (
        f"Selected model: {model_name}. Overdispersion = {output_object['overdispersion']:.2f} "
        "(NB model preferred when >1.5). Coefficients are on the log(rate) scale "
        "(log expected fish per hour). Exponentiated coefficients (rate_ratio) are multiplicative "
        "effects on fish-per-hour: e.g., a rate_ratio of 1.5 means 50% higher catch rate per hour. "
        "'baseline_rate_per_hour' = exp(intercept) is the expected fish/hour for the reference case "
        "(all predictors = 0). Each predictor entry reports the log-coef, p-value, rate ratio, "
        "95% CI for the rate ratio, and whether p < 0.05."
    )

    return {"object": output_object, "description": description}