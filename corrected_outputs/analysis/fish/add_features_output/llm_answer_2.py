def extract_final_answer(model_output):
    """
    Extracts key statistics from the fitted GLM model stored in model_output.

    Returns a dictionary with:
      - "object": a dictionary with per-predictor statistics (coefficient on log-rate scale,
                  p-value, exponentiated coefficient = rate ratio (IRR), 95% CI for IRR),
                  model fit stats (AIC, llf if available), and the average predicted
                  fish-per-hour from the model_output (if present).
      - "description": a short plain-language explanation of what the numbers mean.

    Assumes model_output is the dictionary returned by the modeling function in the prompt,
    with at least key 'model' containing a statsmodels GLMResultsWrapper.
    """
    import numpy as np

    if not isinstance(model_output, dict) or 'model' not in model_output:
        raise ValueError("model_output must be a dict containing a 'model' key.")

    model = model_output['model']

    # Prepare container
    out = {
        'predictors': {},
        'model_fit': {},
        'avg_predicted_rate_per_hour': None
    }

    # Extract model-level fit statistics where available
    try:
        out['model_fit']['aic'] = float(model.aic)
    except Exception:
        out['model_fit']['aic'] = None
    try:
        out['model_fit']['llf'] = float(model.llf)
    except Exception:
        out['model_fit']['llf'] = None

    # Parameters, p-values, and confidence intervals (on coefficient/log scale)
    params = model.params
    pvalues = model.pvalues
    try:
        conf = model.conf_int()
    except Exception:
        conf = None

    # Iterate over coefficients (skip Intercept if present but include it for completeness)
    for name, coef in params.items():
        entry = {}
        entry['coef_log_rate'] = float(coef)  # coefficient on log(rate)
        entry['p_value'] = float(pvalues.get(name, np.nan))
        # Exponentiate to get multiplicative effect on rate (IRR = e^{coef})
        irr = np.exp(coef)
        entry['irr'] = float(irr)
        # 95% CI for coef -> transform to IRR CI if conf available
        if conf is not None and name in conf.index:
            lower, upper = conf.loc[name].values
            entry['irr_95ci'] = (float(np.exp(lower)), float(np.exp(upper)))
            entry['coef_95ci'] = (float(lower), float(upper))
        else:
            entry['irr_95ci'] = (None, None)
            entry['coef_95ci'] = (None, None)

        out['predictors'][name] = entry

    # Include average predicted rate per hour if available
    if 'avg_predicted_rate_per_hour' in model_output and model_output['avg_predicted_rate_per_hour'] is not None:
        try:
            out['avg_predicted_rate_per_hour'] = float(model_output['avg_predicted_rate_per_hour'])
        except Exception:
            out['avg_predicted_rate_per_hour'] = None
    else:
        # Attempt to compute from predicted_count and hours if possible
        if 'predicted_count' in model_output and hasattr(model_output['predicted_count'], 'values') and 'predicted_rate_per_hour' in model_output:
            try:
                avg_rate = float(model_output['predicted_rate_per_hour'].mean())
                out['avg_predicted_rate_per_hour'] = avg_rate
            except Exception:
                out['avg_predicted_rate_per_hour'] = None

    # Compose a concise description for interpretation
    description_lines = [
        "Model type: Negative Binomial GLM with log(hours) offset (so coefficients are on the log-rate scale).",
        "Interpretation: exponentiated coefficients (IRR = exp(coef)) are multiplicative effects on expected fish caught per hour.",
        "For each predictor reported below:",
        "  - coef_log_rate: change in log(expected fish per hour) per unit change in predictor.",
        "  - irr: multiplicative factor for the expected fish-per-hour (IRR > 1 => higher rate; IRR < 1 => lower rate).",
        "  - irr_95ci: 95% confidence interval for the IRR. If it includes 1, the effect is not statistically distinguishable from no effect at alpha=0.05.",
        "Also included: model AIC/llf (when available) and the model's average predicted fish per hour across the dataset."
    ]
    description = " ".join(description_lines)

    return {
        "object": out,
        "description": description
    }