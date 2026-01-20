def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted GLM model output and return a concise
    summary useful for answering "how many fish on average per hour" and how
    predictors influence that rate.

    Returns a dict with:
      - "object": a dict containing numeric results (baseline rate per hour,
                  coefficient table with rate ratios and CIs, dispersion, model used)
      - "description": a brief interpretation of those numbers in plain language
    """
    import numpy as np
    import pandas as pd

    # Basic checks
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the model function")

    chosen = model_output.get('chosen_model')
    if chosen is None:
        raise ValueError("No chosen_model found in model_output")

    # Extract coefficient table
    params = chosen.params.copy()
    try:
        bse = chosen.bse.copy()
    except Exception:
        # fallback if bse unavailable
        bse = pd.Series(index=params.index, data=[np.nan]*len(params))

    try:
        pvalues = chosen.pvalues.copy()
    except Exception:
        pvalues = pd.Series(index=params.index, data=[np.nan]*len(params))

    try:
        ci = chosen.conf_int().copy()  # DataFrame with two columns (lower, upper)
        ci.columns = ['ci_lower', 'ci_upper']
    except Exception:
        ci = pd.DataFrame({'ci_lower': [np.nan]*len(params), 'ci_upper': [np.nan]*len(params)}, index=params.index)

    # Build table with rate ratios (exp(coef)) and their CIs
    rate_ratio = np.exp(params)
    rr_ci_lower = np.exp(ci['ci_lower'])
    rr_ci_upper = np.exp(ci['ci_upper'])

    coef_table = pd.DataFrame({
        'coef': params,
        'se': bse,
        'pvalue': pvalues,
        'rate_ratio': rate_ratio,
        'rr_ci_lower': rr_ci_lower,
        'rr_ci_upper': rr_ci_upper
    })

    # Identify intercept name (common names: 'Intercept' or 'const'), fallback to first index
    idx_names = list(params.index)
    intercept_name = None
    for candidate in ['Intercept', 'intercept', 'const', 'Const']:
        if candidate in idx_names:
            intercept_name = candidate
            break
    if intercept_name is None:
        intercept_name = idx_names[0]

    intercept_coef = params.loc[intercept_name]
    intercept_ci = (ci.loc[intercept_name, 'ci_lower'], ci.loc[intercept_name, 'ci_upper'])

    # Baseline expected rate per hour when all predictors = 0:
    # model: log(E[count]) = intercept + beta*X + log(hours)
    # => rate per hour = E[count]/hours = exp(intercept + beta*X). For X=0: exp(intercept)
    baseline_rate_per_hour = float(np.exp(intercept_coef))
    baseline_rate_ci = (float(np.exp(intercept_ci[0])), float(np.exp(intercept_ci[1])))

    # Assemble numeric object to return
    numeric_result = {
        'model_used': type(chosen.model.family).__name__ if hasattr(chosen, 'model') else str(chosen.__class__.__name__),
        'dispersion': float(model_output.get('dispersion')) if model_output.get('dispersion') is not None else None,
        'baseline_rate_per_hour': baseline_rate_per_hour,
        'baseline_rate_per_hour_ci': baseline_rate_ci,
        # coefficients table as nested dict for easy programmatic use
        'coefficients': coef_table.round(4).to_dict(orient='index')
    }

    # Build a concise description interpreting the main pieces:
    desc_lines = []
    desc_lines.append(f"Chosen model family: {numeric_result['model_used']}.")
    if numeric_result['dispersion'] is not None:
        desc_lines.append(f"Pearson dispersion = {numeric_result['dispersion']:.2f} (>>1 indicates overdispersion; Negative Binomial was used if available).")
    desc_lines.append(f"Baseline expected catch rate (when livebait=0, camper=0, persons=0, child=0) = {baseline_rate_per_hour:.3f} fish/hour "
                      f"(95% CI: {baseline_rate_ci[0]:.3f} to {baseline_rate_ci[1]:.3f}).")
    desc_lines.append("For each predictor, the table 'coefficients' gives: coef (log rate change), se, p-value, rate_ratio = exp(coef) (multiplicative change in fish/hour), and a 95% CI for the rate ratio.")
    desc_lines.append("Interpretation example: a rate_ratio of 1.50 for livebait means using live bait is associated with a 50% higher fish-per-hour rate, all else equal. Use p-values to judge statistical significance (p < 0.05 commonly considered significant).")

    description = " ".join(desc_lines)

    return {'object': numeric_result, 'description': description}