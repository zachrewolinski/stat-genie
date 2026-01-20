def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted GLM model output and return:
      - coefficients, SEs, p-values, 95% CI
      - incidence rate ratios (IRRs = exp(coef)) and their 95% CIs
      - baseline fish/hour (exp(intercept)) and predicted rates when using livebait or when a camper is present
      - model family, dispersion, and sample size

    Returns a dict with keys:
      - "object": dict of numeric results
      - "description": short interpretation of what the numbers mean (fish per hour)
    """
    import numpy as np

    # Locate the fitted model object
    model = None
    if isinstance(model_output, dict):
        model = model_output.get('chosen_model') or model_output.get('poisson_model') or model_output.get('model')
    else:
        model = model_output

    if model is None:
        return {
            "object": None,
            "description": "No fitted model found in model_output."
        }

    # Extract parameter table
    params = model.params.copy()
    bse = model.bse.copy()
    pvalues = model.pvalues.copy()

    # Confidence intervals (2.5%, 97.5%)
    try:
        conf = model.conf_int()
    except Exception:
        # fallback: use normal approx
        conf_low = params - 1.96 * bse
        conf_high = params + 1.96 * bse
        conf = np.column_stack([conf_low, conf_high])
    # Ensure conf is a DataFrame/array aligned with params
    try:
        conf_df = conf.copy()
        conf_low = np.asarray(conf_df.iloc[:, 0]).astype(float)
        conf_high = np.asarray(conf_df.iloc[:, 1]).astype(float)
    except Exception:
        conf_arr = np.asarray(conf)
        conf_low = conf_arr[:, 0].astype(float)
        conf_high = conf_arr[:, 1].astype(float)

    # IRRs and CI on rate ratio scale
    irr = np.exp(params)
    irr_conf_low = np.exp(conf_low)
    irr_conf_high = np.exp(conf_high)

    # Get intercept (constant). Many formulas use 'const' as name for intercept.
    if 'const' in params.index:
        intercept = float(params['const'])
    else:
        # fallback: first parameter assumed intercept
        intercept = float(params.iloc[0])

    baseline_rate_per_hour = float(np.exp(intercept))  # expected fish per hour for baseline: no livebait, no camper, persons_c=0, child_c=0

    # Predicted rates when adding livebait or camper (holding centered controls at 0)
    beta_live = float(params.get('livebait', 0.0)) if 'livebait' in params.index else 0.0
    beta_camper = float(params.get('camper', 0.0)) if 'camper' in params.index else 0.0

    rate_with_livebait = float(np.exp(intercept + beta_live))
    rate_with_camper = float(np.exp(intercept + beta_camper))

    # assemble coefficient table
    coef_table = {}
    for i, name in enumerate(params.index):
        coef_table[name] = {
            "coef": float(params[name]),
            "se": float(bse[name]) if name in bse.index else None,
            "pvalue": float(pvalues[name]) if name in pvalues.index else None,
            "ci_lower": float(conf_low[i]),
            "ci_upper": float(conf_high[i]),
            "irr": float(irr[name]),
            "irr_ci_lower": float(irr_conf_low[i]),
            "irr_ci_upper": float(irr_conf_high[i])
        }

    # Gather metadata if available
    chosen_family = model_output.get('chosen_family') if isinstance(model_output, dict) else None
    dispersion = model_output.get('dispersion') if isinstance(model_output, dict) else None
    n_obs = int(model_output.get('n_obs')) if isinstance(model_output, dict) and model_output.get('n_obs') is not None else (int(model.nobs) if hasattr(model, 'nobs') else None)

    results = {
        "coefficients": coef_table,
        "baseline_rate_per_hour": baseline_rate_per_hour,
        "rate_with_livebait_per_hour": rate_with_livebait,
        "rate_with_camper_per_hour": rate_with_camper,
        "intercept_log_rate": intercept,
        "dispersion": dispersion,
        "chosen_family": chosen_family,
        "n_obs": n_obs
    }

    # Short human-readable description
    description_lines = []
    description_lines.append("Returned: coefficients (log-rate scale), SEs, p-values, 95% CIs, and IRRs (exp(coef)).")
    description_lines.append("Rates are fish per hour because the model used log(hours) as an offset.")
    description_lines.append(f"Baseline (reference) expected fish/hour = exp(intercept) = {baseline_rate_per_hour:.3f}.")
    description_lines.append(f"Expected fish/hour with livebait (keeping centered controls at 0) = {rate_with_livebait:.3f}.")
    description_lines.append(f"Expected fish/hour with a camper present (keeping centered controls at 0) = {rate_with_camper:.3f}.")
    if dispersion is not None:
        description_lines.append(f"Model family chosen: {chosen_family}, dispersion statistic = {dispersion:.3f}.")
    if n_obs is not None:
        description_lines.append(f"Number of observations = {n_obs}.")

    description = " ".join(description_lines)

    return {
        "object": results,
        "description": description
    }