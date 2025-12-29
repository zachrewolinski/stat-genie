def extract_final_answer(model_output):
    """
    Extract interpretable statistics from the fitted GLM model returned by the modeling function.

    Returns a dict with:
      - "object": a structured dictionary containing model_type, dispersion, parameter estimates
                  (coef, se, p-value, 95% CI), incidence rate ratios (IRR) and IRR CIs,
                  and the baseline predicted fish-per-hour rate (and its 95% CI) corresponding
                  to the model intercept (i.e., predictors = 0: no livebait, no camper, average group size).
      - "description": a short human-readable explanation of what the returned object means.
    """
    import numpy as np

    # Defensive checks
    if not isinstance(model_output, dict):
        return {
            "object": None,
            "description": "model_output is not a dict as expected."
        }

    final_model = model_output.get('final_model', None)
    model_type = model_output.get('model_type', None)
    dispersion = model_output.get('dispersion', None)

    if final_model is None:
        return {
            "object": None,
            "description": "No 'final_model' found in model_output."
        }

    # Extract coefficients and inference stats
    try:
        params = final_model.params.copy()
    except Exception:
        return {
            "object": None,
            "description": "Unable to read params from the fitted model object."
        }

    # Prepare containers
    param_summary = {}

    # Attempt to get standard errors, p-values, and confidence intervals
    bse = getattr(final_model, 'bse', None)
    pvalues = getattr(final_model, 'pvalues', None)
    try:
        conf = final_model.conf_int()
    except Exception:
        conf = None

    for name, coef in params.items():
        coef_f = float(np.asarray(coef).squeeze())
        se_f = float(np.nan) if bse is None or name not in bse.index else float(np.asarray(bse[name]).squeeze())
        p_f = float(np.nan) if pvalues is None or name not in pvalues.index else float(np.asarray(pvalues[name]).squeeze())

        if conf is not None and name in conf.index:
            ci_lower = float(np.asarray(conf.loc[name][0]).squeeze())
            ci_upper = float(np.asarray(conf.loc[name][1]).squeeze())
        else:
            ci_lower = float(np.nan)
            ci_upper = float(np.nan)

        # For a log-link model with offset log(hours), coefficients are log(rate ratios).
        irr = float(np.exp(coef_f)) if not np.isnan(coef_f) else float(np.nan)
        irr_ci_lower = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else float(np.nan)
        irr_ci_upper = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else float(np.nan)

        param_summary[name] = {
            "coef_log_rate": coef_f,
            "se": se_f,
            "p_value": p_f,
            "ci_95_log_rate": (ci_lower, ci_upper),
            "incidence_rate_ratio (IRR)": irr,
            "IRR_95_CI": (irr_ci_lower, irr_ci_upper)
        }

    # Baseline predicted fish-per-hour rate:
    # With model: log(E[y]) = X * beta + log_hours  => E[y]/hours = exp(X * beta)
    # So for predictors at 0, baseline rate per hour = exp(intercept)
    baseline_rate = None
    baseline_rate_ci = (np.nan, np.nan)
    intercept_name_candidates = ['const', 'Intercept', 'intercept']
    intercept_name = None
    for candidate in intercept_name_candidates:
        if candidate in params.index:
            intercept_name = candidate
            break
    # If none of the above, try first index as intercept if it looks like one
    if intercept_name is None:
        # many statsmodels use 'const'
        if len(params.index) > 0:
            # fallback: assume the first param is intercept only if it makes sense;
            # we will check. Use 'const' if present in param_summary keys
            if 'const' in param_summary:
                intercept_name = 'const'
            else:
                # no reliable intercept name found; do not guess
                intercept_name = None

    if intercept_name is not None and intercept_name in params.index:
        intercept = float(np.asarray(params[intercept_name]).squeeze())
        baseline_rate = float(np.exp(intercept))
        if conf is not None and intercept_name in conf.index:
            ci_l = float(np.asarray(conf.loc[intercept_name][0]).squeeze())
            ci_u = float(np.asarray(conf.loc[intercept_name][1]).squeeze())
            baseline_rate_ci = (float(np.exp(ci_l)), float(np.exp(ci_u)))

    result_object = {
        "model_type": model_type,
        "dispersion": dispersion,
        "parameters": param_summary,
        "baseline_rate_per_hour": baseline_rate,
        "baseline_rate_per_hour_95_CI": baseline_rate_ci
    }

    # Build a concise description to help interpret the primary outputs
    description_lines = [
        f"Model type used: {model_type}. Dispersion (from Poisson diagnostic): {dispersion}.",
        "Each model coefficient is on the log-rate scale because the model uses a log link with offset=log_hours.",
        "Therefore exp(coef) = incidence rate ratio (IRR): multiplicative change in expected fish caught per hour",
        "associated with a one-unit increase in the predictor, holding other predictors constant.",
        "",
        "Interpretation notes:",
        "- 'const' (intercept) -> baseline expected fish caught per hour when predictors are at zero "
        "(here: no livebait, no camper, and total_people_c = 0 which corresponds to average group size). "
        f"Baseline rate per hour = {result_object['baseline_rate_per_hour']}, "
        f"95% CI = {result_object['baseline_rate_per_hour_95_CI']}.",
        "- For each predictor (livebait, camper, total_people_c) the returned values include "
        "coef (log-rate), standard error, p-value, 95% CI on log-rate, IRR = exp(coef), and IRR 95% CI.",
        "- A p-value (near or below conventional thresholds like 0.05) suggests the predictor is significantly "
        "associated with the rate of fish caught per hour.",
        "",
        "Use the 'parameters' dictionary in the 'object' to read numeric effect sizes and significance for each variable."
    ]

    description = "\n".join(description_lines)

    return {
        "object": result_object,
        "description": description
    }