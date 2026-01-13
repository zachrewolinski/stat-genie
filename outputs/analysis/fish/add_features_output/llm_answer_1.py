def extract_final_answer(model_output):
    """
    Extracts coefficients, incidence rate ratios (IRRs), confidence intervals, p-values,
    and predicted fish-per-hour rates from a fitted statsmodels GLM (NegativeBinomial/Poisson)
    that used a log(hours) offset. Returns a dictionary with an 'object' (detailed numbers)
    and a short 'description' interpreting the key results.
    """
    import numpy as np

    res = model_output  # statsmodels results wrapper
    # Basic parameter summaries
    params = res.params                      # pandas Series
    pvalues = res.pvalues                    # pandas Series
    conf = res.conf_int()                    # DataFrame with [lower, upper]
    ci_lower = conf.iloc[:, 0]
    ci_upper = conf.iloc[:, 1]

    # Incident Rate Ratios and their CIs (exp of coefficients and conf ints)
    irr = np.exp(params)
    irr_ci_lower = np.exp(ci_lower)
    irr_ci_upper = np.exp(ci_upper)

    # Build per-variable summary
    coeffs = {}
    for name in params.index:
        coeffs[name] = {
            "coef_log_rate": float(params[name]),
            "p_value": float(pvalues.get(name, np.nan)),
            "coef_ci_lower": float(ci_lower.get(name, np.nan)),
            "coef_ci_upper": float(ci_upper.get(name, np.nan)),
            "IRR": float(irr[name]),
            "IRR_ci_lower": float(irr_ci_lower[name]),
            "IRR_ci_upper": float(irr_ci_upper[name]),
            "significant_0.05": bool(pvalues.get(name, 1.0) < 0.05)
        }

    # Predicted rate per hour for:
    # 1) reference profile (all predictors = 0, const = 1)
    # 2) average observed predictors (mean of exog columns)
    pred_rates = {}
    try:
        exog = np.asarray(res.model.exog)               # design matrix used for fitting
        exog_names = list(res.model.exog_names)
        cov = res.cov_params()                          # covariance matrix of params

        # Reference vector: all zeros except constant=1 if present
        x_ref = np.zeros(len(exog_names))
        if "const" in exog_names:
            x_ref[exog_names.index("const")] = 1.0
        elif "Intercept" in exog_names:
            x_ref[exog_names.index("Intercept")] = 1.0
        # linear predictor and CI for reference
        eta_ref = float(np.dot(params.values, x_ref))
        var_ref = float(np.dot(x_ref, np.dot(cov.values, x_ref)))
        se_ref = np.sqrt(max(var_ref, 0.0))
        rate_ref = float(np.exp(eta_ref))
        rate_ref_ci = (float(np.exp(eta_ref - 1.96 * se_ref)), float(np.exp(eta_ref + 1.96 * se_ref)))
        pred_rates['reference_zero_predictors'] = {
            "eta": eta_ref,
            "rate_per_hour": rate_ref,
            "rate_ci_lower": rate_ref_ci[0],
            "rate_ci_upper": rate_ref_ci[1]
        }

        # Mean predictor vector (mean of exog rows)
        x_mean = np.mean(exog, axis=0)
        eta_mean = float(np.dot(params.values, x_mean))
        var_mean = float(np.dot(x_mean, np.dot(cov.values, x_mean)))
        se_mean = np.sqrt(max(var_mean, 0.0))
        rate_mean = float(np.exp(eta_mean))
        rate_mean_ci = (float(np.exp(eta_mean - 1.96 * se_mean)), float(np.exp(eta_mean + 1.96 * se_mean)))
        pred_rates['mean_observed_predictors'] = {
            "eta": eta_mean,
            "rate_per_hour": rate_mean,
            "rate_ci_lower": rate_mean_ci[0],
            "rate_ci_upper": rate_mean_ci[1]
        }

        # Also include the names/order of predictors used so user can map back
        predictor_order = exog_names

    except Exception:
        # If any of the matrix-based calculations fail, skip predicted rates
        pred_rates = {}
        predictor_order = list(params.index)

    # Short human-readable interpretation
    description_lines = []
    description_lines.append(
        "The model is a log-link count model with log(hours) as an offset, "
        "so exp(coef) = incidence rate ratio (IRR) interprets multiplicative effect on fish caught per hour."
    )
    # Summarize notable predictors if any
    sig_vars = [n for n, v in coeffs.items() if v["significant_0.05"] and n not in ("const", "Intercept")]
    if sig_vars:
        description_lines.append(
            "Predictors statistically significant at p<0.05: " + ", ".join(sig_vars) + "."
        )
    else:
        description_lines.append("No non-intercept predictor is statistically significant at p<0.05.")

    # Add baseline / mean rate summary if computed
    if pred_rates:
        rr = pred_rates.get('reference_zero_predictors')
        mm = pred_rates.get('mean_observed_predictors')
        if rr:
            description_lines.append(
                f"Baseline (all predictors = 0) estimated fish per hour = {rr['rate_per_hour']:.3f} "
                f"(95% CI {rr['rate_ci_lower']:.3f}–{rr['rate_ci_upper']:.3f})."
            )
        if mm:
            description_lines.append(
                f"Estimated fish per hour for an average observed group = {mm['rate_per_hour']:.3f} "
                f"(95% CI {mm['rate_ci_lower']:.3f}–{mm['rate_ci_upper']:.3f})."
            )

    description = " ".join(description_lines)

    # Assemble final object
    output = {
        "coefficients": coeffs,
        "predictor_order": predictor_order,
        "predicted_rates_per_hour": pred_rates
    }

    return {"object": output, "description": description}