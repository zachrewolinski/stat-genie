def extract_final_answer(model_output):
    """
    Extracts key statistics from the fitted model output and returns:
      - object: dictionary of numeric results (coefficients, SE, p-values,
                95% CIs on coefficient and on rate ratios, baseline rate per hour, etc.)
      - description: a brief human-readable summary of what the numbers mean

    Expects model_output to be the dictionary returned by the provided `model` function,
    containing at least the key 'chosen_model' (a statsmodels results wrapper).
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise TypeError("model_output must be a dictionary as returned by the modeling function")

    if 'chosen_model' not in model_output:
        raise KeyError("model_output must contain key 'chosen_model'")

    res = model_output['chosen_model']  # statsmodels GLMResultsWrapper

    # Extract basic tables
    params = res.params            # coefficient (log rate) estimates
    bse = res.bse                  # standard errors
    pvalues = res.pvalues
    conf = res.conf_int()          # DataFrame with columns [0, 1] = lower, upper

    # Convert to rate ratios (multiplicative effect on fish/hour) and CI
    rate_ratios = np.exp(params)
    rate_ratio_ci = np.exp(conf)

    # Assemble a serializable dictionary of results
    coeffs = {}
    for name in params.index:
        # conf.loc[name, 0] and conf.loc[name, 1] are lower/upper CI on log scale
        lower_log = float(conf.loc[name, 0])
        upper_log = float(conf.loc[name, 1])
        coeffs[name] = {
            'coef_log_rate': float(params[name]),
            'se': float(bse[name]),
            'pvalue': float(pvalues[name]),
            'ci95_log_rate': [lower_log, upper_log],
            'rate_ratio': float(rate_ratios[name]),
            'ci95_rate_ratio': [float(rate_ratio_ci.loc[name, 0]), float(rate_ratio_ci.loc[name, 1])],
            'significant_at_0.05': bool(pvalues[name] < 0.05)
        }

    # Baseline expected fish/hour for the reference (all predictors = 0):
    # exp(intercept) gives expected fish per hour because offset = log_hours
    intercept_name = 'const' if 'const' in params.index else params.index[0]
    baseline_rate = float(rate_ratios[intercept_name])
    baseline_rate_ci = [float(rate_ratio_ci.loc[intercept_name, 0]), float(rate_ratio_ci.loc[intercept_name, 1])]

    # Determine model used
    model_used = "Negative Binomial" if ('nb_model' in model_output and model_output['chosen_model'] is model_output.get('nb_model')) else "Poisson"

    # Dispersion reported (may be NaN if not available)
    dispersion = model_output.get('dispersion', None)

    # Build a concise human-readable interpretation for each predictor
    interpretations = []
    for name in params.index:
        if name == intercept_name:
            continue
        rr = coeffs[name]['rate_ratio']
        ci = coeffs[name]['ci95_rate_ratio']
        p = coeffs[name]['pvalue']
        sig_text = "statistically significant (p<0.05)" if p < 0.05 else "not statistically significant (p≥0.05)"
        interpretations.append(f"{name}: rate ratio = {rr:.3f} (95% CI {ci[0]:.3f}–{ci[1]:.3f}), p={p:.3f} — {sig_text}.")

    description = (
        f"Model used: {model_used}. Dispersion (Poisson deviance/df_resid): {dispersion}.\n"
        f"Baseline expected fish/hour (reference group, predictors=0): {baseline_rate:.3f} "
        f"(95% CI {baseline_rate_ci[0]:.3f}–{baseline_rate_ci[1]:.3f}).\n"
        "Predictor effects (multiplicative on fish/hour): " + " ".join(interpretations)
    )

    result_object = {
        'model_used': model_used,
        'dispersion': float(dispersion) if dispersion is not None and not np.isnan(dispersion) else None,
        'baseline_rate_per_hour': baseline_rate,
        'baseline_rate_per_hour_ci95': baseline_rate_ci,
        'coefficients': coeffs
    }

    return {
        'object': result_object,
        'description': description
    }