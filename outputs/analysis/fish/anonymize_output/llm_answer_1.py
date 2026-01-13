def extract_final_answer(model_output):
    """
    Extracts coefficients, p-values, confidence intervals, and rate ratios
    from the best available fitted model in model_output (prefers Negative Binomial
    if present, otherwise Poisson). Also returns overall fish/hour and model diagnostics.
    Returns a dictionary with keys "object" (structured numeric results) and
    "description" (brief plain-language interpretation).
    """
    import numpy as np

    # Choose the model: prefer Negative Binomial if available (handles overdispersion)
    model = None
    model_key = None
    if 'neg_binomial' in model_output and model_output['neg_binomial'] is not None:
        model = model_output['neg_binomial']
        model_key = 'neg_binomial'
    elif 'poisson' in model_output and model_output['poisson'] is not None:
        model = model_output['poisson']
        model_key = 'poisson'
    else:
        return {
            "object": None,
            "description": "No fitted model (neither 'neg_binomial' nor 'poisson') found in model_output."
        }

    # Safely extract numeric summaries from the statsmodels results wrapper
    try:
        params = model.params  # log-rate coefficients
        bse = model.bse
        pvalues = model.pvalues
        conf = model.conf_int()  # 2-column DataFrame: lower, upper (on log scale)
    except Exception as e:
        return {
            "object": None,
            "description": f"Model found ('{model_key}') but failed to extract statistics: {e}"
        }

    # Build structured estimates for each predictor
    estimates = {}
    for var in params.index:
        coef = float(params[var])
        se = float(bse[var]) if var in bse.index else None
        pval = float(pvalues[var]) if var in pvalues.index else None
        try:
            ci_low = float(conf.loc[var, 0])
            ci_high = float(conf.loc[var, 1])
        except Exception:
            ci_low = ci_high = None

        # Exponentiate to get rate ratios (multiplicative effect on fish-per-hour)
        rr = float(np.exp(coef))
        rr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        rr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

        estimates[var] = {
            "coef_log_rate": coef,
            "std_err": se,
            "p_value": pval,
            "ci_log_rate": [ci_low, ci_high],
            "rate_ratio": rr,
            "ci_rate_ratio": [rr_ci_low, rr_ci_high],
        }

    # Pull descriptives and diagnostics if present
    descriptives = model_output.get('descriptives', {})
    overall_fish_per_hour = descriptives.get('overall_fish_per_hour')
    mean_fish_per_visit = descriptives.get('mean_fish_per_visit')
    mean_hours_per_visit = descriptives.get('mean_hours_per_visit')

    dispersion = model_output.get('poisson_dispersion')  # informative: poisson dispersion
    aic = model_output.get('aic')
    bic = model_output.get('bic')

    # Create a concise interpretation focusing on the main predictors
    # Emphasize LiveBait result if present
    interp_lines = []
    interp_lines.append(f"Model used: {model_key} (coefficients are on the log-rate scale; "
                        "exponentiated coefficients are multiplicative effects on fish-per-hour).")
    if dispersion is not None:
        interp_lines.append(f"Poisson dispersion statistic: {dispersion:.3f} (used to decide on NB vs Poisson).")
    if overall_fish_per_hour is not None:
        interp_lines.append(f"Overall observed fish/hour (descriptive): {overall_fish_per_hour:.3f}.")

    # Summarize each key predictor
    for key in ['LiveBait', 'Camper', 'TotalPeople']:
        if key in estimates:
            est = estimates[key]
            rr = est['rate_ratio']
            rr_lo, rr_hi = est['ci_rate_ratio']
            pval = est['p_value']
            interp_lines.append(
                f"{key}: rate ratio = {rr:.3f} (95% CI: {rr_lo:.3f}–{rr_hi:.3f}), p = {pval:.3g}."
                + (" Values <1 indicate a decrease, >1 an increase in fish-per-hour compared to the reference.")
            )

    description = " ".join(interp_lines)

    # Pack the object to return (structured numeric results + metadata)
    result_object = {
        "model_used": model_key,
        "estimates": estimates,
        "overall_fish_per_hour": overall_fish_per_hour,
        "mean_fish_per_visit": mean_fish_per_visit,
        "mean_hours_per_visit": mean_hours_per_visit,
        "poisson_dispersion": dispersion,
        "aic": aic,
        "bic": bic
    }

    return {"object": result_object, "description": description}