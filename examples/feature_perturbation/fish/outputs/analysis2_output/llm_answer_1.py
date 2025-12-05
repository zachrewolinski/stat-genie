def extract_final_answer(model_output):
    """
    Extract key statistics from the fitted model output and provide an interpretable summary
    focused on the rate of fish caught per hour and the effect of predictors.

    Returns a dict with:
    - "object": a dictionary of numeric results (model used, dispersion, intercept rate, table of coefficients
                with SE, p-value, 95% CI, rate ratios and rate-ratio CIs)
    - "description": a short human-readable interpretation of the main results in context.

    The function expects model_output to be the dictionary returned by the provided `model` function,
    containing at least the keys 'poisson' (GLMResults) and 'chosen'. If 'chosen' == 'negbin' it will
    use the Negative Binomial results when available; otherwise it uses the Poisson results.
    """
    import numpy as np

    # Basic validation
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dictionary as returned by the modeling function.")

    chosen = model_output.get('chosen', 'poisson')
    poisson_res = model_output.get('poisson', None)
    negbin_res = model_output.get('negbin', None)

    # Select the result object to use
    if chosen == 'negbin' and negbin_res is not None:
        res = negbin_res
        model_used = 'negbin'
    elif poisson_res is not None:
        res = poisson_res
        model_used = 'poisson'
    else:
        raise ValueError("No fitted model results found in model_output under 'poisson' or 'negbin'.")

    # Extract parameter table
    try:
        params = res.params.copy()
        pvalues = res.pvalues.copy()
        bse = res.bse.copy()
        ci = res.conf_int().copy()  # DataFrame or ndarray with two columns
    except Exception as e:
        raise RuntimeError(f"Failed to extract statistics from model result object: {e}")

    # Normalize index names and find intercept name if present
    param_index = list(params.index)
    intercept_name = None
    for candidate in ['Intercept', 'const', 'Intercept[0]']:  # common possibilities
        if candidate in param_index:
            intercept_name = candidate
            break
    # If not found, check if first parameter looks like intercept (no predictor characters)
    if intercept_name is None:
        # assume first parameter is intercept only if its name doesn't match any predictors typically provided
        intercept_name = param_index[0] if len(param_index) > 0 else None

    # Build coefficient table
    coef_table = {}
    for name in param_index:
        coef = float(params.loc[name])
        se = float(bse.loc[name]) if name in bse.index else None
        pval = float(pvalues.loc[name]) if name in pvalues.index else None
        try:
            ci_low = float(ci.loc[name][0]) if name in ci.index else float(ci.iloc[param_index.index(name), 0])
            ci_high = float(ci.loc[name][1]) if name in ci.index else float(ci.iloc[param_index.index(name), 1])
        except Exception:
            # fallback if conf_int indexing differs
            ci_low = None
            ci_high = None

        # Rate ratio (exp(coef)) and CI on that scale
        rr = float(np.exp(coef)) if coef is not None else None
        rr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        rr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

        coef_table[name] = {
            'coef_log_rate': coef,
            'std_err': se,
            'p_value': pval,
            'ci_log_rate': [ci_low, ci_high],
            'rate_ratio': rr,
            'rate_ratio_ci': [rr_ci_low, rr_ci_high]
        }

    # Compute baseline rate per hour if intercept present
    intercept_rate = None
    if intercept_name is not None and intercept_name in coef_table:
        try:
            intercept_rate = coef_table[intercept_name]['rate_ratio']  # exp(intercept) => fish per hour when predictors = 0
        except Exception:
            intercept_rate = None

    # Prepare a short textual interpretation focusing on key predictors
    def fmt_float(x, n=3):
        return ("{0:." + str(n) + "f}").format(x) if x is not None else "NA"

    summary_lines = []
    summary_lines.append(f"Model used: {model_used}. (Dispersion reported = {model_output.get('dispersion', None)})")
    summary_lines.append("Model is a log-linear rate model with log(Hours) used as an offset, so exponentiated coefficients")
    summary_lines.append("are multiplicative effects on the fish-per-hour rate (rate ratios).")

    if intercept_rate is not None:
        summary_lines.append(f"Baseline estimated rate (all predictors = 0): {fmt_float(intercept_rate)} fish/hour "
                             f"(exp(intercept) = {fmt_float(coef_table[intercept_name]['coef_log_rate'])}).")

    # For each predictor (exclude intercept) provide rate ratio and significance
    for name, info in coef_table.items():
        if name == intercept_name:
            continue
        rr = info['rate_ratio']
        rr_ci = info['rate_ratio_ci']
        pval = info['p_value']
        # Interpret direction
        if pval is None:
            signif = ""
        else:
            signif = "statistically significant" if pval < 0.05 else "not statistically significant"
        summary_lines.append(
            f"{name}: rate ratio = {fmt_float(rr)} (95% CI {fmt_float(rr_ci[0])} - {fmt_float(rr_ci[1])}); "
            f"p = {fmt_float(pval, 4)} → {signif}."
        )
        # Short explanation for typical variable meaning
        if name in ['UsedLiveBait', 'HadCamper']:
            summary_lines.append(f"  Interpretation: groups with {name} = 1 have a multiplicative change of {fmt_float(rr)} "
                                 f"in fish-per-hour compared to {name} = 0, holding other predictors constant.")
        else:
            summary_lines.append(f"  Interpretation: a one-unit increase in {name} multiplies the fish-per-hour rate by {fmt_float(rr)}.")

    description_text = " ".join(summary_lines)

    # Package numeric object to return
    result_object = {
        'model_used': model_used,
        'dispersion': float(model_output.get('dispersion')) if model_output.get('dispersion') is not None else None,
        'coefficients': coef_table,
        'intercept_rate_fish_per_hour': intercept_rate,
    }

    return {
        "object": result_object,
        "description": description_text
    }