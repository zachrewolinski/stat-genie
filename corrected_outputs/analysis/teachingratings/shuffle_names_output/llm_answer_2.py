def extract_final_answer(model_output):
    """
    Extracts statistics for the beauty variable from a fitted statsmodels regression result.
    Returns a dictionary with keys:
      - "object": a dict with numeric results (variable name, coefficient, SE, t, p-value, 95% CI)
      - "description": a short human-readable interpretation of the effect in context

    Expects model_output to be a statsmodels RegressionResultsWrapper (or an object exposing .params, .bse, .pvalues, .tvalues, .conf_int()).
    """
    # Identify the results object
    results = None
    if model_output is None:
        raise ValueError("model_output is None")
    if hasattr(model_output, "params"):
        results = model_output
    elif hasattr(model_output, "results") and hasattr(model_output.results, "params"):
        results = model_output.results
    else:
        raise ValueError("Unsupported model_output type: expected statsmodels results with .params attribute")

    # Try to find the beauty variable in the parameter index
    param_index = list(results.params.index)
    # Preferred names in order
    candidates = ['BeautyScore_z', 'BeautyScore', 'Beauty']
    target = None
    for name in candidates:
        if name in param_index:
            target = name
            break
    if target is None:
        # fallback: any parameter name containing 'beaut' (case-insensitive)
        for name in param_index:
            if 'beaut' in name.lower():
                target = name
                break
    if target is None:
        raise ValueError("Could not find a beauty-related variable in model parameters. Available params: " + ", ".join(param_index))

    # Extract statistics safely
    def safe_get(series_like, key):
        try:
            return float(series_like[key])
        except Exception:
            return None

    coef = safe_get(results.params, target)
    se = safe_get(results.bse, target) if hasattr(results, 'bse') else None
    tval = safe_get(results.tvalues, target) if hasattr(results, 'tvalues') else None
    pval = safe_get(results.pvalues, target) if hasattr(results, 'pvalues') else None

    # Confidence interval
    try:
        ci_df = results.conf_int()
        # conf_int may be a DataFrame or ndarray-like; handle both
        if hasattr(ci_df, "loc"):
            ci_lower, ci_upper = [float(x) for x in ci_df.loc[target].tolist()]
        else:
            # assume ordering matches params; find index
            idx = param_index.index(target)
            ci_lower, ci_upper = float(ci_df[idx, 0]), float(ci_df[idx, 1])
    except Exception:
        ci_lower = ci_upper = None

    # Build a readable significance statement
    sig_statement = "p-value unavailable"
    if pval is not None:
        if pval < 0.01:
            sig_statement = "statistically significant at p < 0.01"
        elif pval < 0.05:
            sig_statement = "statistically significant at p < 0.05"
        elif pval < 0.10:
            sig_statement = "marginally significant (p < 0.10)"
        else:
            sig_statement = "not statistically significant (p >= 0.05)"

    # Interpret coefficient: BeautyScore_z is standardized, so interpret per 1 SD change
    if coef is not None:
        effect_sentence = (
            f"A one standard-deviation increase in rated attractiveness ({target}) is associated with a "
            f"{coef:.4f} point change in the course evaluation score, holding controls constant."
        )
    else:
        effect_sentence = "Coefficient unavailable."

    # Compose description with available numbers, guarding missing values
    def fmt(x, prec=4):
        return ("{0:.{1}f}".format(x, prec)) if (x is not None) else "NA"

    description = (
        f"Variable: '{target}'. Coefficient = {fmt(coef)}; SE = {fmt(se)}; t = {fmt(tval,3)}; p = {fmt(pval,3)}; "
        f"95% CI = [{fmt(ci_lower)},{fmt(ci_upper)}]. {effect_sentence} This effect is {sig_statement}."
    )

    # The object returned contains numeric pieces suitable for downstream programmatic use
    result_object = {
        "variable": target,
        "coef": coef,
        "se": se,
        "tvalue": tval,
        "pvalue": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        # additional context where available
        "nobs": int(results.nobs) if hasattr(results, "nobs") else None,
        "model_type": getattr(results, "model", None).__class__.__name__ if hasattr(results, "model") else None
    }

    return {"object": result_object, "description": description}