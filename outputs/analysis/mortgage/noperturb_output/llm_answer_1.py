def extract_final_answer(model_output):
    """
    Extracts statistics for the 'female' coefficient from a fitted statsmodels binary model
    (Logit/GLM results wrapper). Returns a dictionary with keys:
      - "object": a dict containing coefficient, p-value, 95% CI, odds ratio and OR 95% CI,
                  significance at 0.05, and sample size (if available).
      - "description": a plain-language interpretation of these statistics in context.

    Usage:
      result = extract_final_answer(fitted_model)
    """
    import numpy as np

    res = model_output

    # Prepare placeholders
    coef = None
    pvalue = None
    ci = None
    odds_ratio = None
    or_ci = None
    nobs = None
    significant_0_05 = None

    # Get params object
    params = getattr(res, "params", None)
    if params is None:
        raise ValueError("Provided model_output does not have a 'params' attribute.")

    # Ensure 'female' is present
    if "female" not in params.index:
        raise KeyError("'female' not found in model parameters. Available parameters: {}".format(list(params.index)))

    # Extract coefficient
    coef = float(params["female"])

    # p-value
    pvals = getattr(res, "pvalues", None)
    if pvals is not None and "female" in pvals.index:
        pvalue = float(pvals["female"])

    # Confidence interval for coefficient (try a few access patterns)
    ci_mat = None
    try:
        ci_mat = res.conf_int()
    except Exception:
        ci_mat = None

    if ci_mat is not None:
        # If conf_int returns a DataFrame with index
        try:
            if hasattr(ci_mat, "loc"):
                ci_row = ci_mat.loc["female"].values.astype(float)
            else:
                # conf_int returned an ndarray; find index of 'female' in params
                idx = list(params.index).index("female")
                ci_row = np.asarray(ci_mat, dtype=float)[idx]
        except Exception:
            ci_row = None

        if ci_row is not None:
            # Ensure shape (lower, upper)
            ci = [float(ci_row[0]), float(ci_row[1])]

    # Odds ratio and its CI (if coef and ci available)
    if coef is not None:
        odds_ratio = float(np.exp(coef))
    if ci is not None:
        or_ci = [float(np.exp(ci[0])), float(np.exp(ci[1]))]

    # Sample size if available
    if hasattr(res, "nobs"):
        try:
            nobs = int(res.nobs)
        except Exception:
            nobs = None
    else:
        # Try to get from model endog length
        try:
            nobs = int(len(res.model.endog))
        except Exception:
            nobs = None

    # Significance at 0.05 (if p-value available)
    if pvalue is not None:
        significant_0_05 = bool(pvalue < 0.05)

    # Build the object to return
    object_dict = {
        "coef_log_odds": coef,
        "coef_95ci": ci,
        "p_value": pvalue,
        "odds_ratio": odds_ratio,
        "odds_ratio_95ci": or_ci,
        "significant_at_0.05": significant_0_05,
        "nobs": nobs,
    }

    # Build human-readable description
    # Explain interpretation of coefficient and odds ratio in context
    desc_parts = []
    desc_parts.append("Extracted statistics for the 'female' indicator from the fitted logistic model.")
    if coef is not None:
        desc_parts.append(
            "Coefficient (log-odds) = {:.4f}.".format(coef)
        )
    if pvalue is not None:
        desc_parts.append(
            "p-value = {:.4g} ({} at α=0.05).".format(pvalue, "statistically significant" if significant_0_05 else "not statistically significant")
        )
    if ci is not None:
        desc_parts.append(
            "95% CI for coefficient = [{:.4f}, {:.4f}].".format(ci[0], ci[1])
        )
    if odds_ratio is not None:
        desc_parts.append(
            "Odds ratio = {:.4f}.".format(odds_ratio)
        )
    if or_ci is not None:
        desc_parts.append(
            "95% CI for odds ratio = [{:.4f}, {:.4f}].".format(or_ci[0], or_ci[1])
        )
    if nobs is not None:
        desc_parts.append("Sample size (n) = {}.".format(nobs))

    # Add interpretation guidance
    desc_parts.append(
        "Interpretation: the coefficient is the change in log-odds of mortgage approval associated with being female (female=1 vs male=0). "
        "An odds ratio >1 indicates higher odds of approval for females, <1 indicates lower odds. "
        "Use the p-value and confidence interval to judge statistical evidence for an effect."
    )

    description = " ".join(desc_parts)

    return {"object": object_dict, "description": description}