def extract_final_answer(model_output):
    """
    Extracts the Female coefficient, its standard error, p-value, 95% confidence interval,
    and the corresponding odds ratio (with 95% CI) from a fitted statsmodels Logit result
    (BinaryResultsWrapper). Returns a dictionary with keys "object" and "description".

    "object" is a dict with numeric results; "description" is a brief plain-English
    interpretation of the Female effect in context (female=1 vs male=0).
    """
    import numpy as np

    res = model_output

    # Ensure the model output exposes expected attributes
    for attr in ("params", "bse", "pvalues", "conf_int"):
        if not hasattr(res, attr) and not (attr == "conf_int" and hasattr(res, "conf_int")):
            raise ValueError(f"Model output does not expose required attribute '{attr}'")

    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    conf_int = res.conf_int()  # 2-column array-like or DataFrame: lower, upper

    # Find the Female variable name (exact 'Female' preferred, else case-insensitive match)
    female_name = None
    if "Female" in params.index:
        female_name = "Female"
    else:
        for name in params.index:
            if "female" in str(name).lower():
                female_name = name
                break
    if female_name is None:
        raise ValueError("No coefficient matching 'Female' found in model parameters.")

    # Extract values robustly (works whether conf_int is DataFrame or ndarray)
    coef = float(params[female_name])
    se = float(bse[female_name])
    pval = float(pvalues[female_name])

    try:
        # If conf_int is a DataFrame
        ci_lower = float(conf_int.loc[female_name][0])
        ci_upper = float(conf_int.loc[female_name][1])
    except Exception:
        # fallback: conf_int as ndarray in same order as params.index
        idx = list(params.index).index(female_name)
        ci_lower = float(conf_int[idx, 0])
        ci_upper = float(conf_int[idx, 1])

    # Convert log-odds coefficient to odds ratio and transform CI
    odds_ratio = float(np.exp(coef))
    odds_ci_lower = float(np.exp(ci_lower))
    odds_ci_upper = float(np.exp(ci_upper))

    significant = bool(pval < 0.05)

    result_object = {
        "variable": female_name,
        "coefficient_log_odds": coef,
        "std_error": se,
        "p_value": pval,
        "coef_95CI": [ci_lower, ci_upper],
        "odds_ratio": odds_ratio,
        "odds_ratio_95CI": [odds_ci_lower, odds_ci_upper],
        "significant_at_0.05": significant
    }

    # Plain-language description
    significance_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"
    description = (
        f"The estimated log-odds coefficient for {female_name} is {coef:.4f} (SE = {se:.4f}), "
        f"p = {pval:.4g}, which is {significance_text}. "
        f"This corresponds to an odds ratio of {odds_ratio:.3f} with a 95% CI of "
        f"[{odds_ci_lower:.3f}, {odds_ci_upper:.3f}]. "
        "Interpretation: holding the listed controls constant, the coefficient measures "
        "the multiplicative change in the odds of mortgage approval for females (female=1) "
        "relative to males (female=0)."
    )

    return {"object": result_object, "description": description}