def extract_final_answer(model_output):
    """
    Extracts statistics for the 'beauty_z' coefficient from a fitted statsmodels
    RegressionResultsWrapper and returns a summary dict.

    Returns:
      {
        "object": {  # numeric results
          "coef": float,
          "std_err": float,
          "t_value": float,
          "p_value": float,
          "ci_lower": float or None,
          "ci_upper": float or None,
          "n_obs": int or float
        },
        "description": str  # plain-language interpretation
      }
    """
    # Defensive checks / imports
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError("Provided model_output does not look like a fitted statsmodels result.") from e

    var = 'beauty_z'
    if var not in params.index:
        raise ValueError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    # Extract core statistics
    coef = float(params[var])

    # Standard error, t, p
    try:
        std_err = float(model_output.bse[var])
    except Exception:
        # fallback if bse is unordered array
        try:
            idx = list(params.index).index(var)
            std_err = float(model_output.bse[idx])
        except Exception:
            std_err = None

    try:
        t_value = float(model_output.tvalues[var])
    except Exception:
        try:
            idx = list(params.index).index(var)
            t_value = float(model_output.tvalues[idx])
        except Exception:
            t_value = None

    try:
        p_value = float(model_output.pvalues[var])
    except Exception:
        try:
            idx = list(params.index).index(var)
            p_value = float(model_output.pvalues[idx])
        except Exception:
            p_value = None

    # Confidence interval (95%)
    try:
        ci = model_output.conf_int().loc[var]
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        try:
            ci_all = model_output.conf_int()
            idx = list(params.index).index(var)
            ci_lower, ci_upper = float(ci_all[idx, 0]), float(ci_all[idx, 1])
        except Exception:
            ci_lower, ci_upper = None, None

    # Number of observations
    try:
        n_obs = int(model_output.nobs)
    except Exception:
        n_obs = None

    # Statistical significance decision (alpha = 0.05), only if p_value available
    significant = None
    if p_value is not None:
        significant = (p_value < 0.05)

    # Build the object to return
    result_object = {
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs
    }

    # Construct an interpretation description
    if p_value is None:
        sig_text = "p-value unavailable, cannot determine statistical significance."
    else:
        sig_text = "statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)"

    desc = (
        f"Estimated effect of instructor attractiveness (beauty_z) on teaching evaluation:\n"
        f"Coefficient = {coef:.4f}; SE = {std_err:.4f} ; t = {t_value:.3f} ; p = {p_value:.4g}.\n"
        f"95% CI = [{ci_lower if ci_lower is not None else 'NA'}, {ci_upper if ci_upper is not None else 'NA'}].\n"
        f"Based on this model (n = {n_obs}), a one standard-deviation increase in standardized attractiveness "
        f"is associated with a {coef:.3f}-point change on the 1–5 evaluation scale. This effect is {sig_text}."
    )

    return {"object": result_object, "description": desc}