def extract_final_answer(model_output):
    """
    Extracts the coefficient, robust SE, z-stat, p-value, odds ratio, and 95% CI
    for the 'female' variable from the supplied model output object.

    Returns:
      {
        "object": {
          "coef": float,            # log-odds coefficient for female
          "std_err": float,         # robust standard error
          "z": float,               # z-statistic
          "p_value": float,         # two-sided p-value
          "odds_ratio": float,      # exp(coef)
          "ci_lower": float,        # lower bound of 95% CI for odds ratio
          "ci_upper": float         # upper bound of 95% CI for odds ratio
        },
        "description": str          # brief interpretation in context
      }
    """
    import numpy as np
    from scipy import stats

    # Helper: obtain parameter names and values robustly
    params = None
    bse = None
    cov = None
    exog_names = None

    # Direct attributes when a statsmodels-like results object is returned
    if hasattr(model_output, "params"):
        params = model_output.params
    if hasattr(model_output, "bse"):
        bse = model_output.bse
    if hasattr(model_output, "cov_params"):
        try:
            cov = model_output.cov_params()
        except Exception:
            cov = None

    # If wrapped object (RobustResultsWrapper) stores base in _base
    if params is None and hasattr(model_output, "_base"):
        base = getattr(model_output, "_base")
        if hasattr(base, "params"):
            params = base.params
        if hasattr(model_output, "bse"):
            bse = model_output.bse
        if cov is None and hasattr(model_output, "cov_params"):
            try:
                cov = model_output.cov_params()
            except Exception:
                cov = None

    # Attempt to get exogenous names (variable names)
    if hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
        exog_names = list(model_output.model.exog_names)
    elif hasattr(model_output, "_base") and hasattr(model_output._base, "model") and hasattr(model_output._base.model, "exog_names"):
        exog_names = list(model_output._base.model.exog_names)

    # Convert params and bse to dict-like using exog_names if they are arrays
    if exog_names is not None:
        # If params is a pandas Series it will be indexable by name; else build mapping
        if not hasattr(params, "get") and not (hasattr(params, "index") and hasattr(params, "__getitem__")):
            # params is array-like; build mapping
            params = dict(zip(exog_names, np.asarray(params).tolist()))
        else:
            # Convert pandas Series to dict for consistent access
            try:
                params = dict(params)
            except Exception:
                # leave as-is if conversion fails
                pass

        if not hasattr(bse, "get") and not (hasattr(bse, "index") and hasattr(bse, "__getitem__")):
            if bse is not None:
                bse = dict(zip(exog_names, np.asarray(bse).tolist()))
        else:
            try:
                bse = dict(bse)
            except Exception:
                pass

    # Ensure 'female' is present
    if params is None:
        raise ValueError("Could not extract model parameters from model_output.")
    if isinstance(params, dict):
        if 'female' not in params:
            raise KeyError("The parameter 'female' was not found in the model output parameters.")
        coef = float(params['female'])
    else:
        # params might be a pandas Series-like; try indexing
        try:
            coef = float(params['female'])
        except Exception:
            raise KeyError("Unable to locate 'female' coefficient in model_output.params.")

    # Extract standard error
    if bse is None:
        # Try to compute se from covariance matrix if available
        if cov is not None:
            # cov may be a DataFrame or ndarray; try to get female entry
            try:
                if hasattr(cov, "loc"):
                    var_f = float(cov.loc['female', 'female'])
                else:
                    # cov is ndarray; need exog_names to map index
                    if exog_names is None:
                        raise ValueError("No exog_names available to map covariance matrix to variables.")
                    idx = exog_names.index('female')
                    var_f = float(np.asarray(cov)[idx, idx])
                se = float(np.sqrt(max(var_f, 0.0)))
            except Exception:
                raise ValueError("Could not extract robust standard error for 'female' from covariance.")
        else:
            raise ValueError("Standard errors not found in model_output (no bse and no cov).")
    else:
        # bse could be dict or Series-like
        if isinstance(bse, dict):
            if 'female' not in bse:
                raise KeyError("The standard error for 'female' was not found in model output.")
            se = float(bse['female'])
        else:
            try:
                se = float(bse['female'])
            except Exception:
                raise KeyError("Unable to locate 'female' standard error in model_output.bse.")

    # Compute z, p-value, odds ratio and 95% CI
    if se == 0:
        z = float('inf') if coef != 0 else 0.0
        p_value = 0.0 if z == float('inf') else 1.0
    else:
        z = coef / se
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    odds_ratio = float(np.exp(coef))
    ci_low_log = coef - 1.96 * se
    ci_high_log = coef + 1.96 * se
    ci_lower = float(np.exp(ci_low_log))
    ci_upper = float(np.exp(ci_high_log))

    # Build the object to return
    result_obj = {
        "coef": float(coef),
        "std_err": float(se),
        "z": float(z),
        "p_value": float(p_value),
        "odds_ratio": float(odds_ratio),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper)
    }

    # Build a concise interpretation
    alpha = 0.05
    direction = "increase" if coef > 0 else "decrease" if coef < 0 else "no change"
    significance = "statistically significant" if p_value < alpha else "not statistically significant"
    desc = (
        f"The estimated log-odds coefficient for 'female' is {coef:.4f} (SE = {se:.4f}), "
        f"z = {z:.2f}, p = {p_value:.4f}. This corresponds to an odds ratio of {odds_ratio:.3f} "
        f"with a 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]. "
        f"At the {alpha:.2f} significance level, this effect is {significance}; "
        f"point estimate implies a {direction} in the odds of mortgage acceptance for female applicants "
        f"(i.e. a multiplicative change of {odds_ratio:.3f} in odds compared to male applicants)."
    )

    return {"object": result_obj, "description": desc}