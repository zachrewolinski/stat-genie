def extract_final_answer(model_output):
    """
    Extract the coefficient, uncertainty, and interpretable effect for the is_dark indicator
    from a fitted statsmodels results object (e.g., GLMResultsWrapper / ResultWrapper
    possibly returned by get_robustcov_results).

    Returns a dict with keys:
      - "object": dict with numeric results:
          - param_name: name of the parameter found
          - coef: estimated coefficient (log rate ratio)
          - se: standard error of the coefficient
          - p_value: two-sided p-value for the coefficient
          - ci_lower, ci_upper: 95% CI on the coefficient (log scale)
          - rate_ratio: exp(coef) — multiplicative effect on red-card rate per game
          - rr_ci_lower, rr_ci_upper: 95% CI for the rate ratio
          - nobs: number of observations (if available)
          - notes: any notes about how the parameter was located
      - "description": brief human-readable interpretation of the estimate
    """
    import numpy as np

    res = model_output

    # Defensive checks
    if res is None:
        return {
            "object": None,
            "description": "No model output provided."
        }

    # Try to find the parameter name corresponding to the darker-skin indicator.
    # Commonly it's exactly "is_dark", but it might be named differently if statsmodels
    # transformed it (unlikely here) — so pick the first parameter containing 'is_dark'.
    param_name = None
    try:
        param_index = None
        if hasattr(res, 'params'):
            # params may be a pandas Series or numpy array with index
            try:
                names = list(res.params.index)
            except Exception:
                # fallback: if params is ndarray, try model.exog_names
                names = []
                if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                    names = list(res.model.exog_names)
            # find candidate
            for n in names:
                if 'is_dark' in str(n):
                    param_name = n
                    break
            if param_name is None and 'is_dark' in names:
                param_name = 'is_dark'
        else:
            # No params attribute
            return {
                "object": None,
                "description": "Model result does not have 'params' attribute; cannot locate coefficient."
            }
    except Exception as e:
        return {
            "object": None,
            "description": f"Error while locating parameter names: {e}"
        }

    if param_name is None:
        return {
            "object": None,
            "description": "Could not find a parameter whose name contains 'is_dark' in the model results. "
                           "Ensure the model included the binary indicator named 'is_dark'."
        }

    # Extract estimates
    try:
        coef = float(res.params[param_name])
    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to extract coefficient for {param_name}: {e}"
        }

    # Standard error and p-value
    se = None
    p_value = None
    try:
        if hasattr(res, 'bse'):
            se = float(res.bse[param_name])
    except Exception:
        se = None

    try:
        if hasattr(res, 'pvalues'):
            p_value = float(res.pvalues[param_name])
    except Exception:
        p_value = None

    # Confidence interval (use .conf_int() if available)
    ci_lower = ci_upper = None
    try:
        if hasattr(res, 'conf_int'):
            ci = res.conf_int()
            # conf_int() often returns a DataFrame or ndarray with indexed rows
            try:
                ci_row = ci.loc[param_name]
                ci_lower = float(ci_row[0])
                ci_upper = float(ci_row[1])
            except Exception:
                # fallback when conf_int returns ndarray in same order as params
                try:
                    # find index position of param_name
                    if hasattr(res.params, 'index'):
                        idx = list(res.params.index).index(param_name)
                        ci_lower = float(ci[idx, 0])
                        ci_upper = float(ci[idx, 1])
                except Exception:
                    # last-resort: attempt to call conf_int(alpha=0.05) and hope it's aligned
                    arr = np.asarray(ci)
                    if arr.ndim == 2 and arr.shape[0] == len(res.params):
                        # try to match by ordering of params
                        try:
                            idx = list(res.params.index).index(param_name)
                            ci_lower = float(arr[idx, 0])
                            ci_upper = float(arr[idx, 1])
                        except Exception:
                            pass
    except Exception:
        ci_lower = ci_upper = None

    # Exponentiate to get multiplicative effect (rate ratio)
    rate_ratio = None
    rr_ci_lower = rr_ci_upper = None
    try:
        rate_ratio = float(np.exp(coef))
        if ci_lower is not None and ci_upper is not None:
            rr_ci_lower = float(np.exp(ci_lower))
            rr_ci_upper = float(np.exp(ci_upper))
    except Exception:
        rate_ratio = rr_ci_lower = rr_ci_upper = None

    # Number of observations if available
    nobs = None
    try:
        if hasattr(res, 'nobs'):
            nobs = int(res.nobs)
        elif hasattr(res, 'model') and hasattr(res.model, 'data') and hasattr(res.model.data, 'nobs'):
            nobs = int(res.model.data.nobs)
    except Exception:
        nobs = None

    result_object = {
        "param_name": str(param_name),
        "coef": coef,
        "se": se,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "rate_ratio": rate_ratio,
        "rr_ci_lower": rr_ci_lower,
        "rr_ci_upper": rr_ci_upper,
        "nobs": nobs,
        "notes": "Coefficient is on log rate (log red-cards per game); exponentiated coefficient is multiplicative effect on red-card rate per game. "
                 "P-value and CI reflect the model's returned covariance (may be cluster-robust if the supplied result used get_robustcov_results)."
    }

    # Short interpretation
    if rate_ratio is not None:
        if p_value is not None:
            sig_text = ("statistically significant" if p_value < 0.05 else "not statistically significant")
        else:
            sig_text = "statistical significance unknown (p-value not available)"

        description = (
            f"The estimated effect of being coded as darker-skinned ({param_name}) is a log rate ratio = {coef:.4f} "
            f"(SE = {se:.4f}) which corresponds to a multiplicative change in red-card rate per game = {rate_ratio:.3f}. "
        )
        if rr_ci_lower is not None and rr_ci_upper is not None:
            description += f"95% CI for the rate ratio = [{rr_ci_lower:.3f}, {rr_ci_upper:.3f}]. "
        description += f"This effect is {sig_text} (p = {p_value:.3f})" if p_value is not None else description
    else:
        description = (
            "Extracted the coefficient for 'is_dark' but could not compute an interpretable rate ratio. "
            "See 'object' for raw coefficient and available diagnostics."
        )

    return {
        "object": result_object,
        "description": description
    }