def extract_final_answer(model_output):
    """
    Extract the coefficient, robust standard error, p-value, 95% CI, and IRR
    for the 'HasChildren' predictor from a fitted statsmodels GLMResultsWrapper.

    Returns a dict with:
      - "object": dict of numeric results (coef, se, pvalue, ci_lower, ci_upper, irr, irr_ci_lower, irr_ci_upper)
      - "description": human-readable interpretation in context

    The function is robust to small variations in the parameter name (tries exact match,
    then case-insensitive substring match).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to find the parameter name in case of minor naming differences
    def _find_param_name(params_index, target='HasChildren'):
        # Exact match
        if target in params_index:
            return target
        # Case-insensitive exact match
        for name in params_index:
            if name.lower() == target.lower():
                return name
        # Substring match (case-insensitive)
        for name in params_index:
            if target.lower() in name.lower():
                return name
        # No match
        return None

    params_index = list(res.params.index)
    param_name = _find_param_name(params_index, 'HasChildren')
    if param_name is None:
        raise KeyError("Could not find a parameter matching 'HasChildren' in model parameters: {}".format(params_index))

    # Extract statistics with fallbacks
    try:
        coef = float(res.params[param_name])
    except Exception:
        coef = float(res.params.loc[param_name])

    # Standard error (robust if model was fit with cov_type)
    try:
        se = float(res.bse[param_name])
    except Exception:
        se = float(res.bse.loc[param_name])

    # p-value
    try:
        pval = float(res.pvalues[param_name])
    except Exception:
        pval = float(res.pvalues.loc[param_name])

    # 95% confidence interval
    try:
        ci = res.conf_int().loc[param_name]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # conf_int might return an array; try indexing by position
        ci_array = res.conf_int()
        # If conf_int returned ndarray with shape (k,2) and params_index aligns, find index
        try:
            idx = params_index.index(param_name)
            ci_lower = float(ci_array[idx, 0])
            ci_upper = float(ci_array[idx, 1])
        except Exception:
            # As a last resort, set NaNs
            ci_lower = float('nan')
            ci_upper = float('nan')

    # Because GLM NB uses a log link by default, exponentiate coef to get IRR (multiplicative effect on expected count)
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower)) if not (pd.isna(ci_lower) or np.isinf(ci_lower)) else float('nan')
    irr_ci_upper = float(np.exp(ci_upper)) if not (pd.isna(ci_upper) or np.isinf(ci_upper)) else float('nan')

    # Build a concise interpretation
    significance = "statistically significant" if (not pd.isna(pval) and pval < 0.05) else "not statistically significant"
    direction = "decrease" if coef < 0 else ("increase" if coef > 0 else "no change")
    description = (
        f"The estimated coefficient for HasChildren is {coef:.4f} (SE={se:.4f}, p={pval:.4g}), "
        f"with a 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"Because this is a Negative Binomial GLM with a log link, this coefficient is on the log scale: "
        f"having children is associated with a multiplicative change in the expected affair frequency = IRR={irr:.4f} "
        f"(95% CI for IRR: [{irr_ci_lower:.4f}, {irr_ci_upper:.4f}]). "
        f"This corresponds to a relative {direction} in expected affair counts for respondents with children versus without, "
        f"controlling for gender, age, years married, education level, and religiosity. "
        f"The effect is {significance} (alpha=0.05)."
    )

    result_object = {
        "coef": coef,
        "se": se,
        "pvalue": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "param_name": param_name
    }

    return {"object": result_object, "description": description}