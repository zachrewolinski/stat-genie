def extract_final_answer(model_output):
    """
    Extracts the estimated effect of 'beauty' from a statsmodels RegressionResultsWrapper.
    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, std_error, t, p_value, ci_lower, ci_upper)
      - "description": human-readable interpretation of the estimate in context

    Expects the model to have been fit with clustered standard errors (so .bse and .pvalues
    reflect the clustered covariance if that was requested when fitting).
    """
    import numpy as np

    res = model_output
    var = 'beauty'

    # Basic validation
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params).")
    if var not in res.params.index:
        raise ValueError(f"Variable '{var}' not found in model results. Available params: {list(res.params.index)}")

    # Coefficient
    coef = float(res.params[var])

    # Standard error (res.bse should reflect clustered SE if used in fit)
    try:
        se = float(res.bse[var])
    except Exception:
        # fallback: compute from covariance matrix
        cov = res.cov_params()
        idx = list(res.params.index).index(var)
        se = float(np.sqrt(np.diag(cov))[idx])

    # t-value and p-value
    t_val = float(res.tvalues[var]) if hasattr(res, 'tvalues') else (coef / se if se != 0 else float('nan'))
    p_val = float(res.pvalues[var]) if hasattr(res, 'pvalues') else None

    # 95% confidence interval
    ci = res.conf_int()
    try:
        # try DataFrame-style extraction
        ci_low = float(ci.loc[var][0])
        ci_high = float(ci.loc[var][1])
    except Exception:
        # assume numpy array, find index
        idx = list(res.params.index).index(var)
        ci_low = float(ci[idx, 0])
        ci_high = float(ci[idx, 1])

    # Build return object
    numeric_object = {
        'variable': var,
        'coef': coef,
        'std_error': se,
        't_value': t_val,
        'p_value': p_val,
        'ci_lower_95': ci_low,
        'ci_upper_95': ci_high
    }

    # Human-readable description
    # Interpret effect: change in evaluation score (eval is 1-5) per one-unit beauty increase.
    p_text = ("p = {:.3g}".format(p_val)) if p_val is not None else "p-value unavailable"
    significance = "statistically significant" if (p_val is not None and p_val < 0.05) else "not statistically significant"
    desc = (
        f"Estimated effect of instructor physical attractiveness ('beauty') on course evaluation ('eval'): "
        f"coefficient = {coef:.4f} (SE = {se:.4f}), 95% CI [{ci_low:.4f}, {ci_high:.4f}], {p_text}. "
        f"This means a one-unit increase in the panel-rated beauty score is associated with an expected "
        f"{coef:.4f} point change in the evaluation score (on a 1–5 scale), holding controls constant. "
        f"The effect is {significance} at the 0.05 level. Standard errors were clustered at the instructor (prof) level."
    )

    return {"object": numeric_object, "description": desc}