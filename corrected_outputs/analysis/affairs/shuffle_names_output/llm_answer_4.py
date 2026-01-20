import numpy as np

def extract_final_answer(model_output):
    """
    Extracts the effect of HasChildren from a fitted statsmodels GLM (NegativeBinomial or Poisson).
    Returns a dict with:
      - "object": dict of extracted statistics (coef, se, pvalue, 95% CI, IRR and IRR CI, nobs)
      - "description": concise interpretation about whether having children is associated with fewer affairs
    """
    # Try to get parameter series
    try:
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        conf = model_output.conf_int()  # DataFrame with 0 and 1 columns
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract model parameters from model_output: {e}"
        }

    # Find the parameter name corresponding to HasChildren
    target_name = None
    if 'HasChildren' in params.index:
        target_name = 'HasChildren'
    else:
        # try to find a parameter whose name contains 'HasChildren' (robustness)
        matches = [n for n in params.index if 'HasChildren' in str(n)]
        if len(matches) == 1:
            target_name = matches[0]
        elif len(matches) > 1:
            # prefer exact match if present, otherwise take first match
            target_name = matches[0]

    if target_name is None:
        return {
            "object": None,
            "description": "The model does not appear to contain a parameter named 'HasChildren'."
        }

    try:
        coef = float(params[target_name])
    except Exception:
        return {
            "object": None,
            "description": f"Could not read coefficient for parameter '{target_name}'."
        }

    se = float(bse[target_name]) if (hasattr(bse, "index") and target_name in bse.index) else None
    pval = float(pvalues[target_name]) if (hasattr(pvalues, "index") and target_name in pvalues.index) else None

    # Confidence interval: conf_int returns DataFrame; columns may be [0,1]
    try:
        row = conf.loc[target_name]
        # Use iloc to be robust to column names
        ci_lower = float(row.iloc[0])
        ci_upper = float(row.iloc[1])
    except Exception:
        ci_lower = None
        ci_upper = None

    # Incidence Rate Ratio (IRR) and CI
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
    irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

    # Number of observations if available
    nobs = getattr(model_output, 'nobs', None)
    if nobs is None:
        try:
            nobs = int(model_output.model.endog.shape[0])
        except Exception:
            nobs = None

    # Significance at alpha = 0.05
    significant = None
    if pval is not None:
        significant = (pval < 0.05)

    # Build object to return
    result_object = {
        "parameter": target_name,
        "coef_log_count": coef,            # log change in expected count
        "std_error": se,
        "p_value": pval,
        "ci_95_log": [ci_lower, ci_upper],
        "incidence_rate_ratio (IRR)": irr,              # exp(coef)
        "IRR_95_CI": [irr_ci_lower, irr_ci_upper],
        "nobs": int(nobs) if (nobs is not None) else None,
        "significant_at_0.05": significant
    }

    # Prepare formatted strings safely for interpretation
    irr_str = f"{irr:.3f}"
    irr_ci_lower_str = f"{irr_ci_lower:.3f}" if irr_ci_lower is not None else "NA"
    irr_ci_upper_str = f"{irr_ci_upper:.3f}" if irr_ci_upper is not None else "NA"
    pval_str = f"{pval:.3g}" if pval is not None else "NA"

    # Short interpretation
    if pval is None:
        interpretation = ("Extracted statistics for 'HasChildren' but p-value not available; "
                          "cannot determine statistical significance.")
    else:
        if significant:
            if irr < 1:
                interpretation = (f"Having children is associated with a statistically significant decrease "
                                  f"in the expected number of extramarital affairs (IRR = {irr_str}, "
                                  f"95% CI [{irr_ci_lower_str}, {irr_ci_upper_str}], p = {pval_str}).")
            else:
                interpretation = (f"Having children is associated with a statistically significant increase "
                                  f"in the expected number of extramarital affairs (IRR = {irr_str}, "
                                  f"95% CI [{irr_ci_lower_str}, {irr_ci_upper_str}], p = {pval_str}).")
        else:
            # not statistically significant
            if irr < 1:
                direction = "decrease"
            elif irr > 1:
                direction = "increase"
            else:
                direction = "no change"
            interpretation = (f"No statistically significant association between having children and the number "
                              f"of extramarital affairs (IRR = {irr_str}, 95% CI [{irr_ci_lower_str}, {irr_ci_upper_str}], p = {pval_str}). "
                              f"The point estimate suggests a {direction} in affairs but it is not statistically significant.")

    return {"object": result_object, "description": interpretation}