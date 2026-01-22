def extract_final_answer(model_output):
    """
    Extracts the effect of 'Children' on the count (non-inflation) part of a
    Zero-Inflated Negative Binomial model fitted with statsmodels.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, se, pvalue, 95% CI, IRR, IRR CI, significance)
      - "description": plain-language interpretation of the effect in context

    This function tries to be robust to parameter naming conventions used by
    statsmodels (e.g., "Children" for the count equation and "inflate_Children"
    for the inflation equation).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Get parameter index / names
    try:
        param_index = list(res.params.index)
    except Exception:
        # Fallback if params isn't a pandas Series with index (unlikely)
        param_index = [str(n) for n in range(len(res.params))]

    # Find the count-equation parameter name for Children.
    # We prefer a name that includes 'Children' but does NOT start with 'inflate' (or contain 'infl')
    name_candidates = [n for n in param_index if 'Children' in n or 'children' in n or n.lower() == 'children']
    count_name = None
    if name_candidates:
        # prefer candidate without 'infl' substring
        for n in name_candidates:
            if 'infl' not in n.lower():
                count_name = n
                break
        if count_name is None:
            # if all candidates are inflation-related, pick the first anyway
            count_name = name_candidates[0]

    if count_name is None:
        raise KeyError(
            "Could not find a parameter name for 'Children' in model params. "
            "Model param names: {}".format(param_index)
        )

    # Extract stats for the count equation coefficient
    coef = float(res.params[count_name])
    try:
        se = float(res.bse[count_name])
    except Exception:
        # Some result objects use .bse as an array aligned with params
        # Try to align by index
        bse_series = getattr(res, 'bse', None)
        if isinstance(bse_series, (list, tuple, np.ndarray)):
            # find position
            pos = param_index.index(count_name)
            se = float(bse_series[pos])
        else:
            se = None

    try:
        pvalue = float(res.pvalues[count_name])
    except Exception:
        # fallback
        pvalue = None

    # Confidence intervals
    try:
        ci = res.conf_int().loc[count_name].astype(float)
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        # fallback: use coef +/- 1.96*se if se available
        if se is not None:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            ci_lower = ci_upper = None

    # Incidence rate ratio (IRR) and CI on exponentiated scale
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
    irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

    # Statistical significance at alpha=0.05 (two-sided), if p-value available
    significance = None
    if pvalue is not None:
        significance = (pvalue < 0.05)

    # Compose a short interpretation string based on sign and significance
    if coef < 0:
        direction = "Having children is associated with a lower expected rate of affairs (negative coefficient)."
    elif coef > 0:
        direction = "Having children is associated with a higher expected rate of affairs (positive coefficient)."
    else:
        direction = "No association (coefficient is 0)."

    if significance is True:
        significance_text = "This association is statistically significant (p = {:.3g}).".format(pvalue)
    elif significance is False:
        significance_text = "This association is not statistically significant (p = {:.3g}).".format(pvalue) if pvalue is not None else "Statistical significance could not be determined (p-value missing)."
    else:
        significance_text = "Statistical significance could not be determined (p-value missing)."

    description = (
        f"Count-equation parameter for 'Children' = {count_name}. "
        f"Coef = {coef:.4f}, SE = {se:.4f}." if se is not None else
        f"Count-equation parameter for 'Children' = {count_name}. Coef = {coef:.4f}."
    )
    description += (
        f" 95% CI for coef = [{ci_lower:.4f}, {ci_upper:.4f}]."
        f" Exponentiated (IRR) = {irr:.4f} with 95% CI = [{irr_ci_lower:.4f}, {irr_ci_upper:.4f}]. "
    )
    description += direction + " " + significance_text
    description += " Interpretation: the coefficient is on the log-count (log incidence) scale; exp(coef) is the multiplicative change in the expected count of affairs for married respondents with children (Children=1) relative to those without children (Children=0), holding other covariates constant."

    # Construct the object to return (numbers and metadata)
    out_obj = {
        "param_name_count": count_name,
        "coef": coef,
        "se": se,
        "pvalue": pvalue,
        "ci_coef_lower": ci_lower,
        "ci_coef_upper": ci_upper,
        "irr": irr,
        "irr_ci_lower": irr_ci_lower,
        "irr_ci_upper": irr_ci_upper,
        "significant_at_0.05": significance,
    }

    return {"object": out_obj, "description": description}