def extract_final_answer(model_output):
    """
    Extract statistics for the 'DarkSkin' coefficient from a fitted statsmodels GLM/ResultsWrapper.
    Returns a dictionary with:
      - "object": dict of numeric results (coef, se, pvalue, 95% CI, IRR and IRR CI, percent change, nobs)
      - "description": short interpretation in the context of whether dark-skinned players are more likely
                       than light-skinned players to receive red cards (based on direction and p-value).
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Helper to get attribute safely
    def _get_attr(obj, name, default=None):
        return getattr(obj, name) if hasattr(obj, name) else default

    # Try to obtain parameter names and values
    params = _get_attr(res, 'params', None)
    if params is None:
        # Some wrappers store results in ._results or .results
        if hasattr(res, '_results'):
            params = _get_attr(res._results, 'params', None)
            res = res._results if params is not None else res
        if params is None and hasattr(res, 'results'):
            params = _get_attr(res.results, 'params', None)
            res = res.results if params is not None else res

    if params is None:
        return {
            "object": None,
            "description": "Could not find model parameters in the provided model_output object."
        }

    # Locate the parameter name for DarkSkin (allow slight name variations)
    param_index = list(params.index)
    target_name = None
    for name in param_index:
        if name == 'DarkSkin' or 'DarkSkin' in name:
            target_name = name
            break

    if target_name is None:
        return {
            "object": None,
            "description": "The coefficient for 'DarkSkin' was not found among model parameters."
        }

    # Extract estimates
    try:
        coef = float(params.loc[target_name])
    except Exception:
        coef = float(params[target_name])

    # Standard error, p-value
    bse = None
    pvalue = None
    try:
        bse = float(res.bse.loc[target_name])
    except Exception:
        try:
            bse = float(res.bse[target_name])
        except Exception:
            bse = None

    try:
        pvalue = float(res.pvalues.loc[target_name])
    except Exception:
        try:
            pvalue = float(res.pvalues[target_name])
        except Exception:
            pvalue = None

    # Confidence interval
    ci_lower = ci_upper = None
    try:
        ci = res.conf_int()
        # conf_int often returns a DataFrame with two columns (lower, upper)
        if isinstance(ci, pd.DataFrame):
            row = ci.loc[target_name]
            # handle either column labels [0,1] or named
            ci_lower = float(row.iloc[0])
            ci_upper = float(row.iloc[1])
        else:
            # numpy array fallback
            idx = param_index.index(target_name)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
    except Exception:
        ci_lower = ci_upper = None

    # Incidence Rate Ratio (IRR) and CI on IRR scale
    irr = np.exp(coef) if coef is not None else None
    irr_ci_lower = np.exp(ci_lower) if ci_lower is not None else None
    irr_ci_upper = np.exp(ci_upper) if ci_upper is not None else None
    percent_change = (irr - 1) * 100 if irr is not None else None  # percent change in rate

    # Number of observations (if available)
    nobs = None
    try:
        nobs = int(getattr(res, 'nobs', None))
    except Exception:
        try:
            nobs = int(getattr(res, 'nobs', np.nan))
        except Exception:
            nobs = None

    result_object = {
        "variable": target_name,
        "coef": coef,
        "se": bse,
        "pvalue": pvalue,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "IRR": irr,
        "IRR_ci_lower": irr_ci_lower,
        "IRR_ci_upper": irr_ci_upper,
        "percent_change": percent_change,
        "nobs": nobs
    }

    # Interpretation: direction + statistical significance at alpha=0.05
    interpretation = []
    interpretation.append(f"Coefficient for {target_name} = {coef:.4g}" if coef is not None else f"Coefficient for {target_name} = NA")
    if bse is not None:
        interpretation.append(f"(SE = {bse:.4g})")
    if pvalue is not None:
        interpretation.append(f", p = {pvalue:.4g}")
    if irr is not None:
        interpretation.append(f"; IRR = {irr:.3f}")
        if irr_ci_lower is not None and irr_ci_upper is not None:
            interpretation.append(f" (95% CI {irr_ci_lower:.3f} to {irr_ci_upper:.3f})")
    interpretation = ''.join(interpretation)

    # Conclude whether dark-skinned players are more likely to receive red cards:
    conclusion = ""
    if pvalue is None:
        conclusion = ("Could not determine statistical significance (p-value unavailable). "
                      "See extracted estimates in 'object'.")
    else:
        alpha = 0.05
        if pvalue < alpha:
            if irr is not None and irr > 1:
                conclusion = ("Yes — statistically significant evidence that dark-skinned players receive red cards "
                              f"at a higher rate than light-skinned players (IRR={irr:.3f}, p={pvalue:.4g}).")
            elif irr is not None and irr < 1:
                conclusion = ("Yes — statistically significant evidence that dark-skinned players receive red cards "
                              f"at a lower rate than light-skinned players (IRR={irr:.3f}, p={pvalue:.4g}).")
            else:
                conclusion = ("The DarkSkin coefficient is statistically significant, but direction/IRR unavailable.")
        else:
            conclusion = ("No — the association is not statistically significant at the 0.05 level "
                          f"(IRR={irr:.3f} if available, p={pvalue:.4g}). Evidence is inconclusive.")

    description = f"{interpretation} {conclusion}"

    return {
        "object": result_object,
        "description": description
    }