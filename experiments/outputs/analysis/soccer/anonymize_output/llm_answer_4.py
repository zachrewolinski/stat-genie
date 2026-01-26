def extract_final_answer(model_output):
    """
    Extract statistics for the 'IsDark' coefficient from a fitted statsmodels GLMResultsWrapper.
    Returns a dictionary with:
      - "object": dict of numeric results (coef, se, pvalue, 95% CI on coef scale, IRR and its 95% CI, nobs)
      - "description": short plain-language interpretation including a yes/no answer to whether
                       dark-skinned players are more likely to receive red cards (based on sign of IRR>1
                       and conventional p<0.05 threshold).
    """
    import numpy as np
    res = model_output

    # Prepare default "missing" return if IsDark not present
    if 'IsDark' not in getattr(res, "params", {}):
        return {
            "object": None,
            "description": "The fitted model object does not contain a parameter named 'IsDark'."
        }

    # Extract coefficient, SE, p-value
    coef = float(res.params['IsDark'])
    # Use robust/clustered SEs if available in result attributes (res.bse reflects cov_type used during fit)
    se = float(res.bse['IsDark'])
    # p-values are available as res.pvalues if the fit produced them
    pvalue = float(res.pvalues['IsDark']) if 'IsDark' in res.pvalues.index else None

    # Confidence interval on coefficient (log rate ratio) (95% by default)
    try:
        ci = res.conf_int().loc['IsDark'].astype(float).values
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        # Fallback if .conf_int() indexing differs
        ci_vals = res.conf_int().values
        idx = list(res.params.index).index('IsDark')
        ci_lower, ci_upper = float(ci_vals[idx, 0]), float(ci_vals[idx, 1])

    # Incidence rate ratio (IRR) and CI on IRR scale
    irr = float(np.exp(coef))
    irr_ci_lower, irr_ci_upper = float(np.exp(ci_lower)), float(np.exp(ci_upper))

    # Number of observations
    try:
        nobs = int(res.nobs)
    except Exception:
        try:
            nobs = int(len(res.model.endog))
        except Exception:
            nobs = None

    # Simple decision rule for the yes/no question
    # We consider "more likely" if IRR > 1 and p < 0.05 (conventional threshold).
    is_more_likely = None
    if pvalue is not None:
        is_more_likely = (irr > 1) and (pvalue < 0.05)
    else:
        # If no p-value, base decision purely on point estimate direction
        is_more_likely = irr > 1

    # Compose a concise description
    direction = "higher" if irr > 1 else "lower" if irr < 1 else "no difference"
    percent_change = (irr - 1) * 100.0
    desc = (
        f"IsDark coefficient = {coef:.4f} (SE = {se:.4f}, p = {pvalue:.4g}). "
        f"This corresponds to an incidence rate ratio (IRR) = {irr:.3f} "
        f"(95% CI: [{irr_ci_lower:.3f}, {irr_ci_upper:.3f}]). "
        f"Interpreted: Dark-skinned players have a {percent_change:.1f}% {direction} rate of red cards "
        f"compared to light-skinned players, conditional on the included controls. "
    )
    if pvalue is not None:
        if pvalue < 0.05 and irr > 1:
            desc += "By conventional standards (p < 0.05), this is statistically significant evidence that dark-skinned players are more likely to receive red cards."
        elif pvalue < 0.05 and irr < 1:
            desc += "By conventional standards (p < 0.05), this is statistically significant evidence that dark-skinned players are less likely to receive red cards."
        else:
            desc += "This effect is not statistically significant at the 0.05 level, so we cannot conclude a reliable difference."
    else:
        desc += "No p-value was available; conclusion is based only on the point estimate direction."

    result_object = {
        "coef": coef,
        "se": se,
        "pvalue": pvalue,
        "conf_int_coef": [ci_lower, ci_upper],
        "irr": irr,
        "conf_int_irr": [irr_ci_lower, irr_ci_upper],
        "nobs": nobs,
        "is_more_likely_by_conventional_test": is_more_likely
    }

    return {"object": result_object, "description": desc}