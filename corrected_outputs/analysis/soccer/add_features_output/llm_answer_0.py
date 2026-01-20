def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, p-value, 95% CI, and incidence-rate-ratio (IRR) for the
    'DarkSkin' variable from a statsmodels GLMResultsWrapper (clustered SEs preserved).
    Returns a dictionary with numeric outputs under "object" and a plain-language
    interpretation under "description".
    """
    import numpy as np
    # Safety checks
    res = model_output
    required_attrs = ('params', 'bse', 'pvalues', 'conf_int')
    for attr in required_attrs:
        if not hasattr(res, attr):
            return {
                "object": None,
                "description": f"Model output missing required attribute '{attr}'. Cannot extract results."
            }

    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    conf = res.conf_int()  # DataFrame-like with index of parameter names

    target = 'DarkSkin'
    if target not in params.index:
        return {
            "object": None,
            "description": f"Variable '{target}' not found in model parameters. Available params: {list(params.index)}"
        }

    # Extract numeric values
    coef = float(params[target])
    se = float(bse[target]) if target in bse.index else None
    pval = float(pvalues[target]) if target in pvalues.index else None
    try:
        ci_low, ci_high = map(float, conf.loc[target])
    except Exception:
        # conf_int may have different structure; try other access patterns
        ci = conf.loc[target].values
        ci_low, ci_high = float(ci[0]), float(ci[1])

    # Incidence Rate Ratio (IRR) and its CI
    irr = float(np.exp(coef))
    irr_ci_low, irr_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    # Simple significance/conclusion summary
    alpha = 0.05
    if pval is None:
        signif_text = "p-value not available"
        conclusion_bool = None
    else:
        signif_text = "statistically significant" if pval < alpha else "not statistically significant"
        conclusion_bool = (coef > 0) and (pval < alpha)

    direction = (
        "higher rate of red cards for darker-skinned players"
        if coef > 0 else
        "lower rate of red cards for darker-skinned players"
        if coef < 0 else
        "no difference in rate"
    )

    # Build output object (numbers) and description (text interpretation)
    out_object = {
        "variable": target,
        "coef": coef,
        "std_error": se,
        "p_value": pval,
        "conf_int_95": [ci_low, ci_high],
        "IRR": irr,
        "IRR_conf_int_95": [irr_ci_low, irr_ci_high],
        "conclusion_darker_more_likely_at_0.05": conclusion_bool
    }

    description = (
        f"DarkSkin coefficient = {coef:.4f} (SE = {se:.4f}, p = {pval:.4f}), 95% CI [{ci_low:.4f}, {ci_high:.4f}]. "
        f"Exponentiated coefficient (IRR) = {irr:.3f}, 95% CI [{irr_ci_low:.3f}, {irr_ci_high:.3f}]. "
        f"A positive coefficient indicates a {direction}. The effect is {signif_text} at alpha={alpha}. "
        f"Note: standard errors were clustered by referee in the model fit (as specified when fitting)."
    )

    return {"object": out_object, "description": description}