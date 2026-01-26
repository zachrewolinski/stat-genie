def extract_final_answer(model_output):
    """
    Extracts the coefficient, uncertainty, and an interpretable effect-size for the
    'DarkSkin' variable from a fitted statsmodels results object (e.g., GLMResultsWrapper).
    
    Returns a dict with:
      - "object": dict of numeric results (coef, se, z, p, 95% CI, IRR, IRR 95% CI, significant, decision)
      - "description": short plain-language interpretation of the result in context.
    """
    import numpy as np

    res = model_output

    # Ensure the object has the usual statsmodels attributes
    if not hasattr(res, "params"):
        raise AttributeError("model_output has no attribute 'params' (not a statsmodels results object)")

    params = res.params
    if "DarkSkin" not in params.index:
        raise KeyError("The fitted model does not contain a parameter named 'DarkSkin'")

    coef = float(params["DarkSkin"])

    # Standard error
    if hasattr(res, "bse"):
        se = float(res.bse["DarkSkin"])
    else:
        # fallback: try to extract from covariance matrix diagonal
        if hasattr(res, "cov_params"):
            cov = res.cov_params()
            se = float(np.sqrt(np.asarray(cov.loc["DarkSkin", "DarkSkin"])))
        else:
            raise AttributeError("Cannot find standard errors for the model results")

    # z / t value: compute from coef/se if not provided
    if hasattr(res, "tvalues") and "DarkSkin" in getattr(res, "tvalues").index:
        zval = float(res.tvalues["DarkSkin"])
    else:
        zval = coef / se if se != 0 else float("nan")

    # p-value
    if hasattr(res, "pvalues") and "DarkSkin" in getattr(res, "pvalues").index:
        pval = float(res.pvalues["DarkSkin"])
    else:
        # two-sided p-value from normal approximation
        from math import erf, sqrt
        import scipy.stats as _ss  # scipy commonly available; if not, use normal cdf
        pval = float(2 * (1 - _ss.norm.cdf(abs(zval))))

    # 95% CI on coefficient scale
    try:
        ci = res.conf_int(alpha=0.05)
        ci_lower = float(ci.loc["DarkSkin", 0])
        ci_upper = float(ci.loc["DarkSkin", 1])
    except Exception:
        # Approximate using normal quantile
        z_crit = 1.96
        ci_lower = coef - z_crit * se
        ci_upper = coef + z_crit * se

    # Incidence Rate Ratio (IRR) and its CI (exp of coef and CI)
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower))
    irr_ci_upper = float(np.exp(ci_upper))

    # Significance at conventional alpha=0.05
    significant = (pval < 0.05)

    # Simple decision in context of the research question
    if significant:
        if irr > 1:
            decision = ("Yes — statistically significant evidence that players categorized as having a dark "
                        "skin tone receive red cards at a higher rate than light-skinned players "
                        f"(IRR = {irr:.3f}, p = {pval:.3g}).")
        else:
            decision = ("No — statistically significant evidence that dark-skinned players receive red cards "
                        "at a lower rate than light-skinned players "
                        f"(IRR = {irr:.3f}, p = {pval:.3g}).")
    else:
        decision = ("Inconclusive / no statistically significant difference detected in red-card rates between "
                    f"dark- and light-skinned players (IRR = {irr:.3f}, p = {pval:.3g}).")

    out_obj = {
        "coef": coef,
        "std_error": se,
        "z_value": zval,
        "p_value": pval,
        "ci_95_coef": [ci_lower, ci_upper],
        "irr": irr,
        "ci_95_irr": [irr_ci_lower, irr_ci_upper],
        "significant": bool(significant),
        "decision_text": decision
    }

    description = (
        "Extracted the DarkSkin coefficient from a negative-binomial GLM (offset for exposure). "
        "Coefficient is on the log-rate scale; exp(coef) = incidence rate ratio (IRR). "
        "If IRR>1 and p<0.05 we conclude dark-skinned players are more likely to receive red cards; "
        "if IRR<1 and p<0.05 they are less likely; otherwise there is no statistically significant difference."
    )

    return {"object": out_obj, "description": description}