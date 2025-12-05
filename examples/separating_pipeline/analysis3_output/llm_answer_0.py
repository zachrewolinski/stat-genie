def extract_final_answer(model_output):
    """
    Extract statistics for the 'femininity_z' coefficient from a fitted statsmodels GLMResults-like object.
    Returns a dictionary with keys:
      - "object": dict of numeric results (coefficient, p-value, 95% CI, IRR and IRR CI, decision)
      - "description": textual explanation of the returned values and their interpretation
    """
    import numpy as np
    from math import erf, sqrt

    # Helper to compute two-sided p-value from z using normal approximation (fallback)
    def _p_from_z(z):
        # p = 2 * (1 - Phi(|z|)), where Phi is standard normal CDF
        # Use erf: Phi(x) = 0.5*(1 + erf(x/sqrt(2)))
        return 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))

    # Ensure the model output provides parameters
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError(f"Provided model_output does not expose .params: {e}")

    var = 'femininity_z'
    if var not in params.index:
        raise KeyError(f"The fitted model does not contain a parameter named '{var}'")

    # Extract coefficient
    coef = float(params[var])

    # Try to get robust p-value; if not available, compute from bse using normal approx
    pval = None
    try:
        pval = float(model_output.pvalues[var])
    except Exception:
        try:
            bse = float(model_output.bse[var])
            z = coef / bse
            try:
                # Prefer scipy if available for accuracy
                import scipy.stats as _st
                pval = float(2.0 * (1.0 - _st.norm.cdf(abs(z))))
            except Exception:
                pval = float(_p_from_z(z))
        except Exception:
            pval = None

    # Try to get 95% CI; fallback to coef +/- 1.96*bse if conf_int unavailable
    try:
        ci_series = model_output.conf_int().loc[var]
        ci = [float(ci_series[0]), float(ci_series[1])]
    except Exception:
        try:
            bse = float(model_output.bse[var])
            ci = [coef - 1.96 * bse, coef + 1.96 * bse]
        except Exception:
            ci = [None, None]

    # Compute incidence rate ratio (IRR) and its CI on original count scale: exp(coef)
    try:
        irr = float(np.exp(coef))
        irr_ci = [float(np.exp(ci[0])) if ci[0] is not None else None,
                  float(np.exp(ci[1])) if ci[1] is not None else None]
    except Exception:
        irr = None
        irr_ci = [None, None]

    # Decision on hypothesis (alpha = 0.05)
    alpha = 0.05
    if pval is None:
        decision = "Could not compute p-value; unable to draw a statistical inference about the effect."
    else:
        if pval < alpha:
            if coef < 0:
                decision = ("Yes — statistically significant evidence (p < 0.05) that higher name femininity "
                            "is associated with fewer fatalities (negative coefficient; IRR < 1).")
            else:
                decision = ("No (but significant) — statistically significant evidence (p < 0.05) that higher name "
                            "femininity is associated with more fatalities (positive coefficient; IRR > 1).")
        else:
            decision = ("No — there is no statistically significant evidence (p >= 0.05) that name femininity "
                        "affects fatalities in this model.")

    result_object = {
        'coef_femininity_z': coef,
        'pvalue_femininity_z': pval,
        'ci_95_femininity_z': ci,
        'irr': irr,
        'irr_ci_95': irr_ci,
        'decision': decision
    }

    description = (
        "Extracted statistics for the predictor 'femininity_z' from the fitted Negative Binomial GLM.\n"
        "- coef_femininity_z: coefficient on the log-count scale (expected change in log fatalities per 1 SD increase in femininity).\n"
        "- pvalue_femininity_z: two-sided p-value testing H0: coef = 0 (uses model-provided robust p-values when available, "
        "otherwise normal approximation from bse).\n"
        "- ci_95_femininity_z: 95% confidence interval for the coefficient (log-count scale).\n"
        "- irr: incidence rate ratio (IRR = exp(coef)); IRR < 1 implies fewer fatalities for more feminine names.\n"
        "- irr_ci_95: 95% CI for the IRR (exp of coefficient CI).\n"
        "- decision: plain-language inference about whether more feminine names lead to fewer fatalities at alpha = 0.05."
    )

    return {"object": result_object, "description": description}