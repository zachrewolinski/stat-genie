def extract_final_answer(model_output):
    """
    Extracts statistics to answer whether 'Reader View' improves reading speed for individuals with dyslexia.

    Returns a dict with:
      - "object": dict of numeric results (coefficients, SE, t, p, 95% CI) for:
          * ReaderView main effect (effect for non-dyslexic when Dyslexia=0)
          * ReaderView:Dyslexia interaction
          * Combined effect of ReaderView for Dyslexia=1 (ReaderView + ReaderView:Dyslexia)
          * nobs and df_resid for reference
      - "description": brief interpretation of the combined effect for people with dyslexia
    """
    import numpy as np

    res = model_output  # statsmodels RegressionResultsWrapper

    # Parameter names in the fitted model
    params = res.params
    pvals = res.pvalues

    # Prepare output container
    out = {}

    # Helper to safely get a parameter value or None
    def get_param(name):
        return params[name] if name in params.index else None

    def get_pval(name):
        return pvals[name] if name in pvals.index else None

    # Extract main terms if present
    coef_reader = get_param('ReaderView')
    p_reader = get_pval('ReaderView')

    coef_inter = get_param('ReaderView:Dyslexia')
    p_inter = get_pval('ReaderView:Dyslexia')

    out['coef_readerview_non_dyslexic'] = coef_reader
    out['p_readerview_non_dyslexic'] = p_reader
    out['coef_interaction_readerview_by_dyslexia'] = coef_inter
    out['p_interaction'] = p_inter

    # Compute combined effect for dyslexic individuals: ReaderView + ReaderView:Dyslexia
    # Use statsmodels t_test for correct SE, t, p and CI for linear combination
    combined_result = None
    try:
        # This creates a contrast "ReaderView + ReaderView:Dyslexia"
        combined_result = res.t_test('ReaderView + ReaderView:Dyslexia')
    except Exception:
        # Fallback: if names differ or t_test fails, try alternative name with ':' replaced by '::' etc.
        # But primarily report None if t_test is not possible.
        combined_result = None

    if combined_result is not None:
        # combined_result.effect is a 1-element array
        combined_effect = float(np.atleast_1d(combined_result.effect)[0])
        combined_se = float(np.atleast_1d(combined_result.sd)[0])
        combined_t = float(np.atleast_1d(combined_result.tvalue)[0])
        combined_p = float(np.atleast_1d(combined_result.pvalue)[0])
        # conf_int returns array [[lower, upper]]
        ci = combined_result.conf_int(alpha=0.05)
        ci_lower, ci_upper = float(ci[0, 0]), float(ci[0, 1])

        out['coef_readerview_for_dyslexic'] = combined_effect
        out['se_readerview_for_dyslexic'] = combined_se
        out['t_readerview_for_dyslexic'] = combined_t
        out['p_readerview_for_dyslexic'] = combined_p
        out['ci95_readerview_for_dyslexic'] = (ci_lower, ci_upper)
    else:
        # If t_test unavailable, compute using covariance matrix manually if both terms present
        cov = res.cov_params()
        if ('ReaderView' in params.index) and ('ReaderView:Dyslexia' in params.index):
            b1 = params['ReaderView']
            b2 = params['ReaderView:Dyslexia']
            combined_effect = b1 + b2
            var_b1 = cov.loc['ReaderView', 'ReaderView']
            var_b2 = cov.loc['ReaderView:Dyslexia', 'ReaderView:Dyslexia']
            cov_b1b2 = cov.loc['ReaderView', 'ReaderView:Dyslexia']
            combined_var = var_b1 + var_b2 + 2 * cov_b1b2
            combined_se = float(np.sqrt(combined_var))
            combined_t = combined_effect / combined_se if combined_se > 0 else None
            # Use large-sample normal approx for p-value if df-based not available
            from scipy import stats
            combined_p = 2 * (1 - stats.norm.cdf(abs(combined_t))) if combined_t is not None else None
            # Approximate 95% CI using normal quantile
            z = 1.96
            ci_lower = combined_effect - z * combined_se
            ci_upper = combined_effect + z * combined_se

            out['coef_readerview_for_dyslexic'] = float(combined_effect)
            out['se_readerview_for_dyslexic'] = float(combined_se)
            out['t_readerview_for_dyslexic'] = float(combined_t) if combined_t is not None else None
            out['p_readerview_for_dyslexic'] = float(combined_p) if combined_p is not None else None
            out['ci95_readerview_for_dyslexic'] = (float(ci_lower), float(ci_upper))
        else:
            out['coef_readerview_for_dyslexic'] = None
            out['se_readerview_for_dyslexic'] = None
            out['t_readerview_for_dyslexic'] = None
            out['p_readerview_for_dyslexic'] = None
            out['ci95_readerview_for_dyslexic'] = (None, None)

    # Add sample size info
    try:
        out['nobs'] = int(res.nobs)
        out['df_resid'] = float(res.df_resid)
    except Exception:
        out['nobs'] = None
        out['df_resid'] = None

    # Construct description interpreting the combined effect
    if out.get('coef_readerview_for_dyslexic') is None:
        description = (
            "Could not compute the simple slope (effect of ReaderView for dyslexic participants) "
            "from the provided model output. Ensure the model includes terms 'ReaderView' and "
            "'ReaderView:Dyslexia' and that the object is a statsmodels results wrapper."
        )
    else:
        coef = out['coef_readerview_for_dyslexic']
        pval = out['p_readerview_for_dyslexic']
        ci_low, ci_high = out['ci95_readerview_for_dyslexic']
        # Interpret direction
        direction = "increase" if coef > 0 else ("decrease" if coef < 0 else "no change")
        significance = "statistically significant" if (pval is not None and pval < 0.05) else "not statistically significant"
        description = (
            f"The estimated effect of enabling Reader View for participants with dyslexia is {coef:.3f} wpm "
            f"(95% CI [{ci_low:.3f}, {ci_high:.3f}]). This represents an expected {direction} in reading speed "
            f"when Reader View is enabled for dyslexic individuals, controlling for covariates. The effect is "
            f"{significance} (two-sided p = {pval:.4f})."
        )

    return {"object": out, "description": description}