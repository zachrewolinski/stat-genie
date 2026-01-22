def extract_final_answer(model_output):
    """
    Extracts the DarkSkin effect from the model_output and returns a concise
    numeric summary plus a plain-language interpretation.

    Returns a dict with:
      - "object": dict with numeric results (coef, se, pvalue, 95% CI, IRR, IRR CI)
      - "description": brief interpretation in the context of whether dark-skinned
                       players are more likely to receive red cards.
    """
    import math

    # Helper to safely get nested values from the common structure in the example.
    def safe_get(d, *keys, default=None):
        try:
            for k in keys:
                d = d[k]
            return d
        except Exception:
            return default

    # If the input is the full dict used in the example, it should contain 'coef','se','pvalues','conf_int'
    coef = safe_get(model_output, 'coef', 'DarkSkin')
    se = safe_get(model_output, 'se', 'DarkSkin')
    pvalue = safe_get(model_output, 'pvalues', 'DarkSkin')
    ci_lower = safe_get(model_output, 'conf_int', '2.5%', 'DarkSkin')
    ci_upper = safe_get(model_output, 'conf_int', '97.5%', 'DarkSkin')

    # Fallback: if the top-level is a statsmodels results wrapper, try to extract params and robust cov if present.
    if coef is None:
        # Try model_output like {'original_result': res, 'cov_matrix': ...}
        res = safe_get(model_output, 'original_result')
        if res is not None:
            try:
                params = res.params
                coef = float(params.get('DarkSkin', params['DarkSkin'])) if 'DarkSkin' in params else None
            except Exception:
                coef = None
        # If still missing, leave as None.

    # If we still don't have se/pvalue/ci, attempt to compute from cov_matrix if present
    if (se is None or ci_lower is None or ci_upper is None) and isinstance(model_output, dict):
        # Try to use cov_matrix + params
        params_dict = model_output.get('coef', {})
        cov = model_output.get('cov_matrix', None)
        if cov is not None and coef is None:
            # try get param order from 'original_result'
            try:
                res = model_output.get('original_result', None)
                if res is not None:
                    params = res.params
                    coef = float(params.get('DarkSkin', params['DarkSkin']))
            except Exception:
                pass
        if cov is not None and se is None and coef is not None:
            # attempt to locate index of DarkSkin in params ordering
            try:
                res = model_output.get('original_result', None)
                if res is not None:
                    param_names = list(res.params.index)
                    if 'DarkSkin' in param_names:
                        idx = param_names.index('DarkSkin')
                        se = float((cov[idx, idx]) ** 0.5)
            except Exception:
                pass
        # compute CI and pvalue if se available
        if se is not None and (ci_lower is None or ci_upper is None):
            import scipy.stats as _stats
            crit = _stats.norm.ppf(0.975)
            if coef is not None:
                ci_lower = coef - crit * se
                ci_upper = coef + crit * se
            try:
                z = coef / se
                pvalue = 2 * (1 - _stats.norm.cdf(abs(z)))
            except Exception:
                pass

    # Final safety: if still missing values, set them to None
    for v in ('coef', 'se', 'pvalue', 'ci_lower', 'ci_upper'):
        if locals()[v] is None:
            locals_dict = locals()
            locals_dict[v] = None

    # Convert to floats when possible
    try:
        coef = float(coef) if coef is not None else None
    except Exception:
        coef = None
    try:
        se = float(se) if se is not None else None
    except Exception:
        se = None
    try:
        pvalue = float(pvalue) if pvalue is not None else None
    except Exception:
        pvalue = None
    try:
        ci_lower = float(ci_lower) if ci_lower is not None else None
    except Exception:
        ci_lower = None
    try:
        ci_upper = float(ci_upper) if ci_upper is not None else None
    except Exception:
        ci_upper = None

    # Compute incidence rate ratio (IRR) and its CI (exp of coef and CI)
    irr = math.exp(coef) if coef is not None else None
    irr_ci_lower = math.exp(ci_lower) if ci_lower is not None else None
    irr_ci_upper = math.exp(ci_upper) if ci_upper is not None else None

    # Prepare the returned numeric object
    result_object = {
        'coef_log_rate': coef,
        'se': se,
        'pvalue': pvalue,
        '95%_CI_log_rate': [ci_lower, ci_upper],
        'IRR': irr,
        '95%_CI_IRR': [irr_ci_lower, irr_ci_upper]
    }

    # Interpretation in context
    # If pvalue is available, use it; otherwise rely on CI including 1
    if pvalue is not None:
        if pvalue < 0.05:
            conclusion = ("Statistically significant difference: the DarkSkin indicator is associated with a change "
                          f"in red-card rate (log-coef={coef:.4g}, p={pvalue:.4g}). See IRR={irr:.4g} (95% CI "
                          f"{irr_ci_lower:.4g}–{irr_ci_upper:.4g}).")
        else:
            conclusion = ("No statistically significant difference: DarkSkin is not associated with a higher red-card "
                          f"rate (log-coef={coef:.4g}, p={pvalue:.4g}). Estimated IRR={irr:.4g} with 95% CI "
                          f"{irr_ci_lower:.4g}–{irr_ci_upper:.4g}, which includes 1.")
    else:
        # pvalue not available; use CI
        if irr_ci_lower is not None and irr_ci_upper is not None:
            if irr_ci_lower > 1 or irr_ci_upper < 1:
                conclusion = ("Confidence interval for the IRR does not include 1, suggesting a statistically significant "
                              "association. See IRR and CI in result_object.")
            else:
                conclusion = ("Confidence interval for the IRR includes 1, so there is no evidence of a meaningful "
                              "difference in red-card rates between dark- and light-skinned players. See IRR and CI "
                              "in result_object.")
        else:
            conclusion = ("Insufficient information to draw a statistical conclusion. Numeric outputs (coef/se/pvalue/CI) "
                          "are provided in result_object when available.")

    description = (
        "Extracted statistics for the DarkSkin indicator from the fitted negative-binomial model. "
        + conclusion
    )

    return {"object": result_object, "description": description}