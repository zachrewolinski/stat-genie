def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of having children on number of affairs
    from a fitted statsmodels GLMResultsWrapper (Negative Binomial) object.

    Returns a dictionary with:
      - "object": a dict containing coefficients, standard errors, p-values,
                  95% CIs on the log scale, and incidence rate ratios (IRRs)
                  with CIs for:
                  * females (effect of HasChildren when Male=0),
                  * males (combined effect HasChildren + HasChildren_Male),
                  * the interaction term (HasChildren_Male) itself.
      - "description": a short interpretation of those results in context.
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy.stats import norm
    except Exception:
        # fallback: basic normal cdf using erf if scipy unavailable
        from math import erf
        def _norm_cdf(x):
            return 0.5 * (1.0 + erf(x / sqrt(2.0)))
        class _Norm:
            @staticmethod
            def cdf(x):
                return _norm_cdf(x)
        norm = _Norm()

    res = model_output

    # Required parameter names
    name_has = 'HasChildren'
    name_int = 'HasChildren_Male'

    params = res.params
    cov = res.cov_params()
    bse = res.bse
    pvalues = res.pvalues
    conf = None
    try:
        conf = res.conf_int()  # default 95%
    except Exception:
        conf = None

    # Helper to safe extract numeric
    def _get(x, key):
        if hasattr(x, 'loc'):
            return float(x.loc[key])
        else:
            return float(x[key])

    # Extract coefficients and SEs for HasChildren and interaction
    if name_has not in params.index or name_int not in params.index:
        raise KeyError(f"Expected parameter names '{name_has}' and '{name_int}' missing in model_params: {list(params.index)}")

    beta_has = _get(params, name_has)
    beta_int = _get(params, name_int)

    se_has = _get(bse, name_has)
    se_int = _get(bse, name_int)

    # Female effect (Male=0): just beta_has
    coef_f = beta_has
    se_f = se_has
    z_f = coef_f / se_f if se_f != 0 else np.nan
    p_f = float(pvalues.loc[name_has]) if name_has in pvalues.index else float(2 * (1 - norm.cdf(abs(z_f))))
    if conf is not None:
        ci_f_low = float(conf.loc[name_has, 0])
        ci_f_high = float(conf.loc[name_has, 1])
    else:
        ci_f_low = coef_f - 1.96 * se_f
        ci_f_high = coef_f + 1.96 * se_f

    irr_f = float(np.exp(coef_f))
    irr_f_ci = (float(np.exp(ci_f_low)), float(np.exp(ci_f_high)))

    # Male effect (Male=1): beta_has + beta_int
    coef_m = beta_has + beta_int
    # Var(beta_has + beta_int) = var(has) + var(int) + 2 cov(has,int)
    var_has = float(cov.loc[name_has, name_has])
    var_int = float(cov.loc[name_int, name_int])
    cov_has_int = float(cov.loc[name_has, name_int])
    var_m = var_has + var_int + 2.0 * cov_has_int
    se_m = sqrt(var_m) if var_m >= 0 else float('nan')
    z_m = coef_m / se_m if se_m != 0 else np.nan
    # p-value for combined effect computed from z
    p_m = float(2 * (1 - norm.cdf(abs(z_m)))) if not np.isnan(z_m) else float('nan')
    # CI on log scale for male combined effect
    ci_m_low = coef_m - 1.96 * se_m
    ci_m_high = coef_m + 1.96 * se_m
    irr_m = float(np.exp(coef_m))
    irr_m_ci = (float(np.exp(ci_m_low)), float(np.exp(ci_m_high)))

    # Interaction term stats (how much the effect differs for males vs females)
    coef_int = beta_int
    se_int = se_int
    z_int = coef_int / se_int if se_int != 0 else np.nan
    p_int = float(pvalues.loc[name_int]) if name_int in pvalues.index else float(2 * (1 - norm.cdf(abs(z_int))))
    if conf is not None:
        ci_int_low = float(conf.loc[name_int, 0])
        ci_int_high = float(conf.loc[name_int, 1])
    else:
        ci_int_low = coef_int - 1.96 * se_int
        ci_int_high = coef_int + 1.96 * se_int
    irr_int = float(np.exp(coef_int))
    irr_int_ci = (float(np.exp(ci_int_low)), float(np.exp(ci_int_high)))

    result_object = {
        'HasChildren_female': {
            'log_coef': float(coef_f),
            'se': float(se_f),
            'z': float(z_f) if not np.isnan(z_f) else None,
            'p_value': float(p_f),
            'ci_95_log': (float(ci_f_low), float(ci_f_high)),
            'IRR': float(irr_f),
            'IRR_95': irr_f_ci
        },
        'HasChildren_male': {
            'log_coef': float(coef_m),
            'se': float(se_m),
            'z': float(z_m) if not np.isnan(z_m) else None,
            'p_value': float(p_m),
            'ci_95_log': (float(ci_m_low), float(ci_m_high)),
            'IRR': float(irr_m),
            'IRR_95': irr_m_ci
        },
        'Interaction_HasChildren_Male': {
            'log_coef': float(coef_int),
            'se': float(se_int),
            'z': float(z_int) if not np.isnan(z_int) else None,
            'p_value': float(p_int),
            'ci_95_log': (float(ci_int_low), float(ci_int_high)),
            'IRR': float(irr_int),
            'IRR_95': irr_int_ci
        },
        # also include raw params for transparency
        'raw_params': {k: float(v) for k, v in params.items()}
    }

    # Short description/interpretation
    desc_lines = []
    desc_lines.append("Interpretation (multiplicative change in expected count of affairs):")
    desc_lines.append(f"- For females (Male=0), having children corresponds to IRR = {result_object['HasChildren_female']['IRR']:.3f} "
                      f"(95% CI {result_object['HasChildren_female']['IRR_95'][0]:.3f} to {result_object['HasChildren_female']['IRR_95'][1]:.3f}); "
                      f"p = {result_object['HasChildren_female']['p_value']:.3g}.")
    desc_lines.append(f"- For males (Male=1), having children corresponds to IRR = {result_object['HasChildren_male']['IRR']:.3f} "
                      f"(95% CI {result_object['HasChildren_male']['IRR_95'][0]:.3f} to {result_object['HasChildren_male']['IRR_95'][1]:.3f}); "
                      f"p = {result_object['HasChildren_male']['p_value']:.3g}.")
    desc_lines.append(f"- The interaction term (HasChildren_Male) tests whether the child effect differs by gender: "
                      f"log-coef = {result_object['Interaction_HasChildren_Male']['log_coef']:.3f}, "
                      f"IRR = {result_object['Interaction_HasChildren_Male']['IRR']:.3f}, "
                      f"p = {result_object['Interaction_HasChildren_Male']['p_value']:.3g}.")
    desc_lines.append("A (log) coefficient < 0 (or IRR < 1) indicates having children is associated with fewer affairs; "
                      "> 0 (IRR > 1) indicates more affairs. Statistical significance indicated by p-values (commonly p < 0.05).")

    description = " ".join(desc_lines)

    return {'object': result_object, 'description': description}