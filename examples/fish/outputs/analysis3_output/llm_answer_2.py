def extract_final_answer(model_output):
    """
    Extract and interpret the effect of the primary predictor 'livebait' on catch rate
    from the provided model_output, which must contain:
      - 'glm_results': a statsmodels GLMResultsWrapper (Gamma family with log link, offset=log_hours)
      - 'ols_results': a statsmodels RegressionResultsWrapper (OLS on fish_per_hour with robust SEs)

    Returns a dictionary:
      {
        "object": { "glm": {...}, "ols": {...} },
        "description": "Plain-language interpretation of the statistics"
      }
    """
    import numpy as np

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict with keys 'glm_results' and 'ols_results'.")

    if 'glm_results' not in model_output or 'ols_results' not in model_output:
        raise ValueError("model_output must contain keys 'glm_results' and 'ols_results'.")

    glm_res = model_output['glm_results']
    ols_res = model_output['ols_results']

    param_name = 'livebait'

    # Helper to safely extract param, pvalue, bse, confint
    def _get_param_info(res, name):
        # params, pvalues, bse are typically pandas Series with index of variable names
        try:
            coef = float(res.params[name])
            pval = float(res.pvalues[name])
            se = float(res.bse[name]) if hasattr(res, 'bse') and name in res.bse.index else None
        except Exception as e:
            raise KeyError(f"Could not extract parameter '{name}' from results: {e}")

        # conf_int may be DataFrame-like or ndarray
        try:
            ci = res.conf_int()
            # try label-based access first
            try:
                lower, upper = ci.loc[name]
            except Exception:
                # fallback: find index position of the parameter in params
                idx = list(res.params.index).index(name)
                lower, upper = ci[idx]
        except Exception:
            lower, upper = (None, None)

        return {
            'coef': coef,
            'pvalue': pval,
            'se': se,
            'ci_lower': float(lower) if lower is not None else None,
            'ci_upper': float(upper) if upper is not None else None
        }

    # Extract info from GLM
    glm_info = _get_param_info(glm_res, param_name)
    # On the GLM (log link) scale, coef is log multiplicative effect on rate; exponentiate to get rate ratio
    glm_rate_ratio = float(np.exp(glm_info['coef']))
    glm_ci_lower = float(np.exp(glm_info['ci_lower'])) if glm_info['ci_lower'] is not None else None
    glm_ci_upper = float(np.exp(glm_info['ci_upper'])) if glm_info['ci_upper'] is not None else None
    glm_pct_change = (glm_rate_ratio - 1.0) * 100.0
    glm_pct_ci_lower = (glm_ci_lower - 1.0) * 100.0 if glm_ci_lower is not None else None
    glm_pct_ci_upper = (glm_ci_upper - 1.0) * 100.0 if glm_ci_upper is not None else None

    glm_result_obj = {
        'log_coef': glm_info['coef'],
        'log_coef_se': glm_info['se'],
        'pvalue': glm_info['pvalue'],
        'rate_ratio': glm_rate_ratio,
        'rate_ratio_95CI': (glm_ci_lower, glm_ci_upper),
        'percent_change': glm_pct_change,
        'percent_change_95CI': (glm_pct_ci_lower, glm_pct_ci_upper),
        'interpretation': (
            "In the GLM (Gamma, log link) with log(hours) as offset, the exponentiated coefficient "
            "is the multiplicative effect on the catch rate (fish per hour)."
        )
    }

    # Extract info from OLS
    ols_info = _get_param_info(ols_res, param_name)
    # In OLS on fish_per_hour, coef is the absolute change in fish/hour associated with livebait (binary).
    ols_ci = (ols_info['ci_lower'], ols_info['ci_upper'])
    ols_result_obj = {
        'coef_fish_per_hour': ols_info['coef'],
        'coef_se': ols_info['se'],
        'pvalue': ols_info['pvalue'],
        'coef_95CI': ols_ci,
        'interpretation': (
            "In the OLS model on fish_per_hour, the coefficient is the estimated absolute change "
            "in fish caught per hour associated with using live bait (compared to not using live bait)."
        )
    }

    # Compose plain-language description
    desc_lines = []
    desc_lines.append(
        "GLM result (primary): Using live bait is associated with a multiplicative change in catch rate per hour. "
        f"Rate ratio = {glm_rate_ratio:.3f} (95% CI: {glm_ci_lower:.3f} to {glm_ci_upper:.3f}) "
        f"→ {glm_pct_change:.1f}% change (95% CI: {glm_pct_ci_lower:.1f}% to {glm_pct_ci_upper:.1f}%). "
        f"p = {glm_info['pvalue']:.3g}."
    )
    desc_lines.append(
        "OLS result (secondary): Using live bait is associated with an absolute change of "
        f"{ols_info['coef']:.3f} fish/hour (95% CI: {ols_info['ci_lower']:.3f} to {ols_info['ci_upper']:.3f}). "
        f"p = {ols_info['pvalue']:.3g}."
    )
    desc_lines.append(
        "Interpretation: If the GLM rate ratio is >1 and statistically significant, live bait is associated with a higher catch rate per hour; "
        "the OLS coefficient gives the approximate increase in fish/hour. Use the GLM as the preferred multiplicative rate model."
    )
    description = " ".join(desc_lines)

    result_object = {
        'glm': glm_result_obj,
        'ols': ols_result_obj
    }

    return {
        'object': result_object,
        'description': description
    }