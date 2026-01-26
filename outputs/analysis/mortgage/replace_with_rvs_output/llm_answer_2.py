def extract_final_answer(model_output):
    """
    Extract statistics about the effect of gender ('female') on mortgage acceptance
    from the provided model_output and produce a concise interpretation.

    Returns:
      {
        "object": {  # numeric results
            "coef": float,         # logistic coefficient for 'female'
            "odds_ratio": float,   # exp(coef)
            "ci_lower": float,     # lower bound of 95% CI for OR
            "ci_upper": float,     # upper bound of 95% CI for OR
            "pvalue": float,       # p-value for 'female'
            "nobs": int,           # sample size used in the model
            "significant": bool    # True if pvalue < 0.05
        },
        "description": str  # short interpretation in context
      }
    """
    # Normalize input: expect either (result_obj, summary_dict) or summary_dict or result_obj
    summary = None
    # Case: tuple with summary dict as second element
    if isinstance(model_output, tuple) and len(model_output) >= 2:
        candidate = model_output[1]
        if isinstance(candidate, dict):
            summary = candidate
        else:
            # maybe the first element is the dict
            if isinstance(model_output[0], dict):
                summary = model_output[0]
    # Case: provided directly a dict
    if summary is None and isinstance(model_output, dict):
        summary = model_output

    # If still None, try to compute from a statsmodels result object
    if summary is None:
        result_obj = None
        if isinstance(model_output, tuple) and len(model_output) >= 1:
            result_obj = model_output[0]
        else:
            result_obj = model_output
        try:
            params = result_obj.params
            pvalues = result_obj.pvalues
            conf = result_obj.conf_int()
            import numpy as _np
            or_vals = _np.exp(params)
            or_lower = _np.exp(conf[0])
            or_upper = _np.exp(conf[1])
            summary = {
                'params': params,
                'odds_ratios': or_vals,
                'or_ci_lower': or_lower,
                'or_ci_upper': or_upper,
                'pvalues': pvalues,
                'nobs': int(result_obj.nobs)
            }
        except Exception as e:
            raise ValueError("Unsupported model_output format; unable to extract summary.") from e

    # Helper to safely get values from pandas Series or dict
    def _get(series_like, key):
        try:
            return series_like.get(key)
        except Exception:
            try:
                return series_like[key]
            except Exception:
                return None

    coef = _get(summary.get('params', {}), 'female')
    odds_ratio = _get(summary.get('odds_ratios', {}), 'female')
    ci_lower = _get(summary.get('or_ci_lower', {}), 'female')
    ci_upper = _get(summary.get('or_ci_upper', {}), 'female')
    pvalue = _get(summary.get('pvalues', {}), 'female')
    nobs = summary.get('nobs', None)

    # Cast to native Python types where possible
    try:
        coef = float(coef)
    except Exception:
        coef = None
    try:
        odds_ratio = float(odds_ratio)
    except Exception:
        odds_ratio = None
    try:
        ci_lower = float(ci_lower)
    except Exception:
        ci_lower = None
    try:
        ci_upper = float(ci_upper)
    except Exception:
        ci_upper = None
    try:
        pvalue = float(pvalue)
    except Exception:
        pvalue = None
    try:
        nobs = int(nobs) if nobs is not None else None
    except Exception:
        nobs = None

    significant = (pvalue is not None) and (pvalue < 0.05)

    # Build concise description
    if odds_ratio is not None and ci_lower is not None and ci_upper is not None and pvalue is not None:
        sign_text = "statistically significant (p = {:.3g})".format(pvalue) if significant else "not statistically significant (p = {:.3g})".format(pvalue)
        description = (
            "Controlling for the listed covariates, the estimated effect of being female is "
            "coef = {:.4f} (odds ratio = {:.3f}, 95% CI [{:.3f}, {:.3f}]). This effect is {}, "
            "based on n = {} observations. In context: female applicants have about a {:.1f}% "
            "higher odds of mortgage approval than male applicants (association, not proof of causation)."
            .format(coef, odds_ratio, ci_lower, ci_upper, sign_text, nobs if nobs is not None else "unknown",
                    (odds_ratio - 1) * 100 if odds_ratio is not None else float('nan'))
        )
    else:
        description = "Could not extract complete statistics for 'female' from model_output."

    result_object = {
        "coef": coef,
        "odds_ratio": odds_ratio,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pvalue": pvalue,
        "nobs": nobs,
        "significant": significant
    }

    return {"object": result_object, "description": description}