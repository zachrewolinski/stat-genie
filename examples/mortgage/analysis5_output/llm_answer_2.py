def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of applicant gender on mortgage acceptance
    from a fitted statsmodels binary model (Logit or GLM).

    Returns a dictionary with:
      - "object": a dict containing coefficient (log-odds), standard error, p-value,
                  95% CI, and odds ratio (with 95% CI) for:
            * female effect when applicant is non-Black (black=0)
            * female effect when applicant is Black (black=1) -- i.e., female + female_black
      - "description": short plain-language interpretation of these statistics.

    The function is defensive about the exact types returned by statsmodels objects
    (Series / DataFrame / numpy arrays) but assumes the parameter names 'female'
    and 'female_black' exist in the fitted model.
    """
    import numpy as np
    from math import sqrt
    try:
        from scipy import stats
        _norm_cdf = stats.norm.cdf
    except Exception:
        # Fallback: use statsmodels' distribution if scipy not available
        import statsmodels.api as sm
        _norm_cdf = sm.distributions.norm.cdf

    res = model_output

    # Helper to extract param, se, pvalue, ci for a single parameter name
    params = getattr(res, "params")
    pvalues = getattr(res, "pvalues", None)
    bse = getattr(res, "bse", None)
    cov = getattr(res, "cov_params")()
    try:
        ci_raw = res.conf_int()
    except Exception:
        ci_raw = None

    def _get_value_from_indexable(obj, name):
        # Works for Series/DataFrame/ndarray. Raises KeyError if not found.
        if obj is None:
            return None
        try:
            return obj[name]
        except Exception:
            # try positional lookup
            try:
                idx = list(params.index).index(name)
                return obj[idx]
            except Exception:
                raise KeyError(f"Parameter '{name}' not found in model output.")

    def _get_ci(name):
        if ci_raw is None:
            return (None, None)
        try:
            # If ci_raw is a DataFrame-like with index matching params
            low_high = ci_raw.loc[name]
            return (float(low_high[0]), float(low_high[1]))
        except Exception:
            try:
                idx = list(params.index).index(name)
                low = float(ci_raw[idx, 0])
                high = float(ci_raw[idx, 1])
                return (low, high)
            except Exception:
                return (None, None)

    def _get_cov(a, b):
        # cov may be DataFrame or ndarray
        try:
            return float(cov.loc[a, b])
        except Exception:
            try:
                ia = list(params.index).index(a)
                ib = list(params.index).index(b)
                return float(cov[ia, ib])
            except Exception:
                raise KeyError(f"Covariance entry for {a},{b} not found.")

    # Ensure required parameter names exist
    param_names = list(params.index)
    if 'female' not in param_names:
        raise KeyError("Model output does not contain a parameter named 'female'.")

    # Extract female (effect when black=0)
    coef_f = float(_get_value_from_indexable(params, 'female'))
    se_f = float(_get_value_from_indexable(bse, 'female')) if bse is not None else None
    p_f = float(_get_value_from_indexable(pvalues, 'female')) if pvalues is not None else None
    ci_low_f, ci_high_f = _get_ci('female')
    or_f = float(np.exp(coef_f))
    or_ci_low_f = float(np.exp(ci_low_f)) if (ci_low_f is not None) else None
    or_ci_high_f = float(np.exp(ci_high_f)) if (ci_high_f is not None) else None

    female_non_black = {
        'coef_log_odds': coef_f,
        'std_err': se_f,
        'p_value': p_f,
        '95%_ci_log_odds': [ci_low_f, ci_high_f],
        'odds_ratio': or_f,
        '95%_ci_odds_ratio': [or_ci_low_f, or_ci_high_f]
    }

    # Now compute effect of female when black=1 (female + female_black)
    if 'female_black' in param_names:
        coef_fb = float(_get_value_from_indexable(params, 'female') + _get_value_from_indexable(params, 'female_black'))
        # variance of sum = Var(female) + Var(female_black) + 2*Cov(female, female_black)
        var_sum = _get_cov('female', 'female') + _get_cov('female_black', 'female_black') + 2 * _get_cov('female', 'female_black')
        se_sum = float(sqrt(var_sum)) if var_sum >= 0 else None

        # Compute two-sided p-value from z-score if se available, else None
        if se_sum is not None and se_sum > 0:
            z = coef_fb / se_sum
            p_fb = float(2 * (1 - _norm_cdf(abs(z))))
        else:
            p_fb = None

        # Confidence interval for the sum (Wald)
        if se_sum is not None:
            ci_low_fb = float(coef_fb - 1.96 * se_sum)
            ci_high_fb = float(coef_fb + 1.96 * se_sum)
        else:
            ci_low_fb, ci_high_fb = (None, None)

        or_fb = float(np.exp(coef_fb))
        or_ci_low_fb = float(np.exp(ci_low_fb)) if (ci_low_fb is not None) else None
        or_ci_high_fb = float(np.exp(ci_high_fb)) if (ci_high_fb is not None) else None

        female_black = {
            'coef_log_odds': coef_fb,
            'std_err': se_sum,
            'p_value': p_fb,
            '95%_ci_log_odds': [ci_low_fb, ci_high_fb],
            'odds_ratio': or_fb,
            '95%_ci_odds_ratio': [or_ci_low_fb, or_ci_high_fb]
        }
    else:
        # No interaction term: effect for Black is same as for non-Black (female coefficient)
        female_black = female_non_black.copy()
        female_black['note'] = "No 'female_black' interaction term in model; effect for Black equals effect for non-Black (female coefficient)."

    # Short interpretation
    def _sig_label(p):
        if p is None:
            return "p-value unavailable"
        if p < 0.01:
            return "statistically significant (p < 0.01)"
        if p < 0.05:
            return "statistically significant (p < 0.05)"
        if p < 0.10:
            return "marginally significant (p < 0.10)"
        return "not statistically significant (p >= 0.10)"

    desc_lines = []
    desc_lines.append("Reported values are logistic regression coefficients (log-odds).")
    desc_lines.append("Odds ratios = exp(coefficient); values >1 indicate higher odds of acceptance for females.")
    desc_lines.append("Effect of female when applicant is non-Black:")
    sig_text_f = _sig_label(female_non_black['p_value'])
    desc_lines.append(f" - coef={female_non_black['coef_log_odds']:.4f}, OR={female_non_black['odds_ratio']:.4f}, {sig_text_f}.")
    if 'note' in female_black:
        desc_lines.append("Model does not include female*black interaction; same effect applies for Black applicants.")
    else:
        sig_text_fb = _sig_label(female_black['p_value'])
        desc_lines.append("Effect of female when applicant is Black (female + female_black):")
        desc_lines.append(f" - coef={female_black['coef_log_odds']:.4f}, OR={female_black['odds_ratio']:.4f}, {sig_text_fb}.")

    description = " ".join(desc_lines)

    # Return numbers as basic python types
    def _to_basic(obj):
        if isinstance(obj, dict):
            return {k: _to_basic(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (list, tuple)):
            return [_to_basic(x) for x in obj]
        return obj

    return {
        "object": _to_basic({
            "female_non_black": female_non_black,
            "female_black": female_black
        }),
        "description": description
    }