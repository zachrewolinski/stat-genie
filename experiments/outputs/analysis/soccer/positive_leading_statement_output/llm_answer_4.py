def extract_final_answer(model_output):
    """
    Extracts the coefficient, clustered SE, z, p-value, 95% CI, and incidence-rate-ratio (IRR)
    for the 'SkinDark' predictor from the model_output object produced by the modeling function.
    
    Returns a dictionary:
      - "object": dict with numeric results:
            { 'coef', 'se', 'z', 'p', 'ci_lower', 'ci_upper', 'irr', 'irr_ci_lower', 'irr_ci_upper',
              'significant' (bool; at alpha=0.05), 'direction' ('higher'/'lower'/'none') }
      - "description": short interpretation answering whether dark-skinned players are more likely
                       to receive red cards (yes/no/uncertain) with the key numbers.
    """
    import numpy as np
    from math import exp
    try:
        from scipy.stats import norm
    except Exception:
        # fallback: approximate using large-sample normal if scipy not available
        norm = None

    # Helper to safely get pandas-like value
    def safe_get(series_like, key):
        try:
            return series_like.get(key)
        except Exception:
            try:
                # maybe it's a numpy array or dict-like
                return series_like[key]
            except Exception:
                return None

    # 1) obtain params Series
    params = None
    if hasattr(model_output, 'params'):
        params = model_output.params
    elif hasattr(model_output, '_base_res') and hasattr(model_output._base_res, 'params'):
        params = model_output._base_res.params
    else:
        raise ValueError("Couldn't find parameter estimates on the supplied model_output object.")

    coef = safe_get(params, 'SkinDark')
    # 2) obtain standard error for SkinDark (prefer clustered bse if present)
    se = None
    if hasattr(model_output, 'bse'):
        se = safe_get(model_output.bse, 'SkinDark')
    # attempt to compute from cov matrix if bse missing or NaN
    if (se is None) or (isinstance(se, float) and np.isnan(se)):
        cov = None
        if hasattr(model_output, 'cov'):
            cov = model_output.cov
        elif hasattr(model_output, '_base_res') and hasattr(model_output._base_res, 'cov_params'):
            try:
                cov = model_output._base_res.cov_params()
            except Exception:
                cov = None
        # If cov is a DataFrame, convert to numpy and find index
        try:
            if cov is not None:
                if hasattr(cov, 'values'):
                    cov_arr = np.asarray(cov.values)
                    cov_index = list(cov.index) if hasattr(cov, 'index') else None
                else:
                    cov_arr = np.asarray(cov)
                    cov_index = list(params.index) if params is not None else None
                if cov_arr.ndim == 2 and cov_index is not None and 'SkinDark' in cov_index:
                    idx = cov_index.index('SkinDark')
                    se = float(np.sqrt(cov_arr[idx, idx]))
        except Exception:
            se = se  # leave as-is

    # final fallback: try base result bse
    if (se is None) or (isinstance(se, float) and np.isnan(se)):
        if hasattr(model_output, '_base_res') and hasattr(model_output._base_res, 'bse'):
            se = safe_get(model_output._base_res.bse, 'SkinDark')

    # Prepare output structure
    result = {
        'coef': None, 'se': None, 'z': None, 'p': None,
        'ci_lower': None, 'ci_upper': None,
        'irr': None, 'irr_ci_lower': None, 'irr_ci_upper': None,
        'significant': None, 'direction': None
    }

    # If coef is missing, return with explanation
    if coef is None or (isinstance(coef, float) and np.isnan(coef)):
        description = "The model output does not contain an estimate for 'SkinDark'. Cannot answer the question."
        return {'object': None, 'description': description}

    # Fill numeric values
    coef = float(coef)
    result['coef'] = coef

    if se is not None:
        se = float(se)
        result['se'] = se
        if se != 0 and not np.isnan(se):
            z = coef / se
            # p-value using normal approximation
            if norm is not None:
                p = 2 * (1 - norm.cdf(abs(z)))
            else:
                # approximate using error function if scipy missing
                from math import erf, sqrt
                p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se

            result.update({'z': float(z), 'p': float(p), 'ci_lower': float(ci_lower), 'ci_upper': float(ci_upper)})

            # IRR (incidence rate ratio) and its CI
            irr = exp(coef)
            irr_ci_lower = exp(ci_lower)
            irr_ci_upper = exp(ci_upper)
            result.update({'irr': float(irr), 'irr_ci_lower': float(irr_ci_lower), 'irr_ci_upper': float(irr_ci_upper)})

            # significance and direction
            alpha = 0.05
            significant = (p < alpha)
            result['significant'] = bool(significant)
            if coef > 0:
                direction = 'higher'  # SkinDark associated with higher red-card rate
            elif coef < 0:
                direction = 'lower'
            else:
                direction = 'none'
            result['direction'] = direction
        else:
            # se is zero or invalid; cannot compute inferential stats
            result['z'] = None
            result['p'] = None
            result['ci_lower'] = None
            result['ci_upper'] = None
            result['irr'] = exp(coef)
            result['irr_ci_lower'] = None
            result['irr_ci_upper'] = None
            result['significant'] = None
            result['direction'] = 'higher' if coef > 0 else ('lower' if coef < 0 else 'none')
    else:
        # se missing -> still report coef and note missing SE
        result['se'] = None
        result['direction'] = 'higher' if coef > 0 else ('lower' if coef < 0 else 'none')

    # Build human-readable description answering the yes/no question
    if result['p'] is None:
        description = (
            f"The model estimate for SkinDark (log rate) = {result['coef']:.4f}. Standard error or inferential "
            "statistics for this estimate are not available, so we cannot determine statistical significance. "
            "Direction: the point estimate indicates that dark-skinned players receive "
            f"{'higher' if result['coef'] > 0 else 'lower' if result['coef'] < 0 else 'the same'} red-card rates."
        )
    else:
        # Interpret based on significance and sign
        if result['significant']:
            if result['coef'] > 0:
                desc_conclusion = "Yes — dark-skinned players are estimated to receive red cards at a higher rate."
            else:
                desc_conclusion = "No — dark-skinned players are estimated to receive red cards at a lower rate."
        else:
            desc_conclusion = "No strong evidence of a difference in red-card rates by skin tone (not statistically significant)."

        description = (
            f"{desc_conclusion} Estimate (log rate) = {result['coef']:.4f}, SE = {result['se']:.4f}, "
            f"z = {result['z']:.3f}, p = {result['p']:.3g}. "
            f"IRR = {result['irr']:.3f} (95% CI: {result['irr_ci_lower']:.3f} – {result['irr_ci_upper']:.3f})."
        )

    return {'object': result, 'description': description}