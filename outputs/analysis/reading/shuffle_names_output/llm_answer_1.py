def extract_final_answer(model_output):
    """
    Extract the estimated effect of Reader View for readers with dyslexia from a fitted statsmodels OLS result.
    Returns a dict with keys:
      - "object": dict with numeric results (estimate, se, p_value, 95% CI, term_used)
      - "description": human-readable interpretation

    The function handles:
      - models with an interaction term reader_view_bin:is_dyslexic (preferred)
      - models without interaction (uses reader_view_bin main effect)
      - cases where the model did not fit or required params are missing
    """
    import math
    from math import sqrt
    try:
        from scipy.stats import norm
    except Exception:
        norm = None

    # Helper to compute p-value from z
    def two_sided_p_from_z(z):
        if norm is not None:
            return float(2 * (1 - norm.cdf(abs(z))))
        # fallback approximation using error function
        p = 2 * (0.5 - 0.5 * math.erf(abs(z) / math.sqrt(2)))
        return float(p)

    # Validate model_output looks like a fitted statsmodels result
    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided (None). Could not extract statistics."
        }

    # Some models (like DummyResults) don't have params or cov_params
    params = getattr(model_output, 'params', None)
    if params is None or len(getattr(params, 'index', params)) == 0:
        # Try stringify fallback message
        text = str(model_output)
        return {
            "object": None,
            "description": (
                "Model did not produce estimable parameters (no observations or model failed). "
                f"Model output: {text!s}. Cannot compute the effect of Reader View for dyslexic readers."
            )
        }

    # Ensure cov_params is available
    try:
        cov = model_output.cov_params()
    except Exception as e:
        return {
            "object": None,
            "description": f"Model has parameters but cov_params() failed with error: {e}. Cannot compute SE/p-value."
        }

    # Parameter name candidates
    main_term = 'reader_view_bin'
    interaction_term = 'reader_view_bin:is_dyslexic'

    # Some formula implementations might use different ordering or whitespace; try tolerant matching
    param_names = [str(n) for n in params.index]

    def find_param(name):
        if name in param_names:
            return name
        # try variants (spaces)
        alt = name.replace(':', ' : ')
        if alt in param_names:
            return alt
        # case-insensitive match
        for p in param_names:
            if p.lower().replace(' ', '') == name.lower().replace(' ', ''):
                return p
        return None

    main_p = find_param(main_term)
    inter_p = find_param(interaction_term)

    # If interaction present, compute combined effect for dyslexic = main + interaction
    if inter_p is not None and main_p is not None:
        b_main = float(params[main_p])
        b_inter = float(params[inter_p])
        est = b_main + b_inter

        # compute variance of sum: Var(b_main) + Var(b_inter) + 2*Cov(b_main, b_inter)
        try:
            var_main = float(cov.loc[main_p, main_p])
            var_inter = float(cov.loc[inter_p, inter_p])
            cov_main_inter = float(cov.loc[main_p, inter_p])
            var_sum = var_main + var_inter + 2.0 * cov_main_inter
            se = math.sqrt(var_sum) if var_sum >= 0 else float('nan')
        except Exception as e:
            return {
                "object": None,
                "description": f"Failed to compute variance/covariance for combined effect: {e}"
            }

        z = est / se if se and not math.isnan(se) else float('nan')
        pval = two_sided_p_from_z(z) if not math.isnan(z) else float('nan')
        ci_low = est - 1.96 * se
        ci_high = est + 1.96 * se

        result_obj = {
            "term": "effect_of_reader_view_for_dyslexic (reader_view_bin + reader_view_bin:is_dyslexic)",
            "estimate_wpm": float(est),
            "se_wpm": float(se),
            "z_value": float(z),
            "p_value": float(pval),
            "95%_ci": [float(ci_low), float(ci_high)]
        }

        desc = (
            "Estimated effect of turning Reader View ON for readers with dyslexia (difference in words-per-minute). "
            f"Positive means Reader View increased reading speed. "
            f"Estimate = {est:.3f} wpm (SE = {se:.3f}), 95% CI [{ci_low:.3f}, {ci_high:.3f}], p = {pval:.4f}."
        )
        return {"object": result_obj, "description": desc}

    # If no interaction but main effect exists, interpret main effect as average effect (applies to both groups)
    if main_p is not None:
        b_main = float(params[main_p])
        try:
            var_main = float(cov.loc[main_p, main_p])
            se = math.sqrt(var_main)
        except Exception as e:
            return {
                "object": None,
                "description": f"Failed to get SE for main effect parameter '{main_p}': {e}"
            }
        z = b_main / se if se and not math.isnan(se) else float('nan')
        pval = two_sided_p_from_z(z) if not math.isnan(z) else float('nan')
        ci_low = b_main - 1.96 * se
        ci_high = b_main + 1.96 * se

        result_obj = {
            "term": "effect_of_reader_view (main effect reader_view_bin)",
            "estimate_wpm": float(b_main),
            "se_wpm": float(se),
            "z_value": float(z),
            "p_value": float(pval),
            "95%_ci": [float(ci_low), float(ci_high)],
            "note": "No interaction term present; this is the average effect across dyslexic and non-dyslexic readers."
        }

        desc = (
            "No reader_view_bin:is_dyslexic interaction found in the model. "
            "Returning the main effect of Reader View (average effect across groups). "
            f"Estimate = {b_main:.3f} wpm (SE = {se:.3f}), 95% CI [{ci_low:.3f}, {ci_high:.3f}], p = {pval:.4f}."
        )
        return {"object": result_obj, "description": desc}

    # Neither main nor interaction found
    return {
        "object": None,
        "description": (
            "Model does not contain parameters for 'reader_view_bin' or the interaction 'reader_view_bin:is_dyslexic'. "
            "Cannot determine the effect of Reader View for dyslexic readers from this model output."
        )
    }