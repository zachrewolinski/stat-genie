def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View on reading speed for individuals with dyslexia
    from a fitted statsmodels MixedLMResults (MixedLMResultsWrapper) object.

    Returns a dictionary with:
      - "object": a dict containing numeric results (log-scale estimate, SE, z, p, 95% CI,
                  and percent-change interpretation with CI)
      - "description": a brief human-readable interpretation of the result in context.

    The function handles two cases:
      1) the model includes an interaction term between reader_view and dyslexia_bin:
           effect_for_dyslexic = coef(reader_view) + coef(reader_view:dyslexia_bin)
      2) the model does NOT include an interaction term: the effect_for_dyslexic is taken
           to be the main reader_view coefficient (with a note in the description).
    """
    import numpy as np
    from scipy.stats import norm

    res = model_output

    # Get parameter names and arrays
    try:
        params = res.params  # pandas Series
        cov = res.cov_params()  # DataFrame
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a statsmodels results object "
                         "with .params and .cov_params(). Error: " + str(e))

    param_names = list(params.index.astype(str))

    # Helper to find parameter name for main reader_view and the interaction
    # Strategy: exact match 'reader_view' for main; for interaction find a param name
    # that contains both 'reader_view' and 'dyslexia_bin'. If not found, try common separators.
    def find_main_reader():
        if 'reader_view' in param_names:
            return 'reader_view'
        # try alternative exact matches
        for n in param_names:
            if n.strip() == 'reader_view':
                return n
        # fallback: choose a name that contains 'reader_view' but not dyslexia
        for n in param_names:
            if ('reader_view' in n) and ('dyslexia' not in n):
                return n
        return None

    def find_interaction():
        # look for name containing both substrings
        for n in param_names:
            if ('reader_view' in n) and ('dyslexia' in n):
                return n
        # try colon or bracket variants
        for sep in [':', '*', '_x_']:
            candidate = f"reader_view{sep}dyslexia_bin"
            if candidate in param_names:
                return candidate
            candidate = f"dyslexia_bin{sep}reader_view"
            if candidate in param_names:
                return candidate
        return None

    main_name = find_main_reader()
    int_name = find_interaction()

    if main_name is None:
        raise KeyError("Could not find a parameter corresponding to 'reader_view' in model params. "
                       "Available params: " + ", ".join(param_names))

    # Extract coefficients
    beta_main = float(params[main_name])
    var_main = float(cov.loc[main_name, main_name])

    if int_name is not None:
        beta_int = float(params[int_name])
        var_int = float(cov.loc[int_name, int_name])
        cov_main_int = float(cov.loc[main_name, int_name])
        # Combined effect for dyslexic readers = main + interaction
        beta_comb = beta_main + beta_int
        var_comb = var_main + var_int + 2.0 * cov_main_int
        note_interaction = True
    else:
        # No interaction term: assume main effect applies to dyslexic readers as well
        beta_comb = beta_main
        var_comb = var_main
        note_interaction = False

    # Standard error, z, p-value (two-sided)
    se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else float('nan')
    if np.isnan(se_comb) or se_comb == 0:
        z_val = float('nan')
        p_val = float('nan')
    else:
        z_val = beta_comb / se_comb
        p_val = 2.0 * norm.sf(abs(z_val))

    # 95% CI on log scale
    ci_low = beta_comb - 1.96 * se_comb
    ci_high = beta_comb + 1.96 * se_comb

    # Convert to multiplicative percent change in reading speed:
    # percent change = (exp(beta) - 1) * 100
    pct_change = (np.exp(beta_comb) - 1.0) * 100.0
    pct_ci_low = (np.exp(ci_low) - 1.0) * 100.0
    pct_ci_high = (np.exp(ci_high) - 1.0) * 100.0

    # Build the return object
    obj = {
        "model_param_names": param_names,
        "reader_view_param_name_used": main_name,
        "interaction_param_name_used": int_name,
        "estimate_log_effect": beta_comb,
        "std_error_log_effect": se_comb,
        "z_value": z_val,
        "p_value": p_val,
        "95CI_log": [ci_low, ci_high],
        "percent_change_in_speed": pct_change,
        "95CI_percent_change": [pct_ci_low, pct_ci_high],
    }

    # Short description interpreting results
    if note_interaction:
        sig_text = "statistically significant" if (not np.isnan(p_val) and p_val < 0.05) else "not statistically significant"
        description = (
            f"The estimated effect of turning ON Reader View for readers with dyslexia is the sum of the "
            f"'reader_view' and 'reader_view:dyslexia_bin' coefficients. The combined estimate on the log speed "
            f"scale is {beta_comb:.4f} (SE = {se_comb:.4f}), z = {z_val:.2f}, p = {p_val:.3g}. "
            f"On the original speed scale this corresponds to a {pct_change:.1f}% change in reading speed "
            f"(95% CI: {pct_ci_low:.1f}% to {pct_ci_high:.1f}%). This effect is {sig_text} at alpha=0.05."
        )
    else:
        sig_text = "statistically significant" if (not np.isnan(p_val) and p_val < 0.05) else "not statistically significant"
        description = (
            f"The model does not contain a reader_view × dyslexia_bin interaction term. The effect of "
            f"Reader View for all readers (including those with dyslexia) is the 'reader_view' coefficient: "
            f"{beta_comb:.4f} (SE = {se_comb:.4f}), z = {z_val:.2f}, p = {p_val:.3g). if np.isnan(p_val) else ''}. "
        )
        # The previous f-string attempted to inline a conditional; correct it for clarity:
        description = (
            f"The model does not contain a reader_view × dyslexia_bin interaction term. The effect of "
            f"Reader View for all readers (including those with dyslexia) is the 'reader_view' coefficient: "
            f"{beta_comb:.4f} (SE = {se_comb:.4f}), z = {z_val:.2f}, p = {p_val:.3g}. "
            f"On the original speed scale this corresponds to a {pct_change:.1f}% change in reading speed "
            f"(95% CI: {pct_ci_low:.1f}% to {pct_ci_high:.1f}%). This effect is {sig_text} at alpha=0.05."
        )

    return {"object": obj, "description": description}