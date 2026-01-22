def extract_final_answer(model_output):
    """
    Extracts statistics relevant to whether Reader View improves reading speed for individuals with dyslexia.
    Expects a fitted statsmodels results object (OLS/MixedLM or a robustified results wrapper).

    Returns:
      {
        "object": {
           "model_type": str,
           "interaction_name": str or None,
           "interaction_coef": float or None,
           "interaction_se": float or None,
           "interaction_pval": float or None,
           "interaction_ci": [low, high] or None,
           "reader_view_coef_non_dyslexic": float or None,
           "reader_view_se_non_dyslexic": float or None,
           "reader_view_pval_non_dyslexic": float or None,
           "reader_view_ci_non_dyslexic": [low, high] or None,
           "reader_view_effect_dyslexic_log": float or None,
           "reader_view_effect_dyslexic_se": float or None,
           "reader_view_effect_dyslexic_pval": float or None,
           "reader_view_effect_dyslexic_pct_change": float or None
        },
        "description": "Concise interpretation"
      }

    The function is defensive: it searches for the interaction term name in the model parameter names
    (looks for any param name containing both 'reader_view' and 'dyslexia' / 'dyslexia_bin'). If covariance
    matrix is available, it computes the standard error and p-value for the combined effect (reader_view + interaction)
    which is the effect of Reader View for dyslexic readers. p-values for combined effect use a normal approximation.
    """
    import math
    from collections import OrderedDict

    # Helper: normal two-sided p-value from z
    def two_sided_p_from_z(z):
        # p = 2*(1 - Phi(|z|)) where Phi is CDF of standard normal
        # Use erf: Phi(x) = 0.5*(1 + erf(x/sqrt(2)))
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    # Prepare return structure
    result_obj = OrderedDict(
        [
            ("model_type", None),
            ("interaction_name", None),
            ("interaction_coef", None),
            ("interaction_se", None),
            ("interaction_pval", None),
            ("interaction_ci", None),
            ("reader_view_coef_non_dyslexic", None),
            ("reader_view_se_non_dyslexic", None),
            ("reader_view_pval_non_dyslexic", None),
            ("reader_view_ci_non_dyslexic", None),
            ("reader_view_effect_dyslexic_log", None),
            ("reader_view_effect_dyslexic_se", None),
            ("reader_view_effect_dyslexic_pval", None),
            ("reader_view_effect_dyslexic_pct_change", None),
        ]
    )

    # Determine model type string
    try:
        result_obj["model_type"] = type(model_output).__name__
    except Exception:
        result_obj["model_type"] = str(model_output)

    # Get params (as a pandas Series or dict-like)
    try:
        params = model_output.params
    except Exception as e:
        return {
            "object": result_obj,
            "description": f"Could not read model parameters from the provided object: {e}",
        }

    # Ensure parameter names are strings
    try:
        param_names = [str(n) for n in params.index]
    except Exception:
        param_names = [str(n) for n in list(params.keys())]

    # Find interaction parameter name: any param containing both reader_view and dyslexia/dyslexia_bin
    interaction_candidates = [
        n
        for n in param_names
        if ("reader_view" in n and ("dyslexia" in n or "dyslexia_bin" in n) and ":" in n)
        or ("reader_view" in n and "dyslexia" in n and ":" not in n and "/" in n)  # just in case
    ]
    # fallback: any param that contains both substrings even without ":" (defensive)
    if not interaction_candidates:
        interaction_candidates = [
            n for n in param_names if ("reader_view" in n and "dyslexia" in n)
        ]

    interaction_name = interaction_candidates[0] if interaction_candidates else None
    result_obj["interaction_name"] = interaction_name

    # Find reader_view main effect parameter: contains reader_view but not dyslexia
    reader_view_candidates = [
        n
        for n in param_names
        if ("reader_view" in n and "dyslexia" not in n)
    ]
    # Prefer exact 'reader_view' if present
    if "reader_view" in param_names:
        reader_view_name = "reader_view"
    elif reader_view_candidates:
        # pick the shortest matching name (most likely the main numeric term)
        reader_view_name = sorted(reader_view_candidates, key=len)[0]
    else:
        reader_view_name = None

    # Extract interaction stats if present
    def safe_get_value(series_like, key):
        try:
            return float(series_like[key])
        except Exception:
            return None

    if interaction_name is not None:
        interaction_coef = safe_get_value(params, interaction_name)
        # SE and p-val may be in bse / pvalues
        try:
            bse = model_output.bse if hasattr(model_output, "bse") else None
            pvalues = model_output.pvalues if hasattr(model_output, "pvalues") else None
            interaction_se = safe_get_value(bse, interaction_name) if bse is not None else None
            interaction_pval = safe_get_value(pvalues, interaction_name) if pvalues is not None else None
        except Exception:
            interaction_se = interaction_pval = None

        # Confidence interval
        try:
            ci_df = model_output.conf_int()
            # conf_int may be a DataFrame with index matching param names
            if interaction_name in ci_df.index:
                ci_low, ci_high = float(ci_df.loc[interaction_name, 0]), float(ci_df.loc[interaction_name, 1])
                interaction_ci = [ci_low, ci_high]
            else:
                interaction_ci = None
        except Exception:
            interaction_ci = None

        result_obj["interaction_coef"] = interaction_coef
        result_obj["interaction_se"] = interaction_se
        result_obj["interaction_pval"] = interaction_pval
        result_obj["interaction_ci"] = interaction_ci
    else:
        interaction_coef = interaction_se = interaction_pval = interaction_ci = None

    # Extract reader_view main effect stats (non-dyslexic effect)
    if reader_view_name is not None:
        rv_coef = safe_get_value(params, reader_view_name)
        try:
            bse = model_output.bse if hasattr(model_output, "bse") else None
            pvalues = model_output.pvalues if hasattr(model_output, "pvalues") else None
            rv_se = safe_get_value(bse, reader_view_name) if bse is not None else None
            rv_pval = safe_get_value(pvalues, reader_view_name) if pvalues is not None else None
        except Exception:
            rv_se = rv_pval = None

        try:
            ci_df = model_output.conf_int()
            if reader_view_name in ci_df.index:
                ci_low, ci_high = float(ci_df.loc[reader_view_name, 0]), float(ci_df.loc[reader_view_name, 1])
                rv_ci = [ci_low, ci_high]
            else:
                rv_ci = None
        except Exception:
            rv_ci = None

        result_obj["reader_view_coef_non_dyslexic"] = rv_coef
        result_obj["reader_view_se_non_dyslexic"] = rv_se
        result_obj["reader_view_pval_non_dyslexic"] = rv_pval
        result_obj["reader_view_ci_non_dyslexic"] = rv_ci
    else:
        rv_coef = rv_se = rv_pval = rv_ci = None

    # Compute combined effect for dyslexic readers: reader_view + interaction
    if (rv_coef is not None) and (interaction_coef is not None):
        combined_coef = rv_coef + interaction_coef
        result_obj["reader_view_effect_dyslexic_log"] = combined_coef

        # Try to compute SE of combined effect using covariance matrix
        try:
            cov = model_output.cov_params()
            # ensure keys present
            if interaction_name in cov.index and reader_view_name in cov.index:
                var_rv = float(cov.loc[reader_view_name, reader_view_name])
                var_int = float(cov.loc[interaction_name, interaction_name])
                cov_rv_int = float(cov.loc[reader_view_name, interaction_name])
                combined_var = var_rv + var_int + 2.0 * cov_rv_int
                combined_se = math.sqrt(max(combined_var, 0.0))
                result_obj["reader_view_effect_dyslexic_se"] = combined_se
                # p-value using normal approx
                z = combined_coef / combined_se if combined_se and combined_se > 0 else None
                combined_p = two_sided_p_from_z(z) if z is not None else None
                result_obj["reader_view_effect_dyslexic_pval"] = combined_p
            else:
                # Cov matrix present but keys missing
                result_obj["reader_view_effect_dyslexic_se"] = None
                result_obj["reader_view_effect_dyslexic_pval"] = None
        except Exception:
            result_obj["reader_view_effect_dyslexic_se"] = None
            result_obj["reader_view_effect_dyslexic_pval"] = None

        # Percent change in wpm (log outcome): (exp(coef)-1)*100
        try:
            pct_change = (math.exp(combined_coef) - 1.0) * 100.0
            result_obj["reader_view_effect_dyslexic_pct_change"] = pct_change
        except Exception:
            result_obj["reader_view_effect_dyslexic_pct_change"] = None
    else:
        # if missing either element, cannot compute combined
        result_obj["reader_view_effect_dyslexic_log"] = None
        result_obj["reader_view_effect_dyslexic_se"] = None
        result_obj["reader_view_effect_dyslexic_pval"] = None
        result_obj["reader_view_effect_dyslexic_pct_change"] = None

    # Build brief description / interpretation
    if result_obj["reader_view_effect_dyslexic_log"] is None:
        description = (
            "Could not locate required parameter(s) to compute the effect of Reader View for dyslexic readers. "
            "Searched model parameters for 'reader_view' and an interaction with 'dyslexia'/'dyslexia_bin'. "
            "Returned object contains whatever statistics were found."
        )
    else:
        coef = result_obj["reader_view_effect_dyslexic_log"]
        pct = result_obj["reader_view_effect_dyslexic_pct_change"]
        p_comb = result_obj["reader_view_effect_dyslexic_pval"]
        # Interpret significance
        if p_comb is None:
            sig_text = "p-value unavailable for the combined effect (could not compute SE/covariance)."
        else:
            sig_text = (
                f"the combined effect has p = {p_comb:.3g}; "
                + ("statistically significant (p < 0.05)." if p_comb < 0.05 else "not statistically significant (p >= 0.05).")
            )
        description = (
            f"Estimated effect of Reader View for dyslexic readers (log scale) = {coef:.4g}. "
            f"This corresponds to ~{pct:.2f}% change in predicted reading speed (wpm). "
            f"According to the model, {sig_text} "
            "Positive percent change means Reader View is associated with faster reading; negative means slower. "
            "All effects are on the natural-log scale of wpm; reported CIs, SEs, and p-values are in the returned 'object'."
        )

    return {"object": dict(result_obj), "description": description}