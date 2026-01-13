def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and confidence intervals for:
      - main effect of ReaderView (ReaderView)
      - interaction ReaderView:Dyslexia
      - combined effect of ReaderView for individuals with Dyslexia (ReaderView + ReaderView:Dyslexia)

    Returns:
      {
        "object": { ...metrics... } or None,
        "description": <string explanation>
      }
    """
    import math
    from collections import OrderedDict

    # Helpers for safe access
    def safe_get(series_like, key):
        try:
            return series_like[key]
        except Exception:
            # Pandas Series will raise KeyError, dict will too; return None if missing
            return None

    # Check for presence of params
    params = getattr(model_output, "params", None)
    if params is None or (hasattr(params, "size") and getattr(params, "size", 0) == 0):
        return {
            "object": None,
            "description": (
                "No model results available in model_output (params is empty). "
                "Cannot extract coefficients or p-values. Ensure the model fitted successfully "
                "and that model_output contains attributes like params and cov_params()."
            ),
        }

    # Identify possible names for interaction term
    interaction_names = [
        "ReaderView:Dyslexia",
        "ReaderView * Dyslexia",
        "ReaderView*Dyslexia",
        "ReaderView:Dyslexia[T.1]",  # unlikely but try
    ]

    # Identify the main ReaderView term name (most likely 'ReaderView')
    reader_name = None
    for name in ("ReaderView",):
        if safe_get(params, name) is not None:
            reader_name = name
            break
    if reader_name is None:
        # fallback: try to find a param name that contains 'ReaderView' (but not 'Dyslexia')
        for k in params.index:
            if "ReaderView" in str(k) and "Dyslexia" not in str(k):
                reader_name = k
                break

    # Find interaction term name
    interaction_name = None
    for nm in interaction_names:
        if safe_get(params, nm) is not None:
            interaction_name = nm
            break
    if interaction_name is None:
        # fallback: search params names for both tokens
        for k in params.index:
            if "ReaderView" in str(k) and "Dyslexia" in str(k):
                interaction_name = k
                break

    # Collect results for main ReaderView
    results = OrderedDict()

    if reader_name is None:
        return {
            "object": None,
            "description": (
                "The model parameters do not contain a 'ReaderView' term. "
                "Cannot evaluate the ReaderView effect. Check that the model formula "
                "and variable names match exactly."
            ),
        }

    beta_rv = float(safe_get(params, reader_name))
    # Attempt to get standard error and p-value
    se_rv = None
    p_rv = None
    ci_rv = None

    # p-values may be in model_output.pvalues
    pvalues = getattr(model_output, "pvalues", None)
    if pvalues is not None:
        p_rv = safe_get(pvalues, reader_name)
        if p_rv is not None:
            p_rv = float(p_rv)

    # standard errors may be in bse
    bse = getattr(model_output, "bse", None)
    if bse is not None:
        se_val = safe_get(bse, reader_name)
        if se_val is not None:
            se_rv = float(se_val)

    # If cov_params available, get se and allow computing combined effect se
    cov = None
    try:
        cov = model_output.cov_params()
    except Exception:
        # Some objects expose cov_params as a method, others as attribute; try attribute
        cov = getattr(model_output, "cov_params", None)
        if callable(cov):
            try:
                cov = cov()
            except Exception:
                cov = None
        else:
            cov = None

    # Compute CI for main effect if possible
    df_resid = getattr(model_output, "df_resid", None)
    use_t = False
    if df_resid is not None and not (math.isnan(df_resid) if isinstance(df_resid, float) else False):
        try:
            import scipy.stats as _st

            crit = _st.t.ppf(1 - 0.025, df_resid)
            use_t = True
        except Exception:
            try:
                import math
                crit = 1.96
            except Exception:
                crit = 1.96
    else:
        crit = 1.96

    if se_rv is None and cov is not None and reader_name in cov.index:
        se_rv = math.sqrt(float(cov.loc[reader_name, reader_name]))

    if se_rv is not None:
        ci_rv = (beta_rv - crit * se_rv, beta_rv + crit * se_rv)
        # Compute p-value if missing using t or normal approx
        if p_rv is None:
            try:
                import scipy.stats as _st

                if use_t:
                    tstat = beta_rv / se_rv
                    p_rv = float(2 * _st.t.sf(abs(tstat), df_resid))
                else:
                    z = beta_rv / se_rv
                    p_rv = float(2 * _st.norm.sf(abs(z)))
            except Exception:
                # fallback: cannot compute p-value
                p_rv = None

    results["ReaderView_main"] = {
        "param_name": reader_name,
        "beta": beta_rv,
        "se": se_rv,
        "p_value": p_rv,
        "95ci": ci_rv,
        "note": "Main effect of ReaderView for the reference (non-dyslexic) group unless interaction present.",
    }

    # Combined effect for dyslexic readers: beta_readerview + beta_interaction
    if interaction_name is None:
        # No interaction term present: effect for dyslexic is same as main
        combined_beta = beta_rv
        combined_se = se_rv
        combined_p = p_rv
        combined_ci = ci_rv
        interaction_present = False
    else:
        interaction_present = True
        beta_int = float(safe_get(params, interaction_name))
        # Try to get se of interaction
        se_int = None
        if bse is not None:
            se_tmp = safe_get(bse, interaction_name)
            if se_tmp is not None:
                se_int = float(se_tmp)
        if se_int is None and cov is not None and interaction_name in cov.index:
            se_int = math.sqrt(float(cov.loc[interaction_name, interaction_name]))

        combined_beta = beta_rv + beta_int

        # Compute variance of sum: var(a+b) = var(a) + var(b) + 2 cov(a,b)
        combined_se = None
        combined_p = None
        combined_ci = None

        if cov is not None and reader_name in cov.index and interaction_name in cov.index:
            var_r = float(cov.loc[reader_name, reader_name])
            var_i = float(cov.loc[interaction_name, interaction_name])
            cov_ri = float(cov.loc[reader_name, interaction_name])
            var_comb = var_r + var_i + 2.0 * cov_ri
            if var_comb < 0:
                # numerical issue
                combined_se = None
            else:
                combined_se = math.sqrt(var_comb)
                # p-value
                try:
                    import scipy.stats as _st

                    if use_t:
                        tstat = combined_beta / combined_se
                        combined_p = float(2 * _st.t.sf(abs(tstat), df_resid))
                    else:
                        z = combined_beta / combined_se
                        combined_p = float(2 * _st.norm.sf(abs(z)))
                except Exception:
                    combined_p = None
                combined_ci = (combined_beta - crit * combined_se, combined_beta + crit * combined_se)
        else:
            # Fallback: if individual ses available but no covariance, we cannot compute combined se properly.
            if se_rv is not None and se_int is not None:
                # Assuming covariance = 0 (conservative/incorrect), compute approximate se
                approx_var = se_rv ** 2 + se_int ** 2
                combined_se = math.sqrt(approx_var)
                try:
                    import scipy.stats as _st

                    if use_t:
                        tstat = combined_beta / combined_se
                        combined_p = float(2 * _st.t.sf(abs(tstat), df_resid))
                    else:
                        z = combined_beta / combined_se
                        combined_p = float(2 * _st.norm.sf(abs(z)))
                except Exception:
                    combined_p = None
                combined_ci = (combined_beta - crit * combined_se, combined_beta + crit * combined_se)
            else:
                combined_se = None
                combined_p = None
                combined_ci = None

        results["ReaderView_interaction"] = {
            "param_name": interaction_name,
            "beta_interaction": beta_int,
            "se_interaction": se_int,
            "note": "Interaction term: additional effect of ReaderView when Dyslexia==1 (on top of main ReaderView).",
        }

    results["ReaderView_effect_for_dyslexic"] = {
        "beta": combined_beta,
        "se": combined_se,
        "p_value": combined_p,
        "95ci": combined_ci,
        "note": (
            "Combined effect of ReaderView for participants with Dyslexia (ReaderView + ReaderView:Dyslexia). "
            "If interaction term is absent, this equals the main ReaderView effect."
        ),
        "interaction_present": interaction_present,
    }

    # Interpretation sentence
    interp = []
    if results["ReaderView_effect_for_dyslexic"]["p_value"] is None:
        interp.append(
            "Could not compute a reliable p-value for the combined ReaderView effect for dyslexic readers "
            "(missing covariance or standard error information)."
        )
    else:
        pval = results["ReaderView_effect_for_dyslexic"]["p_value"]
        beta = results["ReaderView_effect_for_dyslexic"]["beta"]
        if pval < 0.05:
            direction = "increase" if beta > 0 else "decrease"
            interp.append(
                f"ReaderView has a statistically significant {direction} in log(WPM) for dyslexic readers "
                f"(combined beta = {beta:.4g}, p = {pval:.3g})."
            )
        else:
            interp.append(
                f"No statistically significant effect of ReaderView for dyslexic readers (combined beta = {beta:.4g}, p = {pval:.3g})."
            )

    # Additional note about interpretation on the original scale
    interp.append(
        "Note: the dependent variable is log(WPM). Exponentiating a beta gives multiplicative change in WPM: "
        "exp(beta) ≈ multiplicative factor for WPM when ReaderView is activated for dyslexic readers."
    )

    return {"object": results, "description": " ".join(interp)}