def extract_final_answer(model_output):
    """
    Extracts and interprets the effect of 'reader_view' (and its interaction with dyslexia_bin)
    from a fitted statsmodels RegressionResultsWrapper.

    Returns a dictionary:
      {
        "object": {
            "reader_view_non_dyslexic": {coef, se, pvalue, ci_lower, ci_upper, pct_change, pct_CI_lower, pct_CI_upper, significant},
            "reader_view_dyslexic": {same fields for the combined effect},
            "interaction_term": {coef, se, pvalue, ci_lower, ci_upper}  # if present
        },
        "description": "Plain-language interpretation and conclusion about whether Reader View improves reading speed for individuals with dyslexia."
      }

    Notes:
    - Uses the model's reported covariance matrix (cov_params) to compute standard errors for sums
      (i.e., combined effect for dyslexic readers).
    - Two-sided p-values and 95% CIs for the combined effect are computed using a normal approximation
      (common when using cluster-robust covariances). If the model used a different inference method,
      the model_output's own p-values for single coefficients are preserved/returned.
    """
    import numpy as np
    from scipy import stats

    res = model_output

    # Helper to find parameter names robustly
    params = res.params
    index = list(params.index)

    def find_param_exact_or_contains(target_exact, contains_all=None):
        # Try exact match first
        if target_exact in index:
            return target_exact
        # If contains_all specified (list of strings), find a name containing all those substrings
        if contains_all:
            for name in index:
                if all(sub in name for sub in contains_all):
                    return name
        # Falls back to any name that contains the main token
        for name in index:
            if target_exact in name:
                return name
        return None

    # Determine parameter names
    name_rv = find_param_exact_or_contains('reader_view', contains_all=None)
    name_int = None
    # interaction should include both tokens reader_view and dyslexia
    for name in index:
        if 'reader_view' in name and 'dyslexia' in name:
            name_int = name
            break
    name_dys = find_param_exact_or_contains('dyslexia_bin', contains_all=None)

    # Ensure we at least have reader_view
    if name_rv is None:
        raise ValueError("Could not find a parameter name for 'reader_view' in model params: %s" % index)

    # Extract coefficient-level info for available terms
    def coef_info(term_name):
        if term_name is None:
            return None
        coef = float(params[term_name])
        # Prefer model-provided bse/pvalues/conf_int if available
        try:
            se = float(res.bse[term_name])
        except Exception:
            # derive from cov_params if possible
            try:
                cov = res.cov_params()
                se = float(np.sqrt(cov.loc[term_name, term_name]))
            except Exception:
                se = np.nan
        try:
            pval = float(res.pvalues[term_name])
        except Exception:
            # fallback to normal approx
            if not np.isnan(se) and se > 0:
                z = coef / se
                pval = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                pval = np.nan
        try:
            ci = res.conf_int().loc[term_name].to_list()
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            # approximate with normal approx
            if not np.isnan(se):
                ci_lower = coef - 1.96 * se
                ci_upper = coef + 1.96 * se
            else:
                ci_lower = ci_upper = np.nan
        return {
            "term": term_name,
            "coef": coef,
            "se": se,
            "pvalue": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    info_rv = coef_info(name_rv)
    info_int = coef_info(name_int) if name_int is not None else None
    info_dys = coef_info(name_dys) if name_dys is not None else None

    # Compute combined effect for dyslexic readers: beta_rv + beta_interaction (if interaction exists)
    covmat = None
    try:
        covmat = res.cov_params()
    except Exception:
        covmat = None

    beta_rv = info_rv["coef"]
    # If interaction term missing, treat its coef as 0 with 0 covariance
    beta_int = info_int["coef"] if info_int is not None else 0.0

    # Combined coefficient
    beta_dys = beta_rv + beta_int

    # Variance calculations
    if covmat is not None:
        try:
            var_rv = float(covmat.loc[name_rv, name_rv])
        except Exception:
            var_rv = info_rv["se"] ** 2 if not np.isnan(info_rv["se"]) else np.nan
        if info_int is not None:
            try:
                var_int = float(covmat.loc[name_int, name_int])
                cov_ri = float(covmat.loc[name_rv, name_int])
            except Exception:
                var_int = info_int["se"] ** 2 if not np.isnan(info_int["se"]) else np.nan
                cov_ri = 0.0
        else:
            var_int = 0.0
            cov_ri = 0.0
        var_dys = var_rv + var_int + 2.0 * cov_ri
        se_rv = np.sqrt(var_rv) if not np.isnan(var_rv) else info_rv["se"]
        se_dys = np.sqrt(var_dys) if var_dys >= 0 else np.nan
    else:
        # Fallback to sum-in-quadrature of reported SEs (ignores covariance)
        se_rv = info_rv["se"]
        se_int = info_int["se"] if info_int is not None else 0.0
        se_dys = np.sqrt((se_rv ** 2) + (se_int ** 2))
        var_rv = se_rv ** 2
        var_dys = se_dys ** 2

    # p-values and CIs for non-dyslexic (reader_view effect when dyslexia_bin=0)
    p_rv = info_rv["pvalue"] if (info_rv and not np.isnan(info_rv["pvalue"])) else (
        2 * (1 - stats.norm.cdf(abs(beta_rv / se_rv))) if se_rv and se_rv > 0 else np.nan
    )
    ci_rv_lower = beta_rv - 1.96 * se_rv if se_rv and not np.isnan(se_rv) else np.nan
    ci_rv_upper = beta_rv + 1.96 * se_rv if se_rv and not np.isnan(se_rv) else np.nan

    # For dyslexic combined effect
    z_dys = beta_dys / se_dys if se_dys and not np.isnan(se_dys) else np.nan
    p_dys = 2 * (1 - stats.norm.cdf(abs(z_dys))) if not np.isnan(z_dys) else np.nan
    ci_dys_lower = beta_dys - 1.96 * se_dys if not np.isnan(se_dys) else np.nan
    ci_dys_upper = beta_dys + 1.96 * se_dys if not np.isnan(se_dys) else np.nan

    # Convert log-scale effects to approximate percent changes: (exp(beta)-1)*100
    try:
        pct_rv = (np.exp(beta_rv) - 1.0) * 100.0
        pct_rv_ci_lower = (np.exp(ci_rv_lower) - 1.0) * 100.0
        pct_rv_ci_upper = (np.exp(ci_rv_upper) - 1.0) * 100.0
    except Exception:
        pct_rv = pct_rv_ci_lower = pct_rv_ci_upper = np.nan

    try:
        pct_dys = (np.exp(beta_dys) - 1.0) * 100.0
        pct_dys_ci_lower = (np.exp(ci_dys_lower) - 1.0) * 100.0
        pct_dys_ci_upper = (np.exp(ci_dys_upper) - 1.0) * 100.0
    except Exception:
        pct_dys = pct_dys_ci_lower = pct_dys_ci_upper = np.nan

    # Significance flags at alpha=0.05
    sig_rv = (p_rv < 0.05) if (not np.isnan(p_rv)) else None
    sig_dys = (p_dys < 0.05) if (not np.isnan(p_dys)) else None

    # Build return object
    result_object = {
        "reader_view_non_dyslexic": {
            "term": name_rv,
            "coef_log": beta_rv,
            "se": se_rv,
            "pvalue": p_rv,
            "ci_log": [ci_rv_lower, ci_rv_upper],
            "approx_pct_change": pct_rv,
            "approx_pct_change_ci": [pct_rv_ci_lower, pct_rv_ci_upper],
            "significant_at_0.05": sig_rv
        },
        "reader_view_dyslexic": {
            "term_combination": f"{name_rv} + {name_int}" if name_int is not None else name_rv,
            "coef_log": beta_dys,
            "se": se_dys,
            "pvalue": p_dys,
            "ci_log": [ci_dys_lower, ci_dys_upper],
            "approx_pct_change": pct_dys,
            "approx_pct_change_ci": [pct_dys_ci_lower, pct_dys_ci_upper],
            "significant_at_0.05": sig_dys
        },
        "interaction_term": info_int,  # may be None
        "notes": {
            "coef_names": {
                "reader_view": name_rv,
                "interaction": name_int,
                "dyslexia_bin": name_dys
            },
            "inference_details": (
                "P-values and standard errors use the model's reported covariance matrix where available. "
                "Combined effect for dyslexic readers computed as coef(reader_view) + coef(interaction). "
                "Confidence intervals and p-values for the combined effect use a normal approximation; "
                "if the model used cluster-robust SEs these were taken from res.cov_params()."
            )
        }
    }

    # Short plain-language description / conclusion about the study question
    # We focus on whether reader_view improves reading speed for individuals with dyslexia:
    if sig_dys is True:
        direction = "increase" if beta_dys > 0 else "decrease"
        descr = (
            f"For readers with dyslexia, activating Reader View is associated with a statistically significant "
            f"{direction} in log reading speed (coef={beta_dys:.4f}, se={se_dys:.4f}, p={p_dys:.3g}). "
            f"On the original speed scale this corresponds to an approximate {pct_dys:.1f}% change "
            f"(95% CI: {pct_dys_ci_lower:.1f}% to {pct_dys_ci_upper:.1f}%)."
        )
    elif sig_dys is False:
        direction = "increase" if beta_dys > 0 else "decrease"
        descr = (
            f"For readers with dyslexia, the estimated effect of Reader View is a {direction} in log reading speed "
            f"(coef={beta_dys:.4f}, se={se_dys:.4f}), but this effect is not statistically significant at alpha=0.05 "
            f"(p={p_dys:.3g}). The approximate percent change is {pct_dys:.1f}% "
            f"(95% CI: {pct_dys_ci_lower:.1f}% to {pct_dys_ci_upper:.1f}%)."
        )
    else:
        descr = (
            "Could not determine statistical significance for the dyslexia subgroup (missing standard errors or p-values). "
            "Returned raw coefficients and approximate conversions to percent change where possible."
        )

    return {"object": result_object, "description": descr}