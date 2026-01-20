def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, and 95% CIs for:
      - the main effect of reader_view (effect for dyslexia_bin == 0)
      - the interaction reader_view:dyslexia_bin and the combined effect
        of reader_view for dyslexic readers (reader_view + reader_view:dyslexia_bin)

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Brief interpretation of the returned numbers"
      }

    Notes:
      - Assumes parameter names contain 'reader_view' and that the interaction
        parameter name contains both 'reader_view' and 'dyslexia' (e.g.
        'reader_view:dyslexia_bin').
      - Uses the model object's robust covariance (the model was fit with cov_type='HC3').
    """
    import numpy as np
    import scipy.stats as st

    res = model_output
    params = res.params
    cov = res.cov_params()
    pvals = res.pvalues
    ci = res.conf_int()

    # Find parameter names robustly
    name_main = None
    name_inter = None
    for n in params.index:
        if ('reader_view' in n) and ('dyslexia' not in n):
            # choose the main reader_view term (not an interaction)
            name_main = n
        if ('reader_view' in n) and ('dyslexia' in n):
            name_inter = n

    if name_main is None:
        raise KeyError("Could not find a 'reader_view' main effect parameter in model params: " + ", ".join(params.index))

    # Main effect (effect of Reader View when dyslexia_bin == 0)
    coef_main = float(params[name_main])
    p_main = float(pvals[name_main])
    ci_main = [float(ci.loc[name_main, 0]), float(ci.loc[name_main, 1])]
    # approximate percent change on original speed scale (since DV is ln(speed))
    pct_main = (np.exp(coef_main) - 1.0) * 100
    pct_ci_main = [(np.exp(ci_main[0]) - 1.0) * 100, (np.exp(ci_main[1]) - 1.0) * 100]

    result = {
        "nobs": int(getattr(res, "nobs", np.nan)),
        "df_resid": float(getattr(res, "df_resid", np.nan)),
        "reader_view_main": {
            "param_name": name_main,
            "coef": coef_main,
            "p_value": p_main,
            "95%_CI_coef": ci_main,
            "approx_percent_change": pct_main,
            "95%_CI_percent_change": pct_ci_main
        }
    }

    # If interaction exists, compute combined effect for dyslexic readers:
    if name_inter is not None and name_inter in params.index:
        coef_inter = float(params[name_inter])
        # combined coefficient for dyslexic readers
        coef_dys = coef_main + coef_inter

        # compute standard error for the linear combination using covariance matrix
        var_main = float(cov.loc[name_main, name_main])
        var_inter = float(cov.loc[name_inter, name_inter])
        cov_main_inter = float(cov.loc[name_main, name_inter])
        se_dys = np.sqrt(var_main + var_inter + 2.0 * cov_main_inter)

        # t-test and p-value using residual df
        df = res.df_resid
        tval = coef_dys / se_dys if se_dys > 0 else np.nan
        p_dys = float(2.0 * st.t.sf(abs(tval), df)) if not np.isnan(tval) else np.nan

        # 95% CI for combined coef
        t_crit = st.t.ppf(0.975, df)
        ci_dys = [coef_dys - t_crit * se_dys, coef_dys + t_crit * se_dys]

        # percent-change interpretation
        pct_dys = (np.exp(coef_dys) - 1.0) * 100
        pct_ci_dys = [(np.exp(ci_dys[0]) - 1.0) * 100, (np.exp(ci_dys[1]) - 1.0) * 100]

        result["reader_view_interaction"] = {
            "param_name": name_inter,
            "coef_interaction": coef_inter,
            "reader_view_for_dyslexic_coef": coef_dys,
            "reader_view_for_dyslexic_se": se_dys,
            "reader_view_for_dyslexic_t": tval,
            "reader_view_for_dyslexic_p_value": p_dys,
            "reader_view_for_dyslexic_95%_CI_coef": ci_dys,
            "reader_view_for_dyslexic_approx_percent_change": pct_dys,
            "reader_view_for_dyslexic_95%_CI_percent_change": pct_ci_dys
        }
    else:
        # No interaction term found — the effect is the same for dyslexic and non-dyslexic under this model
        result["reader_view_interaction"] = {
            "param_name": None,
            "note": "No reader_view:dyslexia interaction parameter found; model does not estimate a differential effect."
        }

    # Short interpretive description
    if name_inter is not None:
        desc = (
            "Returned: (1) the main effect of Reader View (coefficient, p-value, 95% CI) which equals the effect for "
            "non-dyslexic readers (dyslexia_bin==0), and (2) the interaction term plus the combined effect of Reader View "
            "for dyslexic readers (reader_view + interaction). Coefficients are on the log(speed) scale: exp(coef)-1 "
            "gives the approximate percent change in speed. Compare p-values to your threshold (e.g., 0.05) to decide "
            "whether Reader View significantly affects reading speed overall or differently for dyslexic readers."
        )
    else:
        desc = (
            "Returned: the main effect of Reader View (coefficient, p-value, 95% CI). No interaction term was detected, so "
            "the model does not estimate a different Reader View effect for dyslexic readers; the reported main effect "
            "applies to all readers under this specification. Coefficients are on the log(speed) scale: exp(coef)-1 "
            "gives the approximate percent change in speed."
        )

    return {"object": result, "description": desc}