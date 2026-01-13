def extract_final_answer(model_output):
    """
    Extract the marginal effect of Reader View for individuals with dyslexia
    from a fitted statsmodels RegressionResultsWrapper that included the
    interaction reader_view * dyslexia_bin.

    Returns a dict with:
      - "object": dict with keys:
           'marginal_effect' : estimated effect on log_speed when dyslexia_bin=1
           'se'              : standard error of that linear combination
           't'               : t-statistic
           'p_value'         : two-sided p-value
           'ci_lower'        : lower bound of 95% CI
           'ci_upper'        : upper bound of 95% CI
           'significant'     : boolean, True if p_value < 0.05
      - "description": plain-English interpretation of the result
    """
    import numpy as np
    from scipy import stats

    res = model_output  # expected statsmodels RegressionResultsWrapper

    # Parameter names: primary expects 'reader_view' and 'reader_view:dyslexia_bin'
    params = res.params
    cov = res.cov_params()

    # Helper to find interaction parameter name robustly
    possible_inter_names = [
        "reader_view:dyslexia_bin",
        "dyslexia_bin:reader_view",
        "reader_view*dyslexia_bin",  # unlikely but keep as fallback
        "reader_view:dyslexia_bin[T.1]"  # not likely here
    ]

    # Identify main and interaction names present in the model
    main_name = None
    inter_name = None
    for name in params.index:
        if name == "reader_view":
            main_name = "reader_view"
        # catch case where variable encoding yields something like reader_view[T.1]
        if name.startswith("reader_view") and name != "reader_view":
            # prefer exact match but do not override if exact found
            if main_name is None and ":" not in name and "*" not in name:
                main_name = name
        for pin in possible_inter_names:
            if name == pin:
                inter_name = name
    # If explicit interaction name not found, try to locate any param that contains both tokens
    if inter_name is None:
        for name in params.index:
            if "reader_view" in name and "dyslexia_bin" in name and name != main_name:
                inter_name = name
                break

    if main_name is None:
        raise KeyError("Could not find a parameter named 'reader_view' in model params: "
                       f"found {list(params.index)}")
    if inter_name is None:
        raise KeyError("Could not find an interaction parameter between reader_view and dyslexia_bin "
                       f"in model params: found {list(params.index)}")

    # Coefficients
    beta_main = float(params[main_name])
    beta_inter = float(params[inter_name])
    # Marginal effect of reader_view when dyslexia_bin == 1:
    marginal_effect = beta_main + beta_inter

    # Variance of linear combination: Var(aX + bY) with a=b=1 here
    var_main = float(cov.loc[main_name, main_name])
    var_inter = float(cov.loc[inter_name, inter_name])
    cov_main_inter = float(cov.loc[main_name, inter_name])
    var_comb = var_main + var_inter + 2.0 * cov_main_inter
    se_comb = float(np.sqrt(var_comb)) if var_comb >= 0 else np.nan

    # t-statistic and p-value (two-sided). Use t-distribution with residual df if available.
    t_stat = marginal_effect / se_comb if se_comb and not np.isnan(se_comb) else np.nan
    # degrees of freedom -- fall back to large-sample normal if not present
    try:
        df_resid = float(res.df_resid)
        p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_resid))
        # 95% CI using t critical value
        crit = stats.t.ppf(1.0 - 0.025, df=df_resid)
    except Exception:
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(t_stat)))
        crit = stats.norm.ppf(1.0 - 0.025)

    ci_lower = marginal_effect - crit * se_comb
    ci_upper = marginal_effect + crit * se_comb

    result_obj = {
        "marginal_effect": marginal_effect,
        "se": se_comb,
        "t": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": (p_value < 0.05)
    }

    # Human-readable description
    if np.isnan(marginal_effect) or np.isnan(p_value):
        description = ("Could not compute the marginal effect and inference for Reader View "
                       "among individuals with dyslexia due to missing/invalid estimates.")
    else:
        direction = "increase" if marginal_effect > 0 else ("decrease" if marginal_effect < 0 else "no change")
        sig_text = "statistically significant (p < 0.05)" if result_obj["significant"] else "not statistically significant (p >= 0.05)"
        description = (
            f"The estimated marginal effect of activating Reader View for participants with dyslexia "
            f"is {marginal_effect:.4f} on log_speed (SE = {se_comb:.4f}, t = {t_stat:.3f}, p = {p_value:.3g}). "
            f"The 95% CI is [{ci_lower:.4f}, {ci_upper:.4f}]. This corresponds to a {direction} in log-transformed "
            f"reading speed and is {sig_text}. In plain terms, if the effect is positive and significant, Reader View "
            f"speeds up reading for individuals with dyslexia; if negative and significant, it slows them down; "
            f"if not significant, there is no strong evidence of an effect."
        )

    return {"object": result_obj, "description": description}