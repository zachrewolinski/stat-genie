def extract_final_answer(model_output):
    """
    Extract the estimated effect of ReaderView for individuals with dyslexia
    from a fitted statsmodels results object (robust cov results expected).
    Returns a dict with numeric results under "object" and a textual
    interpretation under "description".
    """
    import numpy as np
    from scipy import stats

    # Helper to safely get params; handle Series (with .index) or ndarray
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError("model_output has no .params attribute") from e

    # Build param_names and a mapping from name->value for robust access
    params_dict = {}
    if hasattr(params, "index"):
        # Likely a pandas Series
        param_names = [str(n) for n in params.index]
        for n in params.index:
            params_dict[str(n)] = float(params.loc[n])
    else:
        # params is array-like; try to obtain names from common locations
        params_arr = np.asarray(params, dtype=float)
        param_names = None
        if hasattr(model_output, "param_names") and model_output.param_names is not None:
            param_names = [str(n) for n in model_output.param_names]
        elif hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
            param_names = [str(n) for n in model_output.model.exog_names]
        elif (
            hasattr(model_output, "model")
            and getattr(model_output.model, "data", None) is not None
            and hasattr(model_output.model.data, "param_names")
            and model_output.model.data.param_names is not None
        ):
            param_names = [str(n) for n in model_output.model.data.param_names]
        else:
            # Fallback generic names
            param_names = [f"param{i}" for i in range(len(params_arr))]
        # Map names to values (assume same ordering)
        for i, name in enumerate(param_names):
            try:
                params_dict[name] = float(params_arr[i])
            except Exception:
                # If indexing fails, set NaN to indicate missing value
                params_dict[name] = float("nan")

    # Identify main effect name for ReaderView
    main_name = None
    if "ReaderView" in param_names:
        main_name = "ReaderView"
    else:
        cand = [n for n in param_names if "ReaderView" in n]
        if len(cand) > 0:
            main_name = cand[0]

    # Identify interaction term name containing ReaderView and dyslexia
    inter_name = None
    for n in param_names:
        if ("ReaderView" in n) and ("dyslexia" in n):
            inter_name = n
            break

    # Get numeric main and interaction coefficients (None if missing)
    main_coef = None
    if main_name is not None:
        main_coef = params_dict.get(main_name, None)
        if main_coef is None:
            # If mapping produced NaN, raise
            raise ValueError("Main 'ReaderView' coefficient was found in names but not in params.")
        main_coef = float(main_coef)

    inter_coef = 0.0
    if inter_name is not None:
        inter_val = params_dict.get(inter_name, None)
        if inter_val is None:
            raise ValueError("Interaction coefficient found in names but not in params.")
        inter_coef = float(inter_val)

    # Covariance matrix for linear combination
    cov = None
    try:
        # cov_params may be a method
        cov = model_output.cov_params()
    except Exception:
        # fallback: try attribute names commonly present
        if hasattr(model_output, "cov_params_default"):
            cov = model_output.cov_params_default
        elif hasattr(model_output, "normalized_cov_params"):
            # normalized_cov_params may be provided as ndarray
            cov = model_output.normalized_cov_params
        else:
            raise ValueError("Could not obtain covariance matrix from model_output")

    # Ensure main_coef exists
    if main_coef is None:
        raise ValueError("Could not find a main 'ReaderView' coefficient in the model parameters.")

    # Compute combined effect and standard error
    if inter_name is None:
        combined = main_coef
        # variance is var(main)
        try:
            var_main = float(cov.loc[main_name, main_name])
        except Exception:
            # fallback: if cov is ndarray-like, find index
            try:
                idx_map = {n: i for i, n in enumerate(param_names)}
                i = idx_map[main_name]
                var_main = float(np.asarray(cov)[i, i])
            except Exception:
                raise ValueError("Could not extract variance for main coefficient from covariance matrix.")
        se = float(np.sqrt(var_main))
    else:
        combined = main_coef + inter_coef
        # compute variance: Var(main + inter) = Var(main) + Var(inter) + 2*Cov(main,inter)
        try:
            var_main = float(cov.loc[main_name, main_name])
            var_inter = float(cov.loc[inter_name, inter_name])
            covar = float(cov.loc[main_name, inter_name])
        except Exception:
            # fallback for ndarray-like cov
            idx_map = {n: i for i, n in enumerate(param_names)}
            try:
                i = idx_map[main_name]
                j = idx_map[inter_name]
                arr = np.asarray(cov)
                var_main = float(arr[i, i])
                var_inter = float(arr[j, j])
                covar = float(arr[i, j])
            except Exception:
                raise ValueError("Could not extract required entries from covariance matrix.")
        var_combined = var_main + var_inter + 2.0 * covar
        # numeric safeguard
        if var_combined < 0 and var_combined > -1e-12:
            var_combined = 0.0
        if var_combined < 0:
            raise ValueError(f"Computed negative variance for combined effect: {var_combined}")
        se = float(np.sqrt(var_combined))

    # t-stat and p-value (two-sided). Use df_resid if available, else normal approx.
    t_stat = combined / se if se > 0 else float("nan")
    df_resid = getattr(model_output, "df_resid", None)
    p_two = None
    ci_lower = None
    ci_upper = None
    try:
        if (df_resid is not None) and (np.isfinite(df_resid)) and (df_resid > 0) and np.isfinite(t_stat):
            p_two = float(2.0 * stats.t.sf(abs(t_stat), df=df_resid))
            t_crit = float(stats.t.ppf(0.975, df=df_resid))
            ci_lower = combined - t_crit * se
            ci_upper = combined + t_crit * se
        else:
            # normal approximation
            if np.isfinite(t_stat):
                p_two = float(2.0 * stats.norm.sf(abs(t_stat)))
            else:
                p_two = float("nan")
            z = float(stats.norm.ppf(0.975))
            ci_lower = combined - z * se
            ci_upper = combined + z * se
    except Exception:
        # fallback to normal approx if something fails
        p_two = float(2.0 * stats.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else float("nan")
        z = float(stats.norm.ppf(0.975))
        ci_lower = combined - z * se
        ci_upper = combined + z * se

    # Translate effect on log1p(WPM) to multiplicative change in (1 + WPM):
    # factor = exp(delta); percent_change = (exp(delta) - 1) * 100
    try:
        pct_change = float((np.exp(combined) - 1.0) * 100.0)
        pct_ci_lower = float((np.exp(ci_lower) - 1.0) * 100.0)
        pct_ci_upper = float((np.exp(ci_upper) - 1.0) * 100.0)
    except Exception:
        pct_change = float("nan")
        pct_ci_lower = float("nan")
        pct_ci_upper = float("nan")

    # Statistical significance at alpha=0.05
    significant = (p_two is not None) and (np.isfinite(p_two)) and (p_two < 0.05)

    # Prepare object to return (JSON-serializable types)
    result_obj = {
        "coef_readerview_main (log1p WPM)": float(main_coef),
        "coef_interaction_readerview:dyslexia (log1p WPM)": float(inter_coef) if inter_name is not None else 0.0,
        "effect_for_dyslexia_readers (log1p WPM)": float(combined),
        "se_effect_for_dyslexia": float(se),
        "t_stat_effect_for_dyslexia": float(t_stat) if np.isfinite(t_stat) else None,
        "p_value_two_sided": float(p_two) if (p_two is not None and np.isfinite(p_two)) else None,
        "95%_ci_log1p_lower": float(ci_lower),
        "95%_ci_log1p_upper": float(ci_upper),
        "approx_percent_change_in_(1+WPM)": float(pct_change),
        "95%_ci_percent_change_lower": float(pct_ci_lower),
        "95%_ci_percent_change_upper": float(pct_ci_upper),
        "significant_at_0.05": bool(significant),
        "notes": (
            "Dependent variable is log1p(WPM)=ln(1+WPM). "
            "The percent change columns show 100*(exp(effect)-1) which is the "
            "multiplicative change in (1+WPM) associated with ReaderView for readers with dyslexia."
        ),
    }

    # Human-readable description
    if significant:
        sig_text = "statistically significant (p < 0.05)"
    else:
        sig_text = "not statistically significant (p >= 0.05)"

    # Safely format numeric values for description (use nan-aware formatting)
    def fmt(x, fmt_spec=".4f"):
        try:
            if x is None or (isinstance(x, float) and not np.isfinite(x)):
                return "NA"
            return format(x, fmt_spec)
        except Exception:
            return "NA"

    description = (
        f"The estimated effect of turning Reader View ON for readers with dyslexia is {fmt(combined, '.4f')} "
        f"on the log1p(WPM) scale (SE = {fmt(se, '.4f')}, t = {fmt(t_stat, '.3f')}, p = {fmt(p_two, '.3g')}). "
        f"This effect is {sig_text}. "
        f"On the original scale of (1 + WPM), this corresponds to an approximate "
        f"{fmt(pct_change, '.2f')}% change (95% CI: {fmt(pct_ci_lower, '.2f')}% to {fmt(pct_ci_upper, '.2f')}%). "
        f"Interpret these percent changes as changes in (1 + WPM); for typical WPM values the percentage "
        f"change in WPM will be very similar. Main and interaction coefficients returned above."
    )

    return {"object": result_obj, "description": description}