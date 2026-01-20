def extract_final_answer(model_output):
    """
    Extracts the estimated effect of Reader View for readers with dyslexia from a fitted statsmodels result.

    Returns a dict with:
      - "object": dict of numeric results (coef, se, t, p, 95% CI) for:
            * effect when dyslexia = 1 (main + interaction)
            * effect when dyslexia = 0 (main effect)
            * percent-change interpretation (exp(coef)-1) and CI
      - "description": textual explanation of what the numbers mean and a simple yes/no conclusion
    """
    import numpy as np
    import pandas as pd
    from math import isnan
    try:
        from scipy import stats
    except Exception:
        stats = None

    res = model_output

    # Get parameter names and parameter series
    try:
        raw_params = res.params
    except Exception as e:
        raise ValueError("model_output has no .params attribute") from e

    # Normalize params into a pandas Series with names
    if hasattr(raw_params, "index"):
        params = raw_params
        param_names = list(params.index)
    else:
        # raw_params is likely a numpy array; attempt to find names from model metadata
        names = None
        model_obj = getattr(res, "model", None)
        if model_obj is not None:
            names = getattr(model_obj, "exog_names", None)
        if names is None:
            # try other possible attributes
            names = getattr(res, "param_names", None)
        if names is None:
            # fallback to generic names
            try:
                length = len(raw_params)
            except Exception:
                raise ValueError("Unable to determine parameter names or length from model_output.params")
            names = [f"param_{i}" for i in range(length)]
        param_names = list(names)
        try:
            params = pd.Series(np.asarray(raw_params).astype(float), index=param_names)
        except Exception:
            # as a last resort, wrap raw_params without casting
            params = pd.Series(raw_params, index=param_names)

    # Covariance of parameter estimates (robust/corrected if provided by model_output)
    try:
        cov = res.cov_params()
        # if cov is ndarray, convert to DataFrame for consistent indexing
        if isinstance(cov, (np.ndarray, list)):
            cov = np.asarray(cov)
            cov = pd.DataFrame(cov, index=param_names, columns=param_names)
    except Exception:
        # fallback: try attribute .normalized_cov_params or build diag from bse
        try:
            bse = res.bse
            bse_arr = np.asarray(bse, dtype=float)
            cov = np.diag(bse_arr ** 2)
            cov = pd.DataFrame(cov, index=param_names, columns=param_names)
        except Exception:
            raise ValueError("Unable to obtain covariance matrix from model_output")

    # Identify the main reader_view parameter and the interaction term with dyslexia.
    # We look for a param that contains 'reader_view' but not 'dyslexia' for main.
    # Interaction should contain both 'reader_view' and 'dyslexia' (colon or other separator).
    main_name = None
    interaction_name = None

    for n in param_names:
        if 'reader_view' in n and 'dyslexia' not in n:
            # prefer exact match if present
            if n == 'reader_view':
                main_name = n
                break
            if main_name is None:
                main_name = n
    # find interaction (contains both tokens)
    for n in param_names:
        if 'reader_view' in n and 'dyslexia' in n:
            interaction_name = n
            break

    if main_name is None:
        # try any param that equals 'reader_view' exactly (defensive)
        if 'reader_view' in param_names:
            main_name = 'reader_view'
        else:
            raise ValueError("Could not find a parameter name for the 'reader_view' main effect. "
                             f"Available params: {param_names}")

    # Extract main coef
    try:
        main_coef = float(params[main_name])
    except Exception:
        # If the params Series cannot be indexed by name for some reason, fall back to positional
        try:
            idx_main = param_names.index(main_name)
            main_coef = float(params.iloc[idx_main])
        except Exception as e:
            raise ValueError("Unable to extract main parameter coefficient") from e

    # variance for main
    try:
        var_main = float(cov.loc[main_name, main_name])
    except Exception:
        # cov might be np.ndarray or DataFrame but indexing failed
        try:
            idx = param_names.index(main_name)
            var_main = float(cov.iloc[idx, idx]) if hasattr(cov, "iloc") else float(np.asarray(cov)[idx, idx])
        except Exception as e:
            raise ValueError("Unable to index covariance for main parameter") from e

    # If interaction exists, compute combined effect for dyslexia=1
    if interaction_name is not None:
        try:
            inter_coef = float(params[interaction_name])
        except Exception:
            try:
                idx_inter = param_names.index(interaction_name)
                inter_coef = float(params.iloc[idx_inter])
            except Exception as e:
                raise ValueError("Unable to extract interaction parameter coefficient") from e

        try:
            var_inter = float(cov.loc[interaction_name, interaction_name])
            cov_main_inter = float(cov.loc[main_name, interaction_name])
        except Exception:
            try:
                i_main = param_names.index(main_name)
                i_inter = param_names.index(interaction_name)
                arr = np.asarray(cov)
                var_inter = float(arr[i_inter, i_inter])
                cov_main_inter = float(arr[i_main, i_inter])
            except Exception as e:
                raise ValueError("Unable to index covariance for interaction parameter") from e

        effect_dys_coef = main_coef + inter_coef
        var_effect_dys = var_main + var_inter + 2.0 * cov_main_inter
    else:
        inter_coef = 0.0
        effect_dys_coef = main_coef
        var_effect_dys = var_main

    # Ensure variances non-negative
    var_effect_dys = max(var_effect_dys, 0.0)
    se_effect_dys = var_effect_dys ** 0.5

    # t-stats and p-values
    if se_effect_dys == 0 or isnan(se_effect_dys):
        t_effect_dys = float('nan')
        p_effect_dys = float('nan')
    else:
        t_effect_dys = effect_dys_coef / se_effect_dys
        # try to use t-distribution with df_resid if available, else normal approx
        df = getattr(res, 'df_resid', None)
        try:
            df_val = float(df) if df is not None else None
        except Exception:
            df_val = None
        if df_val is None or (df_val <= 0 or np.isnan(df_val)):
            # normal approx
            if stats is not None:
                p_effect_dys = 2.0 * (1.0 - stats.norm.cdf(abs(t_effect_dys)))
            else:
                p_effect_dys = float('nan')
        else:
            if stats is not None:
                p_effect_dys = 2.0 * stats.t.sf(abs(t_effect_dys), df_val)
            else:
                p_effect_dys = float('nan')

    # Confidence interval for effect_dys (95%)
    if stats is not None and df_val is not None and not (df_val <= 0 or np.isnan(df_val)):
        tcrit = stats.t.ppf(0.975, df_val)
    elif stats is not None:
        tcrit = stats.norm.ppf(0.975)
    else:
        tcrit = 1.96
    ci_lower = effect_dys_coef - tcrit * se_effect_dys
    ci_upper = effect_dys_coef + tcrit * se_effect_dys

    # Also compute for non-dyslexia (dyslexia = 0): main effect only
    se_main = var_main ** 0.5
    if se_main == 0 or isnan(se_main):
        t_main = float('nan')
        p_main = float('nan')
    else:
        t_main = main_coef / se_main
        df = getattr(res, 'df_resid', None)
        try:
            df_val2 = float(df) if df is not None else None
        except Exception:
            df_val2 = None
        if df_val2 is None or (df_val2 <= 0 or np.isnan(df_val2)):
            if stats is not None:
                p_main = 2.0 * (1.0 - stats.norm.cdf(abs(t_main)))
            else:
                p_main = float('nan')
        else:
            if stats is not None:
                p_main = 2.0 * stats.t.sf(abs(t_main), df_val2)
            else:
                p_main = float('nan')

    # Interpret effect in percent change on original speed scale:
    # For small changes: percent ≈ 100*(exp(coef)-1)
    try:
        pct_effect_dys = 100.0 * (np.exp(effect_dys_coef) - 1.0)
        pct_ci_lower = 100.0 * (np.exp(ci_lower) - 1.0)
        pct_ci_upper = 100.0 * (np.exp(ci_upper) - 1.0)
    except Exception:
        pct_effect_dys = pct_ci_lower = pct_ci_upper = float('nan')

    # Simple yes/no conclusion (alpha=0.05)
    if not np.isnan(p_effect_dys):
        if p_effect_dys < 0.05 and effect_dys_coef > 0:
            conclusion = "Yes: statistically significant evidence that Reader View increases reading speed for readers with dyslexia (alpha=0.05)."
        elif p_effect_dys < 0.05 and effect_dys_coef < 0:
            conclusion = "No: statistically significant evidence that Reader View decreases reading speed for readers with dyslexia (alpha=0.05)."
        else:
            conclusion = "No: no statistically significant evidence that Reader View changes reading speed for readers with dyslexia (alpha=0.05)."
    else:
        conclusion = "Could not determine statistical significance (p-value unavailable)."

    result_object = {
        "main_param_name": main_name,
        "interaction_param_name": interaction_name,
        "effect_dys_coef_log": effect_dys_coef,
        "effect_dys_se_log": se_effect_dys,
        "effect_dys_t": t_effect_dys,
        "effect_dys_p": p_effect_dys,
        "effect_dys_ci95_log": (ci_lower, ci_upper),
        "effect_dys_pct_change": pct_effect_dys,
        "effect_dys_pct_change_ci95": (pct_ci_lower, pct_ci_upper),
        "no_dys_coef_log": main_coef,
        "no_dys_se_log": se_main,
        "no_dys_t": t_main,
        "no_dys_p": p_main,
    }

    description_lines = [
        "This extracts the estimated effect of enabling Reader View on log(reading speed) for readers with dyslexia.",
        f"Identified main reader_view parameter as '{main_name}' and interaction as '{interaction_name}'.",
        f"Effect for dyslexia=1 (log-scale): coef = {effect_dys_coef:.4f}, SE = {se_effect_dys:.4f}, t = {t_effect_dys:.3f}, p = {p_effect_dys:.4f}.",
        f"95% CI on log-scale: [{ci_lower:.4f}, {ci_upper:.4f}].",
        f"Interpreting on original speed scale: estimated percent change = {pct_effect_dys:.2f}%, 95% CI = [{pct_ci_lower:.2f}%, {pct_ci_upper:.2f}%].",
        f"Effect for dyslexia=0 (main effect only): coef = {main_coef:.4f}, SE = {se_main:.4f}, t = {t_main:.3f}, p = {p_main:.4f}.",
        conclusion,
        "Notes: Dependent variable is natural-log of winsorized speed, so coefficients are interpretable as approximate proportionate changes (exp(coef)-1). "
        "If an interaction term exists, the effect for dyslexic readers equals main + interaction; otherwise it's just the main effect.",
    ]

    return {"object": result_object, "description": " ".join(description_lines)}