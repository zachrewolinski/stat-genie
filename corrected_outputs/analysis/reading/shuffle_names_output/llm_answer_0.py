def extract_final_answer(model_output):
    """
    Extracts the estimated effect of ReaderView for readers with dyslexia from a
    statsmodels OLSResults (robust) object that was fit with an interaction:
        LogReadingWPM ~ ReaderView * DyslexiaAny + ...
    
    Returns a dict with:
      - object: dict with coefficients, standard error, t, p, 95% CI, and percent change
                for the combined effect of ReaderView for DyslexiaAny == 1
      - description: human-readable explanation of what the numbers mean
    
    The function is robust to different parameter representations (e.g. params as
    numpy array or pandas Series/DataFrame) and different naming conventions
    (e.g. 'ReaderView', 'ReaderView[T.1]', 'ReaderView:DyslexiaAny', etc.).
    """
    import numpy as np
    import pandas as pd
    from math import exp
    from scipy import stats

    res = model_output  # expected to be a statsmodels results object (robust)

    # Normalize params to a pandas Series with parameter names as the index
    raw_params = getattr(res, "params", None)
    if raw_params is None:
        raise ValueError("Provided model object has no 'params' attribute.")

    if isinstance(raw_params, pd.Series):
        params = raw_params.copy()
    elif isinstance(raw_params, pd.DataFrame):
        # If a DataFrame (e.g., multi-index), try to squeeze to Series
        params = raw_params.squeeze()
        if not isinstance(params, pd.Series):
            # fallback: take the first column
            params = pd.Series(raw_params.iloc[:, 0].values, index=raw_params.columns)
    elif isinstance(raw_params, np.ndarray):
        # Try to get parameter names from the model object
        names = None
        if hasattr(res, "model") and hasattr(res.model, "exog_names"):
            names = list(res.model.exog_names)
        elif hasattr(res, "param_names"):
            # statsmodels sometimes exposes this
            names = list(getattr(res, "param_names"))
        if names is None:
            # fallback generic names
            names = [f"param_{i}" for i in range(len(raw_params))]
        params = pd.Series(raw_params, index=names)
    else:
        # Final fallback: try to cast to Series
        try:
            params = pd.Series(raw_params)
        except Exception:
            raise ValueError("Unable to interpret model params of type: {}".format(type(raw_params)))

    # Normalize covariance matrix to a pandas DataFrame with same index/columns as params
    raw_cov = None
    try:
        raw_cov = res.cov_params()
    except Exception:
        # Some results objects might store cov in different attr
        raw_cov = getattr(res, "cov_params", None)
        if callable(raw_cov):
            raw_cov = raw_cov()
    if raw_cov is None:
        raise ValueError("Provided model object has no covariance information (cov_params).")

    if isinstance(raw_cov, pd.DataFrame):
        cov = raw_cov.copy()
    elif isinstance(raw_cov, np.ndarray):
        # Ensure shape consistency
        kv = len(params)
        if raw_cov.shape[0] != kv or raw_cov.shape[1] != kv:
            # Try to handle cases where covariance contains extra params: attempt to align by names if possible
            cov = pd.DataFrame(raw_cov)
            cov = cov.iloc[:kv, :kv]
            cov.index = params.index[:cov.shape[0]]
            cov.columns = params.index[:cov.shape[1]]
        else:
            cov = pd.DataFrame(raw_cov, index=params.index, columns=params.index)
    else:
        try:
            cov = pd.DataFrame(raw_cov)
            # attempt to set index/columns
            if cov.shape[0] == len(params):
                cov.index = params.index
            if cov.shape[1] == len(params):
                cov.columns = params.index
        except Exception:
            raise ValueError("Unable to interpret covariance matrix of type: {}".format(type(raw_cov)))

    names = list(params.index)

    # Find main ReaderView parameter name (contains 'ReaderView' but not ':' which denotes interaction)
    main_candidates = [n for n in names if ('ReaderView' in n) and (':' not in n)]
    main_name = main_candidates[0] if main_candidates else None

    # Find interaction parameter name (contains both ReaderView and Dyslexia or contains ':' and ReaderView)
    inter_candidates = [
        n for n in names
        if ('ReaderView' in n) and (('Dyslexia' in n) or (':' in n and 'ReaderView' in n))
    ]
    # Prefer candidate that also contains 'Dyslexia'
    interaction_name = None
    for n in inter_candidates:
        if 'Dyslexia' in n:
            # Ensure it's not the same as the main (rare edge case)
            if n != main_name:
                interaction_name = n
                break
    if interaction_name is None:
        # pick first candidate that's not the main (if possible)
        for n in inter_candidates:
            if n != main_name:
                interaction_name = n
                break
    if interaction_name is None and inter_candidates:
        # last resort: allow it to be the same as main (models sometimes encode differently)
        interaction_name = inter_candidates[0]

    if main_name is None:
        raise ValueError("Could not find a main 'ReaderView' parameter in the model parameters: {}".format(names))

    # Extract coefficients (safely)
    beta_main = float(params.loc[main_name]) if main_name in params.index else float(params.iloc[0])
    beta_inter = 0.0
    if interaction_name is not None and interaction_name in params.index:
        beta_inter = float(params.loc[interaction_name])
    else:
        # interaction not present or not found -> assume zero
        beta_inter = 0.0

    # Combined effect for DyslexiaAny == 1: beta_main + beta_inter
    beta_combined = beta_main + beta_inter

    # Variance and standard error for the linear combination
    if interaction_name is not None and interaction_name in cov.index and main_name in cov.index:
        var_main = float(cov.loc[main_name, main_name])
        var_inter = float(cov.loc[interaction_name, interaction_name])
        covar = float(cov.loc[main_name, interaction_name])
        var_combined = var_main + var_inter + 2.0 * covar
    else:
        # No interaction term present or not found in covariance: variance is just var(main)
        if main_name in cov.index:
            var_combined = float(cov.loc[main_name, main_name])
        else:
            # fallback: take diagonal first element
            var_combined = float(np.diag(cov.values)[0])

    # Guard against negative variance due to numerical issues
    if var_combined < 0 and abs(var_combined) < 1e-12:
        var_combined = 0.0
    if var_combined < 0:
        raise ValueError("Computed negative variance for combined effect: {}".format(var_combined))

    se_combined = float(np.sqrt(var_combined))

    # t-statistic and two-sided p-value using t-distribution with df_resid if available
    df_resid = getattr(res, "df_resid", None)
    # df_resid might be array-like; coerce to float if possible
    try:
        if df_resid is not None and np.size(df_resid) == 1:
            df_resid = float(df_resid)
    except Exception:
        df_resid = None

    t_stat = beta_combined / se_combined if se_combined != 0 else np.nan
    if df_resid is None:
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        t_crit = float(stats.norm.ppf(1 - 0.025))
    else:
        # Ensure df_resid > 0
        try:
            if df_resid <= 0 or np.isnan(df_resid):
                raise Exception()
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df_resid))
            t_crit = float(stats.t.ppf(1.0 - 0.025, df=df_resid))
        except Exception:
            p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
            t_crit = float(stats.norm.ppf(1 - 0.025))

    # 95% CI for combined effect (on log WPM scale)
    ci_lower = beta_combined - t_crit * se_combined
    ci_upper = beta_combined + t_crit * se_combined

    # Convert log-scale effect to percent change in WPM: (exp(beta) - 1) * 100
    try:
        pct_change = (exp(beta_combined) - 1.0) * 100.0
        ci_lower_pct = (exp(ci_lower) - 1.0) * 100.0
        ci_upper_pct = (exp(ci_upper) - 1.0) * 100.0
    except OverflowError:
        pct_change = float('inf')
        ci_lower_pct = float('inf')
        ci_upper_pct = float('inf')

    result_obj = {
        "beta_main_ReaderView (log WPM, for DyslexiaAny=0)": beta_main,
        "beta_interaction ReaderView:DyslexiaAny": beta_inter,
        "beta_ReaderView_for_Dyslexia (log WPM)": beta_combined,
        "se_ReaderView_for_Dyslexia": se_combined,
        "t_ReaderView_for_Dyslexia": t_stat,
        "pvalue_ReaderView_for_Dyslexia (two-sided)": p_value,
        "95%_CI_logWPM": [ci_lower, ci_upper],
        "percent_change_in_WPM_for_Dyslexia": pct_change,
        "95%_CI_percent_change": [ci_lower_pct, ci_upper_pct],
        "n_obs": int(getattr(res, "nobs", res.model.nobs if hasattr(res, "model") and hasattr(res.model, "nobs") else None)),
        "parameter_names_used": {"main": main_name, "interaction": interaction_name},
    }

    description_lines = [
        "Returned numbers describe the estimated effect of activating ReaderView for readers with dyslexia (DyslexiaAny == 1).",
        "- Coefficients are on the log(WPM) scale. The key value is 'beta_ReaderView_for_Dyslexia (log WPM)'.",
        "- The percent change in WPM is (exp(beta) - 1) * 100 and gives an interpretable effect on reading speed.",
        "- pvalue_ReaderView_for_Dyslexia is the two-sided p-value for the null hypothesis that the combined effect = 0.",
        "- If p < 0.05, there is evidence that ReaderView significantly changes reading speed for readers with dyslexia.",
        "- The 95% CI is provided both on the log scale and transformed to percent change.",
        "Note: If the model used a different parameter naming convention, the function attempts to locate the main and interaction parameters automatically;"
        " if it cannot find a ReaderView main effect it will raise an error."
    ]
    description = " ".join(description_lines)

    return {"object": result_obj, "description": description}