def extract_final_answer(model_output):
    """
    Extracts statistics needed to answer whether Reader View improves reading speed for individuals
    with dyslexia from a fitted statsmodels OLSResults (robust covariance) object.

    Returns a dict with:
      - "object": dict with coefficients, robust SEs, p-values, 95% CIs for:
            * reader_view effect for non-dyslexic readers
            * reader_view:dyslexia_bin interaction
            * combined reader_view effect for dyslexic readers (reader_view + interaction)
            * model nobs and df_resid
      - "description": short plain-language interpretation answering the task question.

    Notes:
      - Expects parameter names like 'reader_view' and 'reader_view:dyslexia_bin' in model_output.params.index.
      - Uses model_output.cov_params() (robust covariance matrix) to compute SE & p-value for linear combination.
    """
    import numpy as np
    from scipy import stats
    import pandas as pd

    # Ensure model_output has the attributes we need
    required_attrs = ['params', 'cov_params', 'nobs', 'df_resid']
    for a in required_attrs:
        if not hasattr(model_output, a):
            raise ValueError(f"model_output is missing required attribute: {a}")

    # Normalize params to a pandas Series with an index of parameter names
    raw_params = model_output.params
    if isinstance(raw_params, pd.Series):
        params = raw_params.copy()
    else:
        # raw_params may be a numpy array or list; try to obtain names from the model/result
        if hasattr(model_output, 'model') and hasattr(model_output.model, 'exog_names'):
            index = list(model_output.model.exog_names)
        elif hasattr(model_output, 'param_names'):
            index = list(model_output.param_names)
        elif hasattr(model_output, 'params') and hasattr(getattr(model_output, 'params'), 'dtype'):
            # fallback: generic names
            length = len(raw_params)
            index = [f'param_{i}' for i in range(length)]
        else:
            # ultimate fallback
            index = [f'param_{i}' for i in range(len(raw_params))]
        params = pd.Series(np.asarray(raw_params), index=index)

    # Get covariance matrix and ensure it's a DataFrame with matching index/columns
    cov_raw = model_output.cov_params()
    if isinstance(cov_raw, pd.DataFrame):
        cov = cov_raw.copy()
        # Reindex to match params if necessary
        try:
            cov = cov.reindex(index=params.index, columns=params.index)
        except Exception:
            # If reindex fails, attempt to set columns/index directly if shapes align
            if cov.shape == (len(params), len(params)):
                cov.index = params.index
                cov.columns = params.index
            else:
                raise
    else:
        # Assume ndarray-like
        cov_arr = np.asarray(cov_raw)
        if cov_arr.shape[0] != len(params):
            # try to infer names from cov if possible; otherwise raise informative error
            raise ValueError("Covariance matrix shape does not match number of parameters.")
        cov = pd.DataFrame(cov_arr, index=params.index, columns=params.index)

    # Helper to robustly find parameter names (allowing slight naming differences)
    def find_param(possible_names):
        for name in possible_names:
            if name in params.index:
                return name
        # fallback: try substring matching
        for name in params.index:
            try:
                low = str(name).lower()
            except Exception:
                continue
            for cand in possible_names:
                cand_parts = cand.replace(':', ' ').split()
                if all(part.lower() in low for part in cand_parts if part):
                    return name
        return None

    reader_name = find_param(['reader_view', 'reader_view[T.1]', 'reader_view_1', 'reader_view:'])
    interaction_name = find_param(['reader_view:dyslexia_bin', 'reader_view:dyslexia_bin[T.1]',
                                  'reader_view:dyslexia_bin_1', 'reader_view:dyslexia',
                                  'reader_view:dyslexia_bin:T.1', 'reader_view*dyslexia_bin'])

    if reader_name is None:
        # Attempt to find a parameter that contains 'reader_view'
        matches = [n for n in params.index if 'reader_view' in str(n)]
        if matches:
            reader_name = matches[0]
    if interaction_name is None:
        matches = [n for n in params.index if ('reader_view' in str(n).lower() and 'dyslex' in str(n).lower())]
        if matches:
            interaction_name = matches[0]

    if reader_name is None:
        raise ValueError("Could not find a parameter corresponding to 'reader_view' in model_output.params")
    if interaction_name is None:
        # It's possible the interaction was omitted for some reason; handle gracefully
        interaction_present = False
    else:
        interaction_present = True

    # Extract reader_view effect for non-dyslexic (this is the coef on reader_view)
    coef_reader = float(params.loc[reader_name])
    se_reader = float(np.sqrt(float(cov.loc[reader_name, reader_name])))
    df_resid = float(model_output.df_resid)
    t_reader = coef_reader / se_reader if se_reader != 0 else np.nan
    p_reader = 2 * (1 - stats.t.cdf(abs(t_reader), df_resid)) if not np.isnan(t_reader) else np.nan
    tcrit = stats.t.ppf(0.975, df_resid)
    ci_reader = (coef_reader - tcrit * se_reader, coef_reader + tcrit * se_reader)

    # Interaction term (reader_view:dyslexia_bin)
    if interaction_present:
        coef_inter = float(params.loc[interaction_name])
        se_inter = float(np.sqrt(float(cov.loc[interaction_name, interaction_name])))
        t_inter = coef_inter / se_inter if se_inter != 0 else np.nan
        p_inter = 2 * (1 - stats.t.cdf(abs(t_inter), df_resid)) if not np.isnan(t_inter) else np.nan
        ci_inter = (coef_inter - tcrit * se_inter, coef_inter + tcrit * se_inter)
    else:
        coef_inter = 0.0
        se_inter = 0.0
        t_inter = np.nan
        p_inter = np.nan
        ci_inter = (np.nan, np.nan)

    # Combined effect for dyslexic readers: reader_view + interaction
    # Build contrast vector c such that est = c'params
    c = np.zeros(len(params))
    idx = {name: i for i, name in enumerate(params.index)}
    c[idx[reader_name]] = 1.0
    if interaction_present:
        c[idx[interaction_name]] = 1.0

    est_comb = float(np.dot(c, params.values))
    se_comb = float(np.sqrt(np.dot(c, np.dot(cov.values, c))))
    t_comb = est_comb / se_comb if se_comb != 0 else np.nan
    p_comb = 2 * (1 - stats.t.cdf(abs(t_comb), df_resid)) if not np.isnan(t_comb) else np.nan
    ci_comb = (est_comb - tcrit * se_comb, est_comb + tcrit * se_comb)

    # Prepare result object
    result_object = {
        'nobs': int(model_output.nobs) if hasattr(model_output, 'nobs') else None,
        'df_resid': float(df_resid),
        'reader_view_non_dyslexic': {
            'param_name': reader_name,
            'coef_wpm': coef_reader,
            'se': se_reader,
            't': t_reader,
            'p_value': p_reader,
            '95ci': (ci_reader[0], ci_reader[1]),
            'interpretation': 'effect of Reader View on wpm for non-dyslexic readers'
        },
        'interaction_reader_view_x_dyslexia': {
            'param_name': interaction_name if interaction_present else None,
            'coef_wpm': coef_inter,
            'se': se_inter,
            't': t_inter,
            'p_value': p_inter,
            '95ci': (ci_inter[0], ci_inter[1]),
            'interpretation': 'additional effect of Reader View for dyslexic readers (difference vs non-dyslexic)'
        },
        'reader_view_for_dyslexic': {
            'coef_wpm': est_comb,
            'se': se_comb,
            't': t_comb,
            'p_value': p_comb,
            '95ci': (ci_comb[0], ci_comb[1]),
            'interpretation': 'total effect of Reader View on wpm for dyslexic readers (reader_view + interaction)'
        }
    }

    # Simple decision rule: Reader View "improves" for dyslexic if combined coef > 0 and p < 0.05
    signif = (not np.isnan(p_comb)) and (p_comb < 0.05)
    improves = (est_comb > 0) and signif
    if np.isnan(p_comb):
        conclusion = ("Could not compute a valid p-value for the combined effect; unable to conclude whether "
                      "Reader View improves reading speed for dyslexic individuals from this model output.")
    else:
        if improves:
            conclusion = (f"Yes — the estimated effect of Reader View for dyslexic readers is +{est_comb:.2f} wpm "
                          f"(95% CI [{ci_comb[0]:.2f}, {ci_comb[1]:.2f}]), p = {p_comb:.3g}, "
                          "indicating a statistically significant improvement.")
        else:
            # Provide more nuance if interaction matters
            if (est_comb > 0) and (not signif):
                conclusion = (f"The estimated effect of Reader View for dyslexic readers is positive (+{est_comb:.2f} wpm) "
                              f"but not statistically significant (p = {p_comb:.3g}, 95% CI [{ci_comb[0]:.2f}, {ci_comb[1]:.2f}]).")
            elif (est_comb <= 0) and signif:
                conclusion = (f"No — the estimated effect of Reader View for dyslexic readers is non-positive "
                              f"({est_comb:.2f} wpm) and statistically significant (p = {p_comb:.3g}), indicating no improvement.")
            else:
                conclusion = (f"No statistically significant evidence that Reader View improves reading speed for dyslexic readers. "
                              f"Estimated effect = {est_comb:.2f} wpm, p = {p_comb:.3g}, 95% CI [{ci_comb[0]:.2f}, {ci_comb[1]:.2f}].")

    return {
        "object": result_object,
        "description": conclusion
    }