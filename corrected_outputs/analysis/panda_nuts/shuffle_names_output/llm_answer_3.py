def extract_final_answer(model_output):
    """
    Extract key statistics from the model output returned by the modeling function.
    Returns a dictionary with keys:
      - "object": a dict containing parameter table and selected conditional effects
      - "description": brief explanation of the returned values and their interpretation

    The function attempts to use the robust_result (HC3) if present, otherwise falls
    back to the ordinary ols_result or the model object supplied directly.
    """
    import numpy as np
    import pandas as pd

    # Helper to safely get the statsmodels results object
    def _get_result_obj(mo):
        # If caller passed the full dict the modeling function returned:
        if isinstance(mo, dict):
            # prefer robust_result if available
            for key in ('robust_result', 'ols_result', 'result', 'model'):
                if key in mo and mo[key] is not None:
                    return mo[key]
            # If none of the above, try to find any statsmodels-like object in values
            for v in mo.values():
                # crude check for statsmodels result object
                if hasattr(v, 'params') and hasattr(v, 'bse'):
                    return v
        # If a model result object was passed directly, return it
        if hasattr(mo, 'params') and hasattr(mo, 'bse'):
            return mo
        raise ValueError("Could not locate a statsmodels result object in model_output")

    res = _get_result_obj(model_output)

    # Ensure params is a pandas Series with an index
    raw_params = getattr(res, 'params', None)
    if isinstance(raw_params, pd.Series):
        params = raw_params.copy()
    elif raw_params is None:
        params = pd.Series(dtype=float)
    else:
        # raw_params might be a numpy array; try to get names
        names = None
        # statsmodels sometimes has param_names or model.exog_names
        names = getattr(res, 'param_names', None)
        if names is None:
            model_obj = getattr(res, 'model', None)
            if model_obj is not None:
                names = getattr(model_obj, 'exog_names', None)
        if names is None:
            # fallback generic names
            try:
                length = len(raw_params)
            except Exception:
                length = 0
            names = [f'param_{i}' for i in range(length)]
        params = pd.Series(data=list(raw_params), index=list(names))

    # p-values: may be Series or ndarray or missing
    pvals_raw = getattr(res, 'pvalues', None)
    if isinstance(pvals_raw, pd.Series):
        pvalues = pvals_raw.reindex(params.index).copy()
    elif pvals_raw is None:
        pvalues = pd.Series(index=params.index, data=[np.nan] * len(params))
    else:
        # ndarray or other array-like
        try:
            pvalues = pd.Series(data=list(pvals_raw), index=list(getattr(pvals_raw, 'index', params.index)))
            # reindex to params.index if lengths differ
            if not pvalues.index.equals(params.index):
                # if pvals had no index, or different, align by position
                pvalues = pd.Series(data=list(pvals_raw), index=params.index)
        except Exception:
            pvalues = pd.Series(index=params.index, data=[np.nan] * len(params))

    # Confidence intervals: try res.conf_int -> may return ndarray or DataFrame
    try:
        ci_raw = res.conf_int(alpha=0.05)
        if isinstance(ci_raw, pd.DataFrame):
            ci = ci_raw.copy()
            # Ensure names
            if ci.shape[1] >= 2:
                ci = ci.iloc[:, :2]
                ci.columns = ['CI_lower', 'CI_upper']
            else:
                # unexpected shape
                ci = pd.DataFrame(index=params.index, data={'CI_lower': [np.nan]*len(params),
                                                           'CI_upper': [np.nan]*len(params)})
        else:
            # assume numpy array with shape (n_params, 2)
            ci = pd.DataFrame(ci_raw, index=params.index, columns=['CI_lower', 'CI_upper'])
    except Exception:
        # fallback: use NaNs
        ci = pd.DataFrame(index=params.index, data={'CI_lower': [np.nan]*len(params),
                                                   'CI_upper': [np.nan]*len(params)})

    coef_table = pd.DataFrame({
        'estimate': params,
        'pvalue': pvalues,
        'CI_lower': ci['CI_lower'],
        'CI_upper': ci['CI_upper']
    }, index=params.index)

    # Prepare for linear-combination tests using t_test
    param_names = list(params.index)

    def _t_test_contrast(con_dict):
        """
        con_dict: mapping param_name -> coefficient in linear combination.
        Returns dict with estimate, se, tvalue, pvalue, CI_lower, CI_upper.
        """
        # Build contrast vector in the order of param_names
        contrast = np.zeros(len(param_names))
        for k, v in con_dict.items():
            if k in param_names:
                contrast[param_names.index(k)] = v
            else:
                # If requested parameter not present, raise informative error
                raise KeyError(f"Parameter '{k}' not found in model parameters.")
        # Use t_test (uses current covariance, which will be robust if res is robust_result)
        tt = res.t_test(contrast)
        # tt.effect etc. may be arrays; take scalar
        def _to_scalar(x):
            try:
                return float(np.squeeze(np.array(x)))
            except Exception:
                return np.nan
        estimate = _to_scalar(tt.effect)
        se = _to_scalar(getattr(tt, 'sd', None))
        tval = _to_scalar(getattr(tt, 'tvalue', None))
        pval = _to_scalar(getattr(tt, 'pvalue', None))
        # Confidence interval from the test result (tt.conf_int() gives 2D array)
        try:
            conf = tt.conf_int(alpha=0.05)
            # conf expected shape (1,2) or (k,2)
            ci_low, ci_high = float(conf[0, 0]), float(conf[0, 1])
        except Exception:
            ci_low, ci_high = np.nan, np.nan
        return {
            'estimate': estimate,
            'se': se,
            'tvalue': tval,
            'pvalue': pval,
            'CI_lower': ci_low,
            'CI_upper': ci_high
        }

    # Compute mean age (needed to interpret ReceivedHelp effect at a typical age).
    # Try to get original data frame if possible
    mean_age = None
    try:
        # statsmodels stores the original data frame at res.model.data.frame in many cases
        data_obj = getattr(getattr(res, 'model', None), 'data', None)
        if data_obj is not None and hasattr(data_obj, 'frame') and getattr(data_obj, 'frame') is not None:
            df_frame = data_obj.frame
            if isinstance(df_frame, pd.DataFrame) and 'AgeYears' in df_frame:
                mean_age = float(df_frame['AgeYears'].mean())
        elif data_obj is not None and hasattr(data_obj, 'orig_exog'):
            # fallback: try to access orig_exog
            exog_df = data_obj.orig_exog
            if exog_df is not None and 'AgeYears' in exog_df:
                mean_age = float(exog_df['AgeYears'].mean())
    except Exception:
        mean_age = None

    # If we couldn't find the dataframe, compute mean from the design matrix if AgeYears present
    if mean_age is None:
        try:
            model_obj = getattr(res, 'model', None)
            exog = getattr(model_obj, 'exog', None)
            exog_names = getattr(model_obj, 'exog_names', None)
            if exog is not None and exog_names is not None and 'AgeYears' in exog_names:
                idx = exog_names.index('AgeYears')
                mean_age = float(np.asarray(exog)[:, idx].mean())
        except Exception:
            mean_age = None

    # If still None, set to 0 and note caveat later
    if mean_age is None:
        mean_age = 0.0
        mean_age_note = "mean age not available from model object; used 0.0 as placeholder"
    else:
        mean_age_note = f"mean age computed from data = {mean_age:.3f}"

    # Define the terms we expect (names depend on how statsmodels encoded them)
    # Common names given the formula: 'AgeYears', 'ReceivedHelp', 'AgeYears:ReceivedHelp',
    # 'C(Sex)[T.M]', 'C(Sex)[T.M]:ReceivedHelp'
    # We'll compute conditional effects if those parameters exist.
    conditional_effects = {}

    # Age effect when ReceivedHelp = 0 : just coefficient of AgeYears
    if 'AgeYears' in param_names:
        try:
            conditional_effects['Age_when_no_help'] = _t_test_contrast({'AgeYears': 1.0})
        except KeyError as e:
            conditional_effects['Age_when_no_help'] = {'error': str(e)}
    else:
        conditional_effects['Age_when_no_help'] = {'error': "Parameter 'AgeYears' not in model."}

    # Age effect when ReceivedHelp = 1 : AgeYears + AgeYears:ReceivedHelp
    inter_age_name = 'AgeYears:ReceivedHelp'
    if 'AgeYears' in param_names and inter_age_name in param_names:
        try:
            conditional_effects['Age_when_help'] = _t_test_contrast({'AgeYears': 1.0, inter_age_name: 1.0})
        except KeyError as e:
            conditional_effects['Age_when_help'] = {'error': str(e)}
    else:
        # If the interaction term is absent, the effect is same as no_help
        if 'AgeYears' in param_names:
            conditional_effects['Age_when_help'] = conditional_effects['Age_when_no_help']
        else:
            conditional_effects['Age_when_help'] = {'error': "Required parameters for Age_when_help missing."}

    # Sex effect (Male vs Female) when ReceivedHelp = 0
    sex_param = None
    sex_inter_param = None
    # find typical encoding names for male dummy
    for name in param_names:
        if name.startswith('C(Sex)') and ':ReceivedHelp' not in name:
            # pick the first C(Sex) parameter (usually 'C(Sex)[T.M]')
            sex_param = name
            break
    for name in param_names:
        if name.startswith('C(Sex)') and ':ReceivedHelp' in name:
            sex_inter_param = name
            break

    if sex_param is not None:
        try:
            conditional_effects['Male_vs_Female_no_help'] = _t_test_contrast({sex_param: 1.0})
        except KeyError as e:
            conditional_effects['Male_vs_Female_no_help'] = {'error': str(e)}
    else:
        conditional_effects['Male_vs_Female_no_help'] = {'error': "Sex parameter (C(Sex)[T.M]) not found in model."}

    # Sex effect when ReceivedHelp = 1: C(Sex)[T.M] + C(Sex)[T.M]:ReceivedHelp (if interaction exists)
    if sex_param is not None and sex_inter_param is not None:
        try:
            conditional_effects['Male_vs_Female_with_help'] = _t_test_contrast({sex_param: 1.0, sex_inter_param: 1.0})
        except KeyError as e:
            conditional_effects['Male_vs_Female_with_help'] = {'error': str(e)}
    elif sex_param is not None:
        conditional_effects['Male_vs_Female_with_help'] = conditional_effects['Male_vs_Female_no_help']
    else:
        conditional_effects['Male_vs_Female_with_help'] = {'error': "Required sex parameters not found."}

    # Effect of ReceivedHelp at mean age for Female (reference sex)
    # ReceivedHelp effect is: ReceivedHelp + AgeYears:ReceivedHelp * Age + C(Sex)[T.M]:ReceivedHelp * (sex==Male)
    rec_name = 'ReceivedHelp'
    rec_age_inter = inter_age_name  # AgeYears:ReceivedHelp
    rec_sex_inter = None
    if sex_inter_param is not None:
        rec_sex_inter = sex_inter_param  # e.g. 'C(Sex)[T.M]:ReceivedHelp'

    # For female (reference), sex interaction term is 0
    if rec_name in param_names:
        con_female = {rec_name: 1.0}
        if rec_age_inter in param_names:
            con_female[rec_age_inter] = mean_age
        # male interaction term absent or zero for female
        try:
            conditional_effects[f'ReceivedHelp_effect_female_at_mean_age ({mean_age_note})'] = _t_test_contrast(con_female)
        except KeyError as e:
            conditional_effects[f'ReceivedHelp_effect_female_at_mean_age ({mean_age_note})'] = {'error': str(e)}
    else:
        conditional_effects[f'ReceivedHelp_effect_female_at_mean_age ({mean_age_note})'] = {'error': "Parameter 'ReceivedHelp' not found."}

    # For male: include sex-specific ReceivedHelp interaction if present
    if rec_name in param_names:
        con_male = {rec_name: 1.0}
        if rec_age_inter in param_names:
            con_male[rec_age_inter] = mean_age
        if rec_sex_inter is not None:
            con_male[rec_sex_inter] = 1.0
        try:
            conditional_effects[f'ReceivedHelp_effect_male_at_mean_age ({mean_age_note})'] = _t_test_contrast(con_male)
        except KeyError as e:
            conditional_effects[f'ReceivedHelp_effect_male_at_mean_age ({mean_age_note})'] = {'error': str(e)}
    else:
        conditional_effects[f'ReceivedHelp_effect_male_at_mean_age ({mean_age_note})'] = {'error': "Parameter 'ReceivedHelp' not found."}

    # Package output object
    if isinstance(model_output, dict):
        n_obs_val = model_output.get('n_obs', getattr(res, 'nobs', np.nan))
        formula_val = model_output.get('formula') if 'formula' in model_output else getattr(getattr(res, 'model', None), 'formula', None)
    else:
        n_obs_val = getattr(res, 'nobs', np.nan)
        formula_val = getattr(getattr(res, 'model', None), 'formula', None)

    try:
        n_obs = int(n_obs_val)
    except Exception:
        n_obs = n_obs_val

    out_object = {
        'n_obs': n_obs,
        'formula': formula_val,
        'coefficient_table': coef_table,
        'conditional_effects': conditional_effects
    }

    # Short description
    desc_lines = [
        "Extracted coefficient table (estimates, p-values, 95% CIs) and selected conditional effects.",
        "Conditional effects computed using linear combinations (t_test) from the fitted model",
        "- Age effect reported for sessions without help and with help (interaction term included).",
        "- Sex effect reported as Male vs Female without help and with help (if interaction present).",
        f"- ReceivedHelp effect evaluated at mean age of the data (female and male). {mean_age_note}.",
        "All inference uses the returned model object; if a 'robust_result' (HC3) was provided it is used."
    ]
    description = " ".join(desc_lines)

    return {
        "object": out_object,
        "description": description
    }