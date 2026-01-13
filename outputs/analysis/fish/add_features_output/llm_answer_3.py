def extract_final_answer(model_output):
    """
    Extracts coefficient estimates, p-values, confidence intervals, and incidence-rate-ratios (IRRs)
    from the provided model_output dictionary returned by the modeling function.

    Returns a dictionary with:
      - "object": a summary dict containing:
          - model_used: which fitted model was used for inference (NB with offset preferred, else Poisson)
          - observed_mean_rate_per_hour: empirical mean fish caught per hour from the data
          - baseline_rate_per_hour: estimated fish-per-hour for the reference group (all predictors = 0)
          - predictor_summary: per-predictor dict with coef, p-value, 95% CI, IRR and IRR_CI
      - "description": brief explanation of what the object means and how to interpret IRRs.
    """
    import numpy as np

    # Helper to safely get attributes
    def safe_attr(obj, name, default=None):
        try:
            return getattr(obj, name)
        except Exception:
            return default

    poisson_res = model_output.get('poisson_result')
    nb_res = model_output.get('nb_result')

    # Prefer a Negative Binomial result that includes the offset (i.e. was fitted as GLM with offset).
    # If nb_res lacks an offset (discrete NB fallback), prefer the Poisson GLM for baseline rate-per-hour calculation
    def has_offset(result):
        if result is None:
            return False
        model = safe_attr(result, 'model')
        if model is None:
            return False
        # model.offset present for GLM family-based fits (it will be an array)
        return getattr(model, 'offset', None) is not None

    selected_res = None
    selected_name = None
    if nb_res is not None and has_offset(nb_res):
        selected_res = nb_res
        selected_name = 'NegativeBinomial (GLM with offset)'
    elif nb_res is not None and poisson_res is None:
        # If only discrete NB available, we'll still use it for coefficients/IRRs,
        # but use poisson data (if available) to compute observed mean rate.
        selected_res = nb_res
        selected_name = 'NegativeBinomial (discrete, no offset)'
    else:
        selected_res = poisson_res
        selected_name = 'Poisson (GLM with offset)'

    if selected_res is None:
        raise ValueError("No fitted model result available in model_output to extract statistics from.")

    # Try to obtain the original dataframe from the Poisson model (it was always fit there).
    df = None
    if poisson_res is not None:
        df = safe_attr(poisson_res.model, 'data')
        # statsmodels may store data frame under model.data.frame or model.data.orig_endog, try common places
        if df is None:
            df = safe_attr(poisson_res.model, 'dataframe')
        if df is None:
            # try model.data.frame
            md = safe_attr(poisson_res.model, 'data')
            if md is not None:
                df = safe_attr(md, 'frame', None) or safe_attr(md, 'dataframe', None)
        # As a fallback, try model.data.frame attribute directly
        if df is None:
            df = getattr(poisson_res.model, 'data', None)
        # If still none, try model.model.data.frame
        if df is None:
            df = getattr(poisson_res, 'data', None)
    # If still not found, try selected_res.model.data or data.frame
    if df is None:
        df = safe_attr(selected_res.model, 'data')
        if df is None:
            df = safe_attr(selected_res.model, 'dataframe')

    # Compute observed mean fish-per-hour directly from the data if possible
    observed_mean_rate = None
    try:
        if df is not None:
            # df may be a patsy/StatsModels DataWrapper; attempt to get a DataFrame or mapping
            if hasattr(df, 'frame'):
                data_df = df.frame
            else:
                data_df = df if hasattr(df, 'columns') else None
            if data_df is None:
                # try if model has .model.data.frame
                data_df = safe_attr(poisson_res.model, 'data')
            # At this point if still not a DataFrame, attempt direct indexing
            if data_df is not None and 'fish_caught' in data_df and 'log_hours' in data_df:
                hours = np.exp(np.asarray(data_df['log_hours']).astype(float))
                fish = np.asarray(data_df['fish_caught']).astype(float)
                total_hours = np.sum(hours)
                total_fish = np.sum(fish)
                if total_hours > 0:
                    observed_mean_rate = float(total_fish / total_hours)
    except Exception:
        observed_mean_rate = None

    # Extract coefficients, p-values, and confidence intervals from the selected model
    res = selected_res
    params = None
    pvalues = None
    conf_int = None
    try:
        params = res.params
    except Exception:
        # some discrete results store params differently
        params = safe_attr(res, 'params', None)
    try:
        pvalues = res.pvalues
    except Exception:
        pvalues = safe_attr(res, 'pvalues', None)
    try:
        conf_int = res.conf_int()
    except Exception:
        # try building conf_int from params and bse if available
        bse = safe_attr(res, 'bse', None)
        if params is not None and bse is not None:
            try:
                ci_lower = params - 1.96 * bse
                ci_upper = params + 1.96 * bse
                # Build a 2-column array-like aligned with params index
                conf_int = np.vstack([ci_lower, ci_upper]).T
            except Exception:
                conf_int = None

    # Normalize params / conf_int into pandas Series / DataFrame-like structures if possible
    import pandas as pd
    if params is not None and not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            # fallback: convert to Series with integer index
            params = pd.Series(np.asarray(params))

    if conf_int is not None and not isinstance(conf_int, pd.DataFrame):
        try:
            # If conf_int came as numpy array, align with params.index
            conf_int = pd.DataFrame(conf_int, index=params.index, columns=['2.5%', '97.5%'])
        except Exception:
            conf_int = None

    if pvalues is not None and not isinstance(pvalues, pd.Series):
        try:
            pvalues = pd.Series(pvalues, index=params.index)
        except Exception:
            pvalues = pd.Series(np.asarray(pvalues))

    # Focus on the predictors of interest
    predictors = ['livebait', 'camper', 'persons', 'child', 'group_size']
    predictor_summary = {}
    for var in predictors:
        if params is None or var not in params.index:
            predictor_summary[var] = {
                'coef': None,
                'p_value': None,
                'ci_2.5%': None,
                'ci_97.5%': None,
                'IRR': None,
                'IRR_ci_2.5%': None,
                'IRR_ci_97.5%': None,
                'note': 'variable not present in the selected model coefficients'
            }
            continue
        coef = float(params.loc[var])
        pval = float(pvalues.loc[var]) if (pvalues is not None and var in pvalues.index) else None
        if conf_int is not None and var in conf_int.index:
            ci_low = float(conf_int.loc[var, conf_int.columns[0]])
            ci_high = float(conf_int.loc[var, conf_int.columns[-1]])
        else:
            ci_low = None
            ci_high = None
        irr = float(np.exp(coef))
        irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None

        predictor_summary[var] = {
            'coef': coef,
            'p_value': pval,
            'ci_2.5%': ci_low,
            'ci_97.5%': ci_high,
            'IRR': irr,
            'IRR_ci_2.5%': irr_ci_low,
            'IRR_ci_97.5%': irr_ci_high,
            'interpretation': (
                "IRR > 1 indicates a multiplicative increase in expected fish count/rate per unit increase; "
                "IRR < 1 indicates a multiplicative decrease."
            )
        }

    # Compute baseline (reference) fish-per-hour rate estimate if possible.
    # For GLM fits with centered offset: baseline_rate_per_hour = exp(intercept - mean(log_hours))
    baseline_rate_per_hour = None
    try:
        # Identify intercept name in params
        intercept_name = None
        possible_names = ['Intercept', 'intercept', 'const', 'Const']
        for nm in possible_names:
            if params is not None and nm in params.index:
                intercept_name = nm
                break
        if intercept_name is None and params is not None:
            # fallback: assume first entry is intercept (risky but last resort)
            intercept_name = params.index[0]

        intercept = float(params.loc[intercept_name]) if intercept_name is not None else None

        # compute mean log_hours from data if available
        mean_log_hours = None
        if df is not None:
            try:
                # df may be a DataFrame-like object
                if hasattr(df, '__getitem__') and 'log_hours' in df:
                    mean_log_hours = float(np.mean(np.asarray(df['log_hours']).astype(float)))
                else:
                    # try deeper
                    frame = getattr(df, 'frame', None) or getattr(df, 'dataframe', None)
                    if frame is not None and 'log_hours' in frame:
                        mean_log_hours = float(np.mean(np.asarray(frame['log_hours']).astype(float)))
            except Exception:
                mean_log_hours = None

        # If the selected model has an offset (GLM), use intercept - mean_log_hours.
        if has_offset(selected_res) and intercept is not None and mean_log_hours is not None:
            baseline_rate_per_hour = float(np.exp(intercept - mean_log_hours))
        else:
            # If no offset available, we cannot reliably convert intercept to per-hour rate.
            # Instead, provide an estimated baseline by multiplying observed mean rate by intercept-based multiplier:
            # Use IRR for a "baseline" (all other covariates 0) as exp(intercept) relative to exp(0)=1,
            # but warn this is approximate and not strictly per-hour if offset missing.
            if intercept is not None:
                baseline_rate_per_hour = float(np.exp(intercept)) if observed_mean_rate is None else float(observed_mean_rate * np.exp(intercept))
    except Exception:
        baseline_rate_per_hour = None

    result_obj = {
        'model_used': selected_name,
        'observed_mean_rate_per_hour': observed_mean_rate,
        'baseline_rate_per_hour_estimate': baseline_rate_per_hour,
        'predictor_summary': predictor_summary,
        'overdispersion_statistic': float(model_output.get('overdispersion')) if model_output.get('overdispersion') is not None else None
    }

    description = (
        "The returned 'object' contains per-predictor coefficients, p-values, 95% CIs, and incidence-rate-ratios (IRRs). "
        "IRR = exp(coef) is the multiplicative change in expected fish counts (and thus fish-per-hour rate if the model included an exposure offset) "
        "for a one-unit increase in the predictor, holding other variables constant. "
        "Also included: the empirical observed mean fish-per-hour computed from the data and a model-based baseline rate-per-hour estimate "
        "(for the reference group with predictors = 0)."
    )

    return {'object': result_obj, 'description': description}