def extract_final_answer(model_output):
    """
    Extracts information about the effect of 'Female' on mortgage approval from the model_output.
    Returns a dictionary with keys:
      - "object": a dict containing extracted statistics (or None if not available)
      - "description": plain-language explanation of what the extracted object means.
    The function is defensive: it tries to read stats from the fitted_model (if present),
    from the prepared fields in model_output (params, odds_ratios, conf_int_odds), and,
    if the model failed due to complete separation, it will attempt to compute simple
    group-level approval rates by gender from the model's data frame (if available).
    """
    import numpy as np
    import pandas as pd

    result_obj = None
    description = ""

    # Helper to convert numpy/pandas scalars to native python types for JSON-friendliness
    def to_py(x):
        try:
            if isinstance(x, (np.generic,)):
                return x.item()
            if isinstance(x, pd.DataFrame):
                # Convert DataFrame to nested dict for JSON-friendly output
                return x.to_dict()
            if isinstance(x, pd.Series):
                if getattr(x, "shape", ()) == ():
                    return x.item()
                return x.tolist()
            if isinstance(x, pd.Index):
                return x.tolist()
            return x
        except Exception:
            return x

    def contains_key(container, key):
        """Safely check if container (Series/DataFrame/dict/Index) contains key/index."""
        try:
            if container is None:
                return False
            if isinstance(container, dict):
                return key in container
            if isinstance(container, pd.Series):
                return key in container.index
            if isinstance(container, pd.DataFrame):
                return key in container.index
            if isinstance(container, (pd.Index, list, tuple, np.ndarray)):
                return key in container
            return False
        except Exception:
            return False

    # Unpack possible sources (avoid using "or" which can trigger DataFrame truth checks)
    fitted = None
    params = None
    odds_ratios = None
    conf_odds = None
    summary_text = ""

    if isinstance(model_output, dict):
        if 'fitted_model' in model_output:
            fitted = model_output.get('fitted_model')
        elif 'model' in model_output:
            fitted = model_output.get('model')
        else:
            fitted = None

        params = model_output.get('params') if 'params' in model_output else None
        odds_ratios = model_output.get('odds_ratios') if 'odds_ratios' in model_output else None
        if 'conf_int_odds' in model_output:
            conf_odds = model_output.get('conf_int_odds')
        elif 'conf_int' in model_output:
            conf_odds = model_output.get('conf_int')
        summary_text = model_output.get('summary_text', "") if 'summary_text' in model_output else ""
    else:
        # model_output might already be a fitted model
        fitted = model_output
        params = None
        odds_ratios = None
        conf_odds = None
        summary_text = ""

    # If fitted model is available, try to get summary_text from it if not provided
    if not summary_text and hasattr(fitted, 'summary'):
        try:
            summary_text = fitted.summary().as_text()
        except Exception:
            summary_text = ""

    # Try to obtain params/pvalues/conf directly from fitted model if not provided
    if fitted is not None:
        try:
            if params is None and hasattr(fitted, 'params'):
                params = fitted.params
            if odds_ratios is None and params is not None:
                try:
                    odds_ratios = np.exp(params)
                except Exception:
                    odds_ratios = None
            if conf_odds is None:
                # try to obtain the coefficient CI then exponentiate
                if hasattr(fitted, 'conf_int'):
                    try:
                        ci = fitted.conf_int()
                        try:
                            # conf_int may be on logit scale; exponentiate to get odds ratio CI
                            conf_odds = np.exp(ci)
                        except Exception:
                            conf_odds = ci
                    except Exception:
                        conf_odds = None
        except Exception:
            pass

    # Case 1: coefficients for 'Female' are present
    try:
        if contains_key(params, 'Female'):
            # Retrieve coefficient (handle dict/Series)
            if isinstance(params, dict):
                coef_raw = params.get('Female')
            else:
                try:
                    coef_raw = params.loc['Female']
                except Exception:
                    # try positional access or key access
                    coef_raw = params['Female'] if hasattr(params, '__getitem__') and 'Female' in getattr(params, 'index', []) else None
            coef = to_py(coef_raw)

            # try p-value from fitted model or params
            pval = None
            try:
                if hasattr(fitted, 'pvalues') and contains_key(fitted.pvalues, 'Female'):
                    pval = to_py(fitted.pvalues.loc['Female'])
                elif hasattr(params, 'pvalues') and contains_key(params.pvalues, 'Female'):
                    pval = to_py(params.pvalues.loc['Female'])
            except Exception:
                pval = None

            oratio = None
            if contains_key(odds_ratios, 'Female'):
                if isinstance(odds_ratios, dict):
                    oratio = to_py(odds_ratios.get('Female'))
                else:
                    try:
                        oratio = to_py(odds_ratios.loc['Female'])
                    except Exception:
                        try:
                            oratio = to_py(odds_ratios['Female'])
                        except Exception:
                            oratio = None

            ci_or = None
            try:
                if contains_key(conf_odds, 'Female'):
                    # conf_odds might be a DataFrame with columns ['2.5%', '97.5%'] or [0,1]
                    if isinstance(conf_odds, pd.DataFrame) or isinstance(conf_odds, pd.Series):
                        row = conf_odds.loc['Female']
                    elif isinstance(conf_odds, dict):
                        row = conf_odds.get('Female')
                    else:
                        row = None
                    if row is not None:
                        if hasattr(row, 'iloc'):
                            # Series-like
                            try:
                                ci_or = [to_py(row.iloc[0]), to_py(row.iloc[1])]
                            except Exception:
                                # maybe named columns
                                try:
                                    vals = list(row)
                                    ci_or = [to_py(vals[0]), to_py(vals[1])]
                                except Exception:
                                    ci_or = None
                        else:
                            # row may be a list/tuple
                            try:
                                ci_or = [to_py(row[0]), to_py(row[1])]
                            except Exception:
                                ci_or = None
            except Exception:
                ci_or = None

            result_obj = {
                'variable': 'Female',
                'coefficient_logit': coef,
                'p_value': pval,
                'odds_ratio': oratio,
                'odds_ratio_confidence_interval': ci_or
            }
            description = (
                "Logistic regression estimates for the 'Female' indicator. "
                "coefficient_logit is the log-odds effect (negative means lower odds). "
                "odds_ratio is exp(coef); confidence interval (if available) is on the odds ratio scale. "
                "p_value tests null hypothesis coef=0."
            )
            return {'object': result_obj, 'description': description}

    except Exception:
        pass

    # Case 2: No Female param present -> likely model failed / complete separation. Detect and handle.
    sep_detected = False
    if isinstance(summary_text, str) and 'complete separation' in summary_text.lower():
        sep_detected = True
    # also check fitted summary text content
    if not sep_detected and hasattr(fitted, 'mle_retvals'):
        try:
            mle_info = str(fitted.mle_retvals)
            if 'complete' in mle_info.lower() or 'separation' in mle_info.lower():
                sep_detected = True
        except Exception:
            pass

    if sep_detected:
        # Attempt to compute simple gender-specific approval rates from the original dataframe if present
        group_rates = None
        contingency = None
        obs_or = None
        obs_or_corrected = None
        df = None
        try:
            # Statsmodels stores the original data frame in fitted.model.data.frame for formula-based models
            if fitted is not None and hasattr(fitted, 'model') and hasattr(fitted.model, 'data'):
                df = getattr(fitted.model.data, 'frame', None) or getattr(fitted.model.data, 'orig_endog', None)
                if df is None:
                    # sometimes it's named .frame or .orig_endog; try again
                    df = getattr(fitted.model.data, 'frame', None)
            # Also try model_output if it contains the original data
            if df is None and isinstance(model_output, dict):
                for key in ['data', 'df', 'original_df']:
                    if key in model_output:
                        df = model_output[key]
                        break
            # If we have a frame and it contains 'Female' and 'Approved', compute rates
            if isinstance(df, pd.DataFrame) and set(['Female', 'Approved']).issubset(set(df.columns)):
                tab = pd.crosstab(df['Female'], df['Approved'])
                contingency = tab.to_dict()  # convert to dict for returning
                # compute rates (mean of Approved by Female)
                grp = df.groupby('Female')['Approved'].agg(['mean', 'count'])
                group_rates = grp.rename(columns={'mean': 'approval_rate', 'count': 'n'}).to_dict(orient='index')
                # Compute observed odds ratio from 2x2 table if possible
                #  rows: Female=0,1 ; cols: Approved=0,1
                a = tab.loc[1, 1] if (1 in tab.index and 1 in tab.columns) else 0
                b = tab.loc[1, 0] if (1 in tab.index and 0 in tab.columns) else 0
                c = tab.loc[0, 1] if (0 in tab.index and 1 in tab.columns) else 0
                d = tab.loc[0, 0] if (0 in tab.index and 0 in tab.columns) else 0
                # odds ratio = (a/b) / (c/d) = a*d / (b*c)
                try:
                    if (b * c) != 0:
                        obs_or = float((a * d) / (b * c))
                    else:
                        obs_or = None
                    # Haldane-Anscombe correction (add 0.5 to all cells) to handle zeros
                    a_c = a + 0.5
                    b_c = b + 0.5
                    c_c = c + 0.5
                    d_c = d + 0.5
                    obs_or_corrected = float((a_c * d_c) / (b_c * c_c))
                except Exception:
                    obs_or = None
                    obs_or_corrected = None
            else:
                df = None  # no usable data
        except Exception:
            df = None

        if group_rates is not None:
            result_obj = {
                'separation_detected': True,
                'note': "Logistic regression failed due to complete separation; raw group-level summaries are provided instead.",
                'group_rates_by_Female': group_rates,
                'contingency_table': contingency,
                'observed_odds_ratio': obs_or,
                'observed_odds_ratio_with_0.5_correction': obs_or_corrected
            }
            description = (
                "The fitted logistic model reports complete separation (parameters not identified). "
                "Therefore the model's coefficient for 'Female' is not available. "
                "Below are the empirical approval rates and contingency counts by Female (1=female, 0=male). "
                "Observed odds ratio may be undefined if any cell is zero; a corrected odds ratio with 0.5 added "
                "to each cell (Haldane-Anscombe) is also provided."
            )
            return {'object': result_obj, 'description': description}
        else:
            # separation detected but no original data available to compute simple summaries
            result_obj = {
                'separation_detected': True,
                'note': "Logistic regression reports complete separation; no coefficient for 'Female' is available, and original data was not found in the model output to compute group summaries."
            }
            description = (
                "The logistic model could not identify parameters due to complete separation (perfect prediction). "
                "This means we cannot trust or obtain a finite coefficient/odds ratio for 'Female' from this fit. "
                "Suggested next steps: (1) compute raw approval rates by gender, (2) use penalized / Firth logistic regression or exact logistic regression, "
                "or (3) collapse predictors to remove perfect prediction."
            )
            return {'object': result_obj, 'description': description}

    # Case 3: generic fallback: try to report whatever we can for 'Female' from model_output dict fields
    # Try to use raw dictionaries (params/odds_ratios/conf_int_odds)
    try:
        if isinstance(params, dict) and 'Female' in params:
            coef = to_py(params['Female'])
            oratio = None
            if isinstance(odds_ratios, dict) and 'Female' in odds_ratios:
                oratio = to_py(odds_ratios['Female'])
            result_obj = {'variable': 'Female', 'coefficient_logit': coef, 'odds_ratio': oratio}
            description = "Extracted Female coefficient and odds ratio from available model_output fields."
            return {'object': result_obj, 'description': description}
        # If params is a Series
        if isinstance(params, pd.Series) and contains_key(params, 'Female'):
            coef = to_py(params.loc['Female'])
            oratio = None
            if isinstance(odds_ratios, pd.Series) and contains_key(odds_ratios, 'Female'):
                oratio = to_py(odds_ratios.loc['Female'])
            result_obj = {'variable': 'Female', 'coefficient_logit': coef, 'odds_ratio': oratio}
            description = "Extracted Female coefficient and odds ratio from available model_output fields."
            return {'object': result_obj, 'description': description}
    except Exception:
        pass

    # If we reach here, we couldn't extract anything useful
    result_obj = None
    description = (
        "Could not extract an estimate for the effect of 'Female' from the provided model output. "
        "The model output appears to have failed (complete separation or no Female coefficient present). "
        "If possible, provide the original DataFrame or a model fit that converged; otherwise consider using "
        "penalized logistic regression (Firth) or computing simple group-level approval rates by gender."
    )
    return {'object': result_obj, 'description': description}