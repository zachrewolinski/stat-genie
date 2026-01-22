def extract_final_answer(model_output):
    """
    Extracts the effect of 'HasChildren' from the provided model_output dictionary.
    Returns a dict with keys:
      - "object": a dict containing numeric results and metadata (or error messages)
      - "description": a short English interpretation of what the numbers mean
    The function handles the case where the logistic model failed (as in the provided
    model_output) and extracts OLS results (coefficient, p-value, 95% CI, nobs)
    from the positive-sample OLS RegressionResultsWrapper if present.
    """
    import pandas as pd

    out_object = {}
    messages = []

    # 1) Logistic model: check if present or an error was recorded
    if 'logit_model' in model_output and model_output['logit_model'] is not None:
        # If a fitted logit model object is present, attempt to extract stats similarly.
        logit = model_output['logit_model']
        try:
            params = getattr(logit, 'params', None)
            if params is not None:
                names = list(params.index)
                # find parameter name that corresponds to HasChildren
                target = next((n for n in names if 'HasChildren' in n), None)
                if target is not None:
                    coef = float(params[target])
                    pval = float(logit.pvalues[target])
                    # conf_int may be method on result; build DataFrame for indexing
                    ci_arr = logit.conf_int()
                    ci_df = pd.DataFrame(ci_arr, index=names, columns=['ci_lower', 'ci_upper'])
                    ci_lower, ci_upper = float(ci_df.loc[target, 'ci_lower']), float(ci_df.loc[target, 'ci_upper'])
                    out_object['Logit'] = {
                        'param_name': target,
                        'coef': coef,
                        'pvalue': pval,
                        'ci_95': [ci_lower, ci_upper],
                        'interpretation': ('coef is the log-odds change in probability of reporting any affair '
                                           'for HasChildren=1 vs 0, holding controls constant.')
                    }
                else:
                    out_object['Logit'] = {'error': "No parameter name containing 'HasChildren' found in logit params."}
            else:
                out_object['Logit'] = {'error': 'Logit model present but has no params attribute.'}
        except Exception as e:
            out_object['Logit'] = {'error': f'Error extracting from logit model: {e}'}
    else:
        # Logit missing or failed: capture error message if provided
        logit_err = model_output.get('logit_error')
        if logit_err:
            out_object['Logit'] = {'error': f'Logistic model not available. Error: {logit_err}'}
            messages.append('Logistic regression for AnyAffair was not estimated due to the error recorded in model_output.')
        else:
            out_object['Logit'] = {'error': 'Logistic model not provided in model_output.'}
            messages.append('No logistic model present in model_output.')

    # 2) OLS among those with AffairCount > 0
    ols_res = model_output.get('ols_positive_model')
    if ols_res is None:
        out_object['OLS_positive'] = {'error': 'No OLS positive-sample model in model_output.'}
        messages.append('OLS on positive-sample not available.')
    else:
        try:
            # statsmodels RegressionResultsWrapper: params, pvalues, conf_int(), nobs
            params = ols_res.params
            names = list(params.index)
            target = next((n for n in names if 'HasChildren' in n), None)
            if target is None:
                out_object['OLS_positive'] = {'error': "No parameter name containing 'HasChildren' found in OLS params."}
                messages.append("Could not find 'HasChildren' parameter in OLS model parameters.")
            else:
                coef = float(params[target])
                pval = float(ols_res.pvalues[target])
                # confidence intervals: conf_int returns an array; align with param names
                try:
                    ci_arr = ols_res.conf_int()
                    ci_df = pd.DataFrame(ci_arr, index=names, columns=['ci_lower', 'ci_upper'])
                    ci_lower = float(ci_df.loc[target, 'ci_lower'])
                    ci_upper = float(ci_df.loc[target, 'ci_upper'])
                except Exception:
                    # fallback: try conf_int with alpha kw
                    ci = ols_res.conf_int(alpha=0.05)
                    ci_df = pd.DataFrame(ci, index=names, columns=['ci_lower', 'ci_upper'])
                    ci_lower = float(ci_df.loc[target, 'ci_lower'])
                    ci_upper = float(ci_df.loc[target, 'ci_upper'])

                # number of observations used in OLS
                try:
                    n_obs = int(ols_res.nobs)
                except Exception:
                    # fallback: try attribute in model
                    try:
                        n_obs = int(len(ols_res.model.endog))
                    except Exception:
                        n_obs = None

                # direction and significance interpretation
                direction = 'decrease' if coef < 0 else 'increase'
                sig = 'statistically significant (p < 0.05)' if pval < 0.05 else 'not statistically significant (p >= 0.05)'
                interp = (f"Among respondents who reported any affair (positive-sample, n={n_obs}), "
                          f"the OLS coefficient on '{target}' = {coef:.4g} (95% CI [{ci_lower:.4g}, {ci_upper:.4g}], p = {pval:.4g}). "
                          f"This indicates that having children is associated with a {direction} of {abs(coef):.4g} units in the "
                          f"AffairCount measure, controlling for the listed covariates. The effect is {sig}.")

                out_object['OLS_positive'] = {
                    'param_name': target,
                    'coef': coef,
                    'pvalue': pval,
                    'ci_95': [ci_lower, ci_upper],
                    'n_obs': n_obs,
                    'interpretation_short': interp
                }
                messages.append('Extracted OLS positive-sample estimate for HasChildren.')
        except Exception as e:
            out_object['OLS_positive'] = {'error': f'Error extracting from OLS model: {e}'}
            messages.append(f'Error while extracting OLS results: {e}')

    # Build a concise description summarizing results and limitations
    summary_lines = []
    # OLS summary line
    ols_info = out_object.get('OLS_positive')
    if ols_info and 'error' not in ols_info:
        summary_lines.append(ols_info['interpretation_short'])
    else:
        summary_lines.append('No usable OLS positive-sample estimate for HasChildren available.')

    # Logit summary line
    logit_info = out_object.get('Logit')
    if logit_info and 'error' not in logit_info:
        li = logit_info
        summary_lines.append(
            f"Logistic model: parameter '{li['param_name']}' coef = {li['coef']:.4g}, p = {li['pvalue']:.4g}, "
            f"95% CI = [{li['ci_95'][0]:.4g}, {li['ci_95'][1]:.4g}]. Interpretation: {li.get('interpretation','')}"
        )
    else:
        err_msg = logit_info.get('error') if logit_info else 'Logit result missing.'
        summary_lines.append(f"Logistic model unavailable or failed: {err_msg}")

    description = " ".join(summary_lines)

    return {'object': out_object, 'description': description}