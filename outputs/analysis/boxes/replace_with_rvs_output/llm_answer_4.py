def extract_final_answer(model_output):
    """
    Extract interpretable statistics (or diagnostics) from the model_output dictionary
    returned by the modeling function.

    Returns a dict with:
      - "object": a dict keyed by model name. For fitted models, contains params, pvalues,
                  conf_int (as nested dicts) and a targeted summary for 'age_centered'.
                  For failed models, contains error_type and message.
      - "description": a short plain-language summary of what the extracted objects mean
                       for the research question (or why the models could not answer it),
                       plus practical next steps to remedy fitting problems.
    """
    import numpy as np
    import pandas as pd

    out = {}
    any_fitted = False

    for name, res in model_output.items():
        entry = {}
        # If the model object is an exception, capture diagnostics
        if isinstance(res, Exception):
            entry['status'] = 'error'
            entry['error_type'] = type(res).__name__
            entry['message'] = str(res)
            entry['notes'] = []
            # Provide likely reasons and quick checks for LinAlgError singular matrix
            if type(res).__name__ in ('LinAlgError',):
                entry['notes'].append(
                    "Singular matrix suggests perfect multicollinearity or insufficient variation "
                    "in predictors/response (e.g., a predictor is linear combination of others, "
                    "a categorical level has zero observations, or an outcome/predictor is constant)."
                )
                entry['notes'].append(
                    "Check: (1) that y/y_adj has >1 category present, (2) culture_cat has >1 level and "
                    "not collinear with other dummies, (3) SociallyGuided and Majority_vs_Minority "
                    "have variation, (4) you are not adding an intercept twice, and (5) sample size per cell."
                )
                entry['notes'].append(
                    "Remedies: drop problematic predictors or interaction, relevel or combine sparse categories, "
                    "use regularized/logistic regression with penalty or Bayesian models, or fit simpler models "
                    "(e.g., no interaction)."
                )
        else:
            # Assume a fitted statsmodels result-like object
            try:
                any_fitted = True
                entry['status'] = 'fitted'
                # params: may be Series, DataFrame or ndarray
                params = getattr(res, 'params', None)
                if params is not None:
                    try:
                        # Convert to DataFrame for consistent downstream handling
                        if isinstance(params, (pd.Series, pd.DataFrame)):
                            params_df = pd.DataFrame(params)
                        else:
                            # ndarray: try to get index from res.model.exog_names if available
                            exog_names = getattr(getattr(res, 'model', None), 'exog_names', None)
                            if exog_names is not None:
                                params_df = pd.DataFrame([params], columns=exog_names).T
                                params_df.columns = ['coef']
                            else:
                                params_df = pd.DataFrame(params)
                        # Round for readability
                        entry['params'] = params_df.round(4).to_dict()
                    except Exception:
                        entry['params'] = str(params)
                else:
                    entry['params'] = None

                # pvalues
                pvalues = getattr(res, 'pvalues', None)
                if pvalues is not None:
                    try:
                        if isinstance(pvalues, (pd.Series, pd.DataFrame)):
                            p_df = pd.DataFrame(pvalues)
                        else:
                            p_df = pd.DataFrame(pvalues)
                        entry['pvalues'] = p_df.round(4).to_dict()
                    except Exception:
                        entry['pvalues'] = str(pvalues)
                else:
                    entry['pvalues'] = None

                # conf_int
                try:
                    ci = res.conf_int()
                    entry['conf_int'] = pd.DataFrame(ci).round(4).to_dict()
                except Exception:
                    entry['conf_int'] = None

                # Targeted extraction: coefficients and p-values for age_centered (and interactions)
                age_entries = {}
                # Collect candidate names that contain 'age_centered'
                # For multi-indexed params (e.g., multinomial) we try to flatten to strings
                def flatten_names(obj):
                    if obj is None:
                        return []
                    if isinstance(obj, (pd.Series, pd.DataFrame)):
                        names = list(obj.index.astype(str))
                    else:
                        # try to get exog names from model
                        model = getattr(res, 'model', None)
                        names = getattr(model, 'exog_names', None) or []
                    return names

                names = flatten_names(params)
                # If params is a DataFrame with columns (e.g., MNLogit), scan both axes
                if isinstance(params, pd.DataFrame) and params.shape[1] > 1:
                    # For MNLogit, params may have columns for each outcome; iterate
                    age_entries = {}
                    for col in params.columns:
                        col_ser = params[col]
                        matches = [n for n in col_ser.index.astype(str) if 'age_centered' in str(n)]
                        if matches:
                            age_entries[col] = {}
                            for m in matches:
                                coef = float(col_ser.loc[m])
                                pval = None
                                pv_attr = getattr(res, 'pvalues', None)
                                try:
                                    pval = float(pv_attr[col].loc[m]) if pv_attr is not None else None
                                except Exception:
                                    # try alternative shapes
                                    try:
                                        pval = float(pv_attr.loc[m, col])
                                    except Exception:
                                        pval = None
                                age_entries[col][m] = {'coef': round(coef, 4), 'pvalue': None if pval is None else round(pval, 4)}
                else:
                    matches = [n for n in names if 'age_centered' in str(n)]
                    if matches:
                        for m in matches:
                            try:
                                coef = float(params.loc[m]) if isinstance(params, (pd.Series, pd.DataFrame)) else float(params[names.index(m)])
                            except Exception:
                                coef = None
                            try:
                                pv = getattr(res, 'pvalues', None)
                                if pv is not None:
                                    if isinstance(pv, (pd.Series, pd.DataFrame)):
                                        pval = float(pv.loc[m]) if m in pv.index else None
                                    else:
                                        pval = None
                                else:
                                    pval = None
                            except Exception:
                                pval = None
                            age_entries[m] = {'coef': None if coef is None else round(coef, 4), 'pvalue': None if pval is None else round(pval, 4)}
                entry['age_centered_effects'] = age_entries

            except Exception as e:
                entry['status'] = 'error_extracting'
                entry['message'] = f'Error while extracting fields: {type(e).__name__}: {e}'

        out[name] = entry

    # Build overall description
    if not any_fitted:
        description = (
            "No models converged successfully (all returned errors). Therefore we cannot extract "
            "coefficients, p-values, or confidence intervals to answer whether reliance on social "
            "information or preference for the majority changes with age across cultures.\n\n"
            "Observed errors (per model) are included in the 'object' field. Common causes and suggested fixes:\n"
            "- Perfect multicollinearity (e.g., redundant dummies, adding intercept twice). Remove/recode collinear predictors.\n"
            "- Sparse or constant outcomes/predictors (e.g., y or SociallyGuided has only one value). Check frequency tables.\n"
            "- Too many parameters for sample size (especially interactions); simplify model (remove interaction or combine sparse levels).\n"
            "- Use penalized/regularized logistic regression or Bayesian methods if separation is the issue.\n\n"
            "After addressing the above, re-run the models. If you get fitted model objects, pass them back to this function to extract "
            "coefficients and p-values for 'age_centered' and its interactions with culture."
        )
    else:
        # Summarize fitted models and targeted inference on age_centered
        descr_lines = []
        descr_lines.append("Extracted statistics for fitted models are available in 'object'.")
        # For each fitted model include a brief summary about age_centered significance if present
        for mname, info in out.items():
            if info.get('status') == 'fitted':
                ace = info.get('age_centered_effects', {})
                if not ace:
                    descr_lines.append(f"- {mname}: no 'age_centered' term found among coefficients.")
                else:
                    # summarize significance and direction
                    for term, stats in ace.items():
                        coef = stats.get('coef')
                        pval = stats.get('pvalue')
                        if coef is None:
                            descr_lines.append(f"- {mname} {term}: coefficient present but could not be parsed.")
                        else:
                            if pval is None:
                                descr_lines.append(f"- {mname} {term}: coef={coef} (p-value not available).")
                            else:
                                sig = "significant" if pval < 0.05 else "not significant"
                                direction = "positive" if coef > 0 else ("negative" if coef < 0 else "null")
                                descr_lines.append(f"- {mname} {term}: coef={coef}, p={pval} ({sig}, {direction} effect of age).")
        description = "\n".join(descr_lines)

    return {"object": out, "description": description}