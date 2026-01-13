def extract_final_answer(model_output):
    """
    Extracts the effect of 'Children' on extramarital affairs from model_output.

    Expects model_output to be a dict possibly containing:
      - 'tobit' : a fitted Tobit model result object (statsmodels GenericLikelihoodModelResults or similar)
      - 'tobit_error' : an error string (if Tobit failed)
      - 'logit' : a fitted Logit model result object (statsmodels results)
      - 'logit_error' : an error string (if Logit failed)

    Returns a dict with keys:
      - "object": a dict with extracted statistics for Tobit and Logit (or error messages)
      - "description": a short plain-English interpretation about whether having children
                       decreases engagement in extramarital affairs based on available models.
    """
    import numpy as np
    from scipy import stats

    def summarize_result(res, model_name):
        """
        Attempt to extract coefficient, SE, p-value and 95% CI for 'Children'.
        Works if res has .params and either .bse or .cov_params or .pvalues or .conf_int().
        If res.params has an index, use it; otherwise assume ordering:
          ['const'] + ['Children','Age','YearsMarried','Male','Religiosity','Education','Occupation','MaritalSatisfaction']
        """
        summary = {}
        if res is None:
            return {"error": f"No {model_name} result object provided."}

        # get params (as pandas Series or numpy array)
        params = getattr(res, "params", None)
        if params is None:
            # some wrappers store params as .params if present; if not, try attribute 'beta' etc.
            params = getattr(res, "beta", None)
        if params is None:
            return {"error": f"{model_name} result has no params attribute."}

        # Convert to numpy array and obtain names if possible
        try:
            # If pandas Series
            param_names = list(params.index)
            param_values = np.asarray(params.values, dtype=float)
        except Exception:
            # assume numpy array
            param_values = np.asarray(params, dtype=float)
            param_names = None

        # Known covariate order used in the modeling code
        covariates = ['Children', 'Age', 'YearsMarried', 'Male', 'Religiosity', 'Education', 'Occupation', 'MaritalSatisfaction']
        # try to determine name->index mapping
        if param_names is None:
            # common lengths: len(covariates)+1 if constant included, or len(covariates)
            if param_values.size == len(covariates) + 1:
                param_names = ['const'] + covariates
            elif param_values.size == len(covariates):
                param_names = covariates
            else:
                # fallback: create generic names
                param_names = [f"param_{i}" for i in range(param_values.size)]

        # create dict of params
        param_dict = {name: float(val) for name, val in zip(param_names, param_values)}

        if 'Children' not in param_dict:
            # attempt case-insensitive match
            match = None
            for name in param_dict:
                if name.lower() == 'children':
                    match = name
                    break
            if match is None:
                return {"error": f"'Children' not found among parameters for {model_name}. Available: {list(param_dict.keys())}"}
            children_name = match
        else:
            children_name = 'Children'

        beta = param_dict[children_name]
        # standard errors
        se = None
        # try .bse
        bse = getattr(res, 'bse', None)
        if bse is not None:
            try:
                se_val = bse[children_name] if hasattr(bse, 'get') or hasattr(bse, 'index') else np.asarray(bse)[param_names.index(children_name)]
                se = float(se_val)
            except Exception:
                # fallback if bse is array-like without names
                try:
                    se = float(np.asarray(bse)[param_names.index(children_name)])
                except Exception:
                    se = None
        if se is None:
            # try cov_params (matrix)
            try:
                cov = res.cov_params()
                # cov could be DataFrame or ndarray
                if hasattr(cov, 'loc'):
                    se = float(np.sqrt(cov.loc[children_name, children_name]))
                else:
                    idx = param_names.index(children_name)
                    se = float(np.sqrt(np.asarray(cov)[idx, idx]))
            except Exception:
                se = None

        # compute p-value and CI if needed
        pval = None
        ci_lower = ci_upper = None
        if se is not None and se > 0:
            z = beta / se
            pval = float(2 * (1 - stats.norm.cdf(abs(z))))
            ci_lower = float(beta - 1.96 * se)
            ci_upper = float(beta + 1.96 * se)
        else:
            # try pvalues attribute
            pvals = getattr(res, 'pvalues', None)
            if pvals is not None:
                try:
                    pval = float(pvals[children_name] if hasattr(pvals, 'get') or hasattr(pvals, 'index') else np.asarray(pvals)[param_names.index(children_name)])
                except Exception:
                    pval = None
            # try conf_int method
            try:
                conf = res.conf_int()
                if hasattr(conf, 'loc'):
                    ci_lower = float(conf.loc[children_name, 0])
                    ci_upper = float(conf.loc[children_name, 1])
                else:
                    ci_arr = np.asarray(conf)
                    idx = param_names.index(children_name)
                    ci_lower = float(ci_arr[idx, 0])
                    ci_upper = float(ci_arr[idx, 1])
            except Exception:
                pass

        # assemble summary
        summary['parameter_name'] = children_name
        summary['coef'] = float(beta)
        summary['se'] = None if se is None else float(se)
        summary['p_value'] = None if pval is None else float(pval)
        summary['95%_ci'] = None if (ci_lower is None or ci_upper is None) else (ci_lower, ci_upper)

        # brief interpretation for this model
        if summary['p_value'] is not None:
            if summary['p_value'] < 0.05:
                if summary['coef'] < 0:
                    summary['interpretation'] = "Statistically significant negative association: having children is associated with fewer reported extramarital acts."
                else:
                    summary['interpretation'] = "Statistically significant positive association: having children is associated with more reported extramarital acts."
            else:
                if summary['coef'] < 0:
                    summary['interpretation'] = "Negative point estimate but not statistically significant at 0.05; evidence is weak that having children reduces extramarital acts."
                elif summary['coef'] > 0:
                    summary['interpretation'] = "Positive point estimate but not statistically significant at 0.05; evidence is weak that having children increases extramarital acts."
                else:
                    summary['interpretation'] = "Point estimate is zero."
        else:
            # no p-value available
            if summary['coef'] < 0:
                summary['interpretation'] = "Negative point estimate for 'Children' (no p-value available); cannot determine statistical significance."
            elif summary['coef'] > 0:
                summary['interpretation'] = "Positive point estimate for 'Children' (no p-value available); cannot determine statistical significance."
            else:
                summary['interpretation'] = "Point estimate is zero and no p-value available."

        return summary

    final_object = {}
    messages = []

    # Tobit
    if 'tobit' in model_output and model_output['tobit'] is not None:
        try:
            final_object['tobit'] = summarize_result(model_output['tobit'], 'tobit')
        except Exception as e:
            final_object['tobit'] = {"error": f"Exception while extracting Tobit results: {str(e)}"}
    elif 'tobit_error' in model_output:
        final_object['tobit'] = {"error": f"Tobit failed: {model_output['tobit_error']}"}
    else:
        final_object['tobit'] = {"error": "No Tobit result or error message found in model_output."}

    # Logit
    if 'logit' in model_output and model_output['logit'] is not None:
        try:
            final_object['logit'] = summarize_result(model_output['logit'], 'logit')
        except Exception as e:
            final_object['logit'] = {"error": f"Exception while extracting Logit results: {str(e)}"}
    elif 'logit_error' in model_output:
        final_object['logit'] = {"error": f"Logit failed: {model_output['logit_error']}"}
    else:
        final_object['logit'] = {"error": "No Logit result or error message found in model_output."}

    # Build description
    # If either model produced a numeric summary, use that to give a concise answer.
    interpretations = []
    for m in ['tobit', 'logit']:
        info = final_object.get(m, {})
        if 'error' in info:
            messages.append(f"{m.upper()}: {info['error']}")
        else:
            coef = info.get('coef')
            p = info.get('p_value')
            interp = info.get('interpretation', '')
            interpretations.append(f"{m.upper()}: coef={coef}, p={p}. {interp}")

    if len(interpretations) > 0:
        description = " | ".join(interpretations)
        if any("Statistically significant negative" in s for s in interpretations) and not any("Statistically significant positive" in s for s in interpretations):
            description = "Overall: Evidence that having children decreases engagement in extramarital affairs (see model details). " + description
        elif any("Statistically significant positive" in s for s in interpretations) and not any("Statistically significant negative" in s for s in interpretations):
            description = "Overall: Evidence that having children increases engagement in extramarital affairs (see model details). " + description
        else:
            # either no significant results or mixed/uncertain
            description = "Overall: No consistent statistically significant evidence that having children decreases engagement in extramarital affairs based on available model results. " + description
    else:
        # No numeric summaries; report errors
        description = "Modeling failed or no fitted models available. Details: " + " ; ".join(messages) if messages else "No results available."

    return {"object": final_object, "description": description}