def extract_final_answer(model_output):
    """
    Extracts statistics about the association of 'HasChildren' with extramarital affairs
    from the provided model_output dict (expected keys: 'logit' and 'ols' with fitted
    statsmodels result objects).
    
    Returns:
      {
        "object": { ... extracted numeric results ... },
        "description": "A short plain-language interpretation of the results"
      }
    """
    import numpy as np
    results = {}
    description_lines = []
    
    def sig_label(p):
        if p < 0.001:
            return 'p < 0.001'
        return f'p = {p:.3f}'
    
    # Helper to extract from a statsmodels result-like object
    def extract_coef_info(res, varname='HasChildren'):
        info = {}
        try:
            params = res.params
            pvalues = res.pvalues
            bse = res.bse
            ci = res.conf_int()
        except Exception:
            # Some result objects (e.g., older wrappers) may store slightly differently
            try:
                params = res._results.params
                pvalues = res._results.pvalues
                bse = res._results.bse
                ci = res._results.conf_int()
            except Exception:
                return None
        
        if varname not in params.index:
            return None
        
        coef = float(params[varname])
        se = float(bse[varname]) if (hasattr(bse, '__getitem__')) else float(bse.loc[varname])
        pval = float(pvalues[varname])
        try:
            ci_low, ci_high = map(float, ci.loc[varname])
        except Exception:
            # conf_int might return ndarray
            try:
                ci_arr = np.array(ci)
                # find index of varname in params.index
                idx = list(params.index).index(varname)
                ci_low, ci_high = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])
            except Exception:
                ci_low, ci_high = (None, None)
        
        info.update({
            'coef': coef,
            'std_err': se,
            'pvalue': pval,
            'ci_lower': ci_low,
            'ci_upper': ci_high
        })
        return info
    
    # Process logit
    logit_obj = model_output.get('logit')
    logit_info = None
    if logit_obj is not None:
        logit_info = extract_coef_info(logit_obj, 'HasChildren')
        if logit_info is None:
            description_lines.append("Logistic model was present but 'HasChildren' not found among parameters.")
        else:
            # compute odds ratio and CI for OR
            coef = logit_info['coef']
            ci_low = logit_info['ci_lower']
            ci_high = logit_info['ci_upper']
            # exponentiate if numeric
            try:
                or_val = float(np.exp(coef))
            except Exception:
                or_val = None
            try:
                or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
                or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
            except Exception:
                or_ci_low = or_ci_high = None
            logit_info.update({
                'odds_ratio': or_val,
                'or_ci_lower': or_ci_low,
                'or_ci_upper': or_ci_high
            })
            results['logit'] = logit_info
            
            # Interpret
            p = logit_info['pvalue']
            if p < 0.05:
                direction = "decrease" if logit_info['odds_ratio'] < 1 else "increase"
                description_lines.append(
                    f"Logistic: Having children is associated with a statistically significant {direction} "
                    f"in the odds of reporting any extramarital affair (OR = {logit_info['odds_ratio']:.3f}, "
                    f"{sig_label(p)}; 95% CI for OR = [{logit_info['or_ci_lower']:.3f}, {logit_info['or_ci_upper']:.3f}])."
                )
            else:
                description_lines.append(
                    f"Logistic: No statistically significant association between having children and the odds of any affair "
                    f"(OR = {logit_info['odds_ratio']:.3f}, {sig_label(p)}; 95% CI for OR = "
                    f"[{(logit_info['or_ci_lower'] if logit_info['or_ci_lower'] is not None else 'NA')}, "
                    f"{(logit_info['or_ci_high'] if 'or_ci_high' in logit_info else 'NA')}])."
                )
    else:
        description_lines.append("No logistic model found in model_output.")
    
    # Process OLS on LogAffairs
    ols_obj = model_output.get('ols')
    ols_info = None
    if ols_obj is not None:
        ols_info = extract_coef_info(ols_obj, 'HasChildren')
        if ols_info is None:
            description_lines.append("OLS model was present but 'HasChildren' not found among parameters.")
        else:
            results['ols'] = ols_info
            p = ols_info['pvalue']
            coef = ols_info['coef']
            if p < 0.05:
                direction = "decrease" if coef < 0 else "increase"
                description_lines.append(
                    f"OLS (log intensity): Having children is associated with a statistically significant {direction} "
                    f"in log(affairs+1) ({coef:.3f}, {sig_label(p)}; 95% CI = [{ols_info['ci_lower']:.3f}, {ols_info['ci_upper']:.3f}])."
                )
            else:
                description_lines.append(
                    f"OLS (log intensity): No statistically significant association between having children and the logged number of affairs "
                    f"({coef:.3f}, {sig_label(p)}; 95% CI = [{ols_info['ci_lower']:.3f}, {ols_info['ci_upper']:.3f}])."
                )
    else:
        description_lines.append("No OLS model found in model_output.")
    
    # Formulate final short conclusion combining both models
    # Prefer logistic as the primary test of whether having children decreases engagement in affairs.
    conclusion = "Final assessment: "
    if 'logit' in results:
        p = results['logit']['pvalue']
        orv = results['logit']['odds_ratio']
        if p < 0.05:
            if orv < 1:
                conclusion += (f"Having children is associated with a statistically significant decrease in the probability of "
                               f"reporting any extramarital affair (OR = {orv:.3f}, {sig_label(p)}).")
            else:
                conclusion += (f"Having children is associated with a statistically significant increase in the probability of "
                               f"reporting any extramarital affair (OR = {orv:.3f}, {sig_label(p)}).")
        else:
            conclusion += ("There is no statistically significant association between having children and the probability "
                           "of reporting any extramarital affair (logistic model).")
    elif 'ols' in results:
        p = results['ols']['pvalue']
        coef = results['ols']['coef']
        if p < 0.05:
            if coef < 0:
                conclusion += ("Having children is associated with a statistically significant decrease in the logged intensity "
                               f"of affairs (coef = {coef:.3f}, {sig_label(p)}).")
            else:
                conclusion += ("Having children is associated with a statistically significant increase in the logged intensity "
                               f"of affairs (coef = {coef:.3f}, {sig_label(p)}).")
        else:
            conclusion += ("There is no statistically significant association between having children and the logged intensity of."
                           " affairs (OLS).")
    else:
        conclusion += "No usable model found to answer the question."
    
    description = " ".join(description_lines + [conclusion])
    
    return {
        "object": results,
        "description": description
    }