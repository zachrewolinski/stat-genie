def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, confidence intervals,
    and interprets the effects of age, sex_male, and help_yes from the provided
    model_output dictionary.

    Returns a dict with:
      - "object": dict with results for final GLM (rate ratios) and OLS (absolute changes)
      - "description": brief interpretation (significance at alpha=0.05) in context
    """
    import numpy as np

    vars_of_interest = ['age', 'sex_male', 'help_yes']

    # Helper to safely extract statistics from a statsmodels result-like object
    def summarize_result(res, is_glm=False):
        out = {}
        if res is None:
            return None

        # Try to get params, bse, pvalues, conf_int
        try:
            params = res.params
        except Exception:
            params = None
        try:
            bse = res.bse
        except Exception:
            bse = None
        try:
            pvalues = res.pvalues
        except Exception:
            pvalues = None
        try:
            ci = res.conf_int()
        except Exception:
            ci = None

        # Helper to get names for array-like containers
        def get_exog_names():
            # preference: model.exog_names, then params.index if available
            try:
                if hasattr(res, 'model') and hasattr(res.model, 'exog_names'):
                    return list(res.model.exog_names)
            except Exception:
                pass
            try:
                if hasattr(params, 'index'):
                    return list(params.index)
            except Exception:
                pass
            return None

        names = get_exog_names()

        # Helper to fetch a value for a variable from various container types
        def get_value(container, var):
            if container is None:
                return None
            # pandas-like Series or DataFrame row
            try:
                if hasattr(container, 'index') and var in container.index:
                    # Series: container[var] or container.loc[var]
                    try:
                        return container.loc[var]
                    except Exception:
                        return container[var]
            except Exception:
                pass
            # dict-like
            try:
                if isinstance(container, dict):
                    return container.get(var)
            except Exception:
                pass
            # numpy array: use names to find index
            try:
                if isinstance(container, np.ndarray):
                    if names:
                        if var in names:
                            idx = names.index(var)
                            # If 1d array
                            if container.ndim == 1 and idx < container.shape[0]:
                                return container[idx]
                            # If 2d like conf_int with shape (k,2)
                            if container.ndim == 2 and idx < container.shape[0]:
                                return container[idx]
                    # If no names, cannot map variable -> return None
            except Exception:
                pass
            # If container has .get with var
            try:
                if hasattr(container, 'get'):
                    return container.get(var)
            except Exception:
                pass
            return None

        for v in vars_of_interest:
            coef_val = get_value(params, v)
            if coef_val is None:
                out[v] = {
                    'coef': None,
                    'se': None,
                    'pvalue': None,
                    'ci_lower': None,
                    'ci_upper': None,
                    'note': f'{v} not found in model'
                }
                continue

            # Convert coefficient to float if possible
            try:
                # coef_val could be an array element or pandas scalar
                coef = float(np.asarray(coef_val).item())
            except Exception:
                try:
                    coef = float(coef_val)
                except Exception:
                    coef = None

            se_val = get_value(bse, v)
            try:
                se = float(np.asarray(se_val).item()) if se_val is not None else None
            except Exception:
                try:
                    se = float(se_val) if se_val is not None else None
                except Exception:
                    se = None

            p_val = get_value(pvalues, v)
            try:
                p = float(np.asarray(p_val).item()) if p_val is not None else None
            except Exception:
                try:
                    p = float(p_val) if p_val is not None else None
                except Exception:
                    p = None

            # Extract CI: ci might be DataFrame-like, ndarray (k,2), or dict-like
            ci_lower = None
            ci_upper = None
            if ci is not None:
                try:
                    # If ci has index and columns
                    if hasattr(ci, 'loc') and hasattr(ci, 'columns'):
                        if v in ci.index:
                            row = ci.loc[v]
                            # row could be array-like or Series
                            try:
                                ci_lower = float(np.asarray(row.iloc[0]).item())
                                ci_upper = float(np.asarray(row.iloc[1]).item())
                            except Exception:
                                try:
                                    ci_lower = float(np.asarray(row[0]).item())
                                    ci_upper = float(np.asarray(row[1]).item())
                                except Exception:
                                    ci_lower = None
                                    ci_upper = None
                    # If ci is ndarray
                    elif isinstance(ci, np.ndarray):
                        if names and (v in names):
                            idx = names.index(v)
                            if ci.ndim == 2 and idx < ci.shape[0]:
                                try:
                                    ci_lower = float(np.asarray(ci[idx, 0]).item())
                                    ci_upper = float(np.asarray(ci[idx, 1]).item())
                                except Exception:
                                    ci_lower = None
                                    ci_upper = None
                    # If ci is dict-like with var -> [low, high]
                    elif isinstance(ci, dict):
                        cival = ci.get(v)
                        if cival is not None:
                            try:
                                ci_lower = float(np.asarray(cival[0]).item())
                                ci_upper = float(np.asarray(cival[1]).item())
                            except Exception:
                                ci_lower = None
                                ci_upper = None
                except Exception:
                    ci_lower = None
                    ci_upper = None

            entry = {
                'coef': coef,
                'se': se,
                'pvalue': p,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }

            # For GLM with log link: convert to rate ratio interpretation
            if is_glm and (coef is not None):
                try:
                    rr = float(np.exp(coef))
                except Exception:
                    rr = None
                try:
                    rr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
                    rr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
                except Exception:
                    rr_ci_lower = None
                    rr_ci_upper = None
                entry.update({
                    'rate_ratio': rr,
                    'rate_ratio_ci_lower': rr_ci_lower,
                    'rate_ratio_ci_upper': rr_ci_upper,
                    'interpretation': (
                        "Multiplicative change (rate ratio) in nuts-opened-per-second "
                        "for a one-unit increase in the predictor.")
                })
            else:
                entry.update({
                    'interpretation': "Absolute change in nuts_per_second for a one-unit increase in predictor."
                })

            out[v] = entry
        return out

    # Select final GLM result (fall back to poisson_result if missing)
    final_glm = model_output.get('final_glm_result') or model_output.get('poisson_result')
    ols_res = model_output.get('ols_result')

    glm_summary = summarize_result(final_glm, is_glm=True)
    ols_summary = summarize_result(ols_res, is_glm=False)

    # Build an overall conclusion at alpha = 0.05 using final GLM (preferred) and OLS as robustness
    def significance_label(p):
        if p is None:
            return 'unknown'
        return 'significant' if p < 0.05 else 'not_significant'

    conclusion = {}
    for v in vars_of_interest:
        glm_p = None
        ols_p = None
        try:
            glm_p = glm_summary[v]['pvalue']
        except Exception:
            glm_p = None
        try:
            ols_p = ols_summary[v]['pvalue']
        except Exception:
            ols_p = None

        # Determine direction from final GLM coef if available, else OLS
        direction = None
        try:
            coef = glm_summary[v]['coef']
            if coef is not None:
                direction = 'positive' if coef > 0 else ('negative' if coef < 0 else 'zero')
        except Exception:
            coef = None
        if (direction is None) and (ols_summary and v in ols_summary):
            try:
                ocoef = ols_summary[v]['coef']
                if ocoef is not None:
                    direction = 'positive' if ocoef > 0 else ('negative' if ocoef < 0 else 'zero')
            except Exception:
                direction = None

        conclusion[v] = {
            'glm_pvalue': glm_p,
            'glm_significance': significance_label(glm_p),
            'ols_pvalue': ols_p,
            'ols_significance': significance_label(ols_p),
            'direction_based_on_glm_coef': direction,
            'summary': None
        }

        # Short human-readable interpretation
        if (glm_p is not None) and (glm_p < 0.05):
            concl_text = (f"In the (final) GLM, '{v}' has a statistically significant effect "
                          f"(p={glm_p:.3g}). Direction: {direction}.")
        elif (glm_p is not None) and (glm_p >= 0.05):
            concl_text = (f"In the (final) GLM, '{v}' does NOT have a statistically significant effect "
                          f"(p={glm_p:.3g}).")
        else:
            concl_text = f"Significance for '{v}' in the final GLM is unknown."

        # Add robustness note based on OLS
        if (ols_p is not None):
            if ols_p < 0.05:
                concl_text += f" OLS robustness check shows significance (p={ols_p:.3g})."
            else:
                concl_text += f" OLS robustness check does not show significance (p={ols_p:.3g})."

        conclusion[v]['summary'] = concl_text

    # Overall short description
    # Prefer final GLM (Negative Binomial due to overdispersion) for inference.
    overall_lines = []
    overall_lines.append("Preferred model: final GLM (Negative Binomial used because Poisson showed strong overdispersion).")
    for v in vars_of_interest:
        overall_lines.append(conclusion[v]['summary'])
    overall_text = " ".join(overall_lines)

    # Safely coerce dispersion to float if possible
    dispersion = model_output.get('dispersion')
    try:
        dispersion = float(dispersion) if dispersion is not None else None
    except Exception:
        dispersion = None

    result_object = {
        'final_glm': glm_summary,
        'ols': ols_summary,
        'dispersion': dispersion,
        'glm_family': model_output.get('glm_family'),
        'conclusion': conclusion
    }

    return {
        'object': result_object,
        'description': overall_text
    }