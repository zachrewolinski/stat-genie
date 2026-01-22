def extract_final_answer(model_output):
    """
    Extracts the effect of 'Children' from fitted models contained in model_output.

    Expected keys in model_output (any subset may be present):
      - 'nb_model' or 'nb_model_robust' : statsmodels GLMResults (NegativeBinomial) or robust wrapper
      - 'logit_model' or 'logit_model_robust' : statsmodels LogitResults or robust wrapper
      - 'nb_error', 'logit_error' : error strings (if fitting/processing failed)

    Returns dict with:
      - "object": a dict with extracted statistics (or errors if models are absent)
      - "description": human-readable interpretation about whether having children
                       decreases engagement in extramarital affairs.

    The function is defensive: if models are present it extracts coef, p-value,
    95% CI and exponentiated effect (IRR or OR). If models are missing it returns
    the available error messages and suggestions.
    """
    import numpy as np
    import pandas as pd

    def _extract_from_result(res, varname='Children', model_type='nb'):
        """
        Extract coef, pvalue, CI, and exponentiated effect from a statsmodels result object.
        model_type: 'nb' -> interpret exp(coef) as incidence rate ratio (IRR)
                    'logit' -> interpret exp(coef) as odds ratio (OR)
        """
        out = {}
        # get params, pvalues, conf_int robust to object type
        try:
            params = pd.Series(res.params)
        except Exception:
            # some robust wrappers expose .params differently
            try:
                params = pd.Series(res._results.params)
            except Exception:
                raise RuntimeError("Can't access params from the result object.")
        try:
            pvalues = pd.Series(res.pvalues)
        except Exception:
            try:
                pvalues = pd.Series(res._results.pvalues)
            except Exception:
                pvalues = pd.Series(index=params.index, data=[np.nan] * len(params))

        # conf_int may return ndarray or DataFrame
        try:
            ci = res.conf_int()
            ci_df = pd.DataFrame(ci, index=params.index, columns=['ci_low', 'ci_high'])
        except Exception:
            try:
                ci = res._results.conf_int()
                ci_df = pd.DataFrame(ci, index=params.index, columns=['ci_low', 'ci_high'])
            except Exception:
                ci_df = pd.DataFrame(index=params.index, columns=['ci_low', 'ci_high'])

        # Safely get the variable's statistics
        if varname not in params.index:
            raise KeyError(f"Variable '{varname}' not found in model parameters: {list(params.index)}")

        coef = float(params.loc[varname])
        pval = float(pvalues.get(varname, np.nan))
        ci_low = ci_df.loc[varname, 'ci_low'] if varname in ci_df.index else np.nan
        ci_high = ci_df.loc[varname, 'ci_high'] if varname in ci_df.index else np.nan

        effect = float(np.exp(coef)) if np.isfinite(coef) else np.nan
        # exponentiated CI if available and finite
        try:
            effect_ci_low = float(np.exp(ci_low)) if np.isfinite(ci_low) else np.nan
            effect_ci_high = float(np.exp(ci_high)) if np.isfinite(ci_high) else np.nan
        except Exception:
            effect_ci_low = effect_ci_high = np.nan

        out.update({
            'coef': coef,
            'pvalue': pval,
            'ci_95': (ci_low, ci_high),
            'exp_coef': effect,
            'exp_ci_95': (effect_ci_low, effect_ci_high),
            'model_type': model_type
        })
        return out

    result_object = {}
    messages = []

    # If errors were recorded in the model_output, capture them
    if isinstance(model_output, dict):
        if 'nb_error' in model_output:
            messages.append(f"Negative binomial model error: {model_output.get('nb_error')}")
        if 'logit_error' in model_output:
            messages.append(f"Logistic model error: {model_output.get('logit_error')}")
    else:
        # Unexpected structure
        messages.append("model_output is not a dict; cannot proceed.")

    # Try to extract from Negative Binomial (prefer robust if available)
    nb_res = None
    for key in ('nb_model_robust', 'nb_model'):
        if isinstance(model_output, dict) and key in model_output:
            nb_res = model_output[key]
            break

    logit_res = None
    for key in ('logit_model_robust', 'logit_model'):
        if isinstance(model_output, dict) and key in model_output:
            logit_res = model_output[key]
            break

    extracted = {}
    # Extract if available
    try:
        if nb_res is not None:
            extracted['negative_binomial'] = _extract_from_result(nb_res, varname='Children', model_type='nb')
    except Exception as e:
        messages.append(f"Failed to extract from negative binomial result: {e}")

    try:
        if logit_res is not None:
            extracted['logistic_any_affair'] = _extract_from_result(logit_res, varname='Children', model_type='logit')
    except Exception as e:
        messages.append(f"Failed to extract from logistic result: {e}")

    # If we extracted anything, form an interpretation
    interpretation = ""
    if extracted:
        # Prefer negative binomial for count outcome if available
        primary = extracted.get('negative_binomial') or extracted.get('logistic_any_affair')
        coef = primary['coef']
        pval = primary['pvalue']
        exp_coef = primary['exp_coef']
        ci_low, ci_high = primary['ci_95']
        exp_ci_low, exp_ci_high = primary['exp_ci_95']
        model_type = primary['model_type']

        # Interpret sign and significance
        signif = (pval < 0.05) if (pval is not None and not np.isnan(pval)) else False
        direction = "decrease" if coef < 0 else ("increase" if coef > 0 else "no change")

        if model_type == 'nb':
            measure_name = "incidence rate ratio (IRR) for count of affairs"
        else:
            measure_name = "odds ratio (OR) for having any affair"

        interpretation = (
            f"Primary model used: {'Negative binomial' if model_type=='nb' else 'Logistic'}.\n"
            f"The coefficient on 'Children' = {coef:.4g} (p = {pval:.4g}).\n"
            f"Exponentiated effect ({measure_name}) = {exp_coef:.4g} "
            f"with 95% CI [{exp_ci_low:.4g}, {exp_ci_high:.4g}].\n"
        )
        if signif:
            if coef < 0:
                interpretation += "This indicates a statistically significant decrease in extramarital affairs associated with having children (p < 0.05).\n"
            else:
                interpretation += "This indicates a statistically significant increase in extramarital affairs associated with having children (p < 0.05).\n"
        else:
            interpretation += "The effect is not statistically significant at the 0.05 level; there is no strong evidence that having children changes engagement in extramarital affairs.\n"

        # If both models available, summarize agreement
        if 'negative_binomial' in extracted and 'logistic_any_affair' in extracted:
            nb = extracted['negative_binomial']
            lg = extracted['logistic_any_affair']
            interpretation += (
                f"\nBoth models were available. Negative binomial exp(coef)={nb['exp_coef']:.4g} "
                f"(p={nb['pvalue']:.4g}); Logistic exp(coef)={lg['exp_coef']:.4g} "
                f"(p={lg['pvalue']:.4g}). "
            )
            # Simple agreement statement
            same_direction = (nb['coef'] * lg['coef'] >= 0)
            if same_direction:
                interpretation += "Both models agree on the direction of the effect.\n"
            else:
                interpretation += "The two models disagree on the direction of the effect.\n"

        result_object = {
            'extracted_stats': extracted,
            'messages': messages
        }

    else:
        # Nothing extracted
        interpretation = (
            "No fitted model objects were available in model_output to extract statistics.\n"
            "Observed errors (if any):\n  - " + "\n  - ".join(messages) + "\n\n"
            "Likely cause: the original modeling code attempted to call get_robustcov_results on the fitted "
            "models, which raised an AttributeError and caused the try/except blocks to store only error messages "
            "instead of the fitted model objects.\n\n"
            "Recommended fixes:\n"
            "  - Re-run the original model-fitting code but avoid calling get_robustcov_results if it's not available "
            "in the installed statsmodels version. Instead, either use the plain fitted result object or compute robust "
            "covariances manually, e.g.:\n"
            "      cov = sm.stats.sandwich_covariance.cov_hc3(fit_result)\n"
            "      robust_params = fit_result.params\n"
            "      # compute robust SEs from cov if needed\n"
            "  - Ensure the try/except does not swallow the fitted model objects before they are stored in the results dict.\n\n"
            "Once valid model objects (nb_model and/or logit_model) are available in model_output, this function will "
            "extract the coefficient, p-value, 95% CI, and exponentiated effect and return an interpretation about "
            "whether having children decreases engagement in extramarital affairs."
        )
        result_object = {
            'extracted_stats': None,
            'messages': messages
        }

    return {
        "object": result_object,
        "description": interpretation
    }