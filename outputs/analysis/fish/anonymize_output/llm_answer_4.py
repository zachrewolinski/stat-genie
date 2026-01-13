def extract_final_answer(model_output):
    """
    Extract interpretable statistics from the model output returned by the modeling function.
    Returns a dictionary with keys:
      - "object": a nested dict of extracted numeric results (coefficients, SE, p-values,
                  95% CI, exponentiated coefficients = rate ratios, baseline rate per hour, summary).
                  If no fitted model is available, this is None.
      - "description": a short text explaining what the returned object contains and how to interpret it.
    The function is defensive: it handles when no model was fitted, when a Negative Binomial
    result is present (preferred) or a Poisson result, and it constructs CIs from bse if needed.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    # Handle the case where the modeling function returned the "no data / no fit" dict
    if isinstance(model_output, dict):
        # If the modeling routine added an explanatory note that no model was fitted, return that
        if model_output.get('poisson_result') is None and model_output.get('negative_binomial_result') is None:
            desc = model_output.get('notes', 'No fitted model available in model_output.')
            return {
                "object": None,
                "description": f"No fitted model found. Notes from model run: {desc} Please supply data / fit before extracting coefficients."
            }

        # Prefer negative binomial result if present (overdispersion handled)
        model = model_output.get('negative_binomial_result') or model_output.get('poisson_result')
        model_type = 'NegativeBinomial' if model_output.get('negative_binomial_result') is not None else 'Poisson'
        dispersion = model_output.get('dispersion', None)
    else:
        # model_output may directly be a fitted statsmodels result
        model = model_output
        # Try to infer model type; fall back to generic name
        try:
            model_type = type(model.model.family).__name__
        except Exception:
            model_type = 'fitted_model'
        dispersion = None

    if model is None:
        return {
            "object": None,
            "description": "No fitted model object found in model_output."
        }

    # Extract parameters, SE, p-values, and confidence intervals robustly
    try:
        params = model.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Fitted model present but could not read params: {e}"
        }

    # Ensure params is a pandas Series for indexed access
    if not hasattr(params, 'index'):
        try:
            params = pd.Series(params)
        except Exception:
            params = pd.Series(np.asarray(params))

    # Standard errors
    se = getattr(model, 'bse', None)
    if se is None:
        # try attribute names that may exist
        se = getattr(model, 'standard_errors', None)
    if se is not None and not hasattr(se, 'index'):
        try:
            se = pd.Series(se, index=params.index)
        except Exception:
            se = None

    # p-values
    pvalues = getattr(model, 'pvalues', None)
    if pvalues is None and se is not None:
        # approximate p-values via normal approximation
        try:
            z = params / se
            pvalues = pd.Series(2 * (1 - stats.norm.cdf(np.abs(z))), index=params.index)
        except Exception:
            pvalues = None

    # confidence intervals
    try:
        conf = model.conf_int()
        # conf may be a DataFrame with two columns; convert to DataFrame with columns [lower, upper]
        conf_df = pd.DataFrame(conf.values, index=params.index, columns=['lower', 'upper'])
    except Exception:
        # build from se if available
        if se is not None:
            zval = stats.norm.ppf(0.975)
            lower = params - zval * se
            upper = params + zval * se
            conf_df = pd.DataFrame({'lower': lower, 'upper': upper})
        else:
            # cannot compute CI
            conf_df = pd.DataFrame({'lower': [np.nan]*len(params), 'upper': [np.nan]*len(params)},
                                   index=params.index)

    # Build a serializable summary per parameter
    coef_summary = {}
    for name in params.index:
        coef_val = float(params[name])
        se_val = float(se[name]) if (se is not None and name in se.index) else None
        pval = float(pvalues[name]) if (pvalues is not None and name in pvalues.index) else None
        ci_low = float(conf_df.loc[name, 'lower']) if name in conf_df.index else None
        ci_high = float(conf_df.loc[name, 'upper']) if name in conf_df.index else None
        # exponentiated coefficient = multiplicative effect on rate (rate ratio)
        try:
            rr = float(np.exp(coef_val))
            rr_ci_low = float(np.exp(ci_low)) if ci_low is not None and not np.isnan(ci_low) else None
            rr_ci_high = float(np.exp(ci_high)) if ci_high is not None and not np.isnan(ci_high) else None
        except Exception:
            rr = None
            rr_ci_low = None
            rr_ci_high = None

        coef_summary[name] = {
            'coef': coef_val,
            'se': se_val,
            'pvalue': pval,
            'ci_lower': ci_low,
            'ci_upper': ci_high,
            'rate_ratio': rr,
            'rr_ci_lower': rr_ci_low,
            'rr_ci_upper': rr_ci_high
        }

    # Baseline expected fish-per-hour: exp(intercept) if intercept named 'const' or 'Intercept'
    intercept_name = None
    for candidate in ['const', 'Intercept', 'intercept']:
        if candidate in params.index:
            intercept_name = candidate
            break
    baseline_rate = None
    if intercept_name is not None:
        try:
            baseline_rate = float(np.exp(params[intercept_name]))
        except Exception:
            baseline_rate = None

    # Pack summary info
    summary = {
        'model_used': model_type,
        'nobs': float(getattr(model, 'nobs', np.nan)) if hasattr(model, 'nobs') else None,
        'aic': float(getattr(model, 'aic', np.nan)) if hasattr(model, 'aic') else None,
        'dispersion (from modeling routine)': float(dispersion) if dispersion is not None and not (isinstance(dispersion, float) and np.isnan(dispersion)) else None
    }

    result_object = {
        'coefficients': coef_summary,
        'baseline_rate_per_hour': baseline_rate,
        'summary': summary
    }

    # Interpretation guidance
    description_lines = [
        f"Extracted results from a fitted {model_type} model (if NegativeBinomial was available it was preferred).",
        "For each model parameter, 'coef' is the log-rate coefficient from the model; 'rate_ratio' = exp(coef) is the multiplicative effect on the fish-per-hour rate.",
        "Interpretation example: if rate_ratio for 'livebait' is 1.5, groups using livebait are estimated to catch 50% more fish per hour than groups not using livebait, all else equal.",
        "The 'baseline_rate_per_hour' is exp(intercept) and represents the expected fish caught per hour for the reference group (predictors = 0).",
        "P-values and 95% CIs are provided where available to assess statistical evidence and uncertainty.",
    ]
    description = " ".join(description_lines)

    return {
        "object": result_object,
        "description": description
    }