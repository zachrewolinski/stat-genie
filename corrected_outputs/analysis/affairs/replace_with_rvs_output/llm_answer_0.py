def extract_final_answer(model_output):
    """
    Extracts the count-model coefficient for 'HasChildren' from a fitted
    statsmodels ZeroInflatedNegativeBinomialResultsWrapper and interprets it.

    Returns a dict with keys:
      - "object": dict of extracted numeric results (coef, se, z, p, CI, IRR, % change, significant)
      - "description": plain-language interpretation about whether having children
                       is associated with fewer extramarital affairs.

    Assumes the count-model parameter for the 'HasChildren' regressor is labeled
    exactly 'HasChildren' in model_output.params / .pvalues etc. Inflation-model
    parameters are expected to be prefixed with 'inflate_' and will not be used.
    """
    import numpy as np
    import pandas as pd

    # Prepare a helper to safely extract items from result object
    def safe_get(mapping, key):
        try:
            return mapping[key]
        except Exception:
            # try attribute-like access (some wrappers support .params.HasChildren)
            try:
                return getattr(mapping, key)
            except Exception:
                raise KeyError(f"Key/attribute '{key}' not found in the model output.")

    # Pull coefficient, std error, z/t stat, p-value
    try:
        coef = float(safe_get(model_output.params, 'HasChildren'))
    except KeyError:
        raise KeyError("Parameter 'HasChildren' not found in model_output.params. "
                       "Check parameter names in the fitted model.")

    # Standard error
    try:
        se = float(safe_get(model_output.bse, 'HasChildren'))
    except Exception:
        se = None

    # z/t value (statsmodels commonly exposes tvalues; use whichever exists)
    z = None
    for attr in ('tvalues', 'zvalues', 'z_stat'):
        if hasattr(model_output, attr):
            try:
                z = float(safe_get(getattr(model_output, attr), 'HasChildren'))
                break
            except Exception:
                z = None

    # p-value
    try:
        pval = float(safe_get(model_output.pvalues, 'HasChildren'))
    except Exception:
        pval = None

    # Confidence interval for coefficient
    try:
        ci = model_output.conf_int()
        # conf_int may be a DataFrame or ndarray. Handle both.
        if isinstance(ci, (pd.DataFrame, pd.Series)):
            ci_low = float(ci.loc['HasChildren'][0])
            ci_high = float(ci.loc['HasChildren'][1])
        else:
            # array-like: find the row index corresponding to the parameter
            params_index = list(model_output.params.index)
            idx = params_index.index('HasChildren')
            ci_low = float(ci[idx, 0])
            ci_high = float(ci[idx, 1])
    except Exception:
        ci_low = ci_high = None

    # Incidence Rate Ratio (IRR) and CI on IRR scale
    irr = float(np.exp(coef))
    irr_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
    irr_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
    percent_change = (irr - 1.0) * 100.0  # percent change in expected count

    # Determine statistical significance at alpha=0.05 if p-value available
    significant = None
    if pval is not None:
        significant = (pval < 0.05)

    # Build numeric object to return
    numeric_result = {
        'coef_HasChildren': round(coef, 4),
        'se_HasChildren': round(se, 4) if se is not None else None,
        'z_or_t_HasChildren': round(z, 4) if z is not None else None,
        'pvalue_HasChildren': round(pval, 4) if pval is not None else None,
        'ci_coef_low': round(ci_low, 4) if ci_low is not None else None,
        'ci_coef_high': round(ci_high, 4) if ci_high is not None else None,
        'IRR_HasChildren': round(irr, 4),
        'IRR_CI_low': round(irr_ci_low, 4) if irr_ci_low is not None else None,
        'IRR_CI_high': round(irr_ci_high, 4) if irr_ci_high is not None else None,
        'percent_change_in_count': round(percent_change, 2),
        'significant_at_0.05': significant
    }

    # Compose a concise interpretation
    if (coef is not None) and (pval is not None):
        if significant:
            if coef < 0:
                interpretation = (
                    "Estimated coefficient for HasChildren (count model) is negative "
                    f"({numeric_result['coef_HasChildren']}); IRR = {numeric_result['IRR_HasChildren']}. "
                    "This indicates that, holding controls constant, having children is associated "
                    f"with a statistically significant decrease in the expected number of extramarital affairs "
                    f"(about {abs(numeric_result['percent_change_in_count'])}% lower expected count; "
                    f"95% CI for IRR: [{numeric_result['IRR_CI_low']}, {numeric_result['IRR_CI_high']}])."
                )
            else:
                interpretation = (
                    "Estimated coefficient for HasChildren (count model) is positive "
                    f"({numeric_result['coef_HasChildren']}); IRR = {numeric_result['IRR_HasChildren']}. "
                    "This indicates that, holding controls constant, having children is associated "
                    f"with a statistically significant increase in the expected number of extramarital affairs "
                    f"(about {numeric_result['percent_change_in_count']}% higher expected count; "
                    f"95% CI for IRR: [{numeric_result['IRR_CI_low']}, {numeric_result['IRR_CI_high']}])."
                )
        else:
            # Not statistically significant
            if coef < 0:
                interpretation = (
                    "Estimated coefficient for HasChildren is negative "
                    f"({numeric_result['coef_HasChildren']}) but not statistically significant (p = {numeric_result['pvalue_HasChildren']}). "
                    "There is no strong evidence that having children decreases engagement in extramarital affairs "
                    "after controlling for the listed covariates."
                )
            else:
                interpretation = (
                    "Estimated coefficient for HasChildren is positive "
                    f"({numeric_result['coef_HasChildren']}) but not statistically significant (p = {numeric_result['pvalue_HasChildren']}). "
                    "There is no strong evidence that having children increases engagement in extramarital affairs "
                    "after controlling for the listed covariates."
                )
    else:
        interpretation = (
            "Could not fully compute statistical summary for 'HasChildren' (missing p-value or coefficient). "
            "Numeric results are provided where available in the 'object' field."
        )

    return {
        "object": numeric_result,
        "description": interpretation
    }