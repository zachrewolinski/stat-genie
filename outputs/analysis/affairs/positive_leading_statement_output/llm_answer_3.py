def extract_final_answer(model_output):
    """
    Extracts statistics for the 'Children' coefficient from a fitted model output dict.
    Expects model_output to be a dict containing one or more of:
      - 'nb_robust' (preferred)
      - 'nb'
      - 'poisson' (fallback)
    Returns a dict:
      - "object": dict with numeric results (coef, se, pvalue, 95% CI, IRR and its 95% CI, model used)
      - "description": short interpreted summary of whether having children is associated
                       with a decrease in extramarital affairs (statistical direction + significance)
    """
    import numpy as np

    # Choose preferred model: nb_robust -> nb -> poisson
    preferred_keys = ['nb_robust', 'nb', 'poisson']
    res = None
    used_key = None
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict containing fitted model results (e.g., {'nb': ..., 'nb_robust': ...}).")

    for k in preferred_keys:
        if k in model_output and model_output[k] is not None:
            res = model_output[k]
            used_key = k
            break

    if res is None:
        raise ValueError(f"No usable model found in model_output. Expected one of {preferred_keys}.")

    # Ensure the model result has the 'Children' parameter
    params = getattr(res, "params", None)
    if params is None:
        # Some wrappers may store params differently; try .params if available, else error
        raise ValueError("Selected model result does not expose .params.")
    if 'Children' not in params.index:
        raise ValueError("The selected model result does not contain a 'Children' coefficient in .params.")

    # Extract coefficient, se, p-value
    coef = float(params.loc['Children'])
    # robust result wrappers should have .bse and .pvalues; fall back if missing
    bse = None
    pval = None
    try:
        bse = float(res.bse.loc['Children'])
    except Exception:
        # try using params.std_errors if present, else set to NaN
        try:
            bse = float(params.loc['Children'].std())  # unlikely helpful; placeholder
        except Exception:
            bse = float('nan')
    try:
        pval = float(res.pvalues.loc['Children'])
    except Exception:
        pval = float('nan')

    # Confidence interval for coefficient (default 95%)
    try:
        ci = res.conf_int(alpha=0.05)
        ci_lower = float(ci.loc['Children', 0])
        ci_upper = float(ci.loc['Children', 1])
    except Exception:
        ci_lower = float('nan')
        ci_upper = float('nan')

    # Exponentiated coefficient: incidence rate ratio (IRR) since NB/Poisson are log-link models
    try:
        irr = float(np.exp(coef))
    except Exception:
        irr = float('nan')
    try:
        irr_ci_lower = float(np.exp(ci_lower))
        irr_ci_upper = float(np.exp(ci_upper))
    except Exception:
        irr_ci_lower = float('nan')
        irr_ci_upper = float('nan')

    # Create a concise interpretation
    significance = None
    if not np.isnan(pval):
        significance = (pval < 0.05)
    else:
        significance = None

    if significance is True:
        sig_text = "statistically significant (p < 0.05)"
    elif significance is False:
        sig_text = "not statistically significant (p >= 0.05)"
    else:
        sig_text = "significance could not be determined (p-value missing)"

    if coef < 0:
        direction_text = "Having children is associated with a decrease in the expected number of extramarital affairs."
    elif coef > 0:
        direction_text = "Having children is associated with an increase in the expected number of extramarital affairs."
    else:
        direction_text = "No association (coefficient is exactly zero)."

    description = (
        f"Model used: '{used_key}'. Coefficient on 'Children' = {coef:.4g} (SE = {bse:.4g}, p = {pval:.4g}). "
        f"95% CI for coefficient = [{ci_lower:.4g}, {ci_upper:.4g}]. "
        f"Exponentiated (IRR) = {irr:.4g}, 95% CI for IRR = [{irr_ci_lower:.4g}, {irr_ci_upper:.4g}]. "
        f"Interpretation: {direction_text} This effect is {sig_text}."
    )

    output_object = {
        "model_used": used_key,
        "coef": coef,
        "std_error": bse,
        "p_value": pval,
        "ci_95_coef": [ci_lower, ci_upper],
        "irr": irr,
        "ci_95_irr": [irr_ci_lower, irr_ci_upper]
    }

    return {
        "object": output_object,
        "description": description
    }