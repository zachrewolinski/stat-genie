def extract_final_answer(model_output):
    import numpy as np

    # Target variable to extract/interprete
    var = 'livebait'

    # Initialize return values
    coef = None
    pval = None
    irr = None
    irr_ci = None

    # Helper to safe float conversion
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # 1) If model object available, try extracting from statsmodels results
    model = None
    if isinstance(model_output, dict):
        model = model_output.get('model', None)
    else:
        model = model_output

    if model is not None and hasattr(model, 'params'):
        # try coefficient and p-value
        try:
            coef = _to_float(model.params[var])
        except Exception:
            coef = None
        try:
            pval = _to_float(model.pvalues[var])
        except Exception:
            pval = None
        # try conf int on coefficient and convert to IRR CI
        try:
            conf = model.conf_int()
            low = _to_float(conf.loc[var, 0])
            high = _to_float(conf.loc[var, 1])
            if low is not None and high is not None:
                irr_ci = (float(np.exp(low)), float(np.exp(high)))
        except Exception:
            irr_ci = None
        # compute IRR from coef if possible
        if coef is not None:
            try:
                irr = float(np.exp(coef))
            except Exception:
                irr = None

    # 2) Fallback: if model_output dict supplied with 'irrs' and 'irrs_ci', use them
    if (irr is None or irr_ci is None) and isinstance(model_output, dict):
        irr_dict = model_output.get('irrs', {})
        irr_ci_dict = model_output.get('irrs_ci', {})
        if irr is None and var in irr_dict:
            irr = _to_float(irr_dict.get(var))
        if irr_ci is None and var in irr_ci_dict:
            # expect a (low, high) pair
            try:
                low, high = irr_ci_dict[var]
                irr_ci = (float(low), float(high))
            except Exception:
                irr_ci = None
        # also try to get coef/pval from model_output summary if model object not present
        if coef is None and 'model' not in model_output and 'summary' in model_output:
            # Not robust parsing; leave coef/pval as None.
            pass

    # Build object to return
    object_dict = {
        'variable': var,
        'coefficient_log_rate': coef,        # log-rate coefficient from GLM
        'p_value': pval,                     # p-value for the coefficient
        'incident_rate_ratio': irr,          # exp(coef)
        'irr_95ci': irr_ci                   # (lower, upper) for IRR
    }

    # Build human-readable description
    parts = []
    parts.append("Outcome and model: Negative Binomial GLM of total fish caught with log(hours) as offset (rate per hour).")
    if coef is not None:
        parts.append(f"Coefficient (log-rate) for '{var}': {coef:.4f}.")
    else:
        parts.append(f"Coefficient (log-rate) for '{var}' not available.")
    if pval is not None:
        parts.append(f"p-value = {pval:.3g}.")
    else:
        parts.append("p-value unavailable.")
    if irr is not None:
        parts.append(f"Estimated IRR = {irr:.3f}")
        if irr_ci is not None:
            parts.append(f"with 95% CI = ({irr_ci[0]:.3f}, {irr_ci[1]:.3f}).")
        else:
            parts.append("with 95% CI unavailable.")
    else:
        parts.append("IRR unavailable.")

    # Interpret effect in plain language
    if irr is not None and pval is not None:
        sig_text = "statistically significant" if pval < 0.05 else "not statistically significant"
        parts.append(f"Interpretation: Visitors who used live bait had an estimated {irr:.2f}-fold rate of fish caught per hour compared to those who did not, holding other predictors constant; this effect is {sig_text}.")
    else:
        parts.append("Interpretation not fully available due to missing statistics.")

    description = " ".join(parts)

    return {"object": object_dict, "description": description}