def extract_final_answer(model_output):
    """
    Extracts statistics for the effect of skin tone (skin_score) on red card rates
    from the provided model_output (expected to be a dict with keys 'model' and/or 'rate_ratios').

    Returns:
      {
        "object": { ... numeric results ... },
        "description": "Brief interpretation in plain language"
      }
    """
    import numpy as np
    import pandas as pd

    # Prepare empty result
    result_object = {}
    description = ""

    # Try to get the fitted model result object
    res = None
    if isinstance(model_output, dict):
        res = model_output.get('model', None)
        rate_ratios = model_output.get('rate_ratios', None)
    else:
        res = model_output
        rate_ratios = None

    # Helper to safely pull values from the statsmodels result
    def safe_get_from_res(res, name):
        try:
            coef = float(res.params[name])
        except Exception:
            coef = None
        try:
            se = float(res.bse[name])
        except Exception:
            se = None
        try:
            pval = float(res.pvalues[name])
        except Exception:
            pval = None
        try:
            ci = res.conf_int().loc[name].astype(float)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower, ci_upper = None, None
        return coef, se, pval, ci_lower, ci_upper

    # Primary target: continuous skin_score
    name = 'skin_score'
    if res is not None:
        coef, se, pval, ci_low, ci_high = safe_get_from_res(res, name)
    else:
        coef = se = pval = ci_low = ci_high = None

    # If any pieces missing, try to read from provided rate_ratios table (which stores IRR and CI)
    irr = irr_ci_low = irr_ci_high = None
    if rate_ratios is not None:
        try:
            rr_df = rate_ratios.copy()
            if name in rr_df.index:
                irr = float(rr_df.loc[name, 'irr'])
                irr_ci_low = float(rr_df.loc[name, 'ci_lower'])
                irr_ci_high = float(rr_df.loc[name, 'ci_upper'])
        except Exception:
            pass

    # If we have coef from model, compute IRR and CI for IRR
    if coef is not None:
        irr = float(np.exp(coef))
        if ci_low is not None and ci_high is not None:
            irr_ci_low = float(np.exp(ci_low))
            irr_ci_high = float(np.exp(ci_high))

    # If we only have IRR from rate_ratios but no coef, try to invert
    if coef is None and irr is not None:
        try:
            coef = float(np.log(irr))
            if irr_ci_low is not None and irr_ci_high is not None:
                ci_low = float(np.log(irr_ci_low))
                ci_high = float(np.log(irr_ci_high))
        except Exception:
            pass

    # Also check interaction term (skin_score_x_meanIAT) for whether moderation was important
    interaction_name = 'skin_score_x_meanIAT'
    inter_coef = inter_pval = inter_irr = inter_ci_low = inter_ci_high = None
    if res is not None:
        try:
            inter_coef, inter_se, inter_pval, inter_ci_low, inter_ci_high = safe_get_from_res(res, interaction_name)
            if inter_coef is not None:
                inter_irr = float(np.exp(inter_coef))
        except Exception:
            inter_coef = inter_pval = inter_irr = inter_ci_low = inter_ci_high = None
    else:
        # try rate_ratios table
        if rate_ratios is not None and interaction_name in rate_ratios.index:
            try:
                inter_irr = float(rate_ratios.loc[interaction_name, 'irr'])
                inter_ci_low = float(rate_ratios.loc[interaction_name, 'ci_lower'])
                inter_ci_high = float(rate_ratios.loc[interaction_name, 'ci_upper'])
                inter_coef = float(np.log(inter_irr))
            except Exception:
                pass

    # Compose the object with numeric outputs
    result_object = {
        'term': name,
        'coef_log_rate': coef,                 # coefficient on log scale
        'se_log_rate': se,
        'p_value': pval,
        'ci_log_lower': ci_low,
        'ci_log_upper': ci_high,
        'irr': irr,                            # incidence rate ratio = exp(coef)
        'irr_ci_lower': irr_ci_low,
        'irr_ci_upper': irr_ci_high,
        # interaction (if present)
        'interaction_term': interaction_name,
        'interaction_coef_log_rate': inter_coef,
        'interaction_p_value': inter_pval,
        'interaction_irr': inter_irr,
        'interaction_irr_ci_lower': inter_ci_low,
        'interaction_irr_ci_upper': inter_ci_high,
    }

    # Construct a concise interpretation
    # Base conclusion about statistical evidence
    if irr is not None and (irr_ci_low is not None and irr_ci_high is not None):
        # Check whether CI includes 1
        if irr_ci_low > 1:
            sig_text = "statistically significantly associated with higher red card rates (IRR > 1)."
        elif irr_ci_high < 1:
            sig_text = "statistically significantly associated with lower red card rates (IRR < 1)."
        else:
            sig_text = "not statistically significantly associated with red card rates (CI includes 1)."
    else:
        sig_text = "could not determine significance from the provided output."

    # Build description string
    desc_lines = []
    desc_lines.append(f"Effect of skin_score (darker = higher): IRR = {irr:.3f}" if irr is not None else "IRR: not available")
    if irr_ci_low is not None and irr_ci_high is not None:
        desc_lines.append(f"95% CI for IRR = [{irr_ci_low:.3f}, {irr_ci_high:.3f}]")
    if pval is not None:
        desc_lines.append(f"p-value = {pval:.3g}")
    desc_lines.append(f"Interpretation: The estimated effect is {sig_text}")
    # Note about moderation if present
    if inter_coef is not None:
        if inter_pval is not None and inter_pval < 0.05:
            desc_lines.append("The interaction with meanIAT is statistically significant, so the association between skin_score and red cards depends on referee-country implicit bias (meanIAT).")
        else:
            desc_lines.append("The interaction with meanIAT is not statistically significant, so there is no evidence that country-level implicit bias (meanIAT) modifies the skin_score association.")
    else:
        desc_lines.append("No interaction estimate available in the output / interaction term not present.")

    description = " ".join(desc_lines)

    return {
        "object": result_object,
        "description": description
    }