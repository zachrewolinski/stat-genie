def extract_final_answer(model_output):
    """
    Extracts the coefficient and related statistics for the GenusHuman predictor
    from a fitted statsmodels GLMResultsWrapper (or its robustcov result).
    
    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, z/t, p, conf_int, odds_ratio, or_conf_int)
      - "description": human-readable interpretation in the context of the task.
    """
    import numpy as np
    import pandas as pd

    # Obtain parameter names and try to find the GenusHuman-related parameter.
    params = None
    try:
        params = model_output.params
    except Exception:
        raise ValueError("model_output does not expose .params; provide a fitted statsmodels results object.")
    # Look for parameter that includes 'GenusHuman' (covers different naming conventions)
    gen_param = None
    for name in params.index:
        if "GenusHuman" in name:
            gen_param = name
            break
    if gen_param is None:
        # try exact fallback
        if "GenusHuman" in params.index:
            gen_param = "GenusHuman"
        else:
            raise ValueError("Could not find a parameter containing 'GenusHuman' in model params: {}".format(list(params.index)))
    # Extract coefficient (log-odds scale)
    coef = float(params[gen_param])

    # Standard error: try .bse, else derive from covariance matrix
    try:
        se = float(model_output.bse[gen_param])
    except Exception:
        try:
            cov = model_output.cov_params()
            se = float(np.sqrt(np.abs(cov.loc[gen_param, gen_param])))
        except Exception:
            se = None

    # z / t value and p-value (depending on the results object these may be named differently)
    zval = None
    pval = None
    try:
        if gen_param in model_output.tvalues.index:
            # some wrappers expose tvalues
            zval = float(model_output.tvalues[gen_param])
        elif gen_param in model_output.zvalues.index:
            zval = float(model_output.zvalues[gen_param])
    except Exception:
        # try generic attributes
        pass
    try:
        pval = float(model_output.pvalues[gen_param])
    except Exception:
        pval = None

    # Confidence interval on the coefficient (log-odds)
    try:
        ci_df = model_output.conf_int()
        ci_low, ci_high = float(ci_df.loc[gen_param, 0]), float(ci_df.loc[gen_param, 1])
    except Exception:
        ci_low, ci_high = None, None

    # Convert to odds ratio and CI on odds ratio scale
    or_val = float(np.exp(coef))
    or_ci = (np.exp(ci_low) if ci_low is not None else None, np.exp(ci_high) if ci_high is not None else None)

    # Simple significance label
    significance = None
    if pval is not None:
        significance = "statistically significant (p < 0.05)" if pval < 0.05 else "not statistically significant (p >= 0.05)"

    # Directional interpretation
    if coef > 0:
        direction = "Higher AMTL odds in modern humans (Homo sapiens) compared to the reference non-human primates, holding age, prob_male, and tooth class constant."
    elif coef < 0:
        direction = "Lower AMTL odds in modern humans (Homo sapiens) compared to the reference non-human primates, holding age, prob_male, and tooth class constant."
    else:
        direction = "No difference in AMTL odds between modern humans and non-human primates."

    result_object = {
        "parameter_name": gen_param,
        "coef_log_odds": coef,
        "std_error": se,
        "z_or_t_value": zval,
        "p_value": pval,
        "conf_int_log_odds": (ci_low, ci_high),
        "odds_ratio": or_val,
        "conf_int_odds_ratio": or_ci,
        "significance": significance,
        "direction_interpretation": direction,
    }

    # Description explaining what the extracted values mean
    description_lines = [
        f"Extracted parameter '{gen_param}' which represents the effect of being a modern human (GenusHuman) on the log-odds of antemortem tooth loss (AMTL).",
        f"Coefficient (log-odds): {coef:.4f}. This is the change in log-odds of a socket being AMTL for modern humans vs non-human primates, controlling for age (age_c), estimated male probability (prob_male), and tooth_class.",
        f"Standard error: {se if se is not None else 'NA'}.",
        f"Z/T statistic: {zval if zval is not None else 'NA'}. p-value: {pval if pval is not None else 'NA'}.",
        f"95% CI on log-odds: ({ci_low:.4f}, {ci_high:.4f})" if (ci_low is not None and ci_high is not None) else "95% CI on log-odds: NA",
        f"Odds ratio = exp(coef): {or_val:.3f}. 95% CI on odds ratio: ({or_ci[0]:.3f}, {or_ci[1]:.3f})" if (or_ci[0] is not None and or_ci[1] is not None) else f"Odds ratio = {or_val:.3f}. CI: NA",
        f"Interpretation: {direction}",
        f"Statistical significance: {significance if significance is not None else 'NA'}",
        "A statistically significant positive coefficient (and OR > 1) would support the hypothesis that modern humans have higher AMTL frequencies than the non-human primates considered, after accounting for age, sex uncertainty, and tooth class."
    ]
    description = " ".join(description_lines)

    return {"object": result_object, "description": description}