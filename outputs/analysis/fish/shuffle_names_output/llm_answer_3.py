def extract_final_answer(model_output):
    """
    Extracts the effect of 'livebait' on fish-per-hour rate from a fitted statsmodels GLM results object.
    Returns a dictionary with:
      - "object": dict containing coefficient, SE, p-value, 95% CI (log scale), IRR and its 95% CI,
                  baseline rate per hour when predictors are zero (if intercept present), and model family name.
      - "description": human-readable interpretation of the 'livebait' effect in context.
    """
    import numpy as np

    res = model_output

    # Ensure expected attributes exist
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not look like a statsmodels results object (no .params).")

    params = res.params
    if 'livebait' not in params.index:
        raise ValueError("The model does not contain a parameter named 'livebait'.")

    # Extract coefficient, se, p-value
    coef = float(params['livebait'])
    se = float(res.bse['livebait']) if hasattr(res, 'bse') and 'livebait' in res.bse.index else None
    pval = float(res.pvalues['livebait']) if hasattr(res, 'pvalues') and 'livebait' in res.pvalues.index else None

    # Confidence interval on log scale
    try:
        ci_row = res.conf_int().loc['livebait']
        ci_lower = float(ci_row[0])
        ci_upper = float(ci_row[1])
    except Exception:
        ci_lower = ci_upper = None

    # Incidence Rate Ratio (IRR) and CI on original rate scale
    irr = float(np.exp(coef))
    irr_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
    irr_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None

    # Baseline rate per hour (when predictors = 0) if intercept present
    intercept_name = None
    for name in ['Intercept', 'const', 'intercept']:
        if name in params.index:
            intercept_name = name
            break

    baseline_rate = None
    if intercept_name is not None:
        intercept = float(params[intercept_name])
        # Because the model used log(hours) as an offset, the intercept corresponds to log(rate) when predictors=0
        baseline_rate = float(np.exp(intercept))

    # Model family name
    fam_name = None
    if hasattr(res, 'family') and res.family is not None:
        fam_name = res.family.__class__.__name__

    result_object = {
        "livebait_coef_log_rate_per_hour": coef,
        "se": se,
        "p_value": pval,
        "95%_ci_log_scale": [ci_lower, ci_upper],
        "IRR_livebait": irr,
        "95%_ci_IRR": [irr_ci_lower, irr_ci_upper],
        "baseline_rate_per_hour_when_predictors_zero": baseline_rate,
        "model_family": fam_name
    }

    # Human-readable description
    desc_parts = []
    if fam_name:
        desc_parts.append(f"Model family: {fam_name}.")
    desc_parts.append("The model used log(hours) as an offset, so coefficients represent multiplicative effects on fish-per-hour.")
    desc_parts.append(
        f"The estimated coefficient for 'livebait' (log rate ratio) = {coef:.4g}"
        + (f" (SE = {se:.4g})" if se is not None else "") 
        + (f", p = {pval:.4g}" if pval is not None else "") + "."
    )
    desc_parts.append(
        f"Exponentiating gives an incidence rate ratio (IRR) = {irr:.4g}"
        + (f" with 95% CI [{irr_ci_lower:.4g}, {irr_ci_upper:.4g}]" if irr_ci_lower is not None else "") + "."
    )
    desc_parts.append(
        "Interpretation: visitors using live bait are estimated to catch approximately "
        f"{irr:.2f} times as many fish per hour as visitors not using live bait, "
        "holding other model covariates constant."
    )
    if baseline_rate is not None:
        desc_parts.append(f"Baseline catch rate when predictors = 0 is approx. {baseline_rate:.4g} fish per hour (intercept).")

    description = " ".join(desc_parts)

    return {"object": result_object, "description": description}