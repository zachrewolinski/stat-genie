def extract_final_answer(model_output):
    """
    Extracts key statistics from a fitted statsmodels GLM/GLMResultsWrapper that modeled
    fish counts with log(hours) as an offset.

    Returns a dictionary with:
      - "object": a dict containing numeric results (baseline rate per hour, IRRs, CIs, p-values, dispersion, model family)
      - "description": a short plain-language interpretation of the key results

    The function expects `model_output` to be a statsmodels results object (e.g. GLMResultsWrapper).
    """
    import numpy as np

    res = model_output

    # Basic parameter table
    params = res.params  # pandas Series
    bse = getattr(res, 'bse', None)
    pvalues = getattr(res, 'pvalues', None)
    conf_int = getattr(res, 'conf_int', None)
    if conf_int is not None:
        conf_int = res.conf_int()  # DataFrame with two columns [lower, upper]

    # Identify intercept (constant)
    if 'const' in params.index:
        intercept_name = 'const'
    else:
        # fallback to first parameter if constant named differently
        intercept_name = params.index[0]

    intercept_coef = float(params.loc[intercept_name])
    intercept_ci = None
    if conf_int is not None and intercept_name in conf_int.index:
        intercept_ci = conf_int.loc[intercept_name].values.astype(float)

    # Baseline rate per hour (when predictors = 0)
    # For a model with offset log_hours: rate_per_hour = exp(intercept + beta*x). At x=0 this is exp(intercept).
    baseline_rate = float(np.exp(intercept_coef))
    baseline_ci_exp = None
    if intercept_ci is not None:
        baseline_ci_exp = np.exp(intercept_ci).tolist()

    # Prepare predictor summaries (exclude intercept)
    predictor_names = [n for n in params.index if n != intercept_name]
    predictors = {}
    for name in predictor_names:
        coef = float(params.loc[name])
        se = float(bse.loc[name]) if (bse is not None and name in bse.index) else None
        pval = float(pvalues.loc[name]) if (pvalues is not None and name in pvalues.index) else None
        ci = conf_int.loc[name].values.astype(float).tolist() if (conf_int is not None and name in conf_int.index) else None

        # Incidence Rate Ratio (IRR) and CI on rate scale
        irr = float(np.exp(coef))
        irr_ci = np.exp(ci).tolist() if ci is not None else None

        predictors[name] = {
            'coef_log_rate': coef,
            'se': se,
            'p_value': pval,
            'coef_95CI_log_scale': ci,
            'IRR_rate_ratio': irr,
            'IRR_95CI': irr_ci,
            'interpretation': (
                "e^(coef) = IRR: multiplicative change in fish-catch rate per hour "
                "for a one-unit increase in this predictor, holding others constant."
            )
        }

    # Dispersion info (attached by model-fitting function when Poisson initial fit was done)
    dispersion = getattr(res, 'dispersion', None)
    # If negative binomial was returned after detecting overdispersion, the Poisson dispersion may be available under a different attribute
    dispersion_from_poisson = getattr(res, 'dispersion_from_poisson', None)
    model_family = getattr(res.model.family, '__class__', None)
    model_family_name = model_family.__name__ if model_family is not None else str(getattr(res.model, 'family', 'Unknown'))

    # Build the returned object
    result_object = {
        'model_family': model_family_name,
        'baseline_rate_per_hour': baseline_rate,
        'baseline_rate_per_hour_95CI': baseline_ci_exp,
        'predictors': predictors,
        'dispersion': dispersion,
        'dispersion_from_poisson': dispersion_from_poisson,
        'note': (
            "Coefficients are on the log-rate scale. Exponentiated coefficients (IRR) "
            "give multiplicative effects on fish caught per hour. Baseline rate is the "
            "expected fish per hour when all predictors (after centering) equal zero."
        )
    }

    # Short human-readable description
    # Mention how to interpret key numbers succinctly
    description_lines = [
        f"Model family: {model_family_name}.",
        f"Baseline (when predictors = 0): expected {baseline_rate:.3f} fish per hour"
    ]
    if baseline_ci_exp is not None:
        description_lines.append(f"(95% CI: {baseline_ci_exp[0]:.3f} to {baseline_ci_exp[1]:.3f}).")
    else:
        description_lines.append(".")
    description_lines.append("Predictor effects (IRR = multiplicative change in rate per hour):")
    for name, info in predictors.items():
        pstr = f"p={info['p_value']:.3g}" if info['p_value'] is not None else "p=NA"
        ci_str = (
            f"95%CI IRR [{info['IRR_95CI'][0]:.3f}, {info['IRR_95CI'][1]:.3f}]"
            if info['IRR_95CI'] is not None else "95%CI NA"
        )
        description_lines.append(
            f" - {name}: IRR = {info['IRR_rate_ratio']:.3f}, {ci_str}, {pstr}."
        )
    if dispersion is not None:
        description_lines.append(f"Dispersion (Pearson chi2 / df) reported: {dispersion:.3f}.")
    if dispersion_from_poisson is not None:
        description_lines.append(f"Dispersion from initial Poisson fit: {dispersion_from_poisson:.3f} (used to choose NB).")

    description = " ".join(description_lines)

    return {
        "object": result_object,
        "description": description
    }