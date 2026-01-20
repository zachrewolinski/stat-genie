def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted
    statsmodels RegressionResultsWrapper and provides an interpretation.
    
    Returns a dictionary with keys:
      - "object": dict with numeric results (coef, se, t, p, 95% CI, standardized coef)
      - "description": text interpreting whether lower student-teacher ratio is
                       associated with higher academic performance in this model.
    """
    import numpy as np

    # Name of focal variable used in the model code
    focal = 'StudentTeacherRatio'
    
    # Basic checks
    if not hasattr(model_output, 'params'):
        raise ValueError("model_output does not look like a statsmodels fitted result (missing .params).")
    if focal not in model_output.params.index:
        raise KeyError(f"Focal variable '{focal}' not found in model parameters. Available params: {list(model_output.params.index)}")

    # Coefficient, se, t, p
    coef = float(model_output.params[focal])
    # statsmodels stores bse, tvalues, pvalues keyed by variable name
    se = float(model_output.bse[focal]) if hasattr(model_output, 'bse') else None
    tval = float(model_output.tvalues[focal]) if hasattr(model_output, 'tvalues') else None
    pval = float(model_output.pvalues[focal]) if hasattr(model_output, 'pvalues') else None

    # 95% CI
    try:
        ci = model_output.conf_int(alpha=0.05).loc[focal].values
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Standardized coefficient: beta_std = coef * (sd_x / sd_y)
    # Get endogenous (y) and exogenous (X) data from the model object if available
    std_coef = None
    try:
        endog = np.asarray(model_output.model.endog).ravel()
        exog_names = list(model_output.model.exog_names)
        if focal in exog_names:
            col_index = exog_names.index(focal)
            exog = np.asarray(model_output.model.exog)
            x = exog[:, col_index]
            sd_x = np.nanstd(x, ddof=1)
            sd_y = np.nanstd(endog, ddof=1)
            if sd_x > 0 and sd_y > 0:
                std_coef = float(coef * (sd_x / sd_y))
    except Exception:
        std_coef = None

    # Decision rule: coefficient negative and statistically significant at alpha=0.05
    supports_hypothesis = None
    if pval is not None:
        if coef < 0 and pval < 0.05:
            supports_hypothesis = True
        else:
            supports_hypothesis = False

    # Prepare returned object
    result_obj = {
        'variable': focal,
        'coef': coef,
        'std_error': se,
        't_value': tval,
        'p_value': pval,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'standardized_coef': std_coef,
        'supports_hypothesis_at_0.05': supports_hypothesis
    }

    # Human-readable description
    # Interpret direction: StudentTeacherRatio is students per teacher. A negative coef
    # means higher ratio -> lower test scores (so lower ratio -> higher scores).
    direction_desc = ""
    if coef < 0:
        direction_desc = (
            "Coefficient is negative, so increasing the student-teacher ratio (more students per teacher) "
            "is associated with lower average test scores — equivalently, a lower student-teacher ratio "
            "(fewer students per teacher) is associated with higher performance."
        )
    elif coef > 0:
        direction_desc = (
            "Coefficient is positive, so increasing the student-teacher ratio (more students per teacher) "
            "is associated with higher average test scores — this is opposite the hypothesis."
        )
    else:
        direction_desc = "Coefficient is (approximately) zero, indicating no association in the point estimate."

    significance_desc = ""
    if pval is None:
        significance_desc = "P-value is unavailable, so statistical significance cannot be assessed."
    else:
        significance_desc = (
            f"P-value = {pval:.4g}. "
            + ("This effect is statistically significant at the 0.05 level." if pval < 0.05 else "This effect is not statistically significant at the 0.05 level.")
        )

    std_desc = ""
    if std_coef is not None:
        std_desc = f"The standardized coefficient is {std_coef:.4f}, i.e. a one-SD increase in ratio is associated with a {std_coef:.4f}-SD change in AvgTestScore."
    else:
        std_desc = "Standardized coefficient could not be computed from the model object."

    description = (
        f"Estimated effect of {focal} on AvgTestScore: coefficient = {coef:.4f} "
        f"(SE = {se:.4f}, t = {tval:.3f}, p = {pval:.4g}), 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        + direction_desc + " " + significance_desc + " " + std_desc
    )

    return {"object": result_obj, "description": description}