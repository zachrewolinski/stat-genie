def extract_final_answer(model_output):
    """
    Extracts the StudentsPerTeacher_z coefficient and inference from a statsmodels RegressionResultsWrapper.
    Returns a dict with keys:
      - "object": dict with numeric results and a boolean answer to the yes/no question
      - "description": brief human-readable interpretation

    The yes/no question: "Is a lower student-teacher ratio associated with higher academic performance?"
    This function answers that by checking the sign of the StudentsPerTeacher_z coefficient and its statistical significance
    (two-sided test at alpha=0.05). Note: StudentsPerTeacher_z is coded so LOWER values = fewer students per teacher (smaller classes).
    """
    res = model_output

    # Basic validation
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params)")

    param_name = 'StudentsPerTeacher_z'
    if param_name not in res.params.index:
        raise ValueError(f"Parameter '{param_name}' not found in model parameters.")

    # Extract statistics
    coef = float(res.params[param_name])
    se = float(res.bse[param_name]) if param_name in res.bse.index else None
    pvalue = float(res.pvalues[param_name]) if param_name in res.pvalues.index else None

    # Confidence interval (attempt DataFrame indexing, fallback to ndarray indexing)
    ci_df = res.conf_int(alpha=0.05)
    try:
        ci_low = float(ci_df.loc[param_name, 0])
        ci_high = float(ci_df.loc[param_name, 1])
    except Exception:
        # fallback if conf_int returned a numpy array
        idx = list(res.params.index).index(param_name)
        ci_low = float(ci_df[idx, 0])
        ci_high = float(ci_df[idx, 1])

    # Sample size if available
    nobs = int(getattr(res, 'nobs', getattr(res, 'nobsobs', float('nan'))))

    # Interpretation relative to the research question:
    # - coef < 0: increasing StudentsPerTeacher_z (more students per teacher = larger classes) -> lower CompositeScore.
    #   Therefore, LOWER StudentsPerTeacher_z (fewer students per teacher = smaller classes) -> HIGHER CompositeScore.
    # - coef > 0: opposite direction.
    alpha = 0.05
    significant = (pvalue is not None) and (pvalue < alpha)
    lower_ratio_assoc_higher_performance = None
    if coef < 0:
        lower_ratio_assoc_higher_performance = True
    elif coef > 0:
        lower_ratio_assoc_higher_performance = False
    else:
        lower_ratio_assoc_higher_performance = None  # exactly zero

    # Build machine-readable object
    result_object = {
        'parameter': param_name,
        'coef': coef,
        'std_err': se,
        'p_value': pvalue,
        'conf_int_95': (ci_low, ci_high),
        'nobs': nobs,
        'alpha': alpha,
        'significant': significant,
        # Boolean answer to the question (True means: lower student-teacher ratio associated with higher performance)
        'lower_ratio_associated_with_higher_performance': lower_ratio_assoc_higher_performance
    }

    # Human-readable description
    if lower_ratio_assoc_higher_performance is True:
        direction_text = ("The estimated effect is negative: higher StudentsPerTeacher_z (more students per teacher, "
                          "i.e., larger classes) is associated with lower CompositeScore. Equivalently, lower "
                          "StudentsPerTeacher_z (fewer students per teacher, smaller classes) is associated with "
                          "higher academic performance.")
    elif lower_ratio_assoc_higher_performance is False:
        direction_text = ("The estimated effect is positive: higher StudentsPerTeacher_z (more students per teacher, "
                          "i.e., larger classes) is associated with higher CompositeScore. Equivalently, lower "
                          "StudentsPerTeacher_z (fewer students per teacher, smaller classes) would be associated with "
                          "lower academic performance.")
    else:
        direction_text = "The estimated effect is effectively zero."

    sig_text = ("This association is statistically significant at alpha = 0.05."
                if significant else
                "This association is NOT statistically significant at alpha = 0.05.")

    description = (
        f"StudentsPerTeacher_z coefficient = {coef:.4f} (SE = {se:.4f}), 95% CI = [{ci_low:.4f}, {ci_high:.4f}], "
        f"p = {pvalue:.4f}, n = {nobs}. {direction_text} {sig_text}"
    )

    return {"object": result_object, "description": description}