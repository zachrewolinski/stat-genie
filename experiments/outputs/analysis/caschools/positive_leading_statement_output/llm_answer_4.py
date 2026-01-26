def extract_final_answer(model_output):
    """
    Extract key statistics for the StudentTeacherRatio coefficient from a fitted
    statsmodels RegressionResultsWrapper and produce a brief interpretation
    relevant to the question:
      "Is a lower student-teacher ratio associated with higher academic performance?"
    
    Returns:
      dict with keys:
        - "object": dict with extracted numeric results:
            - coef: estimated coefficient (change in AvgScore per 1 additional student per teacher)
            - p_value: p-value for the coefficient (HC3 or model-supplied)
            - ci_lower, ci_upper: 95% confidence interval bounds
            - n_obs: number of observations used in the regression (if available)
            - significant_0.05: boolean, True if p_value < 0.05
            - coef_per_10_students: estimated change in AvgScore per 10 additional students per teacher
        - "description": short interpretation tying sign and significance to the task question.
    """
    # Defensive checks for expected attributes
    if not hasattr(model_output, 'params') or not hasattr(model_output, 'pvalues'):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params and .pvalues")
    
    var = 'StudentTeacherRatio'
    try:
        coef = float(model_output.params[var])
        p_value = float(model_output.pvalues[var])
    except Exception as e:
        raise KeyError(f"Could not find coefficient or p-value for variable '{var}' in model_output: {e}")
    
    # Confidence interval (95% default)
    try:
        ci = model_output.conf_int().loc[var]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # fallback if conf_int returns array-like without index
        try:
            ci_arr = model_output.conf_int()
            # find row index matching var name
            idx = list(model_output.params.index).index(var)
            ci_lower = float(ci_arr[idx, 0])
            ci_upper = float(ci_arr[idx, 1])
        except Exception as e:
            ci_lower = None
            ci_upper = None
    
    # Number of observations if available
    n_obs = None
    if hasattr(model_output, 'nobs'):
        try:
            n_obs = int(model_output.nobs)
        except Exception:
            n_obs = None
    
    significant_0_05 = (p_value < 0.05)
    coef_per_10 = coef * 10.0  # change in AvgScore per 10 additional students per teacher
    
    # Interpretation: negative coefficient implies higher StudentTeacherRatio (more students per teacher)
    # is associated with lower AvgScore. That means lower ratio (fewer students per teacher) is associated
    # with higher AvgScore.
    if coef < 0 and significant_0_05:
        conclusion = (
            "Yes. The StudentTeacherRatio coefficient is negative and statistically significant "
            "(p < 0.05), indicating that higher student-teacher ratios (more students per teacher) "
            "are associated with lower average test scores — equivalently, lower ratios (smaller class sizes) "
            "are associated with higher academic performance."
        )
    elif coef < 0 and not significant_0_05:
        conclusion = (
            "The StudentTeacherRatio coefficient is negative but not statistically significant (p >= 0.05). "
            "The point estimate suggests that lower student-teacher ratios may be associated with higher scores, "
            "but the evidence is weak/inconclusive at conventional significance levels."
        )
    elif coef > 0 and significant_0_05:
        conclusion = (
            "No. The StudentTeacherRatio coefficient is positive and statistically significant (p < 0.05), "
            "indicating that higher student-teacher ratios are associated with higher average test scores — "
            "i.e., in this model, lower ratios are associated with lower performance."
        )
    else:  # coef > 0 and not significant
        conclusion = (
            "The StudentTeacherRatio coefficient is positive but not statistically significant (p >= 0.05). "
            "There is no strong evidence of an association between student-teacher ratio and average test scores."
        )
    
    result_object = {
        "coef": coef,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": n_obs,
        "significant_0.05": significant_0_05,
        "coef_per_10_students": coef_per_10
    }
    
    return {
        "object": result_object,
        "description": conclusion
    }