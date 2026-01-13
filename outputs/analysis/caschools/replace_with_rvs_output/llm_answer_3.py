def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-stat, p-value, 95% CI for the StudentsPerTeacher
    coefficient from a statsmodels RegressionResultsWrapper (with robust SEs already applied).
    Returns a dict with numeric results under "object" and a plain-language interpretation under "description".
    """
    var = 'StudentsPerTeacher'
    res = model_output

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not appear to be a statsmodels results object with 'params'.")

    if var not in res.params.index:
        raise ValueError(f"Variable '{var}' not found in model coefficients. Available vars: {list(res.params.index)}")

    # Extract statistics
    coef = float(res.params[var])
    se = float(res.bse[var]) if hasattr(res, 'bse') else None
    tstat = float(res.tvalues[var]) if hasattr(res, 'tvalues') else None
    pvalue = float(res.pvalues[var]) if hasattr(res, 'pvalues') else None

    # 95% confidence interval
    try:
        ci = res.conf_int().loc[var]
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Significance at alpha = 0.05
    significant = (pvalue is not None) and (pvalue < 0.05)

    # Interpret direction relative to the research question:
    # StudentsPerTeacher = number of students per teacher.
    # A negative coefficient means that increasing StudentsPerTeacher (more students per teacher)
    # is associated with lower AvgScore; equivalently, a lower StudentsPerTeacher (fewer students per teacher)
    # is associated with higher AvgScore.
    if significant:
        if coef < 0:
            conclusion = (
                "Yes — statistically significant evidence that a lower student-teacher ratio "
                "(fewer students per teacher) is associated with higher average academic performance. "
                f"(coef = {coef:.4f}, SE = {se:.4f}, t = {tstat:.3f}, p = {pvalue:.3g}, "
                f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}])"
            )
        else:
            conclusion = (
                "No — statistically significant evidence in the opposite direction: higher student-teacher "
                "ratios (more students per teacher) are associated with higher average scores. "
                f"(coef = {coef:.4f}, SE = {se:.4f}, t = {tstat:.3f}, p = {pvalue:.3g}, "
                f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}])"
            )
    else:
        conclusion = (
            "No statistically significant association between student-teacher ratio and average academic performance "
            f"was detected (coef = {coef:.4f}, SE = {se:.4f}, t = {tstat:.3f}, p = {pvalue:.3g}, "
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]). "
            "The data do not provide reliable evidence that lowering the student-teacher ratio changes average scores."
        )

    # Prepare object to return (numbers for programmatic use)
    result_object = {
        "variable": var,
        "coef": coef,
        "std_error": se,
        "t_stat": tstat,
        "p_value": pvalue,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "significant_at_0.05": significant,
        "conclusion": conclusion
    }

    return {"object": result_object, "description": conclusion}