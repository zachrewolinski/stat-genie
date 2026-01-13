def extract_final_answer(model_output):
    """
    Extract coefficient, standard error, t-stat, p-value, and 95% CI for
    'student_teacher_ratio' from a statsmodels RegressionResultsWrapper, and
    produce a short interpretation regarding whether a lower student-teacher
    ratio is associated with higher academic performance.

    Returns:
      dict with keys:
        - "object": dict containing numeric results (coef, std_err, t, p_value, ci_lower, ci_upper)
        - "description": short interpretation in the context of the task
    """
    try:
        params = model_output.params
        bse = model_output.bse
        tvals = model_output.tvalues
        pvals = model_output.pvalues
        ci = model_output.conf_int(alpha=0.05)
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract results from model_output: {e}"
        }

    var = 'student_teacher_ratio'
    if var not in params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in model parameters."
        }

    coef = float(params[var])
    std_err = float(bse[var]) if var in bse.index else None
    t_stat = float(tvals[var]) if var in tvals.index else None
    p_value = float(pvals[var]) if var in pvals.index else None
    try:
        ci_row = ci.loc[var]
        ci_lower = float(ci_row.iloc[0])
        ci_upper = float(ci_row.iloc[1])
    except Exception:
        ci_lower = ci_upper = None

    # Interpretation: recall student_teacher_ratio = students / teachers.
    # A negative coefficient implies that a lower ratio (fewer students per teacher)
    # is associated with higher avg_test_score.
    alpha = 0.05
    if p_value is None:
        conclusion = "Could not determine statistical significance (p-value unavailable)."
    else:
        if p_value < alpha:
            if coef < 0:
                conclusion = (
                    f"Yes — statistically significant (p = {p_value:.3g}). Coefficient = {coef:.4f} "
                    f"(95% CI [{ci_lower:.4f}, {ci_upper:.4f}]) indicates that a lower student-teacher "
                    "ratio (fewer students per teacher) is associated with higher average test scores."
                )
            else:
                conclusion = (
                    f"No — statistically significant (p = {p_value:.3g}). Coefficient = {coef:.4f} "
                    f"(95% CI [{ci_lower:.4f}, {ci_upper:.4f}]) indicates that a higher student-teacher "
                    "ratio is associated with higher average test scores (opposite of the hypothesized direction)."
                )
        else:
            if coef < 0:
                conclusion = (
                    f"No strong evidence (p = {p_value:.3g}). Coefficient = {coef:.4f} "
                    f"(95% CI [{ci_lower:.4f}, {ci_upper:.4f}]) suggests a negative relationship "
                    "(lower ratio -> higher scores) but it is not statistically significant at α = 0.05."
                )
            else:
                conclusion = (
                    f"No strong evidence (p = {p_value:.3g}). Coefficient = {coef:.4f} "
                    f"(95% CI [{ci_lower:.4f}, {ci_upper:.4f}]) suggests a positive relationship "
                    "but it is not statistically significant at α = 0.05."
                )

    result_object = {
        "coef": coef,
        "std_err": std_err,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

    return {"object": result_object, "description": conclusion}