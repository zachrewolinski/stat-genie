def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels
    regression results object and provides a concise interpretation answering whether
    a lower student-teacher ratio is associated with higher academic performance.

    Returns a dictionary with keys:
      - "object": a dict of extracted numeric results and the inferred conclusion
      - "description": a short human-readable explanation of the numbers and conclusion
    """
    # Ensure model_output looks like a statsmodels RegressionResultsWrapper
    # and contains the coefficient for StudentTeacherRatio
    var = 'StudentTeacherRatio'
    try:
        params = model_output.params
    except Exception as e:
        raise ValueError("Provided model_output does not appear to be a statsmodels results object.") from e

    if var not in params.index:
        raise ValueError(f"Variable '{var}' not found in the model results. Available variables: {list(params.index)}")

    # Extract statistics
    coef = float(params[var])
    # robust standard errors are available via bse
    se = float(model_output.bse[var]) if hasattr(model_output, 'bse') else None
    tstat = float(model_output.tvalues[var]) if hasattr(model_output, 'tvalues') else None
    pval = float(model_output.pvalues[var]) if hasattr(model_output, 'pvalues') else None

    # 95% confidence interval
    try:
        ci = model_output.conf_int(alpha=0.05).loc[var].tolist()
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Determine statistical significance (two-sided alpha=0.05) and direction
    significant = (pval is not None) and (pval < 0.05)
    # Interpretation: StudentTeacherRatio = students per teacher; lower values = fewer students per teacher.
    if coef < 0:
        direction = "Lower student-teacher ratio (fewer students per teacher) is associated with HIGHER AvgScore"
    elif coef > 0:
        direction = "Lower student-teacher ratio (fewer students per teacher) is associated with LOWER AvgScore (i.e., higher ratio -> higher AvgScore)"
    else:
        direction = "No directional association (coefficient is zero)"

    if significant:
        conclusion = "Yes, there is statistically significant evidence supporting the association described by the sign of the coefficient."
    else:
        conclusion = "No, there is not statistically significant evidence at the 5% level to conclude an association."

    # Build returned object
    result_obj = {
        "variable": var,
        "coef": coef,
        "std_error": se,
        "t_stat": tstat,
        "p_value": pval,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "significant_at_0.05": significant,
        "direction_interpretation": direction,
        "conclusion": conclusion,
        # Practical interpretation: change in AvgScore per one-unit increase in StudentTeacherRatio
        "practical_interpretation": (
            f"A one-unit increase in StudentTeacherRatio is associated with a {coef:.3f} point change in AvgScore."
            if coef is not None else None
        )
    }

    # Short human-readable description
    description = (
        f"Coefficient for {var} = {coef:.4f} (SE = {se:.4f}, t = {tstat:.3f}, p = {pval:.4g}). "
        f"95% CI = [{ci_lower if ci_lower is not None else 'NA'}, {ci_upper if ci_upper is not None else 'NA'}]. "
        f"{'Statistically significant at 5%.' if significant else 'Not statistically significant at 5%.'} "
        f"Interpretation: {direction}. "
        f"Practical effect: one-unit increase in StudentTeacherRatio -> AvgScore change of {coef:.3f} points."
    )

    return {"object": result_obj, "description": description}