def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-value, p-value, 95% CI, sample size, and R-squared
    for the 'student_teacher_ratio' variable from a statsmodels RegressionResultsWrapper.

    Returns:
      {
        "object": {
            "coef": float,
            "se": float,
            "t": float,
            "pvalue": float,
            "ci_lower": float,
            "ci_upper": float,
            "nobs": int or None,
            "rsquared": float or None
        },
        "description": str  # brief interpretation in the context of the task
      }
    """
    # Defensive extraction to handle typical statsmodels result objects
    try:
        params = model_output.params
        bse = model_output.bse
        tvals = model_output.tvalues
        pvals = model_output.pvalues
    except Exception as e:
        raise ValueError("Provided model_output does not look like a statsmodels results object: " + str(e))

    var = 'student_teacher_ratio'
    if var not in params.index:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    # Extract numeric values
    coef = float(params[var])
    se = float(bse[var]) if var in bse.index else None
    tval = float(tvals[var]) if var in tvals.index else None
    pval = float(pvals[var]) if var in pvals.index else None

    # Confidence interval (try labeled access first, fallback to positional)
    try:
        ci = model_output.conf_int().loc[var]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        # fallback: find positional index of the variable
        try:
            idx = list(params.index).index(var)
            ci_row = model_output.conf_int().iloc[idx]
            ci_lower = float(ci_row[0])
            ci_upper = float(ci_row[1])
        except Exception as e:
            ci_lower = None
            ci_upper = None

    # Sample size and R-squared (may be absent for some wrappers)
    nobs = None
    try:
        nobs_attr = getattr(model_output, 'nobs', None)
        if nobs_attr is not None:
            # nobs can be a numpy type; cast to int where possible
            nobs = int(nobs_attr)
    except Exception:
        nobs = None

    rsq = None
    try:
        rsq_attr = getattr(model_output, 'rsquared', None)
        if rsq_attr is not None:
            rsq = float(rsq_attr)
    except Exception:
        rsq = None

    # Interpretation in the context of the question:
    # Note: coef is the change in avg_score for a one-unit increase in student_teacher_ratio.
    # A negative coef => higher ratio (more students per teacher) associated with lower scores
    # => lower ratio (fewer students per teacher) associated with higher scores.
    significance = "statistically significant" if (pval is not None and pval < 0.05) else "not statistically significant"
    if coef < 0:
        direction_text = (
            "Higher student-teacher ratio (more students per teacher) is associated with LOWER average scores; "
            "consequently, a LOWER student-teacher ratio (fewer students per teacher) is associated with HIGHER performance."
        )
    else:
        direction_text = (
            "Higher student-teacher ratio (more students per teacher) is associated with HIGHER average scores; "
            "consequently, a LOWER student-teacher ratio would be associated with LOWER performance."
        )

    magnitude_text = f"A one-unit increase in student_teacher_ratio is associated with a change of {coef:.3f} points in avg_score"
    if ci_lower is not None and ci_upper is not None:
        magnitude_text += f" (95% CI [{ci_lower:.3f}, {ci_upper:.3f}])"
    magnitude_text += "."

    desc_parts = [
        f"Coefficient = {coef:.3f}",
        f"SE = {se:.3f}" if se is not None else "SE = N/A",
        f"t = {tval:.3f}" if tval is not None else "t = N/A",
        f"p = {pval:.3f}" if pval is not None else "p = N/A",
        f"{significance}.",
        direction_text,
        magnitude_text,
        f"N = {nobs}" if nobs is not None else "N = N/A",
        f"R-squared = {rsq:.3f}" if rsq is not None else "R-squared = N/A"
    ]
    description = " ".join(desc_parts)

    result_object = {
        "coef": coef,
        "se": se,
        "t": tval,
        "pvalue": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "nobs": nobs,
        "rsquared": rsq
    }

    return {"object": result_object, "description": description}