def extract_final_answer(model_output):
    """
    Extracts the coefficient and inference for the student-teacher ratio variable
    from a fitted statsmodels RegressionResultsWrapper (or similar) object.

    Returns a dictionary with:
      - "object": a dict containing numeric outputs (variable name, coef, se, p-value, 95% CI, nobs)
      - "description": a concise interpretation describing direction and statistical significance
    """
    # Attempt to access typical statsmodels result attributes
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
        conf = model_output.conf_int(alpha=0.05)
        # nobs sometimes stored as float
        nobs = getattr(model_output, "nobs", None)
        if nobs is None:
            # fallback: try df_resid -> nobs = df_model + df_resid + 1, but simplest fallback is None
            nobs = None
        else:
            try:
                nobs = int(nobs)
            except Exception:
                pass
    except Exception as e:
        raise ValueError("Provided model_output does not look like a statsmodels results object with params/pvalues/bse/conf_int.") from e

    # Prefer log ratio coefficient if present; otherwise fall back to raw ratio
    if "LogStudentTeacherRatio" in params.index:
        var = "LogStudentTeacherRatio"
    elif "StudentTeacherRatio" in params.index:
        var = "StudentTeacherRatio"
    else:
        available = list(params.index)
        raise ValueError(f"Neither 'LogStudentTeacherRatio' nor 'StudentTeacherRatio' found in model coefficients. Available coefficients: {available}")

    coef = float(params[var])
    stderr = float(bse[var]) if var in bse.index else None
    pval = float(pvalues[var])
    # conf_int returns a DataFrame/array with two columns (lower, upper)
    try:
        ci_lower = float(conf.loc[var, 0])
        ci_upper = float(conf.loc[var, 1])
    except Exception:
        # different indexing conventions
        ci_lower = float(conf.iloc[params.index.get_loc(var), 0])
        ci_upper = float(conf.iloc[params.index.get_loc(var), 1])

    significant = pval < 0.05

    # Interpret direction given coding: higher ratio = more students per teacher (worse)
    if coef < 0:
        direction_text = (
            "Coefficient is negative: higher student-teacher ratio (more students per teacher) "
            "is associated with LOWER AvgScore. Equivalently, a LOWER student-teacher ratio "
            "(fewer students per teacher) is associated with HIGHER academic performance."
        )
    elif coef > 0:
        direction_text = (
            "Coefficient is positive: higher student-teacher ratio (more students per teacher) "
            "is associated with HIGHER AvgScore. Equivalently, a LOWER student-teacher ratio "
            "would be associated with LOWER academic performance."
        )
    else:
        direction_text = "Coefficient is exactly zero (no estimated association)."

    sig_text = (
        f"The estimate is statistically significant at the 5% level (p = {pval:.3g})."
        if significant
        else f"The estimate is not statistically significant at the 5% level (p = {pval:.3g})."
    )

    description = (
        f"{var}: estimate = {coef:.4f}, SE = {stderr:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}], p = {pval:.3g}, n = {nobs}.\n"
        f"{direction_text} {sig_text}"
    )

    result_object = {
        "variable": var,
        "coef": coef,
        "stderr": stderr,
        "pvalue": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "nobs": nobs,
    }

    return {"object": result_object, "description": description}