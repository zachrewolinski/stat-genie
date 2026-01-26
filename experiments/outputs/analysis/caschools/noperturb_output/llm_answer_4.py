def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted statsmodels OLS results object.

    Returns a dictionary with:
      - "object": dict of extracted numeric results (coef, se, t, p, 95% CI, standardized coef if computable,
                   significance flag, and a short conclusion)
      - "description": plain-language interpretation of what the numbers mean for the hypothesis:
                       "Is a lower student-teacher ratio associated with higher academic performance?"
    """
    import numpy as np

    results = model_output

    var = 'StudentTeacherRatio'

    # Ensure requested coefficient exists
    if var not in results.params.index:
        raise KeyError(f"Variable '{var}' not found in model results. Available params: {list(results.params.index)}")

    # Extract basic statistics
    coef = float(results.params[var])
    se = float(results.bse[var])
    t_val = float(results.tvalues[var])
    p_val = float(results.pvalues[var])
    ci_lower, ci_upper = [float(x) for x in results.conf_int().loc[var].tolist()]

    # Try to compute a standardized (beta) coefficient: coef * sd(x) / sd(y)
    std_coef = None
    try:
        exog = results.model.exog
        endog = results.model.endog
        names = results.model.exog_names
        # find column index for variable (exog includes intercept if present)
        idx = names.index(var)
        x = exog[:, idx]
        std_x = x.std(ddof=1)
        std_y = endog.std(ddof=1)
        if std_y != 0:
            std_coef = float(coef * (std_x / std_y))
    except Exception:
        std_coef = None

    # Determine statistical significance at conventional levels
    alpha = 0.05
    significant = (p_val < alpha)

    # Interpret direction: recall StudentTeacherRatio = students / teachers,
    # so a negative coefficient means higher ratio (more students per teacher)
    # is associated with lower AvgScore => equivalently, a lower ratio is associated with higher AvgScore.
    if coef < 0 and significant:
        conclusion = (
            "Yes: coefficient is negative and statistically significant (p < 0.05). "
            "This indicates that lower student-teacher ratios (fewer students per teacher) are associated "
            "with higher district average scores."
        )
    elif coef < 0 and not significant:
        conclusion = (
            "Suggestive but inconclusive: coefficient is negative but not statistically significant (p >= 0.05). "
            "Point estimate implies lower ratios relate to higher scores, but evidence is weak."
        )
    elif coef > 0 and significant:
        conclusion = (
            "No (opposite): coefficient is positive and statistically significant (p < 0.05). "
            "This indicates higher student-teacher ratios are associated with higher scores (contrary to the hypothesis)."
        )
    else:
        conclusion = (
            "No strong evidence: coefficient is positive or very close to zero and not statistically significant (p >= 0.05). "
            "No clear association between student-teacher ratio and average scores."
        )

    output_object = {
        "variable": var,
        "coef": coef,
        "std_error": se,
        "t_value": t_val,
        "p_value": p_val,
        "95%_CI": [ci_lower, ci_upper],
        "standardized_coef": std_coef,  # may be None if not computable
        "significant_at_0.05": bool(significant),
        "conclusion": conclusion,
        "notes": "Model was fit with robust (HC3) standard errors; coefficient units are change in AvgScore per one-unit change in StudentTeacherRatio (one additional student per teacher)."
    }

    description = (
        f"Extracted statistics for '{var}': coefficient = {coef:.4f}, SE = {se:.4f}, t = {t_val:.3f}, "
        f"p = {p_val:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        f"{'Standardized coefficient = ' + ('{:.4f}'.format(std_coef) if std_coef is not None else 'NA')}. "
        f"Interpretation: {conclusion}"
    )

    return {"object": output_object, "description": description}