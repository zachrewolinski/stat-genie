def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, and 95% confidence interval
    for the StudentTeacherRatio coefficient from a statsmodels RegressionResultsWrapper.
    Returns a dictionary with keys "object" and "description".

    "object" will be a dict with:
      - coef: estimated coefficient (float)
      - std_err: robust standard error (float)
      - pvalue: p-value for the coefficient (float)
      - ci: [ci_lower, ci_upper] (list of floats)
      - significant: boolean indicating whether pvalue < 0.05
      - conclusion_flag: one of {"lower_ratio_associated_with_higher_perf",
                                 "lower_ratio_not_significantly_associated",
                                 "lower_ratio_associated_with_lower_perf"}
      - conclusion: short textual conclusion

    "description" is a brief explanation of what these numbers mean in context.
    """
    # Defensive checks
    if model_output is None:
        return {
            "object": None,
            "description": "No model output supplied."
        }

    # Try to extract stats for the StudentTeacherRatio term
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
        ci_df = model_output.conf_int()  # DataFrame with index matching params

        term = 'StudentTeacherRatio'
        if term not in params.index:
            # Try alternative indexing if statsmodels wrapped differently
            raise KeyError(f"Term '{term}' not found in model params: {list(params.index)}")

        coef = float(params[term])
        std_err = float(bse[term])
        pval = float(pvalues[term])
        ci_lower, ci_upper = [float(x) for x in ci_df.loc[term].tolist()]

        significant = pval < 0.05

        # Interpret direction: StudentTeacherRatio is # students per teacher.
        # A negative coef means higher ratio -> lower scores, so LOWER ratio -> HIGHER scores.
        if significant and coef < 0:
            conclusion_flag = "lower_ratio_associated_with_higher_perf"
            conclusion = (
                "Statistically significant evidence (p < 0.05) that lower student-teacher "
                "ratio is associated with higher district average test scores. "
                f"Estimate: a 1-unit increase in StudentTeacherRatio is associated with a {coef:.3f} point change in AvgTestScore (95% CI [{ci_lower:.3f}, {ci_upper:.3f}])."
            )
        elif significant and coef > 0:
            conclusion_flag = "lower_ratio_associated_with_lower_perf"
            conclusion = (
                "Statistically significant evidence (p < 0.05) that lower student-teacher "
                "ratio is associated with lower district average test scores (unexpected direction). "
                f"Estimate: a 1-unit increase in StudentTeacherRatio is associated with a {coef:.3f} point change in AvgTestScore (95% CI [{ci_lower:.3f}, {ci_upper:.3f}])."
            )
        else:
            conclusion_flag = "lower_ratio_not_significantly_associated"
            conclusion = (
                "No statistically significant evidence (p >= 0.05) that student-teacher ratio is associated "
                "with district average test scores. The estimated effect is "
                f"{coef:.3f} (p = {pval:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}])."
            )

        result_object = {
            "coef": coef,
            "std_err": std_err,
            "pvalue": pval,
            "ci": [ci_lower, ci_upper],
            "significant": significant,
            "conclusion_flag": conclusion_flag,
            "conclusion": conclusion
        }

        description = (
            "This output reports the estimated effect of StudentTeacherRatio on AvgTestScore "
            "from the fitted OLS model (controls: expenditure, income, english, lunch, calworks, "
            "ComputersPerStudent, LogStudents, grade-span and county fixed effects). "
            "If the coefficient is negative and statistically significant, it implies that lower "
            "student-teacher ratios (fewer students per teacher) are associated with higher test scores."
        )

        return {"object": result_object, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"Failed to extract StudentTeacherRatio statistics from model output: {e}"
        }