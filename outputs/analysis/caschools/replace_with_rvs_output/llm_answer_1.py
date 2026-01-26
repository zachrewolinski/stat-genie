def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, t-stat, p-value, and 95% CI for the 'students_per_teacher'
    variable from a statsmodels RegressionResultsWrapper object and returns a short
    interpretation about whether a lower student-teacher ratio is associated with
    higher academic performance.
    Returns a dict with keys:
      - "object": dict of numeric results and a boolean 'significant' and 'conclusion'
      - "description": short textual interpretation
    """
    result = {}
    try:
        # Coefficient, standard error, t-value, p-value
        coef = float(model_output.params['students_per_teacher'])
        se = float(model_output.bse['students_per_teacher'])
        tval = float(model_output.tvalues['students_per_teacher'])
        pval = float(model_output.pvalues['students_per_teacher'])

        # 95% confidence interval (statsmodels conf_int rows indexed by variable name)
        ci = model_output.conf_int(alpha=0.05).loc['students_per_teacher']
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])

        # Significance at alpha = 0.05
        significant = (pval < 0.05)

        # Directional interpretation:
        # - If coef < 0: decreasing students_per_teacher (fewer students per teacher)
        #   is associated with higher AvgScore.
        # - If coef > 0: decreasing students_per_teacher is associated with lower AvgScore.
        if coef < 0:
            direction_text = ("A lower student-teacher ratio (fewer students per teacher) "
                              "is associated with higher average test scores.")
        elif coef > 0:
            direction_text = ("A lower student-teacher ratio (fewer students per teacher) "
                              "is associated with lower average test scores (opposite of the hypothesized direction).")
        else:
            direction_text = "No directional association (coefficient is exactly zero)."

        # Short conclusion about statistical evidence
        if significant:
            conclusion = ("The association is statistically significant at the 5% level.")
        else:
            conclusion = ("The association is not statistically significant at the 5% level.")

        # Put numeric results in the object
        result_object = {
            "coefficient": coef,
            "std_error": se,
            "t_value": tval,
            "p_value": pval,
            "conf_int_95": [ci_lower, ci_upper],
            "significant_at_0.05": significant,
            # Plain-language conclusion (Yes/No) with direction
            "conclusion": ("Yes: " + direction_text) if (coef < 0 and significant) else
                          ("No: " + direction_text) if (coef >= 0 and significant) else
                          ("Inconclusive: " + direction_text)
        }

        description = (
            f"students_per_teacher coefficient = {coef:.4f} (SE = {se:.4f}, t = {tval:.3f}, p = {pval:.3g}). "
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. {direction_text} {conclusion} "
            "Interpretation: the coefficient indicates the change in district average test score "
            "associated with a one-unit change in students per teacher (units = students per FTE teacher)."
        )

        return {"object": result_object, "description": description}

    except Exception as e:
        # If something unexpected happens, return an informative message
        return {
            "object": None,
            "description": f"Could not extract statistics for 'students_per_teacher' from the model output. Error: {e}"
        }