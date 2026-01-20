def extract_final_answer(model_output):
    """
    Extracts key statistics for the effect of StudentTeacherRatio on AvgScore
    from a fitted statsmodels RegressionResultsWrapper.

    Returns a dictionary with:
      - "object": a dict containing numeric results (coef, se, t, p, 95% CI,
                  significance flag, direction, and a plain-English interpretation)
      - "description": a short explanation of what the numbers mean in context

    If extraction fails (e.g., variable not present), returns object=None and an
    explanatory description.
    """
    var = 'StudentTeacherRatio'
    try:
        res = model_output  # expected to be a statsmodels RegressionResultsWrapper

        # Extract statistics
        coef = float(res.params[var])
        se = float(res.bse[var])
        t_value = float(res.tvalues[var])
        p_value = float(res.pvalues[var])

        # 95% confidence interval (statsmodels.conf_int returns a DataFrame/array)
        ci = res.conf_int().loc[var].tolist() if hasattr(res, "conf_int") else None
        # Determine significance at conventional levels
        significant_05 = p_value < 0.05
        significant_01 = p_value < 0.01

        # Directional interpretation:
        # Note: StudentTeacherRatio is defined as students per teacher (higher = larger classes).
        if coef < 0:
            direction = "negative"
            plain_interpretation = (
                "A negative coefficient means that larger student-teacher ratios (bigger classes) "
                "are associated with lower AvgScore; equivalently, a lower student-teacher ratio "
                "(smaller classes) is associated with higher AvgScore."
            )
        elif coef > 0:
            direction = "positive"
            plain_interpretation = (
                "A positive coefficient means that larger student-teacher ratios (bigger classes) "
                "are associated with higher AvgScore; equivalently, a lower student-teacher ratio "
                "(smaller classes) is associated with lower AvgScore."
            )
        else:
            direction = "zero"
            plain_interpretation = "Estimated effect is zero (no association)."

        # Assemble object to return
        result_object = {
            "variable": var,
            "coef": coef,
            "std_error": se,
            "t_value": t_value,
            "p_value": p_value,
            "ci_95": ci,  # [lower, upper]
            "significant_at_0.05": significant_05,
            "significant_at_0.01": significant_01,
            "direction": direction,
            "plain_interpretation": plain_interpretation,
            "meaning_per_unit": (
                f"The point estimate implies a {coef:.3f} change in AvgScore for a one-student "
                "increase in StudentTeacherRatio (increase = larger class)."
            )
        }

        description = (
            "Extracted coefficient and inference for StudentTeacherRatio from the OLS model "
            "with robust (HC1) standard errors. If coef < 0 and p_value < 0.05, this provides "
            "evidence that lower student-teacher ratios (smaller classes) are associated with "
            "higher average test scores, controlling for the listed covariates and county fixed effects."
        )

        return {"object": result_object, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract results for '{var}': {e}"
        }