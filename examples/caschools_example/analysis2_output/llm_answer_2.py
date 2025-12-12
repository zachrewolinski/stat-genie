def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted statsmodels
    RegressionResultsWrapper and returns a brief interpretation.

    Returns a dictionary with:
      - "object": dict with numeric results (coefficient, se, t, p, 95% CI, effect per 10 students,
                   n_obs, R-squared, R-squared adj, significant(boolean))
      - "description": short plain-language interpretation in the context of the question:
                       "Is a lower student-teacher ratio associated with higher academic performance?"
    """
    try:
        # Pull core statistics
        coef = float(model_output.params['StudentTeacherRatio'])
        se = float(model_output.bse['StudentTeacherRatio'])
        tval = float(model_output.tvalues['StudentTeacherRatio'])
        pval = float(model_output.pvalues['StudentTeacherRatio'])
        ci = model_output.conf_int(alpha=0.05).loc['StudentTeacherRatio']
        ci_low = float(ci[0])
        ci_high = float(ci[1])

        # Additional model summaries
        try:
            n_obs = int(model_output.nobs)
        except Exception:
            # fallback: use df_resid + df_model + 1 if available
            n_obs = None
        r_squared = float(getattr(model_output, 'rsquared', float('nan')))
        r_squared_adj = float(getattr(model_output, 'rsquared_adj', float('nan')))

        # Practical effect: change in AvgScore per 10-student increase in ratio
        effect_per_10 = coef * 10.0

        significant = (pval < 0.05)

        # Interpretation regarding the research question:
        # StudentTeacherRatio = students per teacher. A negative coefficient means
        # higher ratio (more students per teacher) is associated with lower AvgScore,
        # equivalently a lower ratio (fewer students per teacher) is associated with higher AvgScore.
        if coef < 0:
            direction_text = ("The estimated association is negative: higher student-teacher "
                              "ratios (more students per teacher) are associated with lower average scores. "
                              "Thus, lower ratios (fewer students per teacher) are associated with higher academic performance.")
        elif coef > 0:
            direction_text = ("The estimated association is positive: higher student-teacher "
                              "ratios (more students per teacher) are associated with higher average scores. "
                              "Thus, lower ratios would be associated with lower academic performance.")
        else:
            direction_text = "The estimated association is effectively zero."

        significance_text = ("This effect is statistically significant at the 0.05 level."
                             if significant else
                             "This effect is not statistically significant at the 0.05 level.")

        description = (
            f"{direction_text} Coefficient = {coef:.4f} (SE = {se:.4f}, t = {tval:.2f}, p = {pval:.3g}). "
            f"95% CI [{ci_low:.4f}, {ci_high:.4f}]. {significance_text} "
            f"Estimated change in AvgScore for a 10-student increase in the ratio = {effect_per_10:.4f}. "
            f"Model N = {n_obs}, R^2 = {r_squared:.4f}, adj. R^2 = {r_squared_adj:.4f}."
        )

        result_object = {
            "coefficient": coef,
            "std_error": se,
            "t_value": tval,
            "p_value": pval,
            "ci_95_lower": ci_low,
            "ci_95_upper": ci_high,
            "effect_per_10_students": effect_per_10,
            "n_obs": n_obs,
            "r_squared": r_squared,
            "r_squared_adj": r_squared_adj,
            "statistically_significant_at_0.05": significant
        }

        return {"object": result_object, "description": description}

    except Exception as e:
        # In case the model_output doesn't have expected attributes
        return {
            "object": None,
            "description": f"Could not extract statistics from model_output: {e}"
        }