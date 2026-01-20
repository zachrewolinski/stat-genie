def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels OLS results object.
    Returns a dictionary with keys:
      - "object": a dict containing coefficient, se, t, p-value, 95% CI, and a short numeric conclusion flag
      - "description": a human-readable interpretation addressing whether a lower student-teacher ratio
                       is associated with higher academic performance.

    Expected input: a statsmodels RegressionResultsWrapper (the object returned by sm.OLS(...).fit(...))
    """
    # Prepare a default error response
    error_resp = {
        "object": None,
        "description": "Could not extract results: input model_output does not appear to be a fitted statsmodels results object "
                       "or does not contain the parameter 'StudentTeacherRatio'."
    }

    try:
        # Ensure params exist
        params = model_output.params
        if 'StudentTeacherRatio' not in params.index:
            return error_resp

        coef = float(model_output.params['StudentTeacherRatio'])
        se = float(model_output.bse['StudentTeacherRatio']) if hasattr(model_output, 'bse') else None
        tval = float(model_output.tvalues['StudentTeacherRatio']) if hasattr(model_output, 'tvalues') else None
        pval = float(model_output.pvalues['StudentTeacherRatio']) if hasattr(model_output, 'pvalues') else None

        # Confidence interval: conf_int() returns an array ordered like params
        ci_array = model_output.conf_int()  # ndarray or DataFrame-like
        # Find index of the parameter
        try:
            idx = list(model_output.params.index).index('StudentTeacherRatio')
            ci_lower, ci_upper = float(ci_array[idx, 0]), float(ci_array[idx, 1])
        except Exception:
            # Fallback: if conf_int returns a DataFrame with index
            try:
                ci_df = model_output.conf_int()
                ci_lower, ci_upper = float(ci_df.loc['StudentTeacherRatio', 0]), float(ci_df.loc['StudentTeacherRatio', 1])
            except Exception:
                ci_lower, ci_upper = None, None

        # Interpret the result in context:
        # Note: StudentTeacherRatio is defined as students / teachers (higher = more students per teacher).
        # A negative coefficient means that an increase in StudentTeacherRatio (more students per teacher)
        # is associated with a decrease in AvgTestScore; equivalently, a lower ratio (fewer students per teacher)
        # is associated with higher AvgTestScore.
        significance = None
        if pval is not None:
            significance = pval < 0.05

        if pval is None:
            conclusion_text = "Could not determine statistical significance (p-value missing)."
            conclusion_flag = None
        else:
            if coef < 0 and significance:
                conclusion_text = (
                    "Yes — there is statistically significant evidence (p = {:.3g}) that a lower student-teacher "
                    "ratio (fewer students per teacher) is associated with higher district average test scores. "
                    "Interpretation: a one-unit increase in StudentTeacherRatio (one more student per teacher) "
                    "is associated with a change of {:+.3f} points in AvgTestScore (SE = {:.3f}, 95% CI [{:.3f}, {:.3f}])."
                ).format(pval, coef, se if se is not None else float('nan'),
                         ci_lower if ci_lower is not None else float('nan'),
                         ci_upper if ci_upper is not None else float('nan'))
                # conclusion_flag: 1 means "Yes, lower ratio -> higher performance"
                conclusion_flag = 1
            elif coef < 0 and not significance:
                conclusion_text = (
                    "The estimated coefficient is negative (coef = {:+.3f}), which would imply that a lower "
                    "student-teacher ratio is associated with higher AvgTestScore, but this effect is not statistically "
                    "significant (p = {:.3g}). We cannot conclude there is an association based on conventional thresholds."
                ).format(coef, pval)
                conclusion_flag = 0
            elif coef > 0 and significance:
                conclusion_text = (
                    "No — the estimated coefficient is positive and statistically significant (coef = {:+.3f}, p = {:.3g}), "
                    "which implies that a lower student-teacher ratio (fewer students per teacher) is associated with "
                    "lower AvgTestScore (contrary to the expectation). Interpretation: a one-unit increase in "
                    "StudentTeacherRatio is associated with a change of {:+.3f} points in AvgTestScore (SE = {:.3f}, "
                    "95% CI [{:.3f}, {:.3f}])."
                ).format(coef, pval, coef, se if se is not None else float('nan'),
                         ci_lower if ci_lower is not None else float('nan'),
                         ci_upper if ci_upper is not None else float('nan'))
                # conclusion_flag: -1 means "Significant but opposite direction"
                conclusion_flag = -1
            else:  # coef > 0 and not significant
                conclusion_text = (
                    "The estimated coefficient is positive (coef = {:+.3f}), suggesting a lower student-teacher ratio "
                    "might be associated with lower AvgTestScore, but this effect is not statistically significant "
                    "(p = {:.3g}). We cannot conclude there is an association."
                ).format(coef, pval)
                conclusion_flag = 0

        # Build the object to return
        result_object = {
            "coefficient": coef,
            "std_error": se,
            "t_value": tval,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            # Numeric conclusion flag: 1 = yes (lower ratio -> higher scores, significant),
            # 0 = no strong evidence (not significant), -1 = significant but opposite direction.
            "conclusion_flag": conclusion_flag
        }

        return {
            "object": result_object,
            "description": conclusion_text
        }

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting results: {e}"
        }