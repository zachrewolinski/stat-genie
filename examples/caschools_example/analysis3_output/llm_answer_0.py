def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, p-value, 95% CI, and a brief conclusion
    about the association between StudentTeacherRatio and AvgScore from a
    statsmodels RegressionResultsWrapper (fitted with robust cov_type if desired).

    Returns:
      {
        "object": {  # numeric results
          "coef": float,
          "std_err": float,
          "p_value": float,
          "ci_lower": float,
          "ci_upper": float,
          "nobs": int
        },
        "description": str  # brief interpretation in context
      }
    """
    import pandas as pd

    try:
        # Extract core quantities
        params = model_output.params
        bse = model_output.bse
        pvalues = model_output.pvalues
        nobs = int(getattr(model_output, "nobs", None)) if getattr(model_output, "nobs", None) is not None else None

        if "StudentTeacherRatio" not in params.index:
            return {
                "object": None,
                "description": "Model output does not contain a parameter named 'StudentTeacherRatio'."
            }

        coef = float(params["StudentTeacherRatio"])
        std_err = float(bse["StudentTeacherRatio"]) if "StudentTeacherRatio" in bse.index else None
        p_value = float(pvalues["StudentTeacherRatio"]) if "StudentTeacherRatio" in pvalues.index else None

        # Confidence interval (handle both DataFrame and ndarray returns)
        ci = model_output.conf_int()
        try:
            if isinstance(ci, pd.DataFrame):
                ci_row = ci.loc["StudentTeacherRatio"]
                ci_lower, ci_upper = float(ci_row.iloc[0]), float(ci_row.iloc[1])
            else:
                # ci is an ndarray; find index of the parameter
                param_index = list(params.index).index("StudentTeacherRatio")
                ci_lower, ci_upper = float(ci[param_index, 0]), float(ci[param_index, 1])
        except Exception:
            # Fallback if indexing fails
            ci_lower, ci_upper = None, None

        # Interpret the result in context:
        # coef = change in AvgScore for a one-unit increase in students-per-teacher.
        # Lower student-teacher ratio (fewer students per teacher) corresponds to a decrease in StudentTeacherRatio.
        if p_value is None:
            significance_text = "could not determine statistical significance (p-value unavailable)."
        else:
            significance_text = "statistically significant (p < 0.05)." if p_value < 0.05 else "not statistically significant (p >= 0.05)."

        if coef < 0:
            direction_text = (
                "The coefficient is negative, so a lower student-teacher ratio (fewer students per teacher) "
                "is associated with higher average academic performance."
            )
        elif coef > 0:
            direction_text = (
                "The coefficient is positive, so a lower student-teacher ratio (fewer students per teacher) "
                "would be associated with lower average academic performance (opposite of the expected sign)."
            )
        else:
            direction_text = "The coefficient is zero (no estimated association)."

        description = (
            f"Estimate for StudentTeacherRatio: coef = {coef:.4f}, SE = {std_err:.4f}."
            f" 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}] (if available). p-value = {p_value:.4g}."
            f" Interpretation: {direction_text} This effect is {significance_text}"
        )

        result_object = {
            "coef": coef,
            "std_err": std_err,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "nobs": nobs
        }

        return {"object": result_object, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting results: {e}"
        }