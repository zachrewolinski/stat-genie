def extract_final_answer(model_output):
    """
    Extracts coefficient, p-values, confidence interval, and related statistics for
    the 'StuTeacherRatio' coefficient from a statsmodels RegressionResultsWrapper.
    Returns a dictionary with keys:
      - "object": a dict of numeric results and a boolean stating whether there is
                  one-sided statistical evidence that coef < 0 at alpha=0.05
      - "description": a short plain-language interpretation in the context of the task.
    """
    # Prepare a default response if the parameter is missing
    result = {
        "coef": None,
        "p_two_sided": None,
        "p_one_sided": None,
        "ci_lower": None,
        "ci_upper": None,
        "nobs": None,
        "r_squared": None,
        "evidence_for_hypothesis": None  # True if one-sided p < 0.05 for H1: coef < 0
    }

    try:
        # Ensure parameter exists
        params = getattr(model_output, "params")
        if "StuTeacherRatio" not in params.index:
            return {
                "object": result,
                "description": "The model output does not contain a coefficient named 'StuTeacherRatio'."
            }

        coef = float(params["StuTeacherRatio"])
        p_two = float(model_output.pvalues["StuTeacherRatio"])

        # Confidence interval: try DataFrame-style access, fallback to index lookup
        try:
            ci = model_output.conf_int().loc["StuTeacherRatio"].values
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            # fallback: conf_int() may be ndarray; find parameter position
            ci_all = model_output.conf_int()
            try:
                pos = list(model_output.params.index).index("StuTeacherRatio")
                ci_lower, ci_upper = float(ci_all[pos, 0]), float(ci_all[pos, 1])
            except Exception:
                ci_lower, ci_upper = None, None

        # One-sided p-value for H1: coef < 0
        if coef < 0:
            p_one = p_two / 2.0
        else:
            # if coef >= 0, one-sided p for coef < 0 is large (fail to reject)
            p_one = 1.0 - p_two / 2.0

        # Other summary stats
        nobs = int(getattr(model_output, "nobs", None)) if getattr(model_output, "nobs", None) is not None else None
        r2 = float(getattr(model_output, "rsquared", float("nan"))) if hasattr(model_output, "rsquared") else None

        # Populate result
        result.update({
            "coef": coef,
            "p_two_sided": p_two,
            "p_one_sided": p_one,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "nobs": nobs,
            "r_squared": r2,
            "evidence_for_hypothesis": (p_one < 0.05)
        })

        # Build a concise interpretation
        if p_one < 0.05:
            interp = (
                "There is statistically significant one-sided evidence (p = "
                f"{p_one:.4f}) that the coefficient on StuTeacherRatio is negative. "
                "This supports the hypothesis that a lower student-teacher ratio "
                "(fewer students per teacher) is associated with higher average academic performance."
            )
        else:
            interp = (
                "There is NOT statistically significant one-sided evidence (p = "
                f"{p_one:.4f}) that the coefficient on StuTeacherRatio is negative. "
                "The data do not provide strong support that a lower student-teacher ratio "
                "is associated with higher average academic performance at conventional levels."
            )

        # Add numeric detail to description
        interp += (
            " Numeric summary: coef = "
            f"{coef:.4f}, two-sided p = {p_two:.4f}, 95% CI = [{ci_lower}, {ci_upper}],"
            f" n = {nobs}, R^2 = {r2}."
        )

        return {"object": result, "description": interp}

    except Exception as e:
        return {
            "object": result,
            "description": f"An error occurred while extracting results: {e}"
        }