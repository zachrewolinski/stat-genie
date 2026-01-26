def extract_final_answer(model_output):
    """
    Extract statistics for the STR_z coefficient from a fitted statsmodels results object
    and provide an interpretation regarding whether a lower student-teacher ratio
    is associated with higher academic performance.

    Returns a dict with:
      - "object": dict of numeric statistics (coefficient, se, p-value, 95% CI, nobs, significant)
      - "description": short textual interpretation in the context of the task
    """
    # Try to access attributes commonly available on statsmodels RegressionResultsWrapper
    res = model_output

    # Verify required pieces exist
    if not hasattr(res, "params") or "STR_z" not in res.params.index:
        raise ValueError("The provided model output does not contain a coefficient named 'STR_z'.")

    coef = float(res.params["STR_z"])
    # Standard error
    try:
        se = float(res.bse["STR_z"])
    except Exception:
        se = None
    # p-value
    try:
        pval = float(res.pvalues["STR_z"])
    except Exception:
        pval = None
    # 95% confidence interval
    try:
        ci = res.conf_int().loc["STR_z"]
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except Exception:
        ci_lower = ci_upper = None
    # number of observations
    try:
        nobs = int(res.nobs)
    except Exception:
        # fallback: try model endog length
        try:
            nobs = int(res.model.endog.shape[0])
        except Exception:
            nobs = None

    significant = (pval is not None) and (pval < 0.05)

    # Interpretation notes:
    # - STR_z is standardized (one unit = one SD increase in student-teacher ratio,
    #   i.e., more students per teacher / larger classes).
    # - AvgScore_z is standardized (one unit = one SD in average test score).
    # Therefore the coefficient is the change in SDs of AvgScore for a 1 SD increase in STR.
    if coef < 0:
        # Negative coefficient: higher STR (more students per teacher) -> lower AvgScore
        # So lower STR (fewer students per teacher) -> higher AvgScore
        direction_text = (
            "The coefficient is negative, so higher student-teacher ratios (more students per teacher) "
            "are associated with lower academic performance; conversely, lower student-teacher ratios "
            "(fewer students per teacher) are associated with higher academic performance."
        )
    elif coef > 0:
        direction_text = (
            "The coefficient is positive, so higher student-teacher ratios (more students per teacher) "
            "are associated with higher academic performance (the opposite of the commonly expected direction)."
        )
    else:
        direction_text = "The coefficient is exactly zero."

    # Construct a concise conclusion that takes statistical significance into account
    if pval is None:
        conclusion = "Unable to determine statistical significance (p-value not available)."
    else:
        if significant:
            conclusion = "This effect is statistically significant at the 0.05 level."
        else:
            conclusion = "There is not strong evidence of a nonzero effect at the 0.05 level (not statistically significant)."

    # Build the numeric object to return
    numeric_object = {
        "coef": coef,
        "std_err": se,
        "p_value": pval,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "nobs": nobs,
        "significant_at_0.05": significant,
        "interpretation_unit": (
            "change in SDs of AvgScore per 1 SD increase in STR (STR higher = more students per teacher)"
        )
    }

    # Final description: combine direction + magnitude + significance + practical meaning
    # Note: coefficient is already on standardized scale: e.g., coef = -0.10 means a 1 SD decrease in STR
    # (fewer students per teacher) is associated with a +0.10 SD increase in AvgScore.
    # Create plain-language final sentence.
    if coef is not None:
        magnitude_sentence = f"The estimated coefficient on STR_z is {coef:.3f}"
        if se is not None:
            magnitude_sentence += f" (SE = {se:.3f})"
        if pval is not None:
            magnitude_sentence += f", p = {pval:.3f}."
        else:
            magnitude_sentence += "."

        # Express effect for a 1 SD decrease in STR (fewer students per teacher)
        effect_on_decrease = -coef  # change in AvgScore_z for 1 SD decrease in STR
        magnitude_sentence += (
            f" This means a 1 SD decrease in student-teacher ratio (fewer students per teacher) "
            f"is associated with a {effect_on_decrease:.3f} SD change in average test score."
        )
    else:
        magnitude_sentence = "Coefficient could not be retrieved."

    description = " ".join([direction_text, magnitude_sentence, conclusion])

    return {"object": numeric_object, "description": description}