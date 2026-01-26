def extract_final_answer(model_output):
    """
    Extract statistics about the effect of StudentTeacherRatio on AvgScore
    from the model_output produced by the provided modeling function.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coef, se, pvalue, 95% CI,
                  effect per 1-unit decrease, boolean conclusion),
      - "description": a short plain-language interpretation of the result.
    """
    result = {
        "object": None,
        "description": None
    }

    try:
        # get primary model
        model = model_output.get('model_avg', None)
        if model is None:
            raise KeyError("model_avg not found in model_output")

        param = 'StudentTeacherRatio'
        # Ensure the parameter exists in the model
        params = model.params
        if param not in params.index:
            raise KeyError(f"Parameter '{param}' not found in model parameters: {list(params.index)}")

        coef = float(params[param])
        se = float(model.bse[param]) if (hasattr(model, 'bse') and param in model.bse.index) else None
        pvalue = float(model.pvalues[param]) if (hasattr(model, 'pvalues') and param in model.pvalues.index) else None

        # Try to get 95% CI from model; fall back to normal approximation if needed
        try:
            ci = model.conf_int().loc[param].tolist()
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            import math
            if se is not None:
                # approximate with 1.96 critical value
                ci_lower, ci_upper = coef - 1.96 * se, coef + 1.96 * se
            else:
                ci_lower, ci_upper = (None, None)

        # Interpret direction:
        # coef = change in AvgScore for a one-unit increase in StudentTeacherRatio.
        # A negative coef => higher ratio (more students per teacher) reduces AvgScore,
        # so a lower ratio (fewer students per teacher) is associated with higher AvgScore.
        significance = (pvalue is not None) and (pvalue < 0.05)
        lower_ratio_better = (coef < 0) and significance

        # Effect size per one-unit decrease in ratio:
        # For a 1-unit decrease, expected change in AvgScore = -coef
        effect_per_one_unit_decrease = -coef

        obj = {
            "parameter": param,
            "coef": coef,
            "std_error": se,
            "p_value": pvalue,
            "95ci_lower": ci_lower,
            "95ci_upper": ci_upper,
            "significant_at_0.05": significance,
            "direction": ("lower_ratio_associated_with_higher_AvgScore" if coef < 0 else
                          "higher_ratio_associated_with_higher_AvgScore" if coef > 0 else
                          "no_association_detected"),
            "lower_ratio_associated_with_higher_scores": bool(lower_ratio_better),
            "effect_per_1_unit_decrease_in_ratio_on_AvgScore": effect_per_one_unit_decrease
        }

        # Short plain-language description
        if pvalue is None:
            description = ("Extracted coefficient for StudentTeacherRatio but p-value not available. "
                           "See returned numeric object for coefficient and CI.")
        else:
            if lower_ratio_better:
                description = (f"The estimated coefficient on StudentTeacherRatio is {coef:.4f} "
                               f"(SE={se:.4f}, p={pvalue:.3g}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]). "
                               "This is negative and statistically significant at the 5% level, "
                               f"indicating that a lower student–teacher ratio is associated with higher "
                               f"district average test scores. A 1-unit decrease in the ratio is associated "
                               f"with an average increase of {effect_per_one_unit_decrease:.4f} points in AvgScore.")
            else:
                # Not a statistically significant negative association
                if significance:
                    # significant but positive coef
                    description = (f"The estimated coefficient on StudentTeacherRatio is {coef:.4f} "
                                   f"(SE={se:.4f}, p={pvalue:.3g}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]). "
                                   "This is statistically significant at the 5% level but positive, "
                                   "meaning higher student–teacher ratios are associated with higher AvgScore "
                                   "(the opposite of the hypothesized direction).")
                else:
                    description = (f"The estimated coefficient on StudentTeacherRatio is {coef:.4f} "
                                   f"(SE={se:.4f}, p={pvalue:.3g}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]). "
                                   "This effect is not statistically significant at conventional levels, "
                                   "so the data do not provide strong evidence that lower student–teacher ratios "
                                   "are associated with higher district average test scores.")

        result["object"] = obj
        result["description"] = description
        return result

    except Exception as e:
        # Return an informative error in the expected structure
        result["object"] = None
        result["description"] = f"Failed to extract answer: {e}"
        return result