def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels OLS result
    (assumed to be a statsmodels.regression.linear_model.RegressionResultsWrapper).
    Returns a dictionary with keys:
      - "object": a dict with numeric results (coef, se, t, p, 95% CI, significance, interpreted effect)
      - "description": a concise interpretation answering whether a lower student-teacher ratio
                       is associated with higher academic performance.
    """
    result = {"object": None, "description": None}
    try:
        res = model_output  # expected statsmodels RegressionResultsWrapper

        varname = "StudentTeacherRatio"
        params = getattr(res, "params", None)
        pvalues = getattr(res, "pvalues", None)
        bse = getattr(res, "bse", None)
        tvalues = getattr(res, "tvalues", None)
        conf = None
        try:
            conf = res.conf_int(alpha=0.05)
        except Exception:
            # fallback: try method call if attribute not present
            conf = res.conf_int(0.05)

        if params is None or varname not in params.index:
            result["description"] = f"Variable '{varname}' not found in model results."
            return result

        coef = float(params.loc[varname])
        se = float(bse.loc[varname]) if bse is not None else None
        tstat = float(tvalues.loc[varname]) if tvalues is not None else None
        pval = float(pvalues.loc[varname]) if pvalues is not None else None
        ci_lower, ci_upper = (None, None)
        if conf is not None and varname in conf.index:
            ci_lower = float(conf.loc[varname, 0])
            ci_upper = float(conf.loc[varname, 1])

        # Determine significance at conventional levels
        sig05 = (pval is not None and pval < 0.05)
        sig10 = (pval is not None and pval < 0.10)

        # Interpretation: recall StudentTeacherRatio higher = more students per teacher.
        # If coef < 0 and significant => higher ratio -> lower scores, so lower ratio -> higher scores.
        if coef < 0:
            direction = "higher_ratio_associated_with_lower_scores"
            implied = ("A decrease of 1 in StudentTeacherRatio is associated with an "
                       f"estimated increase of {abs(coef):.4f} points in AvgScore.")
        elif coef > 0:
            direction = "higher_ratio_associated_with_higher_scores"
            implied = ("A decrease of 1 in StudentTeacherRatio is associated with an "
                       f"estimated decrease of {coef:.4f} points in AvgScore (i.e., the sign is positive).")
        else:
            direction = "no_effect_point_estimate_zero"
            implied = "Point estimate is zero."

        # Final yes/no answer about "Is a lower student-teacher ratio associated with higher academic performance?"
        if coef < 0 and sig05:
            conclusion = ("Yes: the StudentTeacherRatio coefficient is negative and statistically significant "
                          f"(coef = {coef:.4f}, p = {pval:.3g}), indicating that lower student-teacher ratios "
                          "are associated with higher AvgScore.")
        elif coef < 0 and not sig05:
            conclusion = ("Point estimate suggests that lower student-teacher ratios are associated with higher AvgScore "
                          f"(coef = {coef:.4f}), but this relationship is not statistically significant (p = {pval:.3g}).")
        elif coef > 0 and sig05:
            conclusion = ("No: the StudentTeacherRatio coefficient is positive and statistically significant "
                          f"(coef = {coef:.4f}, p = {pval:.3g}), indicating that lower student-teacher ratios are "
                          "NOT associated with higher AvgScore (the estimate goes in the opposite direction).")
        elif coef > 0 and not sig05:
            conclusion = ("The point estimate indicates higher StudentTeacherRatio is associated with higher AvgScore "
                          f"(coef = {coef:.4f}), but the estimate is not statistically significant (p = {pval:.3g}).")
        else:
            conclusion = ("No detectable association between StudentTeacherRatio and AvgScore based on the point "
                          f"estimate (coef = {coef:.4f}, p = {pval:.3g}).")

        obj = {
            "variable": varname,
            "coef": coef,
            "std_error": se,
            "t_stat": tstat,
            "p_value": pval,
            "95ci_lower": ci_lower,
            "95ci_upper": ci_upper,
            "significant_at_0.05": sig05,
            "significant_at_0.10": sig10,
            "direction": direction,
            "interpretation_per_unit": implied,
            "conclusion_yes_no": ("Yes" if (coef < 0 and sig05) else
                                  "No" if (coef > 0 and sig05) else
                                  "Inconclusive")
        }

        result["object"] = obj

        # Build a concise human-readable description
        desc_lines = [
            f"StudentTeacherRatio coefficient = {coef:.4f}",
            f"Standard error = {se:.4f}" if se is not None else "Standard error = NA",
            f"t = {tstat:.3f}" if tstat is not None else None,
            f"p-value = {pval:.3g}",
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]" if (ci_lower is not None and ci_upper is not None) else None,
            "",
            conclusion,
            implied
        ]
        # filter out None lines
        desc = "\n".join([ln for ln in desc_lines if ln is not None])
        result["description"] = desc
        return result

    except Exception as e:
        result["description"] = f"Error extracting results: {e}"
        return result