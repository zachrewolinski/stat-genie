def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted
    statsmodels OLSResults object and provides a brief interpretation.

    Returns a dict with keys:
      - "object": dict with numeric results (coefficient, std_err, t, p_value,
                  95% CI lower/upper, nobs, rsquared)
      - "description": human-readable explanation of the result and whether it
                       supports the hypothesis that a lower student-teacher
                       ratio is associated with higher academic performance.
    """
    try:
        res = model_output

        # Identify the variable name in the model results (allow minor name variation)
        target_var = 'StudentTeacherRatio'
        params_index = list(res.params.index)

        if target_var not in params_index:
            # fallback: pick a parameter name that contains both 'Student' and 'Teacher'
            candidates = [p for p in params_index if 'student' in p.lower() and 'teacher' in p.lower()]
            if len(candidates) >= 1:
                target_var = candidates[0]
            else:
                raise KeyError(f"Variable 'StudentTeacherRatio' not found in model parameters: {params_index}")

        coef = float(res.params[target_var])
        std_err = float(res.bse[target_var]) if hasattr(res, 'bse') else None
        t_value = float(res.tvalues[target_var]) if hasattr(res, 'tvalues') else (coef / std_err if std_err not in (None, 0) else None)
        p_value = float(res.pvalues[target_var]) if hasattr(res, 'pvalues') else None

        ci = res.conf_int().loc[target_var].tolist() if hasattr(res, 'conf_int') else [None, None]
        ci_lower, ci_upper = (float(ci[0]), float(ci[1])) if (ci and ci[0] is not None) else (None, None)

        # Additional model info (if available)
        nobs = int(getattr(res, 'nobs', None)) if getattr(res, 'nobs', None) is not None else None
        rsq = float(getattr(res, 'rsquared', None)) if getattr(res, 'rsquared', None) is not None else None

        # Interpretation logic
        # A negative coefficient means that higher StudentTeacherRatio (more students per teacher)
        # is associated with lower AvgTestScore. Therefore a negative coef supports the claim
        # that a lower student-teacher ratio (fewer students per teacher) is associated with
        # higher academic performance.
        supports_hypothesis = None
        significance = None
        if p_value is not None:
            significance = (p_value < 0.05)
            if coef < 0 and significance:
                supports_hypothesis = True
            elif coef < 0 and not significance:
                supports_hypothesis = "point_estimate_only"  # directionally supportive but not significant
            elif coef >= 0 and significance:
                supports_hypothesis = False
            else:
                supports_hypothesis = "no_evidence"

        # Build description
        desc_parts = []
        desc_parts.append(
            f"StudentTeacherRatio coefficient = {coef:.4f}"
            + (f" (SE = {std_err:.4f})" if std_err is not None else "")
            + (f", t = {t_value:.2f}" if t_value is not None else "")
            + (f", p = {p_value:.3g}" if p_value is not None else "")
        )
        if ci_lower is not None and ci_upper is not None:
            desc_parts.append(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}].")
        if nobs is not None:
            desc_parts.append(f"N = {nobs}.")
        if rsq is not None:
            desc_parts.append(f"R-squared = {rsq:.3f}.")

        # Conclusion sentence
        if p_value is None:
            conclusion = "Could not determine statistical significance (p-value not available)."
        else:
            if supports_hypothesis is True:
                conclusion = ("There is a statistically significant negative association: "
                              "lower student-teacher ratio (fewer students per teacher) is associated "
                              "with higher average test scores (p < 0.05).")
            elif supports_hypothesis == "point_estimate_only":
                conclusion = ("The point estimate is negative (suggesting lower ratio is associated with higher scores), "
                              "but this effect is not statistically significant (p >= 0.05).")
            elif supports_hypothesis is False:
                conclusion = ("There is a statistically significant positive association (p < 0.05), "
                              "meaning higher student-teacher ratio is associated with higher average test scores "
                              "(opposite of the hypothesis).")
            else:  # "no_evidence"
                conclusion = ("No statistically significant association was found between student-teacher ratio "
                              "and average test scores (p >= 0.05).")

        description = " ".join(desc_parts) + " " + conclusion

        return {
            "object": {
                "variable": target_var,
                "coefficient": coef,
                "std_err": std_err,
                "t_value": t_value,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "nobs": nobs,
                "rsquared": rsq,
                "supports_hypothesis": supports_hypothesis
            },
            "description": description
        }

    except Exception as e:
        return {
            "object": None,
            "description": f"Error extracting results for 'StudentTeacherRatio': {e}"
        }