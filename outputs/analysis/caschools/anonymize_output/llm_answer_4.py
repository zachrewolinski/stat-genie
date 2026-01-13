def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, t, p-value, 95% CI, standardized beta, and significance
    for the 'StudentTeacherRatio' variable from a statsmodels RegressionResultsWrapper.

    Returns a dict with:
      - "object": dict of numeric results (coef, std_err, t, p_value, ci_lower, ci_upper,
                  std_beta, significant)
      - "description": text interpretation of the coefficient in the context of the task
    """
    import numpy as np

    try:
        params = model_output.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Provided model_output does not appear to be a fitted statsmodels results object: {e}"
        }

    var = 'StudentTeacherRatio'
    if var not in params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}"
        }

    try:
        coef = float(model_output.params[var])
        se = float(model_output.bse[var])
        tval = float(model_output.tvalues[var])
        pval = float(model_output.pvalues[var])

        # 95% CI
        try:
            ci = model_output.conf_int(alpha=0.05).loc[var]
            ci_lower = float(ci[0])
            ci_upper = float(ci[1])
        except Exception:
            # fallback if conf_int returns different structure
            ci_lower, ci_upper = (None, None)

        # Attempt to compute a standardized beta (requires access to original endog/exog)
        std_beta = None
        try:
            endog = model_output.model.endog
            exog = model_output.model.exog
            exog_names = model_output.model.exog_names
            idx = exog_names.index(var)
            x = exog[:, idx]
            # compute standardized coefficient: beta * (sd_x / sd_y)
            sd_x = np.std(x, ddof=1)
            sd_y = np.std(endog, ddof=1)
            if sd_y != 0:
                std_beta = float(coef * (sd_x / sd_y))
        except Exception:
            std_beta = None

        significant = (pval < 0.05)

        result_obj = {
            "coef": coef,
            "std_err": se,
            "t_value": tval,
            "p_value": pval,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "std_beta": std_beta,
            "significant_at_0.05": significant
        }

        # Interpretation of direction: note that lower StudentTeacherRatio means fewer students per teacher.
        if coef < 0 and significant:
            conclusion = (
                "The estimated coefficient is negative and statistically significant: "
                "this provides evidence that lower student-teacher ratios (fewer students per teacher) "
                "are associated with higher district average test scores."
            )
        elif coef < 0 and not significant:
            conclusion = (
                "The estimated coefficient is negative (suggesting lower ratios relate to higher scores) "
                "but it is not statistically significant at the 0.05 level."
            )
        elif coef > 0 and significant:
            conclusion = (
                "The estimated coefficient is positive and statistically significant: "
                "this provides evidence that higher student-teacher ratios (more students per teacher) "
                "are associated with higher district average test scores (contrary to the hypothesis)."
            )
        elif coef > 0 and not significant:
            conclusion = (
                "The estimated coefficient is positive but not statistically significant at the 0.05 level."
            )
        else:
            conclusion = "The estimated coefficient is essentially zero."

        desc = (
            f"StudentTeacherRatio coefficient = {coef:.4f} (SE={se:.4f}, t={tval:.2f}, p={pval:.3g}). "
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
            f"Standardized beta ≈ {std_beta:.4f}." if (ci_lower is not None and ci_upper is not None) else
            f"StudentTeacherRatio coefficient = {coef:.4f} (SE={se:.4f}, t={tval:.2f}, p={pval:.3g}). "
            f"Standardized beta ≈ {std_beta:.4f}."
        )
        # Combine numeric summary and plain-language conclusion
        full_description = desc + " " + conclusion + (
            " Interpretation: coefficient is the change in AvgScore associated with a one-unit increase in student-teacher ratio."
        )

        return {"object": result_obj, "description": full_description}

    except Exception as e:
        return {"object": None, "description": f"Error extracting statistics for '{var}': {e}"}