def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted
    statsmodels RegressionResultsWrapper and returns a concise interpretation.

    Returns a dict with:
      - "object": dict of numeric results (coef, se, t, p, 95% CI, nobs, adj_r2,
                                   effect_per_1, effect_per_10, significant, conclusion)
      - "description": short plain-English interpretation answering whether a
                       lower student-teacher ratio is associated with higher performance.
    """
    import numpy as np

    var = 'StudentTeacherRatio'
    # Prepare default return if variable missing
    missing_ret = {
        "object": None,
        "description": f"Variable '{var}' not found in the provided model output."
    }

    # Basic validation
    if model_output is None:
        return {
            "object": None,
            "description": "No model output provided."
        }

    # Try to access expected attributes from a statsmodels RegressionResultsWrapper
    try:
        params = model_output.params
        if var not in params.index:
            return missing_ret

        coef = float(params.loc[var])
        se = float(model_output.bse.loc[var])
        t_stat = float(model_output.tvalues.loc[var])
        p_value = float(model_output.pvalues.loc[var])

        # 95% CI
        try:
            ci_lower, ci_upper = model_output.conf_int(alpha=0.05).loc[var].tolist()
            ci_lower = float(ci_lower)
            ci_upper = float(ci_upper)
        except Exception:
            ci_lower, ci_upper = (np.nan, np.nan)

        # sample size and adjusted R-squared if available
        nobs = int(model_output.nobs) if hasattr(model_output, 'nobs') else None
        adj_r2 = float(model_output.rsquared_adj) if hasattr(model_output, 'rsquared_adj') else None

        # Interpret direction and significance
        alpha = 0.05
        significant = (p_value < alpha)

        # Express effect per 1-unit change (one more student per teacher) and per 10-unit change
        effect_per_1 = coef  # change in AvgTestScore per +1 student per teacher
        effect_per_10 = coef * 10

        # Construct conclusion about the research question:
        # The variable is Students per Teacher; a LOWER value means fewer students per teacher.
        if significant:
            if coef < 0:
                conclusion = ("Yes: the estimated coefficient is negative and statistically significant "
                              "(p < {:.2g}), implying that higher student-teacher ratios (more students "
                              "per teacher) are associated with LOWER AvgTestScore. Equivalently, a "
                              "lower student-teacher ratio is associated with HIGHER academic performance."
                              ).format(alpha)
            else:
                conclusion = ("No (opposite): the estimated coefficient is positive and statistically "
                              "significant (p < {:.2g}), implying that higher student-teacher ratios are "
                              "associated with HIGHER AvgTestScore. Equivalently, a lower student-teacher "
                              "ratio would be associated with LOWER academic performance."
                              ).format(alpha)
        else:
            # Not statistically significant
            if coef < 0:
                conclusion = ("Inconclusive (directional but not significant): the coefficient is negative "
                              "but not statistically significant (p = {:.3f}), so we cannot confidently "
                              "conclude that a lower student-teacher ratio is associated with higher performance."
                              ).format(p_value)
            elif coef > 0:
                conclusion = ("Inconclusive (directional but not significant): the coefficient is positive "
                              "but not statistically significant (p = {:.3f}), so we cannot confidently "
                              "conclude that a lower student-teacher ratio is associated with higher performance; "
                              "the (non-significant) point estimate actually suggests the opposite."
                              ).format(p_value)
            else:
                conclusion = ("No effect estimated: coefficient is essentially zero and not statistically significant "
                              "(p = {:.3f}).").format(p_value)

        result_object = {
            "variable": var,
            "coef": coef,
            "std_error": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "95ci": [ci_lower, ci_upper],
            "nobs": nobs,
            "adj_r2": adj_r2,
            "significant_at_0.05": bool(significant),
            "effect_per_1_unit": effect_per_1,
            "effect_per_10_units": effect_per_10,
            "conclusion": conclusion
        }

        description = (
            "Interpretation for StudentTeacherRatio: coef = {coef:.4f} (SE = {se:.4f}, "
            "t = {t:.2f}, p = {p:.3f}), 95% CI = [{cil:.3f}, {ciu:.3f}]. {concl}"
        ).format(coef=coef, se=se, t=t_stat, p=p_value, cil=ci_lower, ciu=ci_upper, concl=conclusion)

        return {"object": result_object, "description": description}

    except Exception as e:
        return {
            "object": None,
            "description": f"Error extracting results from model_output: {e}"
        }