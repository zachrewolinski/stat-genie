def extract_final_answer(model_output):
    """
    Extracts coefficient, robust SE, t-stat, p-value, 95% CI, standardized coefficient,
    and sample size for the 'StudentTeacherRatio' predictor from a fitted statsmodels
    RegressionResultsWrapper. Returns a dictionary with keys:
      - "object": dict of numeric results
      - "description": plain-language interpretation of the results in context

    If 'StudentTeacherRatio' is not present or an error occurs, returns a descriptive message.
    """
    import numpy as np

    res = model_output

    # Defensive checks
    try:
        param_names = list(res.params.index)
    except Exception:
        return {
            "object": None,
            "description": "Input model_output does not appear to be a statsmodels results object with .params."
        }

    var = 'StudentTeacherRatio'
    if var not in param_names:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in the model parameters. Available parameters: {param_names}"
        }

    try:
        coef = float(res.params[var])
        se = float(res.bse[var]) if hasattr(res, 'bse') else None
        tval = float(res.tvalues[var]) if hasattr(res, 'tvalues') else None
        pval = float(res.pvalues[var]) if hasattr(res, 'pvalues') else None

        # 95% confidence interval
        try:
            ci = res.conf_int(alpha=0.05)
            ci_lower = float(ci.loc[var, 0]) if hasattr(ci, 'loc') else float(ci[param_names.index(var), 0])
            ci_upper = float(ci.loc[var, 1]) if hasattr(ci, 'loc') else float(ci[param_names.index(var), 1])
        except Exception:
            # fallback if conf_int returns ndarray
            ci_array = res.conf_int(alpha=0.05)
            idx = param_names.index(var)
            ci_lower = float(ci_array[idx, 0])
            ci_upper = float(ci_array[idx, 1])

        # Sample size
        try:
            nobs = int(res.nobs)
        except Exception:
            nobs = None

        # Standardized coefficient: coef * (sd_x / sd_y)
        std_coef = None
        try:
            exog_names = res.model.exog_names
            # find column index for var in exog (exog has constant possibly)
            col_idx = exog_names.index(var)
            x = np.asarray(res.model.exog)[:, col_idx]
            y = np.asarray(res.model.endog)
            sd_x = np.std(x, ddof=1)
            sd_y = np.std(y, ddof=1)
            if sd_y != 0:
                std_coef = float(coef * (sd_x / sd_y))
            else:
                std_coef = None
        except Exception:
            std_coef = None

        # Interpretation: coefficient sign relative to the question:
        # StudentTeacherRatio is defined as students per teacher; lower ratio = fewer students per teacher.
        # If coef < 0, then a one-unit increase in StudentTeacherRatio is associated with a coef decrease in AvgScore,
        # equivalently a one-unit decrease (i.e., lower ratio) is associated with -coef increase in AvgScore.
        if pval is not None:
            significance = ("statistically significant (p < 0.05)"
                            if pval < 0.05 else
                            "not statistically significant (p >= 0.05)")
        else:
            significance = "p-value unavailable"

        if coef < 0:
            direction_text = ("Negative coefficient: higher student-teacher ratio (more students per teacher) "
                              "is associated with lower average scores. Equivalently, a lower student-teacher ratio "
                              "(fewer students per teacher) is associated with higher average scores.")
        elif coef > 0:
            direction_text = ("Positive coefficient: higher student-teacher ratio (more students per teacher) "
                              "is associated with higher average scores. Equivalently, a lower student-teacher ratio "
                              "(fewer students per teacher) is associated with lower average scores.")
        else:
            direction_text = "Coefficient is zero (no association detected)."

        numeric_result = {
            "variable": var,
            "coefficient": coef,
            "std_error": se,
            "t_value": tval,
            "p_value": pval,
            "95%_CI": (ci_lower, ci_upper),
            "standardized_coefficient": std_coef,
            "n_obs": nobs
        }

        # Build a concise description
        desc_lines = [
            f"Estimate for '{var}': coef = {coef:.4f}, SE = {se:.4f}" if se is not None else f"Estimate for '{var}': coef = {coef:.4f}",
            f"t = {tval:.3f}, p = {pval:.3g}" if tval is not None and pval is not None else "",
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]",
            f"Standardized coef = {std_coef:.4f}" if std_coef is not None else "Standardized coef unavailable",
            f"Sample size (observations used) = {nobs}" if nobs is not None else "Sample size unavailable",
            direction_text,
            f"This effect is {significance}."
        ]
        description = " ".join([s for s in desc_lines if s])

        return {
            "object": numeric_result,
            "description": description
        }

    except Exception as e:
        return {
            "object": None,
            "description": f"An error occurred while extracting statistics: {e}"
        }