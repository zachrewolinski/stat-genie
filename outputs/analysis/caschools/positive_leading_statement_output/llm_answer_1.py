def extract_final_answer(model_output):
    """
    Extracts the estimate for StudentTeacherRatio from a fitted statsmodels OLS result,
    computes a 95% confidence interval, p-value, robust standard error (if present),
    a standardized coefficient, sample size, and a short conclusion about whether
    a lower student-teacher ratio is associated with higher academic performance.

    Returns:
      {
        "object": { ... detailed numeric results ... },
        "description": "Plain-language interpretation and conclusion"
      }
    """
    import numpy as np
    import pandas as pd

    # Ensure model_output looks like a statsmodels results object
    try:
        params = model_output.params
        pvalues = model_output.pvalues
        bse = model_output.bse
    except Exception as e:
        raise ValueError("model_output does not look like a statsmodels results object") from e

    varname = 'StudentTeacherRatio'
    if varname not in params.index:
        raise KeyError(f"Variable '{varname}' not found in model parameters. Available params: {list(params.index)}")

    coef = float(params[varname])
    pval = float(pvalues[varname])
    se = float(bse[varname]) if varname in bse.index else None

    # Confidence interval (95%)
    try:
        ci_df = model_output.conf_int(alpha=0.05)
        if isinstance(ci_df, pd.DataFrame):
            ci_low, ci_high = float(ci_df.loc[varname, 0]), float(ci_df.loc[varname, 1])
        else:
            # conf_int returned ndarray-like; find index
            idx = list(params.index).index(varname)
            ci_low, ci_high = float(ci_df[idx, 0]), float(ci_df[idx, 1])
    except Exception:
        ci_low, ci_high = None, None

    # Sample size
    try:
        nobs = int(model_output.nobs)
    except Exception:
        try:
            nobs = int(len(model_output.model.endog))
        except Exception:
            nobs = None

    # Standardized coefficient: beta_std = beta * (sd_X / sd_Y)
    std_beta = None
    try:
        exog = model_output.model.exog
        exog_names = list(model_output.model.exog_names)
        if varname in exog_names:
            col_idx = exog_names.index(varname)
            x = np.asarray(exog[:, col_idx], dtype=float)
            y = np.asarray(model_output.model.endog, dtype=float)
            sd_x = np.std(x, ddof=1)
            sd_y = np.std(y, ddof=1)
            if sd_x > 0 and sd_y > 0:
                std_beta = float(coef * (sd_x / sd_y))
    except Exception:
        std_beta = None

    # Determine statistical significance at alpha = 0.05
    significant = (pval < 0.05)

    # Interpret direction:
    # StudentTeacherRatio = number of students per teacher. Lower values => fewer students per teacher.
    # A negative coefficient means higher ratio -> lower scores, equivalently lower ratio -> higher scores.
    if coef < 0:
        direction_text = "Lower student-teacher ratio (smaller class sizes) is associated with higher AvgScore (negative coefficient)."
    elif coef > 0:
        direction_text = "Lower student-teacher ratio (smaller class sizes) is associated with lower AvgScore (positive coefficient)."
    else:
        direction_text = "No association (coefficient is zero)."

    # Conclusion text depending on significance
    if significant:
        conclusion = ("Statistically significant association detected at alpha=0.05. "
                      "Interpretation: " + direction_text)
    else:
        conclusion = ("No statistically significant association detected at alpha=0.05. "
                      "The point estimate suggests: " + direction_text +
                      " But the effect is not statistically different from zero, so evidence is inconclusive.")

    result_object = {
        "variable": varname,
        "coef": coef,
        "se": se,
        "p_value": pval,
        "ci_95": (ci_low, ci_high),
        "std_coef": std_beta,
        "nobs": nobs,
        "significant_at_0.05": bool(significant)
    }

    description = (
        f"Estimated effect of {varname} on AvgScore (OLS with controls and county/grade fixed effects; robust SEs): "
        f"coef = {coef:.4f}, se = {se:.4f} " if (se is not None) else
        f"Estimated effect of {varname} on AvgScore (OLS with controls and fixed effects): coef = {coef:.4f}, "
    )
    # Append CI and p-value
    if (ci_low is not None) and (ci_high is not None):
        description += f", 95% CI = [{ci_low:.4f}, {ci_high:.4f}], p = {pval:.4g}. "
    else:
        description += f", p = {pval:.4g}. "

    if std_beta is not None:
        description += f"Standardized coefficient = {std_beta:.4f}. "

    description += conclusion

    return {"object": result_object, "description": description}