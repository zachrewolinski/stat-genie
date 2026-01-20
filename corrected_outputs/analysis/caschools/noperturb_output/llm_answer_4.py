def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a statsmodels
    OLS RegressionResultsWrapper and returns a concise interpretation.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results:
            {
              "coef": float,
              "std_err": float,
              "t_value": float,
              "p_value": float,
              "ci_lower": float,
              "ci_upper": float,
              "standardized_coef": float (beta standardized by SD of x and y)
            }
      - "description": a short plain-language interpretation answering whether
                       a lower student-teacher ratio is associated with higher
                       academic performance, given the sign and significance.
    """
    import numpy as np
    import pandas as pd

    # Parameter name used in the model
    param_name = 'StudentTeacherRatio'

    # Attempt to get parameter-related series (works if params is a pandas Series)
    try:
        params = model_output.params
        b = float(params[param_name])
    except Exception:
        # Fallback: construct params Series from arrays and exog_names
        try:
            names = list(model_output.model.exog_names)
            params = pd.Series(model_output.params, index=names)
            b = float(params[param_name])
        except Exception as e:
            raise RuntimeError(f"Could not extract parameters from model_output: {e}")

    # Standard error, t-value, p-value
    try:
        se = float(model_output.bse[param_name])
        tval = float(model_output.tvalues[param_name])
        pval = float(model_output.pvalues[param_name])
    except Exception:
        # Fallback using positional index
        names = list(model_output.model.exog_names)
        idx = names.index(param_name)
        se = float(model_output.bse[idx])
        tval = float(model_output.tvalues[idx])
        pval = float(model_output.pvalues[idx])

    # Confidence interval (95%)
    try:
        ci = model_output.conf_int(alpha=0.05)
        # ci may be DataFrame/NDArray. Try to index by name first.
        if isinstance(ci, (pd.DataFrame, pd.Series)):
            ci_row = ci.loc[param_name]
            ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
        else:
            # ndarray: find positional index
            names = list(model_output.model.exog_names)
            idx = names.index(param_name)
            ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        # As a last resort compute approximate CI from coef +/- 1.96*se
        ci_lower = b - 1.96 * se
        ci_upper = b + 1.96 * se

    # Compute a standardized coefficient (beta * std(x) / std(y)) if possible
    try:
        exog = np.asarray(model_output.model.exog)
        endog = np.asarray(model_output.model.endog).reshape(-1)
        names = list(model_output.model.exog_names)
        idx = names.index(param_name)
        x = exog[:, idx]
        std_x = np.std(x, ddof=0)
        std_y = np.std(endog, ddof=0)
        if std_y > 0:
            standardized_coef = float(b * (std_x / std_y))
        else:
            standardized_coef = None
    except Exception:
        standardized_coef = None

    # Interpretation: Because StudentTeacherRatio is students per teacher,
    # a negative coefficient implies that higher student-teacher ratio is
    # associated with lower AvgScore, i.e., lower ratio (fewer students per teacher)
    # is associated with higher AvgScore.
    sig_level = 0.05
    if pval < sig_level:
        significance = "statistically significant"
    else:
        significance = "not statistically significant"

    if b < 0:
        direction_text = ("The estimated effect is negative: higher student-teacher "
                          "ratio is associated with LOWER average scores. "
                          "Equivalently, a LOWER student-teacher ratio (fewer students "
                          "per teacher, i.e., smaller class sizes) is associated with "
                          "HIGHER academic performance.")
    elif b > 0:
        direction_text = ("The estimated effect is positive: higher student-teacher "
                          "ratio is associated with HIGHER average scores. "
                          "Equivalently, a LOWER student-teacher ratio would be "
                          "associated with LOWER academic performance.")
    else:
        direction_text = "The estimated effect is exactly zero."

    conclusion = (f"StudentTeacherRatio coef = {b:.4f} (SE = {se:.4f}, t = {tval:.2f}, "
                  f"p = {pval:.3f}), 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
                  f"This effect is {significance}. {direction_text}")

    # Build object to return
    result_object = {
        "coef": b,
        "std_err": se,
        "t_value": tval,
        "p_value": pval,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "standardized_coef": standardized_coef
    }

    return {
        "object": result_object,
        "description": conclusion
    }