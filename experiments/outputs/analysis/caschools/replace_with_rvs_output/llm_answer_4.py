def extract_final_answer(model_output):
    """
    Extracts coef, robust SE, t, p-value, and 95% CI for the StudentTeacherRatio term
    from a statsmodels RegressionResultsWrapper and returns a short interpretation.

    Returns a dict with keys:
      - "object": dict with numeric results (coef, std_err, t, p_value, ci_lower, ci_upper)
      - "description": brief interpretation about whether a lower student-teacher ratio
                       is associated with higher academic performance.
    """
    import pandas as pd
    import numpy as np

    res = model_output
    term = 'StudentTeacherRatio'

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object (missing .params).")

    if term not in res.params.index:
        raise KeyError(f"Term '{term}' not found in model parameters. Available params: {list(res.params.index)}")

    coef = float(res.params[term])
    std_err = float(res.bse[term])          # robust SE should already be in .bse because cov_type was used in fit()
    t_value = float(res.tvalues[term])
    p_value = float(res.pvalues[term])

    # Try to get conf_int from the model; fall back to normal approx if not available
    try:
        ci = res.conf_int()
        # conf_int may be a DataFrame (newer versions) or ndarray
        if isinstance(ci, pd.DataFrame):
            ci_lower = float(ci.loc[term, 0])
            ci_upper = float(ci.loc[term, 1])
        else:
            # ndarray: locate row by parameter index
            idx = list(res.params.index).index(term)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
    except Exception:
        # normal (approximate) 95% CI
        ci_lower = coef - 1.96 * std_err
        ci_upper = coef + 1.96 * std_err

    # Interpretation about direction and significance.
    # Note: StudentTeacherRatio = students per teacher. A negative coef means fewer students per teacher
    # (lower ratio) is associated with higher AvgScore.
    if p_value < 0.05:
        if coef < 0:
            conclusion = ("Yes — statistically significant negative association: "
                          "lower student-teacher ratio (smaller classes / more teachers per student) "
                          "is associated with higher average test scores.")
        else:
            conclusion = ("Yes — statistically significant positive association: "
                          "higher student-teacher ratio is associated with higher average test scores "
                          "(counterintuitive direction).")
    else:
        if coef < 0:
            conclusion = ("No statistically significant association: coefficient is negative (suggesting "
                          "lower ratio -> higher scores) but not statistically significant at conventional levels.")
        else:
            conclusion = ("No statistically significant association: coefficient is positive but not statistically significant.")

    description = (
        f"{conclusion} "
        f"Estimate for {term}: coef = {coef:.4f}, SE = {std_err:.4f}, t = {t_value:.3f}, p = {p_value:.3g}, "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. "
        "Because StudentTeacherRatio is students per teacher, a negative coefficient means that decreasing the ratio "
        "(i.e., smaller class size / more teachers per student) is associated with higher AvgScore."
    )

    return {
        "object": {
            "coef": coef,
            "std_err": std_err,
            "t": t_value,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        },
        "description": description
    }