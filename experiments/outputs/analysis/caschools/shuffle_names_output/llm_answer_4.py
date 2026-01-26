def extract_final_answer(model_output):
    """
    Extract coefficient, SE, t-value, p-value, 95% CI and a short interpretation
    for the 'student_teacher_ratio' variable from a fitted statsmodels results object.

    Returns:
      {
        "object": { ... numeric results and flags ... },
        "description": "Plain-language interpretation of the association and caveats."
      }
    """
    import pandas as pd
    import numpy as np

    result = model_output

    # Check that the result object has the expected attributes
    params = getattr(result, "params", None)
    if params is None:
        return {
            "object": None,
            "description": "The provided model_output does not appear to be a fitted statsmodels results object (missing .params)."
        }

    if 'student_teacher_ratio' not in params.index:
        return {
            "object": None,
            "description": "The model does not include 'student_teacher_ratio' as a coefficient."
        }

    # Extract coefficient, standard error, t-value, p-value
    coef = float(params.get('student_teacher_ratio'))
    se = None
    tvalue = None
    pvalue = None
    try:
        bse = getattr(result, "bse", None)
        if bse is not None and 'student_teacher_ratio' in bse.index:
            se = float(bse['student_teacher_ratio'])
    except Exception:
        se = None

    try:
        t = getattr(result, "tvalues", None)
        if t is not None and 'student_teacher_ratio' in t.index:
            tvalue = float(t['student_teacher_ratio'])
    except Exception:
        tvalue = None

    try:
        p = getattr(result, "pvalues", None)
        if p is not None and 'student_teacher_ratio' in p.index:
            pvalue = float(p['student_teacher_ratio'])
    except Exception:
        pvalue = None

    # Confidence interval (robust if model stored that info)
    ci_lower = ci_upper = None
    try:
        ci = result.conf_int(alpha=0.05)
        # conf_int may return ndarray or DataFrame/Series
        if isinstance(ci, (pd.DataFrame, pd.Series)):
            ci_row = ci.loc['student_teacher_ratio']
            # If it's a Series, entries are [lower, upper] or two columns
            if isinstance(ci_row, pd.Series) or isinstance(ci_row, (list, tuple, np.ndarray)):
                ci_lower = float(ci_row.iloc[0])
                ci_upper = float(ci_row.iloc[1])
            else:
                # fallback
                ci_lower, ci_upper = float(ci_row[0]), float(ci_row[1])
        else:
            # numpy array; find index
            idx = list(params.index).index('student_teacher_ratio')
            ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        ci_lower = ci_upper = None

    # Observations
    nobs = None
    try:
        nobs = int(getattr(result, "nobs", None))
    except Exception:
        nobs = None

    # Determine statistical significance at conventional levels
    significant_05 = (pvalue is not None) and (pvalue < 0.05)
    significant_01 = (pvalue is not None) and (pvalue < 0.01)

    # Interpret direction: recall lower student_teacher_ratio means fewer students per teacher
    # Coefficient is change in AvgTestScore per one-unit increase in student_teacher_ratio.
    if coef < 0:
        direction_text = (
            "Negative coefficient: higher student-teacher ratios (more students per teacher) are associated "
            "with lower AvgTestScore; equivalently, lower student-teacher ratios (fewer students per teacher / smaller classes) "
            "are associated with higher AvgTestScore."
        )
        assoc_flag = True
    elif coef > 0:
        direction_text = (
            "Positive coefficient: higher student-teacher ratios are associated with higher AvgTestScore; "
            "equivalently, lower student-teacher ratios are associated with lower AvgTestScore."
        )
        assoc_flag = False
    else:
        direction_text = "Estimated coefficient is exactly zero."
        assoc_flag = False

    # Build human-readable description
    desc_lines = []
    desc_lines.append(f"Estimated coefficient for student_teacher_ratio = {coef:.4f}")
    if se is not None:
        desc_lines.append(f"(SE = {se:.4f})")
    if tvalue is not None:
        desc_lines.append(f"(t = {tvalue:.3f})")
    if pvalue is not None:
        desc_lines.append(f", p = {pvalue:.4g}")
    if ci_lower is not None and ci_upper is not None:
        desc_lines.append(f", 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
    if nobs is not None:
        desc_lines.append(f"; N = {nobs}")
    # join into single sentence-ish string
    stats_summary = " ".join(desc_lines)

    conclusion = direction_text
    if pvalue is not None:
        if significant_05:
            conclusion += " This association is statistically significant at the 5% level."
            if significant_01:
                conclusion += " It is also significant at the 1% level."
        else:
            conclusion += " This association is not statistically significant at conventional levels (p >= 0.05)."
    else:
        conclusion += " p-value could not be determined, so statistical significance is unknown."

    # Caveat about causality and controls
    caveat = (
        "The model controls for expenditure_per_student, pct_reduced_lunch, pct_english_learners, "
        "num_computers, grade-span (C(school)), and county fixed effects (C(county)). "
        "This result is an association from an observational regression and should not be interpreted as causal without stronger identification."
    )

    description = f"{stats_summary}\n{conclusion}\n{caveat}"

    output_obj = {
        "coefficient": coef,
        "std_error": se,
        "t_value": tvalue,
        "p_value": pvalue,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_obs": nobs,
        "significant_at_0.05": bool(significant_05),
        "significant_at_0.01": bool(significant_01),
        # boolean meaning: True => lower student-teacher ratio is associated with higher AvgTestScore
        "lower_ratio_associated_with_higher_performance": bool(assoc_flag and significant_05),
        "direction_interpretation": direction_text
    }

    return {
        "object": output_obj,
        "description": description
    }