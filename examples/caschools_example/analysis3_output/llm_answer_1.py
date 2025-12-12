def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, p-value, confidence interval, and simple interpretations
    for the effect of LogStudentTeacherRatio on AcademicPerformance from a fitted
    statsmodels RegressionResultsWrapper (or compatible) object.

    Returns a dict with keys:
      - "object": a dictionary of numeric results and interpretation flags
      - "description": a concise human-readable interpretation of the results
    """
    import numpy as np
    import math

    res = model_output

    # Find the parameter name in a case-insensitive / fuzzy way
    target_tokens = ['log', 'student', 'teacher', 'ratio']
    param_key = None
    try:
        param_index = list(res.params.index)
    except Exception as e:
        raise ValueError(f"Could not read params from model_output: {e}")

    for name in param_index:
        lname = str(name).lower()
        if all(tok in lname for tok in ['student', 'teacher']) and ('log' in lname or 'ln' in lname):
            param_key = name
            break
    # Fallback to exact name if not found
    if param_key is None:
        for candidate in ['LogStudentTeacherRatio', 'logstudentteacherratio', 'log_student_teacher_ratio', 'log_student_teacher']:
            for name in param_index:
                if str(name).lower() == candidate.lower():
                    param_key = name
                    break
            if param_key is not None:
                break
    # Final fallback: look for any param containing both 'student' and 'teacher'
    if param_key is None:
        for name in param_index:
            lname = str(name).lower()
            if 'student' in lname and 'teacher' in lname:
                param_key = name
                break

    if param_key is None:
        raise KeyError("Could not locate the model parameter for LogStudentTeacherRatio in model_output.params.")

    # Safely extract statistics
    def safe_get(series_like, key):
        try:
            return float(series_like[key])
        except Exception:
            return None

    coef = safe_get(res.params, param_key)
    se = safe_get(getattr(res, 'bse', {}), param_key)
    pval = safe_get(getattr(res, 'pvalues', {}), param_key)

    # Confidence interval (may return a DataFrame)
    ci_low, ci_high = (None, None)
    try:
        ci = res.conf_int()
        if param_key in ci.index:
            ci_low = float(ci.loc[param_key, 0])
            ci_high = float(ci.loc[param_key, 1])
        else:
            # try by position match
            idx = list(ci.index).index(param_key) if param_key in list(ci.index) else None
    except Exception:
        ci = None

    # Sample size and fit stats
    try:
        nobs = int(res.nobs)
    except Exception:
        try:
            nobs = int(getattr(res.model, 'nobs', np.nan))
        except Exception:
            nobs = None
    try:
        r_squared = float(res.rsquared)
    except Exception:
        r_squared = None
    try:
        adj_r_squared = float(res.rsquared_adj)
    except Exception:
        adj_r_squared = None

    # Interpretations:
    # - Coefficient sign: if coef < 0 then higher student-teacher ratio (more students per teacher)
    #   is associated with lower academic performance, implying that lower ratio (fewer students
    #   per teacher) is associated with higher performance.
    supports_hypothesis = None
    evidence_strength = "insufficient"
    if coef is not None and pval is not None:
        # We consider conventional p < 0.05 as evidence
        supports_hypothesis = (coef < 0) and (pval < 0.05)
        if pval < 0.01:
            evidence_strength = "strong"
        elif pval < 0.05:
            evidence_strength = "moderate"
        elif pval < 0.1:
            evidence_strength = "weak"
        else:
            evidence_strength = "none"
    else:
        supports_hypothesis = None

    # Compute effect sizes for convenience:
    # - Change in AcademicPerformance associated with a 10% decrease in student-teacher ratio:
    #     delta_score_10pct_decrease = coef * log(0.9)
    # - Change associated with a 10% increase: coef * log(1.10)
    effect_10pct_decrease = None
    effect_10pct_increase = None
    if coef is not None:
        effect_10pct_increase = coef * math.log(1.10)  # approx coef * 0.09531
        effect_10pct_decrease = coef * math.log(0.90)  # approx coef * -0.10536

    # Build the object to return
    result_object = {
        'parameter_name': str(param_key),
        'coefficient': coef,
        'std_error': se,
        'p_value': pval,
        'conf_int_95_low': ci_low,
        'conf_int_95_high': ci_high,
        'nobs': nobs,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'effect_10pct_decrease_in_ratio__score_change': effect_10pct_decrease,
        'effect_10pct_increase_in_ratio__score_change': effect_10pct_increase,
        'supports_hypothesis_lower_ratio_associated_with_higher_performance': supports_hypothesis,
        'evidence_strength': evidence_strength
    }

    # Compose a human-readable description
    if coef is None:
        description = "Could not extract the coefficient for LogStudentTeacherRatio from the model output."
    else:
        sign_phrase = ("negative" if coef < 0 else "positive" if coef > 0 else "zero")
        sig_phrase = ""
        if pval is not None:
            sig_phrase = f" (p = {pval:.3g})"
        description_lines = []
        description_lines.append(f"The estimated coefficient on {param_key} is {coef:.4g} with a {sign_phrase} sign{sig_phrase}.")
        if ci_low is not None and ci_high is not None:
            description_lines.append(f"95% CI: [{ci_low:.4g}, {ci_high:.4g}].")
        if effect_10pct_decrease is not None:
            # Interpret direction relative to hypothesis
            direction = "increase" if effect_10pct_decrease > 0 else "decrease" if effect_10pct_decrease < 0 else "no change"
            description_lines.append(
                f"A 10% decrease in the student-teacher ratio is associated with an expected change of "
                f"{effect_10pct_decrease:.4g} points in AcademicPerformance ({direction})."
            )
        if supports_hypothesis is True:
            description_lines.append(
                f"Conclusion: The estimate is consistent with the hypothesis that a lower student-teacher ratio "
                f"is associated with higher academic performance; evidence is {evidence_strength} (p = {pval:.3g})."
            )
        elif supports_hypothesis is False:
            description_lines.append(
                f"Conclusion: The estimate does not support the hypothesis. The coefficient sign and/or significance "
                f"is inconsistent with lower ratios improving performance (p = {pval:.3g})."
            )
        else:
            description_lines.append(
                "Conclusion: Unable to determine strong support due to missing statistical information (e.g., p-value)."
            )
        # include sample size and fit
        if nobs is not None:
            description_lines.append(f"Model sample size: {nobs}. R-squared: {r_squared:.3g} (adj: {adj_r_squared:.3g}).")
        description = " ".join(description_lines)

    return {"object": result_object, "description": description}