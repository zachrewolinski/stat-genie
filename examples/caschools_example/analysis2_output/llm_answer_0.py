def extract_final_answer(model_output):
    """
    Extract statistics for the StudentTeacherRatio_z coefficient from a fitted statsmodels OLS result.

    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coefficient, std err, t, p, 95% CI, significance flag)
      - "description": a short interpretation of what these numbers mean for the question:
                       "Is a lower student-teacher ratio associated with higher academic performance?"
    """
    res = model_output

    # Basic validation
    if res is None:
        raise ValueError("model_output is None")
    if not hasattr(res, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object (missing 'params').")

    key = "StudentTeacherRatio_z"
    if key not in res.params.index:
        raise ValueError(f"Coefficient '{key}' not found in model results. Available params: {list(res.params.index)}")

    # Extract statistics, converting to native Python types
    coef = float(res.params.loc[key])
    se = float(res.bse.loc[key]) if hasattr(res, "bse") else None
    tval = float(res.tvalues.loc[key]) if hasattr(res, "tvalues") else None
    pval = float(res.pvalues.loc[key]) if hasattr(res, "pvalues") else None

    # 95% confidence interval
    try:
        ci = res.conf_int(alpha=0.05).loc[key].tolist()
        # ensure floats
        ci = [float(ci[0]), float(ci[1])]
    except Exception:
        ci = None

    # Interpret significance at alpha = 0.05
    significant = (pval is not None) and (pval < 0.05)

    # Direction interpretation:
    if coef < 0:
        direction_text = "negative"
        conclusion_text = (
            "There is a statistically significant negative association"
            if significant
            else "There is a negative association, but it is not statistically significant"
        )
        implication = "Lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore."
    elif coef > 0:
        direction_text = "positive"
        conclusion_text = (
            "There is a statistically significant positive association"
            if significant
            else "There is a positive association, but it is not statistically significant"
        )
        implication = "Higher student-teacher ratio (more students per teacher) is associated with higher AvgScore."
    else:
        direction_text = "zero"
        conclusion_text = "No association (coefficient is zero)."
        implication = "No directional association detected."

    # Put together the object to return
    result_object = {
        "variable": key,
        "coef": coef,
        "std_err": se,
        "t_value": tval,
        "p_value": pval,
        "95%_CI": ci,
        "significant_at_0.05": bool(significant),
        "direction": direction_text,
        "conclusion_short": conclusion_text,
        "implication": implication,
        "note": (
            "StudentTeacherRatio_z is standardized (z-score). "
            "Coef is the change in AvgScore (raw score) associated with a 1 SD increase in student-teacher ratio. "
            "We use alpha = 0.05 for significance."
        ),
    }

    # Human-readable description
    if significant:
        description = (
            f"The coefficient for {key} = {coef:.4f} (SE = {se:.4f}, t = {tval:.2f}, p = {pval:.3g}). "
            f"The 95% CI is [{ci[0]:.4f}, {ci[1]:.4f}] .\n"
            f"Because the coefficient is {direction_text} and statistically significant (p < 0.05), "
            f"{implication} This effect size is per 1 SD change in the student-teacher ratio."
        )
    else:
        # not significant
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci is not None else "N/A"
        description = (
            f"The coefficient for {key} = {coef:.4f} (SE = {se:.4f}, t = {tval:.2f}, p = {pval:.3g}). "
            f"The 95% CI is {ci_text}.\n"
            f"Because p >= 0.05, there is no strong evidence of a statistically significant association between "
            f"student-teacher ratio and district average academic performance in this model. "
            f"The estimated association is {direction_text} but not statistically distinguishable from zero at alpha = 0.05."
        )

    return {"object": result_object, "description": description}