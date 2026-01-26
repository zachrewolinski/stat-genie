def extract_final_answer(model_output):
    """
    Extract statistics for the z_StudentTeacherRatio coefficient from a fitted
    statsmodels RegressionResultsWrapper (or similar) and produce a brief
    interpretation about whether a lower student-teacher ratio is associated
    with higher academic performance.

    Returns:
      dict with keys:
        - "object": dict of extracted numeric results:
            { "coef", "se", "t", "p_value", "ci_lower", "ci_upper",
              "per_sd_decrease_effect" }
            per_sd_decrease_effect = effect on AvgScore for a one-SD decrease
                                     in student-teacher ratio (i.e., -coef).
        - "description": short text interpreting the numbers in context
    """
    # Name of the variable of interest in the model
    var = "z_StudentTeacherRatio"

    # Try to extract parameters, standard errors, t-values, p-values, CI
    try:
        params = model_output.params
        bse = model_output.bse
        tvals = model_output.tvalues
        pvals = model_output.pvalues
        ci = model_output.conf_int(alpha=0.05)
    except Exception as e:
        raise ValueError("model_output does not expose expected statsmodels attributes: " + str(e))

    if var not in params.index:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    coef = float(params.loc[var])
    se = float(bse.loc[var]) if var in bse.index else None
    t = float(tvals.loc[var]) if var in tvals.index else None
    p = float(pvals.loc[var]) if var in pvals.index else None
    # conf_int may be DataFrame or ndarray; handle both
    try:
        ci_lower = float(ci.loc[var, 0])
        ci_upper = float(ci.loc[var, 1])
    except Exception:
        # fallback for ndarray-like
        try:
            idx = list(params.index).index(var)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
        except Exception as e:
            ci_lower = ci_upper = None

    # Interpretation: coefficient is change in AvgScore per 1 SD increase in ratio.
    # A one-SD decrease in ratio has effect = -coef.
    per_sd_decrease_effect = -coef

    # Determine whether association supports "lower ratio associated with higher performance"
    alpha = 0.05
    if p is not None:
        statistically_significant = (p < alpha)
    else:
        statistically_significant = None

    if statistically_significant is True:
        if coef < 0:
            conclusion = "Yes: statistically significant. The negative coefficient means a lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore."
            conclusion_bool = True
        else:
            conclusion = "No (statistically significant in the opposite direction). The positive coefficient means a lower student-teacher ratio is associated with lower AvgScore."
            conclusion_bool = False
    else:
        conclusion = "No (not statistically significant). There is no strong evidence that student-teacher ratio is associated with AvgScore after controls."
        conclusion_bool = None

    # Prepare the object to return (numeric results + short conclusion flag)
    result_object = {
        "variable": var,
        "coef": round(coef, 4),
        "se": round(se, 4) if se is not None else None,
        "t": round(t, 4) if t is not None else None,
        "p_value": round(p, 4) if p is not None else None,
        "ci_lower": round(ci_lower, 4) if ci_lower is not None else None,
        "ci_upper": round(ci_upper, 4) if ci_upper is not None else None,
        "per_sd_decrease_effect": round(per_sd_decrease_effect, 4),
        "statistically_significant_at_0.05": statistically_significant,
        "conclusion_boolean": conclusion_bool
    }

    # Compose a concise description
    description = (
        f"Coefficient on {var} = {result_object['coef']} (SE = {result_object['se']}, "
        f"t = {result_object['t']}, p = {result_object['p_value']}); 95% CI = "
        f"[{result_object['ci_lower']}, {result_object['ci_upper']}]. "
        f"This coefficient is the change in AvgScore associated with a one-standard-deviation "
        f"increase in student-teacher ratio. Therefore a one-standard-deviation decrease in "
        f"the ratio is associated with a change of {result_object['per_sd_decrease_effect']} in AvgScore. "
        f"Conclusion: {conclusion}"
    )

    return {"object": result_object, "description": description}