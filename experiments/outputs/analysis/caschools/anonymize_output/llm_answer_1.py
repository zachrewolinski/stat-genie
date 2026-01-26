def extract_final_answer(model_output):
    """
    Extract key statistics about the StudentTeacherRatio coefficient from a statsmodels
    RegressionResultsWrapper and provide a short interpretation answering whether a
    lower student-teacher ratio (fewer students per teacher) is associated with higher
    academic performance (AvgTestScore).

    Returns:
      dict with keys:
        - "object": dict of extracted numeric results and a final conclusion flag/text
        - "description": human-readable explanation of the numbers and conclusion
    """
    # Name of the coefficient in the model
    name = 'StudentTeacherRatio'

    # Prepare a friendly error if the parameter is missing
    try:
        params = model_output.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not access model parameters: {e}"
        }

    if name not in params.index:
        return {
            "object": None,
            "description": f"Parameter '{name}' not found in the model output. Available parameters: {list(params.index)}"
        }

    # Extract coefficient and statistics (robust SEs should already be in model_output.bse)
    coef = float(params[name])
    try:
        se = float(model_output.bse[name])
    except Exception:
        # fallback: compute from t-values if available
        try:
            se = float(coef / model_output.tvalues[name])
        except Exception:
            se = None

    try:
        tval = float(model_output.tvalues[name])
    except Exception:
        tval = None

    try:
        pval = float(model_output.pvalues[name])
    except Exception:
        pval = None

    # Confidence interval retrieval with robust handling
    try:
        ci = model_output.conf_int()  # may be DataFrame or ndarray-like
        if hasattr(ci, 'loc'):
            ci_lower, ci_upper = ci.loc[name].astype(float).tolist()
        else:
            # ci is ndarray; find index of parameter
            idx = list(params.index).index(name)
            ci_lower, ci_upper = float(ci[idx, 0]), float(ci[idx, 1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Interpret sign and statistical significance
    significance_level = 0.05
    significant = (pval is not None) and (pval < significance_level)

    if coef < 0:
        direction = "negative"
        direction_text = ("A negative coefficient means that higher student-teacher ratios "
                          "(more students per teacher, i.e., larger classes) are associated "
                          "with lower average test scores. Equivalently, lower student-teacher "
                          "ratios (fewer students per teacher, i.e., smaller classes) are "
                          "associated with higher test scores.")
    elif coef > 0:
        direction = "positive"
        direction_text = ("A positive coefficient means that higher student-teacher ratios "
                          "(more students per teacher) are associated with higher average test scores.")
    else:
        direction = "zero"
        direction_text = "The coefficient is zero (no estimated association)."

    # Formulate a concise conclusion answering the yes/no question
    if significant:
        if coef < 0:
            conclusion_short = "Yes — statistically significant: lower student-teacher ratio is associated with higher academic performance."
        elif coef > 0:
            conclusion_short = "Yes — statistically significant: lower student-teacher ratio is associated with LOWER academic performance (coefficient positive)."
        else:
            conclusion_short = "No — coefficient is (near) zero despite statistical significance (unusual)."
    else:
        # Not statistically significant
        if coef < 0:
            conclusion_short = ("No statistically significant evidence that lower student-teacher ratio is associated "
                                "with higher academic performance (coefficient negative but p >= 0.05).")
        elif coef > 0:
            conclusion_short = ("No statistically significant evidence that lower student-teacher ratio is associated "
                                "with higher academic performance (coefficient positive and p >= 0.05).")
        else:
            conclusion_short = "No statistically significant association (coefficient near zero and p >= 0.05)."

    # Build the object to return (serializable)
    result_object = {
        "variable": name,
        "coef": float(coef),
        "std_err": float(se) if se is not None else None,
        "t_value": float(tval) if tval is not None else None,
        "p_value": float(pval) if pval is not None else None,
        "ci_lower_95": float(ci_lower) if ci_lower is not None else None,
        "ci_upper_95": float(ci_upper) if ci_upper is not None else None,
        "significant_at_0.05": bool(significant),
        "direction": direction,
        "conclusion": conclusion_short
    }

    # Human-readable description
    desc_lines = [
        f"Coefficient for {name}: {result_object['coef']:.4f}",
    ]
    if result_object["std_err"] is not None:
        desc_lines.append(f"Std. error (HC3 robust): {result_object['std_err']:.4f}")
    if result_object["t_value"] is not None:
        desc_lines.append(f"t-value: {result_object['t_value']:.3f}")
    if result_object["p_value"] is not None:
        desc_lines.append(f"p-value: {result_object['p_value']:.4f}")
    if (result_object["ci_lower_95"] is not None) and (result_object["ci_upper_95"] is not None):
        desc_lines.append(f"95% CI: [{result_object['ci_lower_95']:.4f}, {result_object['ci_upper_95']:.4f}]")
    desc_lines.append(direction_text)
    desc_lines.append(f"Statistical significance at alpha={significance_level}: {result_object['significant_at_0.05']}")
    desc_lines.append("Conclusion: " + result_object["conclusion"])

    description = " ".join(desc_lines)

    return {
        "object": result_object,
        "description": description
    }