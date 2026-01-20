def extract_final_answer(model_output):
    """
    Extracts the coefficient, p-value, t-value, and 95% CI for the StudentTeacherRatio
    term from a statsmodels RegressionResultsWrapper and returns a short interpretation.

    Returns:
      {
        "object": {
          "coefficient": float,
          "p_value": float,
          "t_value": float,
          "conf_int_95": [lower, upper],
          "conclusion": "text summary (directional)",
          "stat_significance": "text about significance"
        },
        "description": "A one-sentence explanation of what the numbers mean in context."
      }
    """
    res = model_output
    try:
        coef = float(res.params['StudentTeacherRatio'])
        pval = float(res.pvalues['StudentTeacherRatio'])
        tval = float(res.tvalues['StudentTeacherRatio'])
        ci = res.conf_int().loc['StudentTeacherRatio'].tolist()  # [lower, upper]
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract StudentTeacherRatio from model output: {e}"
        }

    # Determine direction and significance language
    if coef < 0:
        direction_text = ("A negative coefficient: lower student-teacher ratio (fewer students per teacher) "
                          "is associated with higher AvgScore.")
    elif coef > 0:
        direction_text = ("A positive coefficient: higher student-teacher ratio (more students per teacher) "
                          "is associated with higher AvgScore.")
    else:
        direction_text = "Coefficient is exactly zero (no association)."

    if pval < 0.05:
        sig_text = "The relationship is statistically significant at the 5% level (p < 0.05)."
    elif pval < 0.10:
        sig_text = "The relationship is marginally significant at the 10% level (p < 0.10)."
    else:
        sig_text = "The relationship is not statistically significant (p >= 0.10)."

    conclusion = (
        f"StudentTeacherRatio coef = {coef:.4f}, p = {pval:.4f}, t = {tval:.3f}, "
        f"95% CI = ({ci[0]:.4f}, {ci[1]:.4f}). {direction_text} {sig_text} "
        "Interpretation: this coefficient shows the expected change in AvgScore "
        "for a one-unit change in StudentTeacherRatio (one additional student per teacher)."
    )

    object_out = {
        "coefficient": coef,
        "p_value": pval,
        "t_value": tval,
        "conf_int_95": ci,
        "conclusion": direction_text,
        "stat_significance": sig_text
    }

    return {"object": object_out, "description": conclusion}