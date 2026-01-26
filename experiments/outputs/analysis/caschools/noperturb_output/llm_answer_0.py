def extract_final_answer(model_output):
    """
    Extracts statistics about the StudentTeacherRatio_log coefficient from the
    provided model_output (either a dict containing 'fitted_model' and optional 'vif',
    or directly a statsmodels results object).

    Returns a dict with keys:
      - "object": dict with numeric results (coef, p-value, 95% CI, VIF if available,
                  effect for 10% and 20% decreases in student-teacher ratio)
      - "description": plain-English interpretation focused on whether a lower
                       student-teacher ratio is associated with higher academic performance.
    """
    import numpy as np

    # Accept either the dict output or a raw results object
    if isinstance(model_output, dict) and 'fitted_model' in model_output:
        res = model_output['fitted_model']
        vif_df = model_output.get('vif', None)
    else:
        res = model_output
        vif_df = None

    var = 'StudentTeacherRatio_log'

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("Provided model_output does not appear to be a statsmodels results object or dict with 'fitted_model'.")

    if var not in res.params.index:
        raise KeyError(f"Variable '{var}' not found in model coefficients. Available coeffs: {list(res.params.index)}")

    # Extract statistics
    coef = float(res.params[var])
    pval = float(res.pvalues[var]) if var in res.pvalues.index else None

    # 95% CI
    try:
        ci_lower, ci_upper = res.conf_int().loc[var].astype(float).tolist()
    except Exception:
        ci_lower, ci_upper = (None, None)

    # VIF if available
    vif_value = None
    if vif_df is not None:
        # handle DataFrame with columns 'variable' and 'VIF'
        try:
            mask = vif_df['variable'] == var
            if mask.any():
                vif_value = float(vif_df.loc[mask, 'VIF'].iloc[0])
        except Exception:
            vif_value = None

    # Compute approximate absolute effects on AvgScore for example percent reductions
    # (since the predictor is ln(ratio), ΔY ≈ coef * Δln(ratio); for a p% decrease use ln(1-p))
    def effect_for_pct_decrease(pct):
        if not (0 < pct < 1):
            return None
        delta_ln = np.log(1 - pct)
        return float(coef * delta_ln)

    effect_10pct = effect_for_pct_decrease(0.10)  # 10% decrease in student/teacher ratio
    effect_20pct = effect_for_pct_decrease(0.20)  # 20% decrease

    # Statistical significance (two-sided)
    significant = None
    if pval is not None:
        significant = (pval < 0.05)

    # Build output object
    out_obj = {
        'variable': var,
        'coef': coef,
        'p_value': pval,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'significant_at_0.05': significant,
        'vif': vif_value,
        'effect_10pct_decrease_in_ratio__avgscore_points': effect_10pct,
        'effect_20pct_decrease_in_ratio__avgscore_points': effect_20pct,
    }

    # Description / interpretation
    if significant is True:
        sign_text = "negative" if coef < 0 else "positive"
        desc = (
            f"The estimated coefficient on {var} is {coef:.3f} (95% CI [{ci_lower:.3f}, {ci_upper:.3f}], "
            f"p = {pval:.3f}), which is statistically significant at the 0.05 level. "
            f"Because the predictor is ln(students/teachers), a lower student-teacher ratio "
            f"(i.e., a decrease in this logged ratio) is associated with a {sign_text} change "
            f"in AvgScore. For example, a 10% reduction in the student-teacher ratio is "
            f"associated with an average change of about {effect_10pct:.3f} AvgScore points. "
            f"VIF for this predictor is {vif_value if vif_value is not None else 'N/A'}."
        )
    else:
        # Not statistically significant: cannot conclude
        desc = (
            f"The estimated coefficient on {var} is {coef:.3f} (95% CI [{ci_lower:.3f}, {ci_upper:.3f}], "
            f"p = {pval:.3f}). The coefficient is negative, which would imply that a lower "
            f"student-teacher ratio (fewer students per teacher) is associated with higher AvgScore in point estimate, "
            f"but this effect is not statistically significant (p = {pval:.3f}). "
            f"Practical-size examples: a 10% reduction in the student-teacher ratio corresponds to an estimated "
            f"AvgScore increase of about {effect_10pct:.3f} points, and a 20% reduction corresponds to about "
            f"{effect_20pct:.3f} points. The VIF for the predictor is {vif_value if vif_value is not None else 'N/A'}, "
            f"so multicollinearity is unlikely to be driving the null result for this variable."
        )

    return {
        "object": out_obj,
        "description": desc
    }