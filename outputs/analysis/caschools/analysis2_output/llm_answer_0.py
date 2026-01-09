def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, t-stat, p-value, 95% CI, sample size, and R-squared
    for the StudentTeacherRatio variable from a fitted statsmodels OLSResults object.
    Returns a dictionary with keys:
      - "object": dict of numeric results
      - "description": human-readable interpretation about whether lower student-teacher
                       ratio is associated with higher AvgScore.
    """
    result = {
        "object": None,
        "description": ""
    }

    var_base = "StudentTeacherRatio"

    # Helper to find the exact parameter name in the model results
    try:
        param_names = list(model_output.params.index)
    except Exception as e:
        result["description"] = f"Could not read parameter names from model_output: {e}"
        return result

    # Find parameter name matching var_base (exact or containing)
    matches = [n for n in param_names if n == var_base]
    if not matches:
        matches = [n for n in param_names if var_base in n]
    if not matches:
        result["description"] = (
            f"Variable '{var_base}' not found in model parameters. "
            f"Available parameters: {param_names}"
        )
        return result

    var_name = matches[0]

    try:
        coef = float(model_output.params[var_name])
        se = float(model_output.bse[var_name]) if hasattr(model_output, "bse") else None
        tstat = float(model_output.tvalues[var_name]) if hasattr(model_output, "tvalues") else None
        pval = float(model_output.pvalues[var_name]) if hasattr(model_output, "pvalues") else None

        # Confidence interval: handle both ndarray and DataFrame returns
        try:
            ci_all = model_output.conf_int()
            # If conf_int returns a DataFrame with index
            if hasattr(ci_all, "loc"):
                ci_low, ci_high = ci_all.loc[var_name].iloc[0], ci_all.loc[var_name].iloc[1]
            else:
                # ndarray: find index of var_name in exog names
                exog_names = list(model_output.model.exog_names)
                idx = exog_names.index(var_name)
                ci_low, ci_high = ci_all[idx, 0], ci_all[idx, 1]
            ci_low, ci_high = float(ci_low), float(ci_high)
        except Exception:
            ci_low, ci_high = None, None

        nobs = int(getattr(model_output, "nobs", getattr(model_output, "df_resid", None) + getattr(model_output, "df_model", None) + 1)) \
            if getattr(model_output, "nobs", None) is not None else None
        # R-squared if available
        r_squared = float(model_output.rsquared) if hasattr(model_output, "rsquared") else None

        # Interpretation: coefficient is change in AvgScore for a one-unit increase in StudentTeacherRatio
        # (i.e., for one more student per teacher). Therefore, a lower student-teacher ratio
        # (fewer students per teacher) is associated with higher AvgScore if coef < 0.
        sign_text = ("A lower student-teacher ratio (fewer students per teacher) is associated with "
                     "higher AvgScore." if coef < 0 else
                     "A lower student-teacher ratio (fewer students per teacher) is associated with "
                     "lower AvgScore.")
        significance = ("statistically significant (p < 0.05)" if (pval is not None and pval < 0.05)
                        else ("marginal or not statistically significant (p >= 0.05)" if pval is not None else "significance unknown"))

        description = (
            f"Estimated effect of StudentTeacherRatio (variable name in model: '{var_name}'):\n"
            f"  Coefficient = {coef:.4f}\n"
            f"  Std. Error = {se:.4f}" if se is not None else f"  Coefficient = {coef:.4f}\n  Std. Error = None"
        )

        # Append t, p, CI, nobs, R2 to description
        description += (
            f"\n  t-stat = {tstat:.3f}" if tstat is not None else ""
        )
        description += (
            f"\n  p-value = {pval:.4g}" if pval is not None else "\n  p-value = None"
        )
        if ci_low is not None and ci_high is not None:
            description += f"\n  95% CI = [{ci_low:.4f}, {ci_high:.4f}]"
        else:
            description += "\n  95% CI = None"

        if nobs is not None:
            description += f"\n  Observations (n) = {nobs}"
        if r_squared is not None:
            description += f"\n  R-squared = {r_squared:.4f}"

        # Add interpretation lines
        description += (
            f"\n\nInterpretation: The coefficient represents the change in AvgScore associated with a one-unit "
            f"increase in StudentTeacherRatio (one more student per teacher). Therefore, {sign_text} "
            f"The estimated effect is {abs(coef):.4f} points on AvgScore per one-student change in the ratio "
            f"({'increase' if coef>0 else 'decrease'} of one student per teacher corresponds to a {'+' if coef>0 else '-'}{abs(coef):.4f} change in AvgScore). "
            f"This effect is {significance}."
        )

        result["object"] = {
            "variable": var_name,
            "coefficient": coef,
            "std_error": se,
            "t_stat": tstat,
            "p_value": pval,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "nobs": nobs,
            "r_squared": r_squared
        }
        result["description"] = description

        return result

    except Exception as e:
        result["description"] = f"Failed to extract statistics for '{var_name}': {e}"
        return result