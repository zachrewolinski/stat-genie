def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, z-score, p-value, IRR and 95% CI for SkinDark
    from the model_output produced by the provided model() function.

    Returns a dictionary with:
      - "object": a dict with numeric results for SkinDark (log-coef, se, z, p, IRR, IRR_CI)
      - "description": a short plain-language interpretation answering whether dark-skinned
                       players are more likely to receive red cards.
    """
    import numpy as np
    from scipy import stats

    if not isinstance(model_output, dict):
        raise ValueError("model_output must be the dictionary returned by the model function.")

    # Extract robust results and irr_table if available
    robust = model_output.get('robust_results', None)
    irr_table = model_output.get('irr_table', None)

    if robust is None:
        raise ValueError("model_output does not contain 'robust_results'.")

    # Ensure SkinDark is present in the parameter index
    params = robust.params
    if 'SkinDark' not in params.index:
        raise ValueError("SkinDark not found in model parameters.")

    # Extract log-scale coefficient and robust SE
    log_coef = float(params.loc['SkinDark'])
    se = float(robust.bse.loc['SkinDark'])

    # z-statistic and two-sided p-value using normal approximation
    z = log_coef / se if se > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(z))) if not np.isnan(z) else np.nan

    # Confidence interval on log scale from robust results, then exponentiate to IRR scale
    try:
        ci_log = robust.conf_int().loc['SkinDark']
        ci_log_lower = float(ci_log[0])
        ci_log_upper = float(ci_log[1])
    except Exception:
        # Fall back to irr_table if conf_int unavailable
        if irr_table is not None and 'IRR' in irr_table.columns:
            irr_row = irr_table.loc['SkinDark']
            irr = float(irr_row['IRR'])
            irr_ci_lower = float(irr_row['IRR_ci_lower'])
            irr_ci_upper = float(irr_row['IRR_ci_upper'])
            # convert to log-scale for completeness
            ci_log_lower = np.log(irr_ci_lower)
            ci_log_upper = np.log(irr_ci_upper)
        else:
            raise ValueError("Could not obtain confidence intervals for SkinDark from model_output.")

    irr = float(np.exp(log_coef))
    irr_ci_lower = float(np.exp(ci_log_lower))
    irr_ci_upper = float(np.exp(ci_log_upper))

    # Conclusion: check statistical significance and direction
    alpha = 0.05
    if np.isnan(p_value):
        conclusion = "Could not compute p-value; insufficient information."
    else:
        if p_value < alpha:
            if irr > 1:
                conclusion = ("Statistically significant evidence (two-sided p = {:.3g}) that players "
                              "with dark skin receive red cards at a higher rate than light-skin players "
                              "(IRR = {:.3f}, 95% CI [{:.3f}, {:.3f}]).").format(p_value, irr, irr_ci_lower, irr_ci_upper)
            else:
                conclusion = ("Statistically significant evidence (two-sided p = {:.3g}) that players "
                              "with dark skin receive red cards at a lower rate than light-skin players "
                              "(IRR = {:.3f}, 95% CI [{:.3f}, {:.3f}]).").format(p_value, irr, irr_ci_lower, irr_ci_upper)
        else:
            # Not statistically significant
            conclusion = ("No statistically significant evidence that dark-skinned players receive red cards "
                          "at a different rate than light-skinned players (two-sided p = {:.3g}). "
                          "Estimated IRR = {:.3f} with 95% CI [{:.3f}, {:.3f}], which includes 1, "
                          "so the effect is indistinguishable from no difference.").format(p_value, irr, irr_ci_lower, irr_ci_upper)

    result_object = {
        'log_coef': log_coef,                # coefficient on SkinDark (log IRR)
        'se': se,
        'z': z,
        'p_value': p_value,
        'IRR': irr,
        'IRR_ci_lower': irr_ci_lower,
        'IRR_ci_upper': irr_ci_upper,
        'conclusion': conclusion
    }

    description = (
        "Extracted statistics for the effect of SkinDark on red card counts (negative binomial with log(games) offset, "
        "cluster-robust SE by referee). Key result: IRR = {:.3f} (95% CI [{:.3f}, {:.3f}]), two-sided p = {:.3g}. "
        "Interpretation: {}"
    ).format(irr, irr_ci_lower, irr_ci_upper, p_value, ("Dark-skin players are more likely to receive red cards"
                                                         if (p_value < alpha and irr > 1) else
                                                         "No evidence that dark-skin players are more likely to receive red cards"))

    return {
        "object": result_object,
        "description": description
    }