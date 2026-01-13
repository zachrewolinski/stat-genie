def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted
    statsmodels RegressionResultsWrapper (or similar) object and interprets them
    in the context of whether a lower student-teacher ratio is associated with
    higher academic performance (AvgScore).

    Returns:
      {
        "object": {  # machine-friendly numeric summary
           "coef": float,
           "p_value": float,
           "conf_int": (float_low, float_high),
           "std_effect": float,        # effect in SD units: change in AvgScore (SDs) per 1 SD change in ratio
           "n_obs": int
        },
        "description": str  # human-readable interpretation and conclusion
      }
    """
    import numpy as np
    import pandas as pd

    res = model_output

    # Try to access params, pvalues, conf_int
    try:
        params = res.params
        pvalues = res.pvalues
        conf = res.conf_int()
    except Exception as e:
        return {
            "object": None,
            "description": f"Could not extract regression outputs from model_output: {e}"
        }

    # Identify the exact name used for the StudentTeacherRatio regressor
    target_name = None
    candidate_names = ['StudentTeacherRatio']
    # Fallback: try to find any exog name containing the substring (case-sensitive and case-insensitive)
    exog_names = list(getattr(res, 'model').exog_names) if hasattr(res, 'model') and getattr(res.model, 'exog_names', None) is not None else list(params.index)
    for cand in candidate_names:
        if cand in exog_names:
            target_name = cand
            break
    if target_name is None:
        # try case-insensitive or substring matching
        lower_exog = [n.lower() for n in exog_names]
        for n in exog_names:
            if 'studentteacher' in n.lower() or 'student_teacher' in n.lower() or 'studentteacherratio' in n.lower():
                target_name = n
                break

    if target_name is None:
        return {
            "object": None,
            "description": "Could not find a regressor named 'StudentTeacherRatio' (or similar) in the model exogenous variable names."
        }

    # Extract numeric stats
    coef = float(params[target_name])
    pval = float(pvalues[target_name]) if target_name in pvalues.index else float(np.nan)
    try:
        ci = conf.loc[target_name].tolist()
        ci_low, ci_high = float(ci[0]), float(ci[1])
    except Exception:
        # conf_int might be an array without index
        try:
            idx = list(params.index).index(target_name)
            ci_low, ci_high = float(conf[idx, 0]), float(conf[idx, 1])
        except Exception:
            ci_low, ci_high = (None, None)

    # Number of observations
    n_obs = int(getattr(res, 'nobs', getattr(res, 'model').nobs if hasattr(res, 'model') and hasattr(res.model, 'nobs') else (len(res.model.endog) if hasattr(res, 'model') and hasattr(res.model, 'endog') else np.nan)))

    # Compute standardized effect (beta_std): (coef * sd_x) / sd_y if data available in model
    std_effect = None
    try:
        model = res.model
        # model.exog corresponds to design matrix; locate column index
        exog_names = list(model.exog_names)
        idx = exog_names.index(target_name)
        x_col = model.exog[:, idx]
        y_col = model.endog
        sd_x = float(np.std(x_col, ddof=1))
        sd_y = float(np.std(y_col, ddof=1))
        if sd_y != 0:
            std_effect = float((coef * sd_x) / sd_y)
        else:
            std_effect = None
    except Exception:
        std_effect = None

    # Interpretation in context:
    # Reminder: StudentTeacherRatio is students / teachers. Lower values = fewer students per teacher.
    # If coef < 0: increasing ratio (more students per teacher) lowers AvgScore => fewer students per teacher (lower ratio) associated with higher AvgScore.
    # If coef > 0: opposite.
    alpha = 0.05
    significance = (pval is not None) and (not np.isnan(pval)) and (pval < alpha)

    if np.isnan(coef):
        conclusion = "Coefficient is not available."
    else:
        if significance:
            if coef < 0:
                conclusion = ("Statistically significant: coefficient = {:.4f} (p = {:.3g}, 95% CI [{:.4f}, {:.4f}]). "
                              "Interpretation: a LOWER student-teacher ratio (i.e., fewer students per teacher) is associated with HIGHER AvgScore.").format(coef, pval, ci_low, ci_high)
            else:
                conclusion = ("Statistically significant: coefficient = {:.4f} (p = {:.3g}, 95% CI [{:.4f}, {:.4f}]). "
                              "Interpretation: a LOWER student-teacher ratio is associated with LOWER AvgScore (i.e., effect in the opposite direction).").format(coef, pval, ci_low, ci_high)
        else:
            # not statistically significant
            if coef < 0:
                conclusion = ("Coefficient = {:.4f} (p = {:.3g}, 95% CI [{:.4f}, {:.4f}]) suggests that lower student-teacher ratio is associated with higher AvgScore, "
                              "but the association is NOT statistically significant at alpha = 0.05.").format(coef, pval, ci_low, ci_high)
            else:
                conclusion = ("Coefficient = {:.4f} (p = {:.3g}, 95% CI [{:.4f}, {:.4f}]) suggests that lower student-teacher ratio is associated with lower AvgScore (or no beneficial effect), "
                              "but the association is NOT statistically significant at alpha = 0.05.").format(coef, pval, ci_low, ci_high)

    # Add standardized-effect sentence if available
    if std_effect is not None:
        conclusion += " The standardized effect is {:.3f} (change in AvgScore standard deviations per one standard deviation change in StudentTeacherRatio).".format(std_effect)

    # Build returned object
    ret_obj = {
        "coef": coef,
        "p_value": pval,
        "conf_int": (ci_low, ci_high),
        "std_effect": std_effect,
        "n_obs": n_obs
    }

    return {
        "object": ret_obj,
        "description": conclusion
    }