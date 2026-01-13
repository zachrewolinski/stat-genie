def extract_final_answer(model_output):
    """
    Extracts key statistics about the StudentTeacherRatio coefficient from a fitted
    statsmodels OLS RegressionResultsWrapper (fitted with robust SEs).
    
    Returns:
      {
        "object": {   # numeric results
          "coef": float,
          "std_err": float,
          "t_value": float,
          "p_value": float,
          "ci_lower": float,
          "ci_upper": float,
          "std_effect_beta": float,   # standardized regression coefficient
          "n_obs": int,
          "r_squared": float
        },
        "description": str   # brief interpretation in context of the question
      }
    """
    import numpy as np

    res = model_output

    # Find the exact parameter name for StudentTeacherRatio (robust to slight naming differences)
    param_name = None
    for name in res.params.index:
        if name.lower().startswith('studentteacherratio'.lower()) or name == 'StudentTeacherRatio':
            param_name = name
            break
    if param_name is None:
        # try contains
        for name in res.params.index:
            if 'student' in name.lower() and 'teacher' in name.lower():
                param_name = name
                break

    if param_name is None:
        raise KeyError("Could not find a parameter matching 'StudentTeacherRatio' in model_output.params.")

    coef = float(res.params[param_name])
    std_err = float(res.bse[param_name])
    t_value = float(res.tvalues[param_name])
    p_value = float(res.pvalues[param_name])

    # 95% confidence interval
    try:
        ci = res.conf_int(alpha=0.05).loc[param_name].values
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        # fallback if .conf_int() returns ndarray
        ci_array = res.conf_int(alpha=0.05)
        names = res.params.index.tolist()
        idx = names.index(param_name)
        ci_lower, ci_upper = float(ci_array[idx, 0]), float(ci_array[idx, 1])

    # Standardized (beta) effect: beta * (sd_x / sd_y)
    std_effect_beta = None
    try:
        exog_names = res.model.exog_names
        # find column index in exog corresponding to param_name (exact match)
        if param_name in exog_names:
            col_idx = exog_names.index(param_name)
        else:
            # sometimes names differ (e.g., use 'StudentTeacherRatio' vs 'StudentTeacherRatio[T.whatever]')
            # try fuzzy match
            col_idx = None
            for i, nm in enumerate(exog_names):
                if nm.lower().startswith('studentteacherratio'):
                    col_idx = i
                    break
            if col_idx is None:
                raise ValueError
        exog = np.asarray(res.model.exog)
        xcol = exog[:, col_idx]
        y = np.asarray(res.model.endog)
        sd_x = np.std(xcol, ddof=1)
        sd_y = np.std(y, ddof=1)
        if sd_x > 0 and sd_y > 0:
            std_effect_beta = float(coef * (sd_x / sd_y))
    except Exception:
        std_effect_beta = None

    n_obs = int(getattr(res, 'nobs', getattr(res.model, 'nobs', None)))
    r_squared = float(getattr(res, 'rsquared', np.nan))

    # Interpretation: direction and significance
    if p_value < 0.05:
        sig_text = "statistically significant (p < 0.05)."
    else:
        sig_text = "not statistically significant (p >= 0.05)."

    # Determine direction: since higher StudentTeacherRatio = more students per teacher,
    # a negative coefficient implies that lower ratio (fewer students per teacher) is associated with higher AvgScore.
    if coef < 0:
        direction_text = ("The coefficient is negative, so higher student-teacher ratios "
                          "(more students per teacher) are associated with LOWER average scores; "
                          "equivalently, lower ratios are associated with higher performance.")
    elif coef > 0:
        direction_text = ("The coefficient is positive, so higher student-teacher ratios "
                          "(more students per teacher) are associated with HIGHER average scores; "
                          "equivalently, lower ratios are associated with lower performance.")
    else:
        direction_text = "The coefficient is approximately zero (no association)."

    description = (
        f"StudentTeacherRatio coefficient = {coef:.4f} (SE = {std_err:.4f}, t = {t_value:.2f}, p = {p_value:.3f}); "
        f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]. {direction_text} This effect is {sig_text} "
        f"n = {n_obs}, R^2 = {r_squared:.3f}."
    )
    if std_effect_beta is not None:
        description += f" Standardized effect (beta) ≈ {std_effect_beta:.4f}."

    output_obj = {
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_effect_beta": std_effect_beta,
        "n_obs": n_obs,
        "r_squared": r_squared,
        "param_name": param_name
    }

    return {"object": output_obj, "description": description}