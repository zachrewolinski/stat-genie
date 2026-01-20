def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels OLS result.
    Returns a dictionary with keys "object" (detailed numeric results) and "description" (brief interpretation).
    """
    import numpy as np
    import pandas as pd

    res = model_output
    param = 'StudentTeacherRatio'

    if param not in res.params.index:
        raise KeyError(f"Parameter '{param}' not found in model parameters: {list(res.params.index)}")

    coef = float(res.params[param])
    std_err = float(res.bse[param])
    t_value = float(res.tvalues[param])
    p_value = float(res.pvalues[param])

    # Confidence intervals (uses the model's configured cov_type when available)
    ci_df = pd.DataFrame(res.conf_int(), index=res.params.index, columns=['ci_lower', 'ci_upper'])
    ci_lower = float(ci_df.loc[param, 'ci_lower'])
    ci_upper = float(ci_df.loc[param, 'ci_upper'])

    # Sample size
    try:
        nobs = int(res.nobs)
    except Exception:
        nobs = int(res.model.endog.shape[0])

    # Compute a standardized (beta) effect: (coef * sd_x) / sd_y, if possible
    standardized_beta = None
    try:
        exog_names = list(res.model.exog_names)
        idx = exog_names.index(param)
        x = res.model.exog[:, idx]
        y = res.model.endog
        sd_x = x.std(ddof=1)
        sd_y = y.std(ddof=1)
        if sd_y != 0:
            standardized_beta = float((coef * sd_x) / sd_y)
    except Exception:
        standardized_beta = None

    # Simple significance-based conclusion (alpha = 0.05)
    alpha = 0.05
    is_significant = p_value < alpha
    if is_significant:
        if coef < 0:
            conclusion = ("Yes — statistically significant negative association: lower student-teacher "
                          "ratio (fewer students per teacher) is associated with higher AvgScore "
                          f"(coef={coef:.4f}, p={p_value:.3g}).")
        else:
            conclusion = ("Statistically significant association, but positive: higher student-teacher "
                          "ratio associated with higher AvgScore (unexpected direction) "
                          f"(coef={coef:.4f}, p={p_value:.3g}).")
    else:
        if coef < 0:
            conclusion = ("No strong evidence at alpha=0.05. Point estimate is negative (lower ratio -> higher AvgScore) "
                          f"but not statistically significant (coef={coef:.4f}, p={p_value:.3g}).")
        else:
            conclusion = ("No strong evidence at alpha=0.05. Point estimate is positive (higher ratio -> higher AvgScore) "
                          f"and not statistically significant (coef={coef:.4f}, p={p_value:.3g}).")

    result_object = {
        'parameter': param,
        'coefficient': coef,
        'std_err': std_err,
        't_value': t_value,
        'p_value': p_value,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'nobs': nobs,
        'standardized_beta': standardized_beta,
        'significant_at_0.05': is_significant,
        'conclusion': conclusion
    }

    description = (
        f"Extracted results for '{param}': coefficient={coef:.4f}, SE={std_err:.4f}, t={t_value:.2f}, "
        f"p={p_value:.3g}, 95% CI=[{ci_lower:.3f}, {ci_upper:.3f}], n={nobs}. "
        "A negative coefficient implies that a lower student-teacher ratio (fewer students per teacher) "
        "is associated with higher average Stanford-9 scores. " + conclusion
    )

    return {"object": result_object, "description": description}