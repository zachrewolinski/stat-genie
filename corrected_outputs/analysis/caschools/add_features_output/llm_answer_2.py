def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels OLS result.
    Returns a dictionary with:
      - "object": dict of extracted numeric values and a boolean conclusion flag
      - "description": brief interpretation in the context of whether a lower student-teacher
                       ratio is associated with higher academic performance.

    Expected input: a statsmodels RegressionResultsWrapper (the object returned by .fit()).
    """
    import numpy as np

    result = model_output

    # Name of the variable in the model
    var = 'StudentTeacherRatio'

    # Basic checks
    try:
        params = result.params
    except Exception as e:
        return {
            "object": None,
            "description": f"Input does not appear to be a fitted statsmodels results object: {e}"
        }

    if var not in params.index:
        return {
            "object": None,
            "description": f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}"
        }

    # Extract coefficient, robust SE, t-stat, p-value, and 95% CI
    coef = float(params[var])
    try:
        se = float(result.bse[var])
    except Exception:
        # fallback if bse not indexable the usual way
        se = float(result.bse[result.model.exog_names.index(var)])

    try:
        tstat = float(result.tvalues[var])
    except Exception:
        tstat = float(result.tvalues[result.model.exog_names.index(var)])

    try:
        pvalue = float(result.pvalues[var])
    except Exception:
        pvalue = float(result.pvalues[result.model.exog_names.index(var)])

    try:
        ci = result.conf_int(alpha=0.05).loc[var].values
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        # fallback using array indexing
        ci_arr = result.conf_int(alpha=0.05).values
        idx = result.model.exog_names.index(var)
        ci_lower, ci_upper = float(ci_arr[idx, 0]), float(ci_arr[idx, 1])

    # Compute standardized (beta) coefficient: (coef * sd_x) / sd_y
    try:
        exog_names = result.model.exog_names
        idx = exog_names.index(var)
        x_col = result.model.exog[:, idx]
        y = result.model.endog
        sd_x = float(np.std(x_col, ddof=1))
        sd_y = float(np.std(y, ddof=1))
        if sd_y == 0:
            standardized_beta = None
        else:
            standardized_beta = float(coef * sd_x / sd_y)
    except Exception:
        standardized_beta = None

    # Number of observations
    try:
        nobs = int(result.nobs)
    except Exception:
        nobs = None

    # Interpret whether lower student-teacher ratio is associated with higher performance.
    # A negative coefficient implies that a higher ratio (more students per teacher) is associated with lower scores,
    # i.e., a lower ratio (fewer students per teacher) is associated with higher scores.
    direction = "negative" if coef < 0 else ("positive" if coef > 0 else "zero")
    significant = (pvalue < 0.05)

    conclusion_bool = (coef < 0) and significant
    if conclusion_bool:
        conclusion_text = (
            "Statistically significant evidence (alpha=0.05) that a lower student-teacher ratio "
            "is associated with higher average academic performance."
        )
    elif coef < 0 and not significant:
        conclusion_text = (
            "The estimated effect is negative (lower ratio → higher performance) but it is not "
            "statistically significant at the 5% level."
        )
    elif coef > 0 and significant:
        conclusion_text = (
            "Statistically significant evidence (alpha=0.05) that a higher student-teacher ratio "
            "is associated with higher average academic performance (direction opposite to hypothesis)."
        )
    else:
        conclusion_text = (
            "No statistically significant association detected between student-teacher ratio and average academic performance."
        )

    # Assemble object to return
    output_object = {
        "variable": var,
        "coefficient": round(coef, 4),
        "std_error": round(se, 4),
        "t_stat": round(tstat, 4),
        "p_value": round(pvalue, 4),
        "95%_CI": (round(ci_lower, 4), round(ci_upper, 4)),
        "standardized_beta": (round(standardized_beta, 4) if standardized_beta is not None else None),
        "nobs": nobs,
        "direction": direction,
        "significant_at_0.05": significant,
        # Final yes/no: True means supports hypothesis that lower ratio -> higher performance
        "lower_ratio_associated_with_higher_performance": conclusion_bool
    }

    description = (
        f"The model coefficient for '{var}' is {output_object['coefficient']} (SE={output_object['std_error']}, "
        f"t={output_object['t_stat']}, p={output_object['p_value']}). 95% CI = {output_object['95%_CI']}. "
        f"This coefficient is {direction}. {conclusion_text} "
        f"Standardized effect (beta) = {output_object['standardized_beta']}. "
        f"Observations used = {nobs}."
    )

    return {
        "object": output_object,
        "description": description
    }