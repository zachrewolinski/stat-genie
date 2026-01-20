def extract_final_answer(model_output):
    """
    Extract relevant statistics for the 'Femininity' predictor from a statsmodels
    RegressionResultsWrapper object and give a brief interpretation relative to
    the hypothesis that more feminine hurricane names lead to different fatalities.

    Returns a dictionary:
      - "object": dict of extracted numeric results (coef, se, t, p, conf_int, pct_change)
      - "description": short interpretation of these stats in the context of the task
    """
    import numpy as np
    # Prepare a safe failure message
    not_found_msg = {
        "object": None,
        "description": "The model output does not contain a parameter named 'Femininity'."
    }

    # Validate input has the expected attributes (statsmodels RegressionResults-like)
    try:
        params = model_output.params
    except Exception:
        return {
            "object": None,
            "description": "Provided model_output does not appear to be a statsmodels results object with .params"
        }

    if 'Femininity' not in params.index:
        return not_found_msg

    # Extract statistics
    try:
        coef = float(params['Femininity'])
        se = float(model_output.bse['Femininity']) if hasattr(model_output, 'bse') else None
        tval = float(model_output.tvalues['Femininity']) if hasattr(model_output, 'tvalues') else None
        pval = float(model_output.pvalues['Femininity']) if hasattr(model_output, 'pvalues') else None

        # 95% conf int
        try:
            ci = model_output.conf_int().loc['Femininity'].tolist()
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        except Exception:
            ci_lower = ci_upper = None

        # Transform coefficient from log scale to percent change in (Deaths + 1)
        # approximate percent change = (exp(coef) - 1) * 100
        pct_change = (np.expm1(coef)) * 100.0

    except Exception as e:
        return {
            "object": None,
            "description": f"Error extracting Femininity stats from model_output: {e}"
        }

    # Build a concise conclusion regarding the hypothesis direction:
    alpha = 0.05
    if pval is None:
        significance = "p-value unavailable; cannot assess statistical significance."
        conclusion = "Unable to determine statistical support for the hypothesis."
    else:
        if pval < alpha:
            if coef > 0:
                significance = f"Significant at p = {pval:.3g}."
                conclusion = (
                    "Positive and statistically significant association: higher femininity is associated "
                    "with higher log(Deaths+1). This is consistent with the hypothesis that more feminine "
                    "names lead to fewer precautions and therefore more fatalities."
                )
            else:
                significance = f"Significant at p = {pval:.3g}."
                conclusion = (
                    "Negative and statistically significant association: higher femininity is associated "
                    "with lower log(Deaths+1). This contradicts the stated hypothesis."
                )
        else:
            significance = f"Not statistically significant (p = {pval:.3g})."
            conclusion = (
                "No statistically significant evidence that name femininity is associated with fatalities "
                "at conventional levels; the data do not provide support for the hypothesis."
            )

    # Compose returned object
    result_object = {
        "coef": round(coef, 6),
        "std_error": round(se, 6) if se is not None else None,
        "t_value": round(tval, 6) if tval is not None else None,
        "p_value": round(pval, 6) if pval is not None else None,
        "conf_int_95": [round(ci_lower, 6) if ci_lower is not None else None,
                        round(ci_upper, 6) if ci_upper is not None else None],
        "percent_change_in_Deaths_plus1_per_unit_femininity": round(pct_change, 3)
    }

    description = (
        f"Extracted coefficient for 'Femininity' = {result_object['coef']}, SE = {result_object['std_error']}, "
        f"t = {result_object['t_value']}, p = {result_object['p_value']}, 95% CI = {result_object['conf_int_95']}. "
        f"Interpreting the log outcome: a one-unit increase in femininity is associated with an estimated "
        f"{result_object['percent_change_in_Deaths_plus1_per_unit_femininity']}% change in (Deaths + 1). "
        f"{significance} {conclusion}"
    )

    return {"object": result_object, "description": description}