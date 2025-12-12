def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted statsmodels OLS result
    and returns a dictionary with numeric outputs ("object") and a short interpretation ("description").

    Returned structure:
    {
      "object": {
        "variable": "StudentTeacherRatio",
        "coef": float,
        "std_err": float,
        "t_value": float,
        "p_value": float,
        "conf_int": [lower, upper],
        "nobs": int or None,
        "significant_at_0.05": bool
      },
      "description": str
    }
    """
    variable = 'StudentTeacherRatio'
    result = {
        "variable": variable,
        "coef": None,
        "std_err": None,
        "t_value": None,
        "p_value": None,
        "conf_int": [None, None],
        "nobs": None,
        "significant_at_0.05": None,
    }

    try:
        # Coefficient and standard statistics
        params = getattr(model_output, 'params', None)
        bse = getattr(model_output, 'bse', None)
        tvalues = getattr(model_output, 'tvalues', None)
        pvalues = getattr(model_output, 'pvalues', None)

        if params is None or variable not in params.index:
            raise KeyError(f"Variable '{variable}' not found in model parameters.")

        result["coef"] = float(params[variable])
        # Some models store bse/tvalues/pvalues as Series; guard access
        if bse is not None and variable in bse.index:
            result["std_err"] = float(bse[variable])
        if tvalues is not None and variable in tvalues.index:
            result["t_value"] = float(tvalues[variable])
        if pvalues is not None and variable in pvalues.index:
            result["p_value"] = float(pvalues[variable])

        # Confidence interval
        try:
            ci = model_output.conf_int()
            if variable in ci.index:
                lower, upper = ci.loc[variable].tolist()
                result["conf_int"] = [float(lower), float(upper)]
        except Exception:
            # fallback: conf_int as array-like
            try:
                ci_arr = model_output.conf_int()
                idx = list(params.index).index(variable)
                lower, upper = ci_arr[idx]
                result["conf_int"] = [float(lower), float(upper)]
            except Exception:
                pass

        # Number of observations if available
        if hasattr(model_output, 'nobs'):
            try:
                result["nobs"] = int(model_output.nobs)
            except Exception:
                result["nobs"] = None

        # Significance at 0.05
        if result["p_value"] is not None:
            result["significant_at_0.05"] = (result["p_value"] < 0.05)

    except Exception as e:
        # If something goes wrong, return whatever we could collect plus an explanatory description
        description = (
            f"Failed to extract complete statistics for '{variable}'. Error: {e}. "
            "Returned partial results in 'object'."
        )
        return {"object": result, "description": description}

    # Formulate a concise interpretation specific to the research question:
    # "Is a lower student-teacher ratio associated with higher academic performance?"
    coef = result["coef"]
    pval = result["p_value"]
    sig = result["significant_at_0.05"]

    if coef is None or pval is None:
        description = (
            "Could not determine effect: coefficient or p-value for StudentTeacherRatio is missing. "
            "See 'object' for available extracted values."
        )
    else:
        # Negative coefficient means higher ratio (more students per teacher) predicts lower scores,
        # so lower ratio (fewer students per teacher) is associated with higher performance.
        direction = "negative" if coef < 0 else ("positive" if coef > 0 else "zero")
        significance_text = "statistically significant (p < 0.05)" if sig else "not statistically significant (p >= 0.05)"

        conclusion_yesno = "Yes" if (coef < 0 and sig) else "No" if (coef >= 0 and sig) else "Inconclusive"
        # Build description
        description = (
            f"The estimated effect of StudentTeacherRatio on AvgScore is {coef:.4f} "
            f"(SE={result['std_err']:.4f}, t={result['t_value']:.3f}, p={pval:.4f}, "
            f"95% CI=[{result['conf_int'][0]:.4f}, {result['conf_int'][1]:.4f}]). "
            f"The coefficient is {direction} and {significance_text}. "
            f"Interpretation: a one-unit increase in StudentTeacherRatio (one more student per teacher) "
            f"is associated with a {abs(coef):.4f}-point {'decrease' if coef < 0 else 'increase' if coef > 0 else 'change'} "
            f"in AvgScore. Answering the posed question directly: '{conclusion_yesno}' — "
            f"{'A lower student-teacher ratio is associated with higher academic performance.' if conclusion_yesno == 'Yes' else 'The evidence does not support that conclusion.'}"
        )

    return {"object": result, "description": description}