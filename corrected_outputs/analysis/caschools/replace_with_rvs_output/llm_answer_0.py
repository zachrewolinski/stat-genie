def extract_final_answer(model_output):
    """
    Extracts the coefficient, standard error, p-value, 95% CI, sample size, and a concise
    conclusion about whether a lower student-teacher ratio is associated with higher
    average academic performance.

    Returns a dictionary with keys:
      - "object": dict with numeric results and a short conclusion label ('yes'/'no'/'inconclusive')
      - "description": human-readable explanation of the result in context
    """
    # Name of the independent variable of interest in the model
    var = 'StudentTeacherRatio'

    # Basic checks
    if not hasattr(model_output, 'params'):
        raise ValueError("The provided model_output does not look like a fitted statsmodels result (missing .params).")

    params = model_output.params
    if var not in params.index:
        raise ValueError(f"Variable '{var}' not found in model results. Available params: {list(params.index)}")

    # Extract statistics
    coef = float(params[var])
    # standard error
    try:
        se = float(model_output.bse[var])
    except Exception:
        # fallback if bse not available in that form
        se = float(model_output.bse.loc[var]) if hasattr(model_output.bse, 'loc') else float(model_output.bse[list(params.index).index(var)])
    # p-value
    try:
        pval = float(model_output.pvalues[var])
    except Exception:
        pval = float(model_output.pvalues.loc[var]) if hasattr(model_output.pvalues, 'loc') else float(model_output.pvalues[list(params.index).index(var)])

    # 95% confidence interval (handle both ndarray and DataFrame-like returns)
    try:
        ci_all = model_output.conf_int(alpha=0.05)
        if hasattr(ci_all, 'loc') and var in ci_all.index:
            ci_lower, ci_upper = [float(x) for x in ci_all.loc[var].values]
        else:
            # assume numpy ndarray with rows in same order as params.index
            idx = list(params.index).index(var)
            ci_lower, ci_upper = [float(x) for x in ci_all[idx]]
    except Exception:
        ci_lower, ci_upper = (None, None)

    # Sample size (nobs)
    try:
        nobs = int(model_output.nobs)
    except Exception:
        # fallback
        try:
            nobs = int(model_output.model.endog.shape[0])
        except Exception:
            nobs = None

    # Interpretation logic:
    # StudentTeacherRatio: higher values = more students per teacher (larger class sizes).
    # Therefore a negative coefficient implies that larger ratios -> lower AvgScore,
    # i.e., lower ratio (fewer students per teacher) -> higher AvgScore.
    alpha = 0.05
    significant = (pval < alpha) if (pval is not None) else False
    if coef < 0 and significant:
        conclusion_label = 'yes'
        conclusion = ("Yes: coefficient is negative and statistically significant "
                      "(higher student-teacher ratio → lower AvgScore). This implies "
                      "that lower student-teacher ratios (fewer students per teacher) "
                      "are associated with higher average academic performance.")
    elif coef > 0 and significant:
        conclusion_label = 'no'
        conclusion = ("No: coefficient is positive and statistically significant "
                      "(higher student-teacher ratio → higher AvgScore). This implies "
                      "that lower student-teacher ratios are associated with lower performance.")
    else:
        conclusion_label = 'inconclusive'
        conclusion = ("Inconclusive: the coefficient is not statistically significant at "
                      f"alpha={alpha}, so we cannot confidently state an association "
                      "between student-teacher ratio and average academic performance.")

    # Package numeric results in a dictionary
    result_object = {
        'variable': var,
        'coef': round(coef, 4),
        'std_error': round(se, 4) if se is not None else None,
        'p_value': round(pval, 4) if pval is not None else None,
        'ci_lower_95': round(ci_lower, 4) if ci_lower is not None else None,
        'ci_upper_95': round(ci_upper, 4) if ci_upper is not None else None,
        'nobs': nobs,
        'significant': bool(significant),
        'conclusion_label': conclusion_label,  # 'yes' / 'no' / 'inconclusive'
        'conclusion': conclusion
    }

    description = (
        f"Extracted results for '{var}': coefficient={result_object['coef']}, "
        f"SE={result_object['std_error']}, p={result_object['p_value']}, "
        f"95% CI=[{result_object['ci_lower_95']}, {result_object['ci_upper_95']}], "
        f"n={result_object['nobs']}. Interpretation: {result_object['conclusion']}"
    )

    return {"object": result_object, "description": description}