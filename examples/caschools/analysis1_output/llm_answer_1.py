def extract_final_answer(model_output):
    """
    Extracts coefficient, p-value, 95% CI, and (if possible) a standardized effect for the
    'log_ST_Ratio' variable from a statsmodels RegressionResultsWrapper.

    Returns a dictionary with keys:
      - "object": dict containing numeric results and a simple decision about association
      - "description": brief explanation of what was extracted and how to interpret it
    """
    import pandas as pd

    res = model_output
    var = 'log_ST_Ratio'

    # Extract coefficient and p-value
    try:
        coef = float(res.params[var])
    except Exception as e:
        raise KeyError(f"Could not extract parameter '{var}' from model_output: {e}")

    try:
        pvalue = float(res.pvalues[var])
    except Exception as e:
        raise KeyError(f"Could not extract p-value for '{var}' from model_output: {e}")

    # Extract 95% confidence interval
    try:
        ci_array = res.conf_int(alpha=0.05)  # returns (k x 2) array
        ci_df = pd.DataFrame(ci_array, index=res.params.index, columns=['ci_lower', 'ci_upper'])
        ci_lower = float(ci_df.loc[var, 'ci_lower'])
        ci_upper = float(ci_df.loc[var, 'ci_upper'])
    except Exception:
        # Fallback if indexing fails
        try:
            ci = res.conf_int(alpha=0.05)
            # find position of var in params index
            idx = list(res.params.index).index(var)
            ci_lower = float(ci[idx, 0])
            ci_upper = float(ci[idx, 1])
        except Exception as e:
            raise RuntimeError(f"Could not extract confidence interval for '{var}': {e}")

    # Attempt to compute a standardized (beta) effect if original data are available
    std_effect = None
    std_info = None
    try:
        df = res.model.data.frame  # statsmodels stores the DataFrame used for the model here
        if var in df.columns and 'AvgScore' in df.columns:
            std_x = float(df[var].std(ddof=0))
            std_y = float(df['AvgScore'].std(ddof=0))
            if std_y != 0:
                std_effect = float(coef * std_x / std_y)
                std_info = {'std_x': std_x, 'std_y': std_y}
    except Exception:
        # If anything fails, leave standardized effect as None (not critical)
        std_effect = None
        std_info = None

    # Formulate decision about association: we interpret negative coef as "lower ratio -> higher AvgScore"
    alpha = 0.05
    if (coef < 0) and (pvalue < alpha):
        decision = ("Yes — statistically significant negative association: "
                    "lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore.")
    elif (coef < 0) and (pvalue >= alpha):
        decision = ("Negative point estimate (lower ratio associated with higher AvgScore) but not statistically significant "
                    f"(p = {pvalue:.3f}).")
    elif (coef > 0) and (pvalue < alpha):
        decision = ("No — statistically significant positive association: higher student-teacher ratio associated with higher AvgScore.")
    else:
        decision = ("No statistically significant association detected between student-teacher ratio and AvgScore "
                    f"(coef = {coef:.4f}, p = {pvalue:.3f}).")

    result_object = {
        'variable': var,
        'coefficient': coef,
        'p_value': pvalue,
        'ci_95': [ci_lower, ci_upper],
        'standardized_effect': std_effect,   # None if not computable
        'standardized_info': std_info,       # None if not computable
        'decision': decision
    }

    description = (
        f"Extracted statistics for '{var}': coefficient = {coef:.4f}, p-value = {pvalue:.4f}, "
        f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. A negative coefficient indicates that a lower "
        "student-teacher ratio (fewer students per teacher) is associated with higher AvgScore. "
        "If available, a standardized effect (beta) is also provided. The 'decision' field summarizes "
        "whether the association is statistically significant and in which direction."
    )

    return {"object": result_object, "description": description}