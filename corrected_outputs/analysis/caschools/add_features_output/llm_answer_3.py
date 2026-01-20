def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted
    statsmodels RegressionResultsWrapper (or similar) and returns a summary.

    Returns:
        dict with keys:
          - "object": dict of numeric values (coef, se, t, p-value, CI, std_coef, N, significance)
          - "description": human-readable interpretation string about whether
                           lower student-teacher ratio is associated with higher AvgScore
    """
    res = model_output
    var = 'StudentTeacherRatio'

    # Basic checks
    if not hasattr(res, 'params'):
        raise ValueError("model_output does not appear to be a fitted statsmodels result (missing .params)")

    params = res.params
    if var not in params.index:
        raise ValueError(f"Variable '{var}' not found in model parameters. Available params: {list(params.index)}")

    # Extract point estimate and usual inferential stats (if available)
    coef = float(params[var])

    se = None
    if hasattr(res, 'bse') and var in res.bse.index:
        se = float(res.bse[var])

    tval = None
    if hasattr(res, 'tvalues') and var in res.tvalues.index:
        tval = float(res.tvalues[var])

    pval = None
    if hasattr(res, 'pvalues') and var in res.pvalues.index:
        pval = float(res.pvalues[var])

    ci = None
    try:
        ci_mat = res.conf_int(alpha=0.05)
        # conf_int may return DataFrame or ndarray
        if hasattr(ci_mat, 'loc'):
            lower = float(ci_mat.loc[var, 0])
            upper = float(ci_mat.loc[var, 1])
        else:
            idx = list(params.index).index(var)
            lower = float(ci_mat[idx, 0])
            upper = float(ci_mat[idx, 1])
        ci = (lower, upper)
    except Exception:
        ci = None

    # Sample size
    n = None
    try:
        # statsmodels stores nobs as float sometimes
        n = int(getattr(res, 'nobs'))
    except Exception:
        try:
            n = int(res.model.endog.shape[0])
        except Exception:
            n = None

    # Standardized coefficient (if the dataframe used for the regression is attached)
    std_coef = None
    try:
        df = getattr(res, 'model_data', None)
        if df is None:
            # try alternative locations
            try:
                df = res.model.data.frame
            except Exception:
                df = None
        if df is not None and var in df.columns and 'AvgScore' in df.columns:
            x = df[var].astype(float)
            y = df['AvgScore'].astype(float)
            # Use population std (ddof=0) to be consistent with many standardized coef definitions
            std_x = x.std(ddof=0)
            std_y = y.std(ddof=0)
            if std_x > 0 and std_y > 0:
                std_coef = coef * (std_x / std_y)
    except Exception:
        std_coef = None

    # Determine significance at alpha = 0.05 if p-value is available
    significant = None
    if pval is not None:
        significant = (pval < 0.05)

    # Interpretation of direction:
    if coef < 0:
        direction_text = ("The estimated coefficient is negative: higher StudentTeacherRatio (more students per teacher) "
                          "is associated with lower AvgScore. Equivalently, a lower student-teacher ratio "
                          "(fewer students per teacher) is associated with higher academic performance.")
    elif coef > 0:
        direction_text = ("The estimated coefficient is positive: higher StudentTeacherRatio (more students per teacher) "
                          "is associated with higher AvgScore. This means lower student-teacher ratio would be associated "
                          "with lower performance (opposite of the hypothesized direction).")
    else:
        direction_text = "The estimated coefficient is zero (no association detected)."

    # Build description text
    desc_parts = []
    desc_parts.append(f"Variable: {var}")
    desc_parts.append(f"Coefficient = {coef:.6g}")
    if se is not None:
        desc_parts.append(f"SE = {se:.6g}")
    if tval is not None:
        desc_parts.append(f"t = {tval:.4g}")
    if pval is not None:
        desc_parts.append(f"p = {pval:.4g}")
    if ci is not None:
        desc_parts.append(f"95% CI = [{ci[0]:.6g}, {ci[1]:.6g}]")
    if std_coef is not None:
        desc_parts.append(f"Standardized coefficient ≈ {std_coef:.6g}")
    if n is not None:
        desc_parts.append(f"N = {n}")
    if significant is not None:
        desc_parts.append(f"Statistically significant at α=0.05: {'yes' if significant else 'no'}")
    desc_parts.append(direction_text)

    description = "; ".join(desc_parts)

    output_object = {
        'variable': var,
        'coef': coef,
        'se': se,
        't_value': tval,
        'p_value': pval,
        '95%_ci': ci,
        'standardized_coef': std_coef,
        'n': n,
        'significant_0.05': significant
    }

    return {"object": output_object, "description": description}