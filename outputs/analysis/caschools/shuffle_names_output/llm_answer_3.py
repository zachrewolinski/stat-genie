def extract_final_answer(model_output):
    """
    Extracts statistics for the StudentTeacherRatio coefficient from a fitted statsmodels
    RegressionResultsWrapper and returns a summary object and a plain-language description.

    Returns:
      {
        "object": {
            "coef": float,
            "std_err": float,
            "t_value": float,
            "p_value": float,
            "ci_lower": float,
            "ci_upper": float,
            "significant": bool,
            "alpha": 0.05,
            "standardized_coef": float or None
        },
        "description": str
      }
    """
    # Basic validation
    if model_output is None:
        raise ValueError("model_output is None")

    # Ensure model_output looks like a statsmodels results object
    if not hasattr(model_output, "params") or not hasattr(model_output, "pvalues"):
        raise ValueError("model_output does not look like a statsmodels results object.")

    param_name = "StudentTeacherRatio"
    params = model_output.params
    if param_name not in params.index:
        raise ValueError(f"Coefficient '{param_name}' not found in model parameters. Available params: {list(params.index)}")

    # Extract coefficient and inferential stats
    coef = float(params[param_name])
    # Some results objects return pandas Series; use .bse/.tvalues/.pvalues
    std_err = float(model_output.bse[param_name]) if hasattr(model_output, "bse") else None
    t_value = float(model_output.tvalues[param_name]) if hasattr(model_output, "tvalues") else None
    p_value = float(model_output.pvalues[param_name])

    # Confidence interval (handle both ndarray and DataFrame outputs)
    try:
        conf = model_output.conf_int()
        # conf may be a DataFrame or ndarray. Try label-based access first.
        if hasattr(conf, "loc"):
            ci_lower, ci_upper = float(conf.loc[param_name, 0]), float(conf.loc[param_name, 1])
        else:
            # ndarray: find index of parameter
            idx = list(model_output.params.index).index(param_name)
            ci_lower, ci_upper = float(conf[idx, 0]), float(conf[idx, 1])
    except Exception:
        ci_lower, ci_upper = (None, None)

    # Determine statistical significance at alpha = 0.05
    alpha = 0.05
    significant = (p_value < alpha)

    # Attempt to compute a standardized coefficient (beta) if original data frame is available
    standardized_coef = None
    try:
        # model_output.model.data.frame is present when statsmodels was given a DataFrame
        df = getattr(model_output.model.data, "frame", None)
        if df is None:
            # fallback: try model.data.orig_endog / orig_exog not always accessible as DataFrame
            df = getattr(model_output.model.data, "orig_endog", None)
            # If still not a DataFrame, skip standardized coef
        if isinstance(df, (dict, list)) or df is None:
            df = getattr(model_output.model.data, "frame", None)
        if isinstance(df, type(model_output.model.data.frame)) or hasattr(df, "columns"):
            # confirm columns exist
            if ("AvgScore" in df.columns) and (param_name in df.columns):
                std_x = float(df[param_name].std(ddof=1))
                std_y = float(df["AvgScore"].std(ddof=1))
                if std_y != 0:
                    standardized_coef = coef * (std_x / std_y)
    except Exception:
        standardized_coef = None

    # Build the object to return
    result_object = {
        "coef": coef,
        "std_err": std_err,
        "t_value": t_value,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": bool(significant),
        "alpha": alpha,
        "standardized_coef": standardized_coef
    }

    # Plain-language description and conclusion
    # Interpret direction: negative coef means higher ratio (more students per teacher) -> lower AvgScore.
    if significant:
        if coef < 0:
            conclusion = (
                f"Yes — statistically significant at alpha={alpha}: a one-unit increase in StudentTeacherRatio "
                f"(one more student per teacher) is associated with a {coef:.3f} point change in AvgScore (95% CI: "
                f"{ci_lower:.3f} to {ci_upper:.3f}, p = {p_value:.3g}). Because the coefficient is negative, "
                f"lower student-teacher ratios (fewer students per teacher) are associated with higher average academic performance."
            )
        else:
            conclusion = (
                f"No (but significant in the opposite direction) — statistically significant at alpha={alpha}: a one-unit increase in StudentTeacherRatio "
                f"is associated with a {coef:.3f} point increase in AvgScore (95% CI: {ci_lower:.3f} to {ci_upper:.3f}, p = {p_value:.3g}). "
                f"That implies higher student-teacher ratios are associated with higher average academic performance."
            )
    else:
        # Not statistically significant
        direction = "negative" if coef < 0 else "positive"
        conclusion = (
            f"No — the estimated association is {direction} (coef = {coef:.3f}) but not statistically significant at alpha={alpha} "
            f"(p = {p_value:.3g}; 95% CI: {ci_lower:.3f} to {ci_upper:.3f}). "
            f"This means we do not have sufficient evidence to conclude that student-teacher ratio is associated with AvgScore."
        )

    # Append standardized interpretation if available
    if standardized_coef is not None:
        conclusion += f" The standardized coefficient is {standardized_coef:.3f}, which expresses the effect in standard-deviation units."

    return {"object": result_object, "description": conclusion}