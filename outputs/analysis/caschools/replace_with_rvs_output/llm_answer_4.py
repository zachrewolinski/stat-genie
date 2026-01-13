def extract_final_answer(model_output):
    """
    Extracts the coefficient, p-value, 95% CI, and related summary for the
    'student_teacher_ratio' variable from a statsmodels regression results object
    (including objects returned by get_robustcov_results).
    Returns a dictionary with keys:
      - "object": dict with numeric results
      - "description": short plain-language interpretation answering whether
                       a lower student-teacher ratio is associated with higher
                       academic performance.
    """
    # Ensure the expected variable exists in the results
    var = 'student_teacher_ratio'
    # Prepare defaults in case something is missing
    obj = {
        'variable': var,
        'coefficient': None,
        'std_err': None,
        't_value': None,
        'p_value': None,
        'conf_int_95': (None, None),
        'nobs': None,
        'significant_at_0.05': None
    }

    # Helper to safely extract attributes
    try:
        params = getattr(model_output, 'params', None)
        if params is None:
            # Some wrappers store results in .raw_result or similar
            raise AttributeError("Model output has no 'params' attribute.")
        if var not in params.index:
            raise KeyError(f"Variable '{var}' not found in model results. Available vars: {list(params.index)}")

        coef = float(params[var])
        obj['coefficient'] = coef

        # Standard error (robust if model_output came from get_robustcov_results)
        bse = getattr(model_output, 'bse', None)
        if bse is not None and var in bse.index:
            obj['std_err'] = float(bse[var])

        # t-value and p-value
        tvals = getattr(model_output, 'tvalues', None) or getattr(model_output, 'tvalue', None) or getattr(model_output, 't', None)
        if tvals is not None and var in tvals.index:
            obj['t_value'] = float(tvals[var])
        pvals = getattr(model_output, 'pvalues', None)
        if pvals is not None and var in pvals.index:
            obj['p_value'] = float(pvals[var])

        # 95% confidence interval
        try:
            ci = model_output.conf_int(alpha=0.05)
            # conf_int returns a DataFrame or array with rows indexed by param names
            if hasattr(ci, 'loc'):
                low, high = float(ci.loc[var].iloc[0]), float(ci.loc[var].iloc[1])
            else:
                # assume order of params matches
                idx = list(params.index).index(var)
                low, high = float(ci[idx, 0]), float(ci[idx, 1])
            obj['conf_int_95'] = (low, high)
        except Exception:
            # If conf_int fails, leave as None
            pass

        # Number of observations
        if hasattr(model_output, 'nobs'):
            try:
                obj['nobs'] = int(model_output.nobs)
            except Exception:
                obj['nobs'] = model_output.nobs

        # Determine significance at 0.05 if p-value available
        if obj['p_value'] is not None:
            obj['significant_at_0.05'] = (obj['p_value'] < 0.05)

    except Exception as e:
        # Return an object indicating failure to extract
        description = (
            "Failed to extract statistics for 'student_teacher_ratio' from the provided "
            f"model_output. Error: {e}"
        )
        return {"object": obj, "description": description}

    # Construct interpretation:
    # If coefficient is negative and statistically significant -> yes (lower ratio -> higher scores)
    if obj['coefficient'] is None:
        description = "Unable to extract the coefficient for 'student_teacher_ratio'."
    else:
        coef = obj['coefficient']
        sig = obj['significant_at_0.05']
        ci_low, ci_high = obj['conf_int_95']

        if sig is True and coef < 0:
            direction = "Yes"
            reasoning = (
                f"{direction}: the estimated coefficient on student_teacher_ratio is {coef:.4f} "
                f"(p = {obj['p_value']:.3g}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]). "
                "Because the coefficient is negative and statistically significant, a lower "
                "student-teacher ratio (fewer students per teacher) is associated with higher "
                "average academic performance. The point estimate implies that a one-unit decrease "
                f"in student-teacher ratio is associated with an average score increase of {abs(coef):.4f} points."
            )
        elif sig is True and coef > 0:
            direction = "No (opposite)"
            reasoning = (
                f"{direction}: the estimated coefficient on student_teacher_ratio is {coef:.4f} "
                f"(p = {obj['p_value']:.3g}, 95% CI [{ci_low:.4f}, {ci_high:.4f}]). "
                "Because the coefficient is positive and statistically significant, a lower "
                "student-teacher ratio would be associated with lower average performance (the opposite "
                "of the hypothesized direction)."
            )
        else:
            # Not statistically significant
            direction = "Inconclusive"
            p_text = f"p = {obj['p_value']:.3g}" if obj['p_value'] is not None else "p-value unavailable"
            ci_text = f"95% CI [{ci_low:.4f}, {ci_high:.4f}]" if (ci_low is not None and ci_high is not None) else "95% CI unavailable"
            reasoning = (
                f"{direction}: the estimated coefficient on student_teacher_ratio is {coef:.4f} "
                f"({p_text}, {ci_text}). The coefficient is not statistically significant at the 0.05 level, "
                "so the data do not provide strong evidence that a lower student-teacher ratio is associated with "
                "higher academic performance after controlling for the included covariates and fixed effects."
            )

        description = reasoning

    return {"object": obj, "description": description}