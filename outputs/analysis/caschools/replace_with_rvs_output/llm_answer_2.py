def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, t-value, p-value, 95% CI, sample size,
    and a brief interpretation about whether a lower student-teacher ratio
    is associated with higher academic performance.

    Returns:
      {
        "object": { ... numeric results ... , "conclusion": <str> },
        "description": <str explanation>
      }
    """
    # Name of the predictor of interest
    var = 'StudentTeacherRatio'
    # Default result if extraction fails
    result_obj = {
        "coef": None,
        "std_err": None,
        "t_value": None,
        "p_value": None,
        "ci_2.5": None,
        "ci_97.5": None,
        "n_obs": None,
        "significant_at_0.05": None,
        "conclusion": None  # brief yes/no statement about the task question
    }

    try:
        # Coefficient and statistics
        params = model_output.params
        bse = model_output.bse
        tvals = model_output.tvalues
        pvals = model_output.pvalues
        ci = model_output.conf_int(alpha=0.05)
        nobs = getattr(model_output, "nobs", None)

        if var not in params.index:
            raise KeyError(f"Variable '{var}' not found in model parameters")

        coef = float(params[var])
        std_err = float(bse[var]) if var in bse.index else None
        t_value = float(tvals[var]) if var in tvals.index else None
        p_value = float(pvals[var]) if var in pvals.index else None
        ci_row = ci.loc[var] if var in ci.index else [None, None]
        ci_lower = float(ci_row[0]) if ci_row[0] is not None else None
        ci_upper = float(ci_row[1]) if ci_row[1] is not None else None
        nobs_int = int(nobs) if (nobs is not None) else None

        # Fill result object
        result_obj.update({
            "coef": coef,
            "std_err": std_err,
            "t_value": t_value,
            "p_value": p_value,
            "ci_2.5": ci_lower,
            "ci_97.5": ci_upper,
            "n_obs": nobs_int,
        })

        # Interpret sign: a negative coef means that increasing StudentTeacherRatio
        # (more students per teacher) is associated with lower AvgScore,
        # equivalently, a lower ratio (fewer students per teacher) is associated
        # with higher AvgScore.
        sign_interp = ("A negative coefficient indicates that lower student-teacher "
                       "ratios (fewer students per teacher) are associated with higher "
                       "academic performance; a positive coefficient indicates the opposite.")
        # Statistical significance at alpha=0.05
        sig = None
        if p_value is not None:
            sig = (p_value < 0.05)
            result_obj["significant_at_0.05"] = bool(sig)
        else:
            result_obj["significant_at_0.05"] = None

        # Build conclusion string
        if coef is not None:
            if sig is True:
                if coef < 0:
                    conclusion = ("Yes — statistically significant evidence (p < 0.05) that "
                                  "lower student-teacher ratios are associated with higher AvgScore. "
                                  f"Estimated effect: {coef:.4g} points change in AvgScore per one-unit change in StudentTeacherRatio "
                                  f"(95% CI [{ci_lower:.4g}, {ci_upper:.4g}]).")
                else:
                    conclusion = ("No — statistically significant evidence (p < 0.05) that "
                                  "higher student-teacher ratios are associated with higher AvgScore (opposite direction). "
                                  f"Estimated effect: {coef:.4g} (95% CI [{ci_lower:.4g}, {ci_upper:.4g}]).")
            elif sig is False:
                # Not statistically significant
                if coef < 0:
                    conclusion = ("No strong evidence (p >= 0.05) that lower student-teacher ratios "
                                  "are associated with higher AvgScore. Point estimate is negative "
                                  f"({coef:.4g}) but not statistically significant (p = {p_value:.4g}).")
                else:
                    conclusion = ("No strong evidence (p >= 0.05) that lower student-teacher ratios "
                                  "are associated with higher AvgScore. Point estimate is positive "
                                  f"({coef:.4g}) and not statistically significant (p = {p_value:.4g}).")
            else:
                conclusion = ("Unable to determine statistical significance (p-value missing). "
                              f"Point estimate: {coef:.4g}, 95% CI [{ci_lower:.4g}, {ci_upper:.4g}].")
        else:
            conclusion = "Could not extract coefficient for StudentTeacherRatio."

        result_obj["conclusion"] = conclusion

        # Final return object and a brief description
        description = (
            "Extracted OLS estimate and inference for the StudentTeacherRatio coefficient from the fitted model. "
            "Coefficient is the estimated change in AvgScore (test-score points) associated with a one-unit increase "
            "in StudentTeacherRatio (students per teacher). A negative coefficient implies that lower ratios (fewer "
            "students per teacher) are associated with higher academic performance. Significance is evaluated at alpha=0.05."
        )

        return {"object": result_obj, "description": description}

    except Exception as e:
        # Return error information in a consistent format
        return {
            "object": result_obj,
            "description": f"Failed to extract statistics for '{var}': {str(e)}"
        }