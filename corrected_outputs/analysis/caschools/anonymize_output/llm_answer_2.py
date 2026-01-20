def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio_z' coefficient from a fitted model output
    (expected to be a statsmodels RegressionResultsWrapper). Returns a dictionary with:
      - "object": dict of extracted numeric results
      - "description": a short plain-language interpretation answering whether a lower
                       student-teacher ratio is associated with higher academic performance.
    """
    var = 'StudentTeacherRatio_z'
    # Prepare result container
    result_obj = {}
    try:
        # Expecting a statsmodels RegressionResultsWrapper
        params = model_output.params
        pvalues = model_output.pvalues
        conf = model_output.conf_int()
        tvalues = getattr(model_output, 'tvalues', None)
        nobs = getattr(model_output, 'nobs', None)

        if var not in params.index:
            raise KeyError(f"Variable '{var}' not found in model parameters: {list(params.index)}")

        coef = float(params[var])
        pval = float(pvalues[var])
        ci_lower, ci_upper = map(float, conf.loc[var])
        tstat = float(tvalues[var]) if tvalues is not None else None

        # Interpret sign: predictor is standardized such that higher StudentTeacherRatio_z = more students per teacher.
        # So a negative coef means higher ratio -> lower scores, i.e., lower ratio -> higher scores.
        lower_ratio_associated_with_higher_scores = coef < 0
        significant = pval < 0.05

        # Fill object with extracted statistics (rounded for readability)
        result_obj = {
            'variable': var,
            'coef': round(coef, 4),
            'p_value': round(pval, 4),
            't_value': round(tstat, 4) if tstat is not None else None,
            'ci_lower_95': round(ci_lower, 4),
            'ci_upper_95': round(ci_upper, 4),
            'n_obs': int(nobs) if nobs is not None else None,
            'significant_at_0.05': bool(significant),
            'direction_lower_ratio_higher_scores': bool(lower_ratio_associated_with_higher_scores),
            'interpretation_detail': (
                "Coefficient is the change in AvgScore associated with a 1 SD increase in student-teacher ratio "
                "(since StudentTeacherRatio_z is standardized)."
            )
        }

        # Build a concise human-readable description
        if lower_ratio_associated_with_higher_scores:
            dir_text = ("A lower student-teacher ratio (fewer students per teacher) is associated with "
                        "higher AvgScore (coefficient negative).")
        else:
            dir_text = ("A lower student-teacher ratio (fewer students per teacher) is associated with "
                        "lower AvgScore (coefficient positive).")

        sig_text = ("This effect is statistically significant at alpha=0.05."
                    if significant else
                    "This effect is NOT statistically significant at alpha=0.05.")

        description = (
            f"{dir_text} Coefficient = {round(coef,4)}, 95% CI = [{round(ci_lower,4)}, {round(ci_upper,4)}], "
            f"p = {round(pval,4)}. {sig_text}"
        )

        return {"object": result_obj, "description": description}

    except Exception as e:
        # Fallback: attempt to handle sklearn-like objects (coef_, intercept_) if provided
        try:
            coef_attr = getattr(model_output, 'coef_', None)
            if coef_attr is None:
                raise e  # re-raise original if fallback not possible
            # Unable to get variable names from sklearn pipeline here, so return best-effort message
            return {
                "object": None,
                "description": (
                    "Model appears not to be a statsmodels RegressionResultsWrapper. "
                    "Fitted object has coef_ but variable-to-coefficient mapping is unknown; "
                    "please provide a statsmodels result or include feature names."
                )
            }
        except Exception:
            raise ValueError(f"Could not extract results for '{var}' from the provided model object. Original error: {e}")