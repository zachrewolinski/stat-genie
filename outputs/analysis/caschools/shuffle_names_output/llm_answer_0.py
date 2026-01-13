def extract_final_answer(model_output):
    """
    Extracts the coefficient and inference for the STR_z predictor from a fitted statsmodels OLS result.

    Returns a dictionary with keys:
      - "object": dict with numeric outputs (coef, se, t, pvalue, 95% CI, n, df_resid, significant)
      - "description": human-readable interpretation answering whether a lower student-teacher ratio
                       (fewer students per teacher) is associated with higher academic performance.

    Expects model_output to be a statsmodels RegressionResultsWrapper (or similar) with named parameters
    that include 'STR_z'.
    """
    try:
        res = model_output

        # Name of the predictor in the model
        var = 'STR_z'

        # Ensure params exist
        if not hasattr(res, 'params'):
            raise AttributeError("model_output has no attribute 'params'")

        params_index = list(res.params.index)
        if var not in params_index:
            raise KeyError(f"Variable '{var}' not found in model parameters: {params_index}")

        # Extract core statistics
        coef = float(res.params[var])
        # Some result objects might not have bse/tvalues/pvalues if not fitted in usual way; guard access
        se = float(res.bse[var]) if hasattr(res, 'bse') else None
        t_val = float(res.tvalues[var]) if hasattr(res, 'tvalues') else None
        p_val = float(res.pvalues[var]) if hasattr(res, 'pvalues') else None

        # Confidence interval (attempt to index by name; fall back to positional)
        try:
            ci = res.conf_int(alpha=0.05).loc[var].tolist()
        except Exception:
            # conf_int may return ndarray with rows matching params order
            ci_array = res.conf_int(alpha=0.05)
            pos = params_index.index(var)
            ci = [float(ci_array[pos, 0]), float(ci_array[pos, 1])]

        ci_lower, ci_upper = float(ci[0]), float(ci[1])

        # Sample size and residual df if available
        n_obs = int(getattr(res, 'nobs', res.model.endog.shape[0] if hasattr(res, 'model') else None))
        df_resid = int(res.df_resid) if hasattr(res, 'df_resid') else None

        # Determine statistical significance at alpha = 0.05 if p-value exists
        significant = None
        if p_val is not None:
            significant = bool(p_val < 0.05)

        # Interpret direction in context: remember STR_z is (students per teacher) z-scored,
        # and lower STR_z means fewer students per teacher. The model coefficient b gives
        # the change in AvgTestScore for a one SD increase in STR (more students per teacher).
        # Therefore, a negative coef means that increasing STR (more students/teacher) lowers scores,
        # equivalently a one SD decrease in STR (fewer students/teacher) increases scores by |coef|.
        if coef < 0:
            direction_text = ("The estimated coefficient is negative, so a one SD decrease in STR "
                              f"(fewer students per teacher) is associated with an estimated increase "
                              f"of {abs(coef):.3f} test-score units.")
        elif coef > 0:
            direction_text = ("The estimated coefficient is positive, so a one SD decrease in STR "
                              f"(fewer students per teacher) is associated with an estimated decrease "
                              f"of {abs(coef):.3f} test-score units.")
        else:
            direction_text = "The estimated coefficient is zero, implying no association."

        sig_text = ("This association is statistically significant at alpha=0.05."
                    if significant is True else
                    "This association is NOT statistically significant at alpha=0.05."
                    if significant is False else
                    "Statistical significance could not be determined (p-value unavailable).")

        description = (
            f"STR_z coefficient = {coef:.4f} (SE = {se:.4f} ; t = {t_val:.3f} ; p = {p_val:.4g}). "
            f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]. {direction_text} {sig_text} "
            f"Sample size n = {n_obs}, df_resid = {df_resid}."
        )

        result_object = {
            'variable': var,
            'coef': coef,
            'se': se,
            't_value': t_val,
            'p_value': p_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_obs': n_obs,
            'df_resid': df_resid,
            'significant_at_0.05': significant
        }

        return {'object': result_object, 'description': description}

    except Exception as e:
        return {
            'object': None,
            'description': f"Error extracting results for STR_z: {e}"
        }