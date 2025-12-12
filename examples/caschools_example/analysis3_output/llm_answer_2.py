def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a statsmodels OLS results object.
    Returns a dict with keys "object" (detailed numeric results) and "description" (interpreting the results).
    """
    import numpy as np

    res = model_output

    # Prepare a safe container for results
    result_obj = {}
    try:
        # Parameter name expected in the fitted model
        var_name = 'StudentTeacherRatio'

        # Access parameter table
        params = res.params
        if var_name not in params.index:
            # try case-insensitive match as a fallback
            matches = [n for n in params.index if n.lower() == var_name.lower()]
            if matches:
                var_name = matches[0]
            else:
                raise KeyError(f"Variable '{var_name}' not found in model parameters. Available params: {list(params.index)}")

        coef = float(res.params[var_name])
        stderr = float(res.bse[var_name]) if hasattr(res, 'bse') else float('nan')
        t_value = float(res.tvalues[var_name]) if hasattr(res, 'tvalues') else float('nan')
        p_value = float(res.pvalues[var_name]) if hasattr(res, 'pvalues') else float('nan')
        ci = res.conf_int().loc[var_name].tolist() if hasattr(res, 'conf_int') else [float('nan'), float('nan')]
        ci_lower, ci_upper = float(ci[0]), float(ci[1])

        # Model-level info
        r_squared = float(getattr(res, 'rsquared', np.nan))
        nobs = int(getattr(res, 'nobs', np.nan)) if getattr(res, 'nobs', None) is not None else None

        # Compute standardized coefficient (beta) if possible using model endog/exog
        std_beta = float('nan')
        try:
            exog_names = res.model.exog_names
            idx = exog_names.index(var_name)
            x = res.model.exog[:, idx]
            y = res.model.endog
            sx = np.std(x, ddof=1)
            sy = np.std(y, ddof=1)
            if sy != 0:
                std_beta = coef * (sx / sy)
        except Exception:
            # leave std_beta as nan if any issue
            std_beta = float('nan')

        # Populate object to return
        result_obj = {
            "variable": var_name,
            "coef": coef,
            "std_err": stderr,
            "t_value": t_value,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "standardized_beta": std_beta,
            "r_squared": r_squared,
            "nobs": nobs
        }

        # Build human-readable interpretation
        significance = "statistically significant" if (p_value is not None and p_value < 0.05) else "not statistically significant"
        if coef < 0:
            direction = "negative"
            implication = "smaller student-teacher ratios (smaller classes) are associated with higher average scores"
        elif coef > 0:
            direction = "positive"
            implication = "larger student-teacher ratios (larger classes) are associated with higher average scores"
        else:
            direction = "zero"
            implication = "no association detected"

        desc = (
            f"Coefficient on {var_name}: {coef:.4f} (SE={stderr:.4f}, t={t_value:.2f}, p={p_value:.3g}). "
            f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}]. This coefficient is {direction} and is {significance} at the 0.05 level. "
            f"In substantive terms, a one-unit increase in StudentTeacherRatio (one additional student per teacher) is associated "
            f"with a {coef:.4f} point change in AvgScore. Interpretation: {implication} if the coefficient is negative and statistically significant. "
            f"Standardized effect (beta) = {std_beta:.4f}. Model R-squared = {r_squared:.3f} (n = {nobs})."
        )

        return {"object": result_obj, "description": desc}

    except Exception as e:
        # If anything goes wrong, return the exception message for debugging
        return {
            "object": None,
            "description": f"Could not extract statistics for StudentTeacherRatio from the model output. Error: {e}"
        }