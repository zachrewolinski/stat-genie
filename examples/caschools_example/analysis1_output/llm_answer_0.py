import numpy as np

def extract_final_answer(model_output):
    """
    Extracts the coefficient, robust SE, t-value, p-value, and 95% CI for the
    StudentTeacherRatio coefficient from a statsmodels OLS results object
    (including robustcov results returned by get_robustcov_results).

    Returns a dictionary with keys:
      - "object": dict of numeric statistics for StudentTeacherRatio
      - "description": a short interpretation answering whether a lower
                       student-teacher ratio is associated with higher performance.
    """
    var = 'StudentTeacherRatio'

    # Ensure the model output has the expected attributes
    try:
        params_raw = model_output.params
        bse_raw = model_output.bse
        tvalues_raw = model_output.tvalues
        pvalues_raw = model_output.pvalues
        ci_raw = model_output.conf_int()
    except Exception as e:
        raise ValueError("Provided model_output does not have expected statsmodels attributes: " + str(e))

    # Determine parameter names in a robust way
    names = None
    try:
        if hasattr(params_raw, 'index'):
            # pandas Series or similar
            names = list(params_raw.index)
    except Exception:
        names = None

    if names is None:
        # Try to get names from the model specification
        if hasattr(model_output, 'model'):
            m = model_output.model
            if hasattr(m, 'exog_names'):
                try:
                    names = list(m.exog_names)
                except Exception:
                    names = None
            if names is None and hasattr(m, 'data') and hasattr(m.data, 'param_names'):
                try:
                    names = list(m.data.param_names)
                except Exception:
                    names = None

    if names is None and hasattr(model_output, 'names'):
        try:
            names = list(model_output.names)
        except Exception:
            names = None

    if names is None:
        # As a last resort, try to infer length from params_raw
        try:
            length = len(params_raw)
            names = [i for i in range(length)]
        except Exception:
            raise ValueError("Could not determine parameter names/indexes from model_output.")

    # Ensure variable exists
    if var not in names:
        raise KeyError(f"Variable '{var}' not found in model parameters. Available parameters: {list(names)}")

    idx = list(names).index(var)

    # Helper to robustly extract a numeric value by name or by index
    def _extract_numeric(raw, name_idx, idx):
        # Try by name first (works for pandas Series)
        try:
            return float(raw[name_idx])
        except Exception:
            pass
        # Then try by numeric index (works for numpy arrays)
        try:
            return float(raw[idx])
        except Exception as e:
            raise ValueError(f"Unable to extract numeric value for '{name_idx}' at index {idx}: {e}")

    coef = _extract_numeric(params_raw, var, idx)
    std_err = _extract_numeric(bse_raw, var, idx)
    t_value = _extract_numeric(tvalues_raw, var, idx)
    p_value = _extract_numeric(pvalues_raw, var, idx)

    # Extract 95% CI, handling possible formats (DataFrame-like or ndarray)
    ci_lower, ci_upper = None, None
    try:
        # DataFrame-like with .loc
        if hasattr(ci_raw, 'loc'):
            try:
                ci_row = ci_raw.loc[var]
            except Exception:
                # maybe index is positional
                ci_row = ci_raw.iloc[idx]
            try:
                ci_lower = float(ci_row.iloc[0])
                ci_upper = float(ci_row.iloc[1])
            except Exception:
                ci_lower = float(ci_row[0])
                ci_upper = float(ci_row[1])
        else:
            # ndarray-like: assume shape (k, 2)
            ci_arr = np.asarray(ci_raw)
            ci_lower = float(ci_arr[idx, 0])
            ci_upper = float(ci_arr[idx, 1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Formulate a concise conclusion using a 5% significance threshold
    alpha = 0.05
    if p_value < alpha:
        if coef < 0:
            conclusion = (
                "Yes. The StudentTeacherRatio coefficient is negative and statistically significant "
                f"(coef = {coef:.4f}, SE = {std_err:.4f}, t = {t_value:.3f}, p = {p_value:.3g}). "
                "This indicates that a lower student-to-teacher ratio (fewer students per teacher) "
                "is associated with higher district average academic performance."
            )
        else:
            conclusion = (
                "No. The StudentTeacherRatio coefficient is positive and statistically significant "
                f"(coef = {coef:.4f}, SE = {std_err:.4f}, t = {t_value:.3f}, p = {p_value:.3g}). "
                "This indicates that a lower student-to-teacher ratio is associated with lower performance "
                "(i.e., higher ratios are associated with higher scores)."
            )
    else:
        if coef < 0:
            conclusion = (
                "No strong evidence. The StudentTeacherRatio coefficient is negative but not statistically significant "
                f"(coef = {coef:.4f}, SE = {std_err:.4f}, t = {t_value:.3f}, p = {p_value:.3g}). "
                "We cannot conclude a reliable association between lower student-teacher ratios and higher performance."
            )
        else:
            conclusion = (
                "No strong evidence. The StudentTeacherRatio coefficient is positive and not statistically significant "
                f"(coef = {coef:.4f}, SE = {std_err:.4f}, t = {t_value:.3f}, p = {p_value:.3g}). "
                "We cannot conclude a reliable association between student-teacher ratio and performance."
            )

    return {
        "object": {
            "variable": var,
            "coef": coef,
            "std_err": std_err,
            "t_value": t_value,
            "p_value": p_value,
            "95%_ci_lower": ci_lower,
            "95%_ci_upper": ci_upper
        },
        "description": conclusion
    }