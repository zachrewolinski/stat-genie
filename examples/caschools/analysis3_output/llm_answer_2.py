import numpy as np

def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio_z' predictor from a fitted statsmodels results object
    (regular or robust results returned by get_robustcov_results) and interprets whether a lower
    student-teacher ratio is associated with higher AvgTestScore.

    Returns a dict with:
      - "object": a dict of extracted numeric statistics (coef, se, t, p, 95% CI, effect interpretation)
      - "description": a concise explanation of what the numbers mean for the yes/no question
    """
    results = model_output
    param_name = 'StudentTeacherRatio_z'

    # Helper: get parameter names robustly
    param_names = None
    # Try typical pandas Series index
    if hasattr(results, 'params'):
        params_obj = results.params
        if hasattr(params_obj, 'index'):
            param_names = list(params_obj.index)
        else:
            # Try model exog names or other attributes
            if hasattr(results, 'model'):
                exog_names = getattr(results.model, 'exog_names', None)
                if exog_names:
                    param_names = list(exog_names)
            if param_names is None:
                # Try result-level param_names
                param_names = list(getattr(results, 'param_names', [])) or None

            # Fallback: if params is array-like, create numeric names
            if param_names is None:
                try:
                    length = len(params_obj)
                    param_names = [str(i) for i in range(length)]
                except Exception:
                    param_names = []
    else:
        raise ValueError("Provided model_output does not have 'params' attribute expected from statsmodels results.")

    if param_name not in param_names:
        raise ValueError(f"Predictor '{param_name}' not found in the model parameters: {param_names}")

    param_idx = param_names.index(param_name)

    # Helper to extract a value that may be stored as a Series (indexable by name) or ndarray (indexable by position)
    def _get_stat_attr(results, attr, idx, name):
        obj = getattr(results, attr, None)
        if obj is None:
            return None
        try:
            # If it's a pandas-like Series/DataFrame row access
            if hasattr(obj, 'index') and name in obj.index:
                return float(obj[name])
        except Exception:
            pass
        try:
            # Array-like access
            return float(obj[idx])
        except Exception:
            # Last resort: try attribute lookup
            try:
                return float(getattr(obj, name))
            except Exception:
                return None

    # Extract coefficient
    try:
        coef = _get_stat_attr(results, 'params', param_idx, param_name)
    except Exception:
        coef = None

    se = _get_stat_attr(results, 'bse', param_idx, param_name)
    tval = _get_stat_attr(results, 'tvalues', param_idx, param_name)
    pval = _get_stat_attr(results, 'pvalues', param_idx, param_name)

    # Confidence interval
    lower = upper = None
    try:
        ci = results.conf_int(alpha=0.05)
        # If ci is a DataFrame-like with index
        if hasattr(ci, 'loc') and param_name in getattr(ci, 'index', []):
            row = ci.loc[param_name]
            lower = float(row.iloc[0])
            upper = float(row.iloc[1])
        else:
            # Assume ndarray-like
            ci_arr = np.asarray(ci)
            if ci_arr.ndim == 2 and ci_arr.shape[0] > param_idx:
                lower = float(ci_arr[param_idx, 0])
                upper = float(ci_arr[param_idx, 1])
    except Exception:
        lower, upper = (None, None)

    # Interpretation
    if coef is None:
        direction = "unknown"
    else:
        direction = "negative" if coef < 0 else ("positive" if coef > 0 else "null")

    if pval is None:
        significance_text = "p-value unavailable; cannot determine statistical significance."
        significant = None
    else:
        significant = (pval < 0.05)
        significance_text = ("statistically significant (p < 0.05)" if significant else "not statistically significant (p >= 0.05)")

    # Construct conclusion
    if coef is None:
        conclusion = "No clear relationship detected (coefficient is missing)."
    else:
        if coef < 0 and significant is True:
            conclusion = "Yes — lower student-teacher ratio (fewer students per teacher) is associated with higher AvgTestScore; coefficient is negative and statistically significant."
        elif coef < 0 and (significant is False):
            conclusion = "No strong evidence — the coefficient is negative (consistent with lower ratio → higher performance) but it is not statistically significant."
        elif coef > 0 and significant is True:
            conclusion = "No — the model shows a statistically significant positive association: higher student-teacher ratio is associated with higher AvgTestScore (contrary to the hypothesized direction)."
        elif coef > 0 and (significant is False):
            conclusion = "No strong evidence of the expected relationship — coefficient is positive (opposite direction) and not statistically significant."
        else:
            conclusion = "No clear relationship detected (coefficient is approximately zero)."

    # Safe formatting helpers
    def _fmt(x, digits=4):
        return f"{x:.{digits}f}" if (x is not None) else "N/A"

    # Effect magnitude explanation
    if coef is None:
        effect_text = "Effect size could not be computed."
    else:
        if lower is None or upper is None:
            ci_text = "95% CI unavailable"
        else:
            ci_text = f"95% CI [{_fmt(lower,3)}, {_fmt(upper,3)}]"
        if coef < 0:
            effect_text = f"A 1 SD decrease in StudentTeacherRatio_z is associated with an estimated increase of {_fmt(abs(coef),3)} units in AvgTestScore ({ci_text})."
        else:
            effect_text = f"A 1 SD increase in StudentTeacherRatio_z is associated with an estimated change of {_fmt(coef,3)} units in AvgTestScore ({ci_text})."

    output_object = {
        "predictor": param_name,
        "coef": coef,
        "std_err": se,
        "t_value": tval,
        "p_value": pval,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "direction": direction,
        "significant_at_0.05": significant,
        "conclusion_text": conclusion,
        "effect_interpretation": effect_text
    }

    description = (
        f"Extracted coefficient and inference for '{param_name}'. Coefficient = {_fmt(coef)}, "
        f"SE = {_fmt(se)} (if available), t = {_fmt(tval)}, p = {_fmt(pval)}. "
        f"95% CI = [{_fmt(lower)}, {_fmt(upper)}]. Direction: {direction}. {significance_text} {conclusion} {effect_text}"
    )

    return {"object": output_object, "description": description}