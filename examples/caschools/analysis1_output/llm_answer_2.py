import numpy as np

def extract_final_answer(model_output):
    """
    Extracts statistics for the 'StudentTeacherRatio' coefficient from a fitted statsmodels OLS results object
    (including robust-covariance results returned by get_robustcov_results).

    Returns a dictionary with keys:
      - "object": a dict containing numeric results (coefficient, robust SE, t, p-value, 95% CI, significance flag,
                  and a plain-language short interpretation).
      - "description": a brief explanation of what the numbers mean in context.

    This function is robust to model_output.params being either a pandas Series (with an index of parameter names)
    or a numpy array. If params is array-like, the function attempts to obtain parameter names from:
      - model_output.model.exog_names
      - model_output.param_names
      - model_output.names
    """
    # Ensure the model_output has the expected attributes/methods
    required_attrs = ['params', 'bse', 'tvalues', 'pvalues', 'conf_int']
    for attr in required_attrs:
        if not hasattr(model_output, attr):
            raise AttributeError(f"model_output missing required attribute: {attr}")

    param_name = 'StudentTeacherRatio'

    params = model_output.params

    # Determine parameter names and index of the target parameter
    if hasattr(params, 'index'):  # likely a pandas Series
        param_names = list(params.index)
        if param_name not in param_names:
            raise KeyError(f"Coefficient '{param_name}' not found in model output. Available params: {param_names}")
        idx = param_names.index(param_name)
        coef = float(params[param_name])
    else:
        # params is array-like (e.g., numpy.ndarray). Try to find parameter names elsewhere.
        param_names = None
        if hasattr(model_output, 'model') and hasattr(model_output.model, 'exog_names'):
            param_names = list(model_output.model.exog_names)
        elif hasattr(model_output, 'param_names'):
            param_names = list(model_output.param_names)
        elif hasattr(model_output, 'names'):
            param_names = list(model_output.names)

        if param_names is None:
            raise AttributeError(
                "model_output.params is array-like and parameter names are not available. "
                "Expected model_output.model.exog_names or model_output.param_names or model_output.names."
            )

        if param_name not in param_names:
            raise KeyError(f"Coefficient '{param_name}' not found in model output. Available params: {param_names}")
        idx = param_names.index(param_name)
        coef = float(np.asarray(params)[idx])

    # Helper to extract a scalar value from model_output attributes that may be Series or array-like
    def _get_scalar(attr_name):
        attr = getattr(model_output, attr_name)
        # If it's callable (unlikely for these), call it without args
        if callable(attr) and not isinstance(attr, (np.ndarray, list, tuple)):
            try:
                attr = attr()
            except TypeError:
                # not callable without args; fall through
                pass

        if hasattr(attr, 'loc') and param_name in getattr(attr, 'index', []):
            return float(attr.loc[param_name])
        if hasattr(attr, 'get') and param_name in getattr(attr, 'keys', lambda: [])():
            # dict-like
            return float(attr.get(param_name))
        # array-like
        arr = np.asarray(attr)
        return float(arr[idx])

    se = _get_scalar('bse')
    tstat = _get_scalar('tvalues')
    pval = _get_scalar('pvalues')

    # Handle conf_int which may be a method or an attribute. Expect a 2-column structure.
    conf_func_or_attr = getattr(model_output, 'conf_int')
    if callable(conf_func_or_attr):
        conf = conf_func_or_attr(alpha=0.05)
    else:
        conf = conf_func_or_attr

    # conf might be a DataFrame-like, ndarray, or similar
    if hasattr(conf, 'loc') and param_name in getattr(conf, 'index', []):
        ci_row = conf.loc[param_name]
        ci_lower = float(ci_row.iloc[0])
        ci_upper = float(ci_row.iloc[1])
    else:
        conf_arr = np.asarray(conf)
        # If conf_arr is 2D with rows corresponding to params
        if conf_arr.ndim == 2 and conf_arr.shape[0] == len(param_names):
            ci_lower = float(conf_arr[idx, 0])
            ci_upper = float(conf_arr[idx, 1])
        # If conf_arr is e.g. shape (n_params, 2) but param_names unknown, still use idx mapping
        elif conf_arr.ndim == 2 and conf_arr.shape[1] == 2:
            ci_lower = float(conf_arr[idx, 0])
            ci_upper = float(conf_arr[idx, 1])
        else:
            raise ValueError("Unable to interpret conf_int output structure.")

    # Determine statistical significance at conventional levels
    significant_05 = pval < 0.05
    significant_01 = pval < 0.01

    # Interpret direction in context:
    # StudentTeacherRatio = students / teachers. Higher ratio => more students per teacher (larger class sizes).
    if coef < 0:
        direction_text = (
            "Negative coef: higher student-teacher ratio (more students per teacher) is associated with LOWER average test scores; "
            "equivalently, a LOWER student-teacher ratio (fewer students per teacher, smaller classes) is associated with HIGHER scores."
        )
    elif coef > 0:
        direction_text = (
            "Positive coef: higher student-teacher ratio (more students per teacher) is associated with HIGHER average test scores; "
            "equivalently, a LOWER student-teacher ratio is associated with LOWER scores."
        )
    else:
        direction_text = "Coefficient is exactly zero (no association)."

    significance_text = (
        "This association is statistically significant at the 5% level."
        if significant_05 else
        "This association is NOT statistically significant at the 5% level."
    )

    short_interpretation = f"{direction_text} {significance_text}"

    result_object = {
        "variable": param_name,
        "coef": coef,
        "std_err": se,
        "t_stat": tstat,
        "p_value": pval,
        "ci_2.5%": ci_lower,
        "ci_97.5%": ci_upper,
        "significant_at_0.05": bool(significant_05),
        "significant_at_0.01": bool(significant_01),
        # plain-language summary useful for automated decisions
        "interpretation": short_interpretation
    }

    description = (
        "Extracted the estimate and inference for the student-teacher ratio from the OLS results (robust HC3 SEs). "
        "coef = estimated change in district average test score (AvgScore) for a one-unit increase in StudentTeacherRatio "
        "(i.e., one additional student per teacher). The p-value and 95% CI assess statistical significance and precision. "
        "A negative coefficient implies that reducing the student-teacher ratio (fewer students per teacher) is associated "
        "with higher average test scores; the description field above states whether this association is statistically significant."
    )

    return {"object": result_object, "description": description}