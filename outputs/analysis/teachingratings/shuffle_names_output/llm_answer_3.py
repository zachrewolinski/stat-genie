import numpy as np

def extract_final_answer(model_output):
    """
    Extract key statistics for the predictor 'Beauty_c' from a statsmodels results object
    (ordinary or robust covariance results).

    Returns a dictionary with:
      - "object": dict with numeric results (coef, se, t, p, 95% CI, significance)
      - "description": human-readable interpretation of the effect of Beauty_c on TeachingEval

    The function is robust to model_output.params being a pandas Series or a numpy.ndarray
    and to associated statistics (bse, tvalues, pvalues, conf_int) being either array-like
    or pandas objects.
    """
    # Helper to retrieve attribute or raise informative error
    if not hasattr(model_output, "params"):
        raise ValueError("model_output does not appear to be a statsmodels results object with .params")

    params_attr = getattr(model_output, "params")

    # Determine parameter names and parameter values array
    if hasattr(params_attr, "index"):
        param_names = list(params_attr.index)
        params_values = np.asarray(params_attr)
    else:
        # params is likely an ndarray; try to get names from common locations
        params_values = np.asarray(params_attr)
        if hasattr(model_output, "model") and hasattr(model_output.model, "exog_names"):
            param_names = list(model_output.model.exog_names)
        elif hasattr(model_output, "param_names"):
            param_names = list(model_output.param_names)
        elif hasattr(model_output, "names"):
            param_names = list(model_output.names)
        else:
            raise ValueError(
                "Could not determine parameter names from model_output. "
                "params is an ndarray and no parameter name source was found."
            )

    if "Beauty_c" not in param_names:
        raise KeyError("The fitted model does not contain a parameter named 'Beauty_c'")

    idx = param_names.index("Beauty_c")

    # Safely get coefficient value
    try:
        coef = float(params_values[idx])
    except Exception:
        # Fallback: if params_attr is a pandas Series-like but without index
        try:
            coef = float(params_attr["Beauty_c"])
        except Exception:
            raise ValueError("Unable to extract coefficient value for 'Beauty_c'")

    # Helper to extract stat either by name or by index
    def _extract_stat(attr_name):
        if not hasattr(model_output, attr_name):
            return None
        stat = getattr(model_output, attr_name)
        # Try name-based access (pandas Series / DataFrame)
        try:
            if hasattr(stat, "loc"):
                val = stat.loc["Beauty_c"]
                return float(val)
        except Exception:
            pass
        # Try dict-like / key access
        try:
            val = stat["Beauty_c"]
            return float(val)
        except Exception:
            pass
        # Try index-based access
        try:
            val = stat[idx]
            return float(val)
        except Exception:
            pass
        return None

    se = _extract_stat("bse")
    t_stat = _extract_stat("tvalues")
    p_value = _extract_stat("pvalues")

    # Confidence interval
    ci_lower = ci_upper = None
    try:
        ci = model_output.conf_int(alpha=0.05)
        # DataFrame-like
        if hasattr(ci, "loc"):
            try:
                row = ci.loc["Beauty_c"]
                # row may be a Series or array-like of length 2
                ci_lower = float(row.iloc[0]) if hasattr(row, "iloc") else float(row[0])
                ci_upper = float(row.iloc[1]) if hasattr(row, "iloc") else float(row[1])
            except Exception:
                # try alternative indexing
                try:
                    ci_lower = float(ci.loc["Beauty_c", 0])
                    ci_upper = float(ci.loc["Beauty_c", 1])
                except Exception:
                    ci_lower = ci_upper = None
        else:
            # ndarray-like
            ci_arr = np.asarray(ci)
            ci_lower = float(ci_arr[idx, 0])
            ci_upper = float(ci_arr[idx, 1])
    except Exception:
        ci_lower = ci_upper = None

    significant = None
    if p_value is not None:
        try:
            significant = bool(p_value < 0.05)
        except Exception:
            significant = None

    result_object = {
        "parameter": "Beauty_c",
        "coef": coef,
        "std_err": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "significant_at_0.05": significant
    }

    # Formatting helpers
    def _fmt_num(x):
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.3f}"
        except Exception:
            return str(x)

    # Build human-readable description
    if p_value is None:
        significance_text = "p-value unavailable; cannot assess statistical significance."
    else:
        if significant:
            direction = "positive" if coef > 0 else "negative"
            significance_text = (
                f"The association is statistically significant (p = {p_value:.3g}). "
                f"Higher beauty scores are associated with {direction} teaching evaluation scores."
            )
        else:
            significance_text = (
                f"No statistically significant association detected (p = {p_value:.3g}). "
                "We do not have evidence that beauty is associated with teaching evaluation scores."
            )

    magnitude_text = (
        f"The estimated coefficient is {_fmt_num(coef)} "
        f"(95% CI [{_fmt_num(ci_lower)}, {_fmt_num(ci_upper)}]). "
        f"This means a one-unit increase in centered beauty is associated with a "
        f"{'+' if coef >= 0 else ''}{_fmt_num(coef)} change in TeachingEval on the 1-5 scale."
    )

    description = "Effect of Beauty on Teaching Evaluations: " + significance_text + " " + magnitude_text

    return {"object": result_object, "description": description}