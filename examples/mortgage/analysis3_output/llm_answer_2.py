import numpy as np
from scipy import stats


def extract_final_answer(model_output):
    """
    Extract statistics for the 'female' coefficient from the provided model_output (e.g., a statsmodels RobustResults).
    Returns a dict with:
      - "object": dict of numeric results (coef, se, z, p, ci, odds_ratio, or_ci)
      - "description": plain-language interpretation of the result and caveats.
    The function is defensive about whether attributes are stored as callables (methods) or attributes,
    and about various possible formats for confidence intervals.
    """

    def _maybe_call(x):
        """If x is callable, call it and return the result; otherwise return x."""
        try:
            return x() if callable(x) else x
        except Exception:
            return None

    def _get_attr_value(obj, attr, key):
        """
        Try to retrieve obj.attr[key], calling attr if it's callable.
        Returns None if anything fails.
        """
        if not hasattr(obj, attr):
            return None
        val = _maybe_call(getattr(obj, attr))
        if val is None:
            return None
        # Try various ways to index
        try:
            return val[key]
        except Exception:
            try:
                return val.loc[key]
            except Exception:
                try:
                    # if val is a dict-like or has get
                    return val.get(key, None)
                except Exception:
                    return None

    def _get_conf_int(obj, key):
        """
        Try to get a two-element confidence interval (lower, upper) for `key`.
        Supports conf_int as method or attribute, with columns named 2.5%/97.5% or 0/1.
        Returns (lower, upper) or (None, None).
        """
        if not hasattr(obj, "conf_int"):
            return None, None
        df = _maybe_call(getattr(obj, "conf_int"))
        if df is None:
            return None, None
        try:
            # Prefer named percentiles
            lower = df.loc[key, "2.5%"]
            upper = df.loc[key, "97.5%"]
            return float(lower), float(upper)
        except Exception:
            pass
        try:
            # Numeric column positions 0 and 1
            lower = df.loc[key].iat[0]
            upper = df.loc[key].iat[1]
            return float(lower), float(upper)
        except Exception:
            pass
        try:
            # If df is indexed by integers or keys differently
            row = df.loc[key]
            vals = list(row)
            if len(vals) >= 2:
                return float(vals[0]), float(vals[1])
        except Exception:
            pass
        return None, None

    def _get_cov_se(obj, key):
        """
        Try to get standard error from covariance matrix diagonal for `key`.
        cov_params may be callable.
        """
        if not hasattr(obj, "cov_params"):
            return None
        cov = _maybe_call(getattr(obj, "cov_params"))
        if cov is None:
            return None
        try:
            var = cov.loc[key, key]
            return float(np.sqrt(var))
        except Exception:
            try:
                # If cov is an ndarray with an index mapping in params
                params = _maybe_call(getattr(obj, "params")) or {}
                idx = list(params.index).index(key) if hasattr(params, "index") and key in params.index else None
                if idx is not None and hasattr(cov, "iat"):
                    return float(np.sqrt(cov.iat[idx, idx]))
            except Exception:
                return None
        return None

    def safe_exp(x):
        try:
            if x is None:
                return None
            x = float(x)
            if np.isnan(x):
                return None
            # prevent overflow
            if x > 700:
                return float("inf")
            if x < -700:
                return 0.0
            return float(np.exp(x))
        except Exception:
            return None

    # Ensure the expected keys/attributes exist
    params = _maybe_call(getattr(model_output, "params", None))
    if params is None or ("female" not in getattr(params, "index", params) and "female" not in params):
        # Try alternative lookup if params is a dict-like with keys directly
        try:
            if isinstance(params, dict) and "female" in params:
                pass
            else:
                raise ValueError("model_output does not contain a 'female' parameter in .params")
        except Exception:
            raise ValueError("model_output does not contain a 'female' parameter in .params")

    # Extract coefficient
    coef_val = _get_attr_value(model_output, "params", "female")
    try:
        coef = float(coef_val)
    except Exception:
        raise ValueError("Could not extract numeric coefficient for 'female' from model_output.params")

    # Standard error: prefer bse, then cov_params diagonal
    se = _get_attr_value(model_output, "bse", "female")
    if se is not None:
        try:
            se = float(se)
        except Exception:
            se = None
    if se is None:
        se = _get_cov_se(model_output, "female")

    # z-stat and p-value
    z = None
    p_value = None
    if se is not None and not np.isnan(se) and se != 0:
        z = coef / se
        # Prefer provided pvalues
        pv = _get_attr_value(model_output, "pvalues", "female")
        if pv is not None:
            try:
                p_value = float(pv)
            except Exception:
                p_value = None
        else:
            p_value = float(2 * stats.norm.sf(abs(z)))
    else:
        pv = _get_attr_value(model_output, "pvalues", "female")
        if pv is not None:
            try:
                p_value = float(pv)
            except Exception:
                p_value = None

    # Confidence interval on log-odds
    ci_lower, ci_upper = _get_conf_int(model_output, "female")
    if (ci_lower is None or ci_upper is None) and (se is not None and not np.isnan(se) and se != 0):
        zcrit = stats.norm.ppf(0.975)
        ci_lower = coef - zcrit * se
        ci_upper = coef + zcrit * se

    # Odds ratio and CI
    odds_ratio = safe_exp(coef)
    or_ci_lower = safe_exp(ci_lower) if ci_lower is not None else None
    or_ci_upper = safe_exp(ci_upper) if ci_upper is not None else None

    result_object = {
        "coef_log_odds": coef,
        "std_error": se,
        "z_stat": float(z) if z is not None else None,
        "p_value": float(p_value) if p_value is not None else None,
        "ci_log_odds": {"2.5%": ci_lower, "97.5%": ci_upper},
        "odds_ratio": odds_ratio,
        "odds_ratio_ci": {"2.5%": or_ci_lower, "97.5%": or_ci_upper},
    }

    # Interpretation text
    sign_text = "negative (coefficient < 0)" if coef < 0 else "positive (coefficient > 0)" if coef > 0 else "zero (coefficient = 0)"

    if p_value is None:
        significance_text = "unknown (p-value not available)"
    else:
        significance_text = f"statistically significant (p = {p_value:.3g})" if p_value < 0.05 else f"not statistically significant (p = {p_value:.3g})"

    # Safe formatting helpers
    def fmt(x):
        try:
            return f"{x:.6g}"
        except Exception:
            return str(x)

    odds_ratio_text = fmt(odds_ratio) if odds_ratio is not None else "unknown"
    ci_log_odds_text = f"[{fmt(ci_lower)}, {fmt(ci_upper)}]" if (ci_lower is not None and ci_upper is not None) else "not available"
    ci_or_text = f"[{fmt(or_ci_lower)}, {fmt(or_ci_upper)}]" if (or_ci_lower is not None and or_ci_upper is not None) else "not available"
    se_text = fmt(se) if se is not None else "not available"
    z_text = fmt(z) if z is not None else "not available"
    p_text = fmt(p_value) if p_value is not None else "not available"

    # Construct description as a concise paragraph
    description = (
        f"The estimated coefficient for 'female' is {fmt(coef)} on the log-odds scale ({sign_text}). "
        f"Robust standard error = {se_text}. z = {z_text}. p-value = {p_text}. "
        f"95% CI for log-odds = {ci_log_odds_text}. Estimated odds ratio = {odds_ratio_text}. "
        f"95% CI for odds ratio = {ci_or_text}. Interpretation: Being female is associated with a "
        f"{'decrease' if coef < 0 else 'increase' if coef > 0 else 'no change'} in the odds of mortgage approval "
        f"by a factor of {odds_ratio_text} relative to males, but this effect is {significance_text}. "
        f"Caveat: If standard errors or confidence intervals are very large or unavailable, the estimate may be imprecise "
        f"or the model may have estimation issues (e.g., separation or sparse data)."
    )

    return {"object": result_object, "description": description}