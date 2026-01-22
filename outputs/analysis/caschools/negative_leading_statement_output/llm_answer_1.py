import numpy as np

def extract_final_answer(model_output):
    """
    Extracts key statistics for the 'StudentTeacherRatio' coefficient from a statsmodels
    results object (robust results returned by get_robustcov_results or a standard results object).
    Returns a dictionary with:
      - "object": a dict of numeric results (coef, se, t, p, 95% CI, nobs, significant)
      - "description": a short plain-language interpretation in the context of the task.
    The implementation is robust to res.params being a pandas Series or a numpy.ndarray.
    """
    var = 'StudentTeacherRatio'
    res = model_output

    if not hasattr(res, 'params'):
        raise ValueError("Model output does not have a 'params' attribute.")

    def get_param_index():
        # Try to find the index position of var among parameter names
        params = res.params
        # If params has an index (e.g., pandas Series), use it
        if hasattr(params, 'index'):
            try:
                return list(params.index).index(var)
            except ValueError:
                return None
        # Fall back to model.exog_names if available
        model = getattr(res, 'model', None)
        if model is not None:
            exog_names = getattr(model, 'exog_names', None)
            if exog_names is not None:
                try:
                    return list(exog_names).index(var)
                except ValueError:
                    return None
            # Some models store names under different attributes
            for attr in ('data',):
                data = getattr(model, attr, None)
                if data is not None:
                    for name_attr in ('param_names', 'orig_exog_names', 'xnames'):
                        names = getattr(data, name_attr, None)
                        if names:
                            try:
                                return list(names).index(var)
                            except ValueError:
                                pass
        return None

    def get_value(attr_name):
        """
        Get the value for variable `var` from res.<attr_name>.
        Returns None if attribute missing or variable not present.
        """
        attr = getattr(res, attr_name, None)
        if attr is None:
            return None
        # Try direct keyed access (pandas Series or dict-like)
        try:
            val = attr[var]
            # If it's a numpy scalar, convert to Python float
            if isinstance(val, (np.generic,)):
                return float(val)
            return float(val)
        except Exception:
            pass
        # Fall back to positional lookup if attr is array-like
        if isinstance(attr, (list, tuple, np.ndarray)):
            idx = get_param_index()
            if idx is None:
                return None
            try:
                return float(attr[idx])
            except Exception:
                return None
        # If attr has .loc (DataFrame), try using it
        if hasattr(attr, 'loc'):
            try:
                row = attr.loc[var]
                if isinstance(row, (list, tuple, np.ndarray, np.generic)):
                    return float(row)
            except Exception:
                pass
        return None

    # Extract coefficient (raise if missing)
    coef = get_value('params')
    if coef is None:
        raise ValueError(f"Model output does not contain the parameter '{var}'.")

    # Other statistics
    se = get_value('bse')
    tstat = get_value('tvalues')
    pvalue = get_value('pvalues')

    # Confidence interval (res.conf_int() may return DataFrame or ndarray)
    ci_lo, ci_hi = None, None
    try:
        ci = res.conf_int()
        # If DataFrame-like with .loc
        try:
            row = None
            if hasattr(ci, 'loc'):
                row = ci.loc[var]
                ci_lo, ci_hi = float(row.iloc[0]), float(row.iloc[1])
            else:
                # ndarray or list of rows
                if isinstance(ci, (list, tuple, np.ndarray)):
                    idx = get_param_index()
                    if idx is not None and idx < len(ci):
                        row = ci[idx]
                        ci_lo, ci_hi = float(row[0]), float(row[1])
        except Exception:
            # final fallback: try interpreting ci as a 2D ndarray and find matching column by name if possible
            pass
    except Exception:
        ci_lo, ci_hi = None, None

    # Sample size if available
    nobs = None
    try:
        nobs_attr = getattr(res, 'nobs', None)
        if nobs_attr is not None:
            nobs = int(nobs_attr)
    except Exception:
        nobs = None

    # Significance at conventional alpha=0.05 (two-sided)
    significant = (pvalue is not None) and (pvalue < 0.05)

    # Plain-language interpretation
    # Note: coef is change in AvgScore per one-unit increase in StudentTeacherRatio (one more student per teacher).
    if coef < 0:
        direction = "lower student-teacher ratio (fewer students per teacher) is associated with higher AvgScore"
    elif coef > 0:
        direction = "lower student-teacher ratio (fewer students per teacher) is associated with lower AvgScore"
    else:
        direction = "no association detected"

    def fmt(x, fmt_spec):
        if x is None:
            return "NA"
        try:
            return format(x, fmt_spec)
        except Exception:
            return str(x)

    fcoef = fmt(coef, ".4f")
    fse = fmt(se, ".4f")
    ftstat = fmt(tstat, ".3f")
    fpvalue = fmt(pvalue, ".3g")
    fci_lo = fmt(ci_lo, ".4f")
    fci_hi = fmt(ci_hi, ".4f")
    fnobs = str(nobs) if nobs is not None else "NA"

    description = (
        f"StudentTeacherRatio coefficient = {fcoef} (SE = {fse}, t = {ftstat}, p = {fpvalue}). "
        f"95% CI = [{fci_lo}, {fci_hi}]. "
        f"Interpretation: a one-unit increase in StudentTeacherRatio (one more student per teacher) is associated "
        f"with a {fcoef}-point change in district AvgScore. Therefore, {direction}. "
        f"Statistically significant at alpha=0.05: {significant}. "
        f"Robust (HC3) standard errors may have been used to compute the SE, t, p-value and CI if the model was fit with robust covariance."
    )

    return {
        "object": {
            "coef": float(coef),
            "std_err": float(se) if se is not None else None,
            "t_value": float(tstat) if tstat is not None else None,
            "p_value": float(pvalue) if pvalue is not None else None,
            "95%_CI": [float(ci_lo) if ci_lo is not None else None, float(ci_hi) if ci_hi is not None else None],
            "nobs": nobs,
            "significant_at_0.05": bool(significant)
        },
        "description": description
    }