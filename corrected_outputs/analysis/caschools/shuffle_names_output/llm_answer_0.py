def extract_final_answer(model_output):
    """
    Extracts coefficient, standard error, p-value, and 95% CI for the
    independent variable 'LogStudentTeacherRatio' from a statsmodels
    results object (including robust-covariance results).
    Returns a dict with:
      - "object": a dict containing numeric results and a short conclusion
      - "description": A plain-language explanation of what the numbers mean
    """
    import numpy as np
    import pandas as pd

    res = model_output
    iv = 'LogStudentTeacherRatio'

    # Ensure results object has the expected attributes
    if not hasattr(res, 'params') or not hasattr(res, 'pvalues'):
        raise ValueError("model_output does not appear to be a fitted statsmodels results object with .params/.pvalues")

    params = res.params
    pvalues = res.pvalues
    bse = getattr(res, 'bse', None)

    # Determine parameter names in a robust way
    param_names = None
    try:
        if hasattr(params, 'index'):
            param_names = list(params.index)
        else:
            # Try model metadata
            if hasattr(res, 'model'):
                model = res.model
                if hasattr(model, 'exog_names'):
                    param_names = list(model.exog_names)
                elif hasattr(model, 'data') and hasattr(model.data, 'param_names'):
                    param_names = list(model.data.param_names)
                elif hasattr(model, 'data') and hasattr(model.data, 'orig_exog_names'):
                    param_names = list(model.data.orig_exog_names)
            # Try pvalues index as fallback
            if param_names is None and hasattr(pvalues, 'index'):
                param_names = list(pvalues.index)
            # Final fallback: create numeric string names if length available
            if param_names is None:
                try:
                    length = len(params)
                    param_names = [str(i) for i in range(length)]
                except Exception:
                    param_names = None
    except Exception:
        param_names = None

    if not param_names:
        raise ValueError("Unable to determine parameter names from the model_output.")

    if iv not in param_names:
        raise KeyError(f"Independent variable '{iv}' not found in model results. Available params: {param_names}")

    iv_pos = param_names.index(iv)

    # Helper to safely extract a scalar from various container types by name or position
    def _extract_scalar(obj, name, pos):
        if obj is None:
            return None
        # Pandas Series/DataFrame with index
        try:
            if hasattr(obj, 'loc') and hasattr(obj, 'index') and name in obj.index:
                val = obj.loc[name]
                # If a DataFrame row returned, try first element
                if isinstance(val, (pd.Series, pd.DataFrame)):
                    val = val.iloc[0]
                return float(val)
        except Exception:
            pass
        # Pandas with iloc
        try:
            if hasattr(obj, 'iloc'):
                val = obj.iloc[pos]
                # If val is Series or array-like, take first element
                if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray, list, tuple)):
                    # If DataFrame row, reduce to scalar
                    if isinstance(val, pd.Series):
                        return float(val.iloc[0])
                    if isinstance(val, (np.ndarray, list, tuple)):
                        return float(val[0])
                return float(val)
        except Exception:
            pass
        # Numpy array or list-like
        try:
            if isinstance(obj, (np.ndarray, list, tuple)):
                return float(obj[pos])
        except Exception:
            pass
        # Dict-like
        try:
            if isinstance(obj, dict):
                if name in obj:
                    return float(obj[name])
                # try positional key
                return float(obj.get(pos))
        except Exception:
            pass
        # Last resort: try index access by name then pos
        try:
            val = obj[name]
            return float(val)
        except Exception:
            pass
        try:
            val = obj[pos]
            return float(val)
        except Exception:
            pass
        return None

    coef = _extract_scalar(params, iv, iv_pos)
    se = _extract_scalar(bse, iv, iv_pos)
    pval = _extract_scalar(pvalues, iv, iv_pos)

    # Attempt to get 95% confidence interval
    ci_lower = ci_upper = None
    try:
        ci = res.conf_int()  # usually returns DataFrame indexed by param names with two columns or ndarray
        if isinstance(ci, np.ndarray):
            ci_lower, ci_upper = float(ci[iv_pos, 0]), float(ci[iv_pos, 1])
        else:
            # DataFrame-like
            if hasattr(ci, 'loc') and iv in list(ci.index):
                cols = list(ci.columns)
                # take the first two columns in order
                ci_lower = float(ci.loc[iv, cols[0]])
                ci_upper = float(ci.loc[iv, cols[1]])
            else:
                # try positional access
                row = None
                try:
                    row = ci.iloc[iv_pos]
                except Exception:
                    row = None
                if row is not None:
                    # row may be Series; take first two elements
                    try:
                        ci_lower = float(row.iloc[0])
                        ci_upper = float(row.iloc[1])
                    except Exception:
                        ci_lower = ci_upper = None
    except Exception:
        ci_lower = ci_upper = None

    # Interpretation
    alpha = 0.05
    significant = (pval is not None and pval < alpha)
    if coef is None:
        conclusion = "Coefficient unavailable; cannot form conclusion."
    else:
        if coef < 0 and significant:
            conclusion = ("Yes — statistically significant: a lower student-teacher ratio (smaller classes) "
                          "is associated with higher district average test scores.")
        elif coef < 0 and not significant:
            conclusion = ("No strong evidence (coefficient negative but not statistically significant): "
                          "point estimate suggests smaller classes are associated with higher scores, "
                          "but the effect is not statistically significant at alpha=0.05.")
        elif coef > 0 and significant:
            conclusion = ("No — statistically significant in the opposite direction: a lower student-teacher ratio "
                          "(smaller classes) is associated with lower district average test scores (counterintuitive).")
        else:
            conclusion = ("No strong evidence (coefficient positive but not statistically significant): "
                          "no clear association between student-teacher ratio and test scores in the sample.")

    # Pack numeric results
    result_object = {
        "variable": iv,
        "coef": float(coef) if coef is not None else None,
        "std_err": float(se) if se is not None else None,
        "p_value": float(pval) if pval is not None else None,
        "ci_lower_95": float(ci_lower) if ci_lower is not None else None,
        "ci_upper_95": float(ci_upper) if ci_upper is not None else None,
        "significant_at_0.05": bool(significant) if pval is not None else None,
        "conclusion": conclusion
    }

    # Human-readable description
    desc_lines = []
    if coef is not None:
        desc_lines.append(f"Extracted coefficient for '{iv}': {coef:.4g}")
    else:
        desc_lines.append(f"Extracted coefficient for '{iv}': unavailable")
    desc_lines.append((f"Standard error: {se:.4g}" if se is not None else "Standard error: unavailable"))
    desc_lines.append((f"p-value: {pval:.4g}" if pval is not None else "p-value: unavailable"))
    if ci_lower is not None and ci_upper is not None:
        desc_lines.append(f"95% CI: [{ci_lower:.4g}, {ci_upper:.4g}]")
    desc_lines.append(f"Interpretation note: '{iv}' is the natural log of the student-teacher ratio.")
    desc_lines.append(conclusion)
    description = " ".join(desc_lines)

    return {"object": result_object, "description": description}