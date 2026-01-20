def extract_final_answer(model_output):
    """
    Extracts coefficient, SE, t-stat, p-value, and 95% CI for the predictor 'Beauty_z'
    from a statsmodels OLS results object (optionally already adjusted for clustered
    standard errors). Returns a dictionary with an 'object' (numeric results) and a
    short 'description' interpreting the effect on EvalScore (1-5 scale).
    This implementation is robust to model_output attributes being numpy arrays,
    pandas Series/DataFrame, or dict-like.
    """
    import numpy as np

    def to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    # Safe attribute / key getter
    def safe_get(obj, name):
        try:
            return getattr(obj, name)
        except Exception:
            try:
                return obj[name]
            except Exception:
                return None

    # Try to extract parameter names from the model_output or params object
    def get_param_names(params_obj, model_out):
        # If params_obj has an index (e.g., pandas Series)
        try:
            idx = getattr(params_obj, "index", None)
            if idx is not None:
                return list(idx)
        except Exception:
            pass

        # Try common locations on model object
        try:
            model = getattr(model_out, "model", None)
            if model is not None:
                # common attribute names in statsmodels
                for attr in ("exog_names", "data"):
                    if hasattr(model, attr):
                        val = getattr(model, attr)
                        # model.exog_names is often a list
                        if attr == "exog_names" and isinstance(val, (list, tuple)):
                            return list(val)
                        # model.data.param_names or model.data.orig_exog_names
                        if attr == "data":
                            for subattr in ("param_names", "orig_exog_names", "param_names"):
                                if hasattr(val, subattr):
                                    names = getattr(val, subattr)
                                    if isinstance(names, (list, tuple)):
                                        return list(names)
                                    # sometimes an Index/array-like
                                    try:
                                        return list(names)
                                    except Exception:
                                        pass
        except Exception:
            pass

        # As a last resort, try keys() if params_obj is dict-like
        try:
            if hasattr(params_obj, "keys"):
                return list(params_obj.keys())
        except Exception:
            pass

        return None

    # Generic extractor: given an object (params, bse, etc.) and a target name,
    # return the numeric value if possible.
    def extract_by_name(obj, name, params_obj, model_out):
        if obj is None:
            return None
        # If it's dict-like
        try:
            if hasattr(obj, "get") and name in obj:
                return to_float(obj.get(name))
        except Exception:
            pass
        # If it's pandas-like with index
        try:
            idx = getattr(obj, "index", None)
            if idx is not None and name in idx:
                return to_float(obj[name])
        except Exception:
            pass
        # If it's numpy/sequence-like, try to map parameter names to index
        try:
            if isinstance(obj, (list, tuple, np.ndarray)):
                names = get_param_names(params_obj, model_out)
                if names is not None and name in names:
                    idx = names.index(name)
                    return to_float(obj[idx])
                # fallback: if obj has same length as names, try last resort
        except Exception:
            pass
        # If it's pandas DataFrame column/row-like
        try:
            # try .loc[name] if supported
            if hasattr(obj, "loc"):
                try:
                    v = obj.loc[name]
                    return to_float(v) if np.isscalar(v) else to_float(v.values.tolist())
                except Exception:
                    pass
        except Exception:
            pass
        return None

    # Fetch raw objects
    params = safe_get(model_output, "params")
    bse = safe_get(model_output, "bse")
    tvalues = safe_get(model_output, "tvalues")
    pvalues = safe_get(model_output, "pvalues")

    # Determine parameter names robustly
    param_names = get_param_names(params, model_output)

    # Ensure 'Beauty_z' exists among parameter names (if available)
    if param_names is not None:
        if "Beauty_z" not in param_names:
            raise ValueError("The provided model_output does not contain parameter estimates for 'Beauty_z'.")
    else:
        # If we don't know names but params exists and is dict/Series/array, try to extract directly;
        # if we cannot find any way to reference 'Beauty_z', raise an error.
        # Use a conservative approach: if params is numpy array we cannot map without names.
        if params is None:
            raise ValueError("The provided model_output does not contain parameter estimates for 'Beauty_z'.")
        # if params is dict-like, check membership
        try:
            if hasattr(params, "keys") and "Beauty_z" not in params.keys():
                raise ValueError("The provided model_output does not contain parameter estimates for 'Beauty_z'.")
        except Exception:
            # unknown structure; attempt extraction and if fails later we'll raise
            pass

    # Extract coefficient and statistics
    coef = extract_by_name(params, "Beauty_z", params, model_output)
    se = extract_by_name(bse, "Beauty_z", params, model_output)
    t = extract_by_name(tvalues, "Beauty_z", params, model_output)
    p = extract_by_name(pvalues, "Beauty_z", params, model_output)

    # Extract/confidence interval handling
    conf_int = None
    try:
        ci = None
        # try method conf_int if available
        if hasattr(model_output, "conf_int"):
            try:
                ci = model_output.conf_int()
            except Exception:
                ci = None
        if ci is not None:
            # ci may be DataFrame-like or ndarray
            # If DataFrame-like and has .loc
            try:
                if hasattr(ci, "loc") and "Beauty_z" in getattr(ci, "index", []):
                    row = ci.loc["Beauty_z"]
                    conf_int = (to_float(row.iloc[0]) if hasattr(row, "iloc") else to_float(row[0]),
                                to_float(row.iloc[1]) if hasattr(row, "iloc") else to_float(row[1]))
                else:
                    # try names mapping
                    names = get_param_names(params, model_output)
                    if names is not None and "Beauty_z" in names:
                        idx = names.index("Beauty_z")
                        # ci may be ndarray-like
                        if isinstance(ci, (list, tuple, np.ndarray)):
                            row = ci[idx]
                            conf_int = (to_float(row[0]), to_float(row[1]))
                        else:
                            # DataFrame without index containing name; try iloc
                            try:
                                row = ci.iloc[idx]
                                conf_int = (to_float(row.iloc[0]), to_float(row.iloc[1]))
                            except Exception:
                                pass
            except Exception:
                conf_int = None
    except Exception:
        conf_int = None

    # If CI missing but se and coef present, approximate 95% CI using normal approx
    if conf_int is None and se is not None and coef is not None:
        crit = 1.96
        conf_int = (coef - crit * se, coef + crit * se)

    # If coefficient is None at this point, raise error
    if coef is None:
        raise ValueError("Could not extract coefficient for 'Beauty_z' from model_output.")

    interpretation = {
        "coefficient": to_float(coef),
        "std_error": to_float(se),
        "t_stat": to_float(t),
        "p_value": to_float(p),
        "95%_CI": (to_float(conf_int[0]), to_float(conf_int[1])) if conf_int is not None else None,
        "significant_at_0.05": (p is not None and to_float(p) < 0.05),
        "notes": (
            "Coefficient is the change in course evaluation score (1-5 scale) associated "
            "with a one standard-deviation increase in instructor attractiveness (Beauty_z). "
            "Standard errors/p-values reflect the model's reported covariance (clustered at instructor level "
            "if the model was created with get_robustcov_results)."
        )
    }

    # Plain-language description
    sig_text = "statistically significant" if interpretation["significant_at_0.05"] else "not statistically significant"
    se_val = interpretation["std_error"]
    t_val = interpretation["t_stat"]
    p_val = interpretation["p_value"]
    ci_val = interpretation["95%_CI"]

    desc = f"Beauty_z coefficient = {interpretation['coefficient']:.4f}; SE = {se_val:.4f}" if se_val is not None else f"Beauty_z coefficient = {interpretation['coefficient']:.4f}; SE = NA"
    if p_val is not None:
        desc += f"; t = {t_val:.3f}; p = {p_val:.4f}."
    else:
        desc += "."
    if ci_val is not None:
        desc += f" 95% CI = [{ci_val[0]:.4f}, {ci_val[1]:.4f}]."
    desc += (
        f" Interpretation: A one standard-deviation increase in instructor attractiveness is associated with a "
        f"{interpretation['coefficient']:.3f}-point change in the average teaching evaluation (on a 1-5 scale). "
        f"This effect is {sig_text} at the 5% level."
    )

    return {"object": interpretation, "description": desc}