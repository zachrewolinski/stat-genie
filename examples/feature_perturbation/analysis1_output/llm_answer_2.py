def extract_final_answer(model_output):
    """
    Extract final statistics (typically model coefficients) from a model output object.

    Returns a dictionary with keys:
      - "object": the extracted object (converted into Python-native types where possible)
      - "description": brief explanation of what was extracted and its meaning

    The function is defensive: it tries several common attribute/key names for parameters
    and other statistics, handles both attribute access and dict-like access, and converts
    pandas/numpy-like objects into plain Python types where possible.
    """
    from collections.abc import Mapping, Sequence

    # Helper: try to get attribute/key from an object
    def _get_attr(obj, name):
        if obj is None:
            return None
        # dict-like access
        try:
            if isinstance(obj, Mapping) and name in obj:
                return obj[name]
        except Exception:
            pass
        # attribute access
        try:
            return getattr(obj, name)
        except Exception:
            pass
        # numeric/index access for sequences with string keys (rare)
        try:
            return obj[name]
        except Exception:
            return None

    # Helper: recursively convert common container types (pandas, numpy, lists, tuples, dicts)
    def _to_py(obj):
        # Avoid treating strings as sequences
        if obj is None or isinstance(obj, (str, bytes, bool)):
            return obj
        # If it's a mapping (dict-like), convert keys and values
        if isinstance(obj, Mapping):
            return {str(k): _to_py(v) for k, v in obj.items()}
        # If it's a sequence (list/tuple) but not a string, convert items
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            try:
                return [_to_py(x) for x in obj]
            except Exception:
                # fallthrough to other checks
                pass
        # Duck-typed conversions for numpy/pandas/etc. without importing them explicitly
        # If object has a to_dict method, use it
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return _to_py(obj.to_dict())
            except Exception:
                pass
        # If object has a tolist method (numpy arrays, pandas Index), use it
        if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
            try:
                return _to_py(obj.tolist())
            except Exception:
                pass
        # If object has an item() method and it's a 0-d array/np scalar, get Python scalar
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                scalar = obj.item()
                # item() may still return numpy types; try to convert further
                return _to_py(scalar)
            except Exception:
                pass
        # If it's a numeric scalar (int/float), return as-is
        if isinstance(obj, (int, float, complex)):
            return obj
        # Fallback: try to stringify, but prefer the raw object if it's already simple
        try:
            return obj
        except Exception:
            return str(obj)

    # If model_output is a sequence (tuple/list) commonly models might be wrapped, pick a candidate
    res = model_output
    if isinstance(res, (list, tuple)):
        # prefer first non-string element
        chosen = None
        for cand in res:
            if cand is None:
                continue
            if isinstance(cand, (str, bytes)):
                continue
            chosen = cand
            break
        if chosen is not None:
            res = chosen
        elif len(res) > 0:
            res = res[0]

    # Candidate attribute/key names for coefficients/parameters and p-values etc.
    params_names = ["params", "coef", "coefs", "betas", "coefficients", "parameters", "estimates", "estimate"]
    pvalue_names = ["pvalues", "pvals", "p_value", "p_values", "pvalue", "p"]
    se_names = ["bse", "std_err", "stderr", "se", "std_errors", "std_error"]
    tstat_names = ["tvalues", "tstats", "t_stat", "t_statistic", "t"]

    def _find_first(obj, names):
        for n in names:
            val = _get_attr(obj, n)
            if val is not None:
                return val, n
        return None, None

    params_val, params_key = _find_first(res, params_names)
    pvalues_val, pvalues_key = _find_first(res, pvalue_names)
    stderr_val, stderr_key = _find_first(res, se_names)
    tstat_val, tstat_key = _find_first(res, tstat_names)

    # If no params found, try checking top-level keys if model_output itself is mapping
    if params_val is None and isinstance(res, Mapping):
        # pick first mapping item that looks numeric-like
        for k, v in res.items():
            # skip metadata keys
            if k.lower() in ("model", "data", "nobs", "summary"):
                continue
            if v is not None:
                params_val = v
                params_key = k
                break

    # Prepare the object to return: canonical mapping of available stats
    extracted = {}
    if params_val is not None:
        extracted["params_key"] = params_key
        extracted["params"] = _to_py(params_val)
    if pvalues_val is not None:
        extracted["pvalues_key"] = pvalues_key
        extracted["pvalues"] = _to_py(pvalues_val)
    if stderr_val is not None:
        extracted["stderr_key"] = stderr_key
        extracted["stderr"] = _to_py(stderr_val)
    if tstat_val is not None:
        extracted["tstat_key"] = tstat_key
        extracted["tstat"] = _to_py(tstat_val)

    # If nothing was extracted, attempt to return the whole object converted
    if not extracted:
        extracted = {"raw": _to_py(res)}

    # Build a human-readable description
    desc_parts = []
    if "params" in extracted:
        desc_parts.append(
            f"Extracted model parameters from key '{extracted.get('params_key', params_key)}'. "
            "These represent estimated coefficients for the model predictors."
        )
    if "pvalues" in extracted:
        desc_parts.append(
            f"Extracted p-values from key '{extracted.get('pvalues_key', pvalues_key)}'. "
            "P-values indicate the statistical significance of the corresponding coefficients."
        )
    if "stderr" in extracted:
        desc_parts.append(
            f"Extracted standard errors from key '{extracted.get('stderr_key', stderr_key)}'. "
            "Standard errors measure the uncertainty of the coefficient estimates."
        )
    if "tstat" in extracted:
        desc_parts.append(
            f"Extracted t-statistics from key '{extracted.get('tstat_key', tstat_key)}'. "
            "T-statistics are the coefficients divided by their standard errors."
        )
    if not desc_parts:
        desc = "Returned the raw model output converted to Python-native types."
    else:
        desc = " ".join(desc_parts)

    return {"object": extracted, "description": desc}