import numpy as np

def extract_final_answer(model_output):
    """
    Extract a sensible final answer from a fitted model-like object.

    The function attempts to locate a parameter related to "Femininity" (the original
    intended variable) but falls back to searching for related terms if the exact
    name is not present. It returns a dictionary with:
      - "object": a dict with keys "term", "coef", "stderr", "pvalue" (or None if not available)
      - "description": a short human-readable interpretation of the returned statistic

    The function is defensive: it handles statsmodels-like result objects (with .params,
    .pvalues, .bse), plain dict/Mapping of coefficients, and returns a clear message if
    no appropriate variable is found.
    """
    # Helper: extract mapping-like params, pvalues, bse from model_output
    def _to_dict_like(attr):
        if attr is None:
            return {}
        # If pandas Series or similar mapping
        try:
            # Try to iterate like mapping from keys to values
            return {str(k): v for k, v in attr.items()}
        except Exception:
            pass
        # If it's an ndarray or list with named index not available, give up
        return {}

    # Attempt to read attributes in several common forms
    params = {}
    pvalues = {}
    bse = {}

    # If the model_output itself is a mapping of coefficients
    if isinstance(model_output, dict):
        params = {str(k): v for k, v in model_output.items()}
    else:
        # Try attributes commonly found on statsmodels regression results
        params = _to_dict_like(getattr(model_output, "params", None))
        pvalues = _to_dict_like(getattr(model_output, "pvalues", None))
        bse = _to_dict_like(getattr(model_output, "bse", None))

        # Some objects might store these as pandas Series accessible via .params (Series)
        # _to_dict_like above will handle Series.

    # Ensure keys are strings and create lowercase mapping for searching
    param_names = list(params.keys())
    if not param_names:
        description = (
            "No parameters could be extracted from the provided model_output. "
            "Ensure model_output is a fitted model result (e.g., statsmodels) or a dict of coefficients."
        )
        return {"object": None, "description": description}

    lower_map = {name.lower(): name for name in param_names}

    # Candidate variable names in order of preference
    candidates = [
        "Femininity_z", "femininity_z", "Femininity", "femininity",
        "masfem_mturk", "masfem", "gender_mf", "masculinity", "masculinity_z"
    ]

    found_name = None

    # 1) Exact (case-insensitive) match with candidates
    for cand in candidates:
        if cand.lower() in lower_map:
            found_name = lower_map[cand.lower()]
            break

    # 2) Substring match: look for any param name that contains candidate as substring
    if found_name is None:
        for cand in candidates:
            for pname in param_names:
                if cand.lower() in pname.lower():
                    found_name = pname
                    break
            if found_name is not None:
                break

    # 3) Generic fallback: look for anything with 'fem', 'femin', or 'mas' in its name
    if found_name is None:
        for substr in ("fem", "femin", "mas"):
            for pname in param_names:
                if substr in pname.lower():
                    found_name = pname
                    break
            if found_name is not None:
                break

    # 4) If still nothing, report available parameters and return None
    if found_name is None:
        description = (
            "Could not find a variable related to 'Femininity' or similar in the fitted model. "
            f"Available parameters: {param_names}"
        )
        return {"object": None, "description": description}

    # Extract numeric values safely, converting numpy types to Python floats where possible
    def _safe_float(mapping, key):
        try:
            v = mapping.get(key, None)
            if v is None:
                return None
            # convert numpy scalar to python float
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            # If pandas / numpy scalar
            try:
                return float(v)
            except Exception:
                return None
        except Exception:
            return None

    coef = _safe_float(params, found_name)
    stderr = _safe_float(bse, found_name)
    pval = _safe_float(pvalues, found_name)

    # Build the object to return
    result_object = {
        "term": found_name,
        "coef": coef,
        "stderr": stderr,
        "pvalue": pval
    }

    # Compose a concise description
    if coef is None:
        description = f"Found parameter '{found_name}', but its numeric value could not be retrieved."
    else:
        parts = [f"Estimated coefficient for '{found_name}' = {coef:.6g}"]
        if stderr is not None:
            parts.append(f"(SE = {stderr:.6g})")
        if pval is not None:
            parts.append(f"(p = {pval:.6g})")
        # Add brief interpretation
        interpretation = (
            "A positive coefficient indicates that higher values of this predictor are associated "
            "with higher values of the outcome, while a negative coefficient indicates the opposite."
        )
        description = " ".join(parts) + ". " + interpretation

    return {"object": result_object, "description": description}