def extract_final_answer(model_output):
    """
    Attempt to extract a single, sensible "final answer" (coefficient, p-value, estimate, etc.)
    from a variety of possible model_output shapes.

    The function returns a dictionary with two keys:
      - "object": the extracted value (could be a scalar, nested object, or the entire model_output)
      - "description": short explanation of what was extracted and why

    The function handles:
      - primitives (str, int, float, bool)
      - dicts with common keys like 'final_answer', 'estimate', 'coef', 'p_value', 'result', 'answer'
      - nested dicts under keys like 'summary', 'results', 'output'
      - objects with attributes of the same common names
      - fallback heuristics (first scalar-like value in a dict, vars() for objects)
      - last resort: string representation of model_output
    """
    import numbers

    # Helper: create the return dict
    def _ret(obj, desc):
        return {"object": obj, "description": desc}

    # Null
    if model_output is None:
        return _ret(None, "model_output is None.")

    # Primitive values (string/number/bool) -> return as-is
    if isinstance(model_output, (str, numbers.Number, bool)):
        return _ret(model_output, "model_output is a primitive value; returned directly.")

    # Common candidate keys/attributes to look for
    candidate_keys = [
        "final_answer", "final", "answer", "result", "estimate",
        "estimate_value", "coef", "coeff", "coefficient", "coefficients",
        "p_value", "pval", "p-value", "p", "t", "statistic", "value"
    ]
    nested_containers = ["summary", "results", "output", "details"]

    # If it's a dict, try to find a sensible value
    if isinstance(model_output, dict):
        # direct candidate keys
        for key in candidate_keys:
            if key in model_output:
                return _ret(
                    model_output[key],
                    f"Extracted value from model_output['{key}']."
                )

        # check nested containers
        for cont in nested_containers:
            if cont in model_output and isinstance(model_output[cont], dict):
                for key in candidate_keys:
                    if key in model_output[cont]:
                        return _ret(
                            model_output[cont][key],
                            f"Extracted value from model_output['{cont}']['{key}']."
                        )

        # fallback: return the first scalar-like entry in the dict
        for k, v in model_output.items():
            if isinstance(v, (str, numbers.Number, bool)):
                return _ret(
                    v,
                    f"Extracted first scalar-like value from model_output['{k}'] as a fallback."
                )

        # fallback: if dict has a single entry, return its value
        if len(model_output) == 1:
            only_key = next(iter(model_output))
            return _ret(
                model_output[only_key],
                f"Returned the only entry model_output['{only_key}'] because the dict had a single key."
            )

        # last resort for dicts: return entire dict
        return _ret(
            model_output,
            "Could not find a specific scalar/statistic key; returned the entire dict."
        )

    # If it's an object with attributes, inspect attributes
    for attr in candidate_keys + nested_containers:
        if hasattr(model_output, attr):
            val = getattr(model_output, attr)
            return _ret(
                val,
                f"Extracted attribute '{attr}' from model_output."
            )

    # If object exposes a __dict__/vars(), try that (recursively)
    try:
        obj_vars = vars(model_output)
    except TypeError:
        obj_vars = None

    if isinstance(obj_vars, dict):
        # Reuse the dict handling by calling this function recursively
        return extract_final_answer(obj_vars)

    # Final fallback: string representation
    try:
        string_repr = str(model_output)
    except Exception:
        string_repr = "<unrepresentable object>"

    return _ret(
        string_repr,
        "Could not identify a specific statistic; returned string representation of model_output as a last resort."
    )