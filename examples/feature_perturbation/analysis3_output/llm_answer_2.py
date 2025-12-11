def extract_final_answer(model_output):
    """
    Extracts the effect of the independent variable ("name" / "name_z") on the dependent variable
    from the provided model_output dict. Handles three possible outputs:
      - 'nb_model' (statsmodels GLM NegativeBinomial result)
      - 'ols_model' (statsmodels OLS result)
      - 'ols_name_z' (statsmodels OLS result using standardized IV)
    If any of these keys contains an error string (e.g., '*_error'), the function reports that error.

    Returns:
      {
        "object": <dict or None>   # extracted statistics or error info
        "description": <str>       # brief interpretation in context
      }
    """
    import numpy as np

    # Helper to extract stats from a fitted statsmodels result
    def extract_from_result(res, varname):
        out = {}
        try:
            params = res.params
            pvals = res.pvalues
            conf = res.conf_int()
            coef = float(params[varname]) if varname in params.index else None
            pval = float(pvals[varname]) if varname in pvals.index else None
            ci_lower, ci_upper = (None, None)
            if varname in conf.index:
                ci_lower, ci_upper = float(conf.loc[varname, 0]), float(conf.loc[varname, 1])
            out.update({
                "coef": coef,
                "pvalue": pval,
                "conf_int": (ci_lower, ci_upper)
            })
        except Exception as e:
            out["error"] = f"Failed extracting from result: {e}"
        return out

    # If there are explicit error messages, collect and return them
    error_keys = [k for k in model_output.keys() if k.endswith("_error")]
    if error_keys:
        errors = {k: model_output[k] for k in error_keys}
        desc = (
            "No models were successfully fitted. The modeling step returned error messages "
            "indicating a zero-size dataset (e.g., 'zero-size array to reduction operation maximum which has no identity'). "
            "This means there were no observations available for estimation after the preprocessing/filters, "
            "so no coefficient, p-value, or confidence interval can be produced. "
            "Errors: " + "; ".join(f"{k}: {errors[k]}" for k in errors)
        )
        return {"object": errors, "description": desc}

    # Otherwise, try to extract from any available fitted models
    extracted = {}
    # Negative binomial (interpret coefficient as log count change; exp(coef)=incidence rate ratio)
    if "nb_model" in model_output:
        nb = model_output["nb_model"]
        # try 'name' first, fallback to 'name_z'
        target = "name" if "name" in getattr(nb, "params", {}).index else ("name_z" if "name_z" in getattr(nb, "params", {}).index else None)
        if target:
            stats = extract_from_result(nb, target)
            if "coef" in stats and stats.get("coef") is not None:
                stats["irr"] = np.exp(stats["coef"])  # incidence rate ratio
                if stats.get("conf_int") != (None, None):
                    stats["irr_conf_int"] = (np.exp(stats["conf_int"][0]) if stats["conf_int"][0] is not None else None,
                                             np.exp(stats["conf_int"][1]) if stats["conf_int"][1] is not None else None)
            extracted["nb_model"] = {"variable": target, "stats": stats}
        else:
            extracted["nb_model"] = {"error": "fitted nb_model present but target variable ('name'/'name_z') not found in params"}

    # OLS on log(deaths + 1)
    if "ols_model" in model_output:
        ols = model_output["ols_model"]
        target = "name" if "name" in getattr(ols, "params", {}).index else ("name_z" if "name_z" in getattr(ols, "params", {}).index else None)
        if target:
            stats = extract_from_result(ols, target)
            extracted["ols_model"] = {"variable": target, "stats": stats}
        else:
            extracted["ols_model"] = {"error": "fitted ols_model present but target variable ('name'/'name_z') not found in params"}

    # OLS with name_z specifically
    if "ols_name_z" in model_output:
        ols_z = model_output["ols_name_z"]
        target = "name_z"
        if hasattr(ols_z, "params") and target in getattr(ols_z, "params", {}).index:
            stats = extract_from_result(ols_z, target)
            extracted["ols_name_z"] = {"variable": target, "stats": stats}
        else:
            extracted["ols_name_z"] = {"error": "fitted ols_name_z present but 'name_z' not found in params"}

    if not extracted:
        # No error keys and no recognized fitted models found
        desc = (
            "Model output contains neither fitted model objects nor explicit error messages. "
            "No statistics could be extracted. Please provide the fitted statsmodels results or error messages."
        )
        return {"object": None, "description": desc}

    # Build a short interpretation when coefficient(s) present
    interpretations = []
    for mname, info in extracted.items():
        if "stats" in info and info["stats"].get("coef") is not None:
            coef = info["stats"]["coef"]
            pval = info["stats"].get("pvalue")
            var = info["variable"]
            if mname == "nb_model":
                irr = info["stats"].get("irr")
                interpretations.append(
                    f"{mname}: {var} coef={coef:.4g}, IRR={irr:.4g}, p={pval:.4g} (positive coef => higher name score => higher death counts)"
                )
            else:
                interpretations.append(
                    f"{mname}: {var} coef={coef:.4g}, p={pval:.4g} (dependent variable is log(ndam15+1); positive coef => higher name score => higher log-deaths)"
                )
        else:
            interpretations.append(f"{mname}: {info.get('error', 'no usable statistics available')}")

    description = (
        "Extracted statistics for the independent variable where available. "
        "Interpretation notes: a positive coefficient means that a more feminine name (higher 'name' or 'name_z') "
        "is associated with higher deaths (contrary to the hypothesis that feminine names are perceived as less threatening). "
        "A negative coefficient would support the hypothesis. "
        "Details: " + " | ".join(interpretations)
    )

    return {"object": extracted, "description": description}