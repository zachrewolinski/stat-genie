def extract_final_answer(model_output):
    """
    Inspect the model_output dict returned by the modeling function and extract statistics
    about the effect of hurricane name femininity on fatalities.

    Returns dict with keys:
      - "object": a dict with extracted numeric results (or None if nothing extractable)
      - "description": a short human-readable interpretation of the result in context
    """
    import numpy as np
    import pandas as pd
    import math

    # Defensive checks
    if not model_output or not isinstance(model_output, dict) or len(model_output) == 0:
        return {
            "object": None,
            "description": "No models were fit (model_output is empty). Cannot extract statistics."
        }

    # Preferred extraction order: continuous MasFem_z, binary NameGender, MTurkMasFem_z
    preferred = [
        ("MasFem_z", "masfem_nb"),
        ("NameGender", "genderbin_nb"),
        ("MTurkMasFem_z", "mturk_masfem_nb")
    ]

    # Find a model key in model_output matching a preferred specification (allow variants like *_poisson_robust)
    chosen_key = None
    chosen_var = None
    for var_name, base_key in preferred:
        for k in model_output.keys():
            # match exact or startswith base_key (to catch *_poisson_robust fallback names)
            if k == base_key or k.startswith(base_key) or (base_key in k):
                chosen_key = k
                chosen_var = var_name
                break
        if chosen_key is not None:
            break

    if chosen_key is None:
        # No preferred models found; try to pick any model and a variable that looks like MasFem or NameGender
        for k, res in model_output.items():
            # try to detect variable names in the result.params index if available
            try:
                params_index = list(res.params.index)
            except Exception:
                params_index = []
            fallback_var = None
            for candidate in ["MasFem_z", "NameGender", "MTurkMasFem_z", "MasFem"]:
                if candidate in params_index:
                    fallback_var = candidate
                    break
            if fallback_var is not None:
                chosen_key = k
                chosen_var = fallback_var
                break

    if chosen_key is None:
        return {
            "object": None,
            "description": "No suitable fitted model found in model_output; cannot extract femininity effect."
        }

    res = model_output[chosen_key]

    # Helper to safely extract numeric results from a statsmodels-like results object
    def safe_get_param(results, name):
        try:
            # params may be a Series with index
            val = results.params[name]
        except Exception:
            try:
                # maybe params is dict-like
                val = results.params.get(name, None)
            except Exception:
                val = None
        return val

    def safe_get_pvalue(results, name):
        try:
            return results.pvalues[name]
        except Exception:
            try:
                return results.pvalues.get(name, None)
            except Exception:
                return None

    def safe_get_confint(results, name):
        try:
            ci = results.conf_int()
        except Exception:
            return (None, None)
        # ci might be a DataFrame with index names or a numpy array in parameter order
        try:
            if isinstance(ci, (pd.DataFrame, pd.Series)):
                row = ci.loc[name]
                return float(row.iloc[0]), float(row.iloc[1])
            else:
                # assume ndarray; need index of parameter
                idx = list(results.params.index).index(name)
                row = ci[idx]
                return float(row[0]), float(row[1])
        except Exception:
            # fallback: try to match by order if only one parameter
            try:
                if ci.shape[0] == 1:
                    return float(ci[0, 0]), float(ci[0, 1])
            except Exception:
                pass
        return (None, None)

    def safe_get_nobs(results):
        try:
            return int(results.nobs)
        except Exception:
            try:
                return int(results.model.endog.shape[0])
            except Exception:
                try:
                    return int(len(results.model.endog))
                except Exception:
                    return None

    coef = safe_get_param(res, chosen_var)
    pval = safe_get_pvalue(res, chosen_var)
    ci_low, ci_high = safe_get_confint(res, chosen_var)
    nobs = safe_get_nobs(res)

    # If coef is missing, cannot proceed
    if coef is None:
        return {
            "object": None,
            "description": f"Model '{chosen_key}' found but coefficient for variable '{chosen_var}' not present in the fitted result."
        }

    # For count models with log link (NegativeBinomial/Poisson), exponentiate interpretation
    try:
        exp_coef = float(np.exp(coef))
    except Exception:
        exp_coef = None
    try:
        exp_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        exp_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
    except Exception:
        exp_ci_low = exp_ci_high = None

    # Formulate conclusion: does the sign and significance support the hypothesis that more feminine names -> fewer fatalities?
    supports_hypothesis = None
    conclusion_text = ""
    try:
        if pval is not None:
            if coef < 0 and pval < 0.05:
                supports_hypothesis = True
                conclusion_text = ("Result: coefficient is negative and statistically significant (p < 0.05), "
                                   "consistent with the hypothesis that more feminine names are associated with fewer fatalities.")
            elif coef < 0 and pval >= 0.05:
                supports_hypothesis = False
                conclusion_text = ("Coefficient is negative but not statistically significant (p >= 0.05); "
                                   "no strong evidence to conclude a relationship.")
            elif coef > 0 and pval < 0.05:
                supports_hypothesis = False
                conclusion_text = ("Coefficient is positive and statistically significant (p < 0.05), "
                                   "which contradicts the hypothesis (more feminine names associated with MORE fatalities).")
            else:
                supports_hypothesis = False
                conclusion_text = ("Coefficient is positive and not statistically significant (p >= 0.05); "
                                   "no evidence supporting the hypothesis.")
        else:
            # p-value missing: rely on sign only (very weak)
            supports_hypothesis = None
            conclusion_text = ("Could not obtain a p-value. Coefficient sign is {}. "
                               "Cannot make a confidence-based conclusion.").format("negative" if coef < 0 else "positive")
    except Exception:
        supports_hypothesis = None
        conclusion_text = "Unable to form a conclusion due to missing or malformed statistics."

    # Build the object to return (numbers for programmatic inspection)
    result_object = {
        "model_key": chosen_key,
        "variable": chosen_var,
        "coef": float(coef) if coef is not None else None,
        "p_value": float(pval) if pval is not None else None,
        "ci_lower": float(ci_low) if ci_low is not None else None,
        "ci_upper": float(ci_high) if ci_high is not None else None,
        "exp_coef": float(exp_coef) if exp_coef is not None else None,
        "exp_ci_lower": float(exp_ci_low) if exp_ci_low is not None else None,
        "exp_ci_upper": float(exp_ci_high) if exp_ci_high is not None else None,
        "n_obs": int(nobs) if nobs is not None else None,
        "supports_hypothesis": supports_hypothesis
    }

    # Compose a succinct description for humans
    human_desc = (
        f"Extracted from model '{chosen_key}' for variable '{chosen_var}': coef = {result_object['coef']:.4g}, "
        f"p = {result_object['p_value']:.4g} (if available), 95% CI for coef = "
        f"[{result_object['ci_lower']:.4g}, {result_object['ci_upper']:.4g}] (if available). "
    )
    if result_object['exp_coef'] is not None:
        human_desc += (
            f"Exponentiated coef (multiplicative effect on expected fatalities) = {result_object['exp_coef']:.4g}, "
            f"95% CI = [{result_object['exp_ci_lower']:.4g}, {result_object['exp_ci_upper']:.4g}]. "
        )
    if nobs is not None:
        human_desc += f"Sample size (n) = {nobs}. "

    human_desc += conclusion_text

    return {"object": result_object, "description": human_desc}