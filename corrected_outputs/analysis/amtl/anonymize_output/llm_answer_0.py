def extract_final_answer(model_output):
    """
    Extracts the effect of IsHuman on AMTL from a fitted statsmodels GLMResults-like object
    (or a dict containing 'fit' and/or 'clustered_fit_by_region').

    Returns a dictionary with:
      - "object": a dict with numeric results for the IsHuman coefficient (coef, se, z, p,
                  95% CI on link scale, odds ratio and its 95% CI).
      - "description": a short plain-language interpretation answering whether modern humans
                       have higher AMTL after accounting for age, sex, and tooth class.
    """
    import numpy as np
    import pandas as pd

    # Normalize input: accept either the result object itself or a dict containing fits
    res = None
    if isinstance(model_output, dict):
        # Prefer clustered robust results if present
        if 'clustered_fit_by_region' in model_output and model_output['clustered_fit_by_region'] is not None:
            res = model_output['clustered_fit_by_region']
        elif 'fit' in model_output and model_output['fit'] is not None:
            res = model_output['fit']
        else:
            # Fallback: try to find any result-like object in dict
            for v in model_output.values():
                if hasattr(v, 'params'):
                    res = v
                    break
    else:
        res = model_output

    if res is None:
        return {
            "object": None,
            "description": "No valid model result found in model_output input."
        }

    # Ensure object has expected attributes
    if not hasattr(res, 'params'):
        return {
            "object": None,
            "description": "Provided model output does not look like a statsmodels results object (missing .params)."
        }

    # Attempt to get parameter names/index in a safe way
    try:
        param_index = res.params.index
    except Exception:
        # If params has no index (e.g., plain numpy), try to coerce to pandas Series
        try:
            param_index = pd.Series(res.params).index
        except Exception:
            param_index = []

    # Find the parameter name corresponding to the IsHuman predictor.
    # Patsy/statsmodels may name it exactly "IsHuman" or as a dummy like "IsHuman[T.True]" etc.
    param_candidates = [p for p in param_index if (p == 'IsHuman' or (isinstance(p, str) and p.startswith('IsHuman')))]
    if len(param_candidates) == 0:
        # Try other common variants (e.g., if variable named 'Is_Human' etc.)
        param_candidates = [p for p in param_index if isinstance(p, str) and ('IsHuman' in p or 'Is_Human' in p)]
    if len(param_candidates) == 0:
        return {
            "object": None,
            "description": "Could not find a parameter corresponding to 'IsHuman' in the model parameters: "
                           f"{list(param_index)}"
        }

    # Choose the first matching parameter name
    param_name = param_candidates[0]

    # Extract statistics
    # Safe access to params (in case params is Series or dict-like)
    try:
        coef = float(res.params[param_name])
    except Exception:
        # Try accessing by position if param_name is an integer index
        try:
            coef = float(pd.Series(res.params).loc[param_name])
        except Exception:
            coef = None

    # statsmodels robust result objects sometimes store bse/pvalues differently; try to access safely
    se = None
    try:
        se = float(res.bse[param_name])
    except Exception:
        try:
            if hasattr(res.bse, 'loc'):
                se = float(res.bse.loc[param_name])
            else:
                se = float(pd.Series(res.bse).loc[param_name])
        except Exception:
            se = None

    # Compute z and p if available, otherwise compute z from coef/se
    p_value = None
    z_value = None
    try:
        if hasattr(res, 'pvalues') and param_name in getattr(res, 'pvalues').index:
            p_value = float(res.pvalues[param_name])
    except Exception:
        # fallback: attempt Series access
        try:
            p_value = float(pd.Series(res.pvalues).loc[param_name])
        except Exception:
            p_value = None

    try:
        if hasattr(res, 'tvalues') and param_name in getattr(res, 'tvalues').index:
            z_value = float(res.tvalues[param_name])
    except Exception:
        try:
            z_value = float(pd.Series(res.tvalues).loc[param_name])
        except Exception:
            z_value = None

    if z_value is None and se is not None and se != 0 and coef is not None:
        z_value = coef / se

    # Confidence intervals: try res.conf_int()
    ci_lower = None
    ci_upper = None
    try:
        ci_df = res.conf_int()
        # conf_int may be DataFrame with numeric column indices 0 and 1
        if param_name in ci_df.index:
            ci_lower = float(ci_df.loc[param_name, 0])
            ci_upper = float(ci_df.loc[param_name, 1])
        else:
            # try positional access
            ci_ser = pd.Series(ci_df).loc[param_name]
            ci_lower = float(ci_ser[0])
            ci_upper = float(ci_ser[1])
    except Exception:
        # If conf_int not available, approximate using coef +/- 1.96*se
        if coef is not None and se is not None:
            ci_lower = coef - 1.96 * se
            ci_upper = coef + 1.96 * se
        else:
            ci_lower = ci_upper = None

    # Convert to odds ratio scale (exp(coef)) and its CI
    try:
        or_coef = float(np.exp(coef)) if coef is not None else None
        or_ci_lower = float(np.exp(ci_lower)) if ci_lower is not None else None
        or_ci_upper = float(np.exp(ci_upper)) if ci_upper is not None else None
    except Exception:
        or_coef = or_ci_lower = or_ci_upper = None

    # Make a brief inference: positive coef -> higher AMTL in humans on log-odds scale.
    # Use p<0.05 if p_value is available.
    if p_value is not None:
        if p_value < 0.05:
            if coef is not None and coef > 0:
                conclusion = "Yes — the IsHuman coefficient is positive and statistically significant (p < 0.05), " \
                             "indicating modern humans have higher AMTL after accounting for age, sex, and tooth class."
            elif coef is not None and coef < 0:
                conclusion = "No — the IsHuman coefficient is negative and statistically significant (p < 0.05), " \
                             "indicating modern humans have lower AMTL after accounting for age, sex, and tooth class."
            else:
                conclusion = "No — the IsHuman coefficient is essentially zero."
        else:
            # Not statistically significant
            if coef is not None and coef > 0:
                conclusion = "No strong evidence — the IsHuman coefficient is positive but not statistically significant (p >= 0.05). " \
                             "We cannot conclude modern humans have higher AMTL after accounting for the covariates."
            elif coef is not None and coef < 0:
                conclusion = "No strong evidence — the IsHuman coefficient is negative but not statistically significant (p >= 0.05). " \
                             "We cannot conclude modern humans have lower AMTL after accounting for the covariates."
            else:
                conclusion = "No — the IsHuman coefficient is essentially zero."
    else:
        # If p-value not available, give cautious interpretation based on coefficient sign
        if coef is not None and coef > 0:
            conclusion = "Coefficient for IsHuman is positive (higher log-odds of AMTL for humans), but p-value is unavailable — interpret cautiously."
        elif coef is not None and coef < 0:
            conclusion = "Coefficient for IsHuman is negative (lower log-odds of AMTL for humans), but p-value is unavailable — interpret cautiously."
        else:
            conclusion = "IsHuman coefficient is zero (no effect found)."

    result_object = {
        "parameter_name": param_name,
        "coef_log_odds": coef,
        "se": se,
        "z_or_t": z_value,
        "p_value": p_value,
        "ci_log_odds_lower": ci_lower,
        "ci_log_odds_upper": ci_upper,
        "odds_ratio": or_coef,
        "or_ci_lower": or_ci_lower,
        "or_ci_upper": or_ci_upper,
        "note": "Model included controls for age, sex_female, and tooth_class as specified."
    }

    # Safe string formatting for description (avoid format errors when values are None)
    se_str = f"{se:.4f}" if se is not None else "NA"
    z_str = f"{z_value:.3f}" if z_value is not None else "NA"
    p_str = f"{p_value:.4f}" if p_value is not None else "NA"
    ci_lower_str = f"{ci_lower:.4f}" if ci_lower is not None else "NA"
    ci_upper_str = f"{ci_upper:.4f}" if ci_upper is not None else "NA"
    or_str = f"{or_coef:.3f}" if or_coef is not None else "NA"
    or_ci_lower_str = f"{or_ci_lower:.3f}" if or_ci_lower is not None else "NA"
    or_ci_upper_str = f"{or_ci_upper:.3f}" if or_ci_upper is not None else "NA"
    coef_str = f"{coef:.4f}" if coef is not None else "NA"

    description = (
        f"Extracted parameter '{param_name}': coef (log-odds) = {coef_str}, SE = {se_str}, "
        f"z/t = {z_str}, p = {p_str}. "
        f"95% CI (log-odds) = [{ci_lower_str}, {ci_upper_str}]. "
        f"Odds ratio = {or_str} (95% CI = [{or_ci_lower_str}, {or_ci_upper_str}]). "
        f"Interpretation: {conclusion} "
        "All estimates are conditional on the model covariates (age, sex_female, tooth_class)."
    )

    return {"object": result_object, "description": description}