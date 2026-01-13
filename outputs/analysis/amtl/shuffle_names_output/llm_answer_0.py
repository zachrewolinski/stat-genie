def extract_final_answer(model_output):
    """
    Extract statistics testing whether modern humans (Homo sapiens) have different
    antemortem tooth loss (AMTL) than non-human primates, controlling for age,
    sex, and tooth class.

    Expects model_output to be the dict returned by the model() function:
      {
        'full_genus_model': GLMResultsWrapper,
        'human_contrast_model': GLMResultsWrapper
      }

    Returns a dict with:
      - "object": dict of extracted numeric results (coef, se, z, p, conf int, OR, OR CI, n)
      - "description": short interpretation of the result in the context of the task
    """
    import numpy as np

    # Try to find the human-contrast model first (it directly includes IsHuman)
    res = None
    if isinstance(model_output, dict):
        # prefer explicit human model if present
        if 'human_contrast_model' in model_output:
            res = model_output['human_contrast_model']
        elif 'full_genus_model' in model_output:
            # fallback to full model if human_contrast_model missing
            res = model_output['full_genus_model']
    else:
        # If a raw results object was passed accidentally, treat it as the result
        res = model_output

    if res is None:
        return {
            "object": None,
            "description": "No model results found in model_output. Expected keys 'human_contrast_model' or 'full_genus_model'."
        }

    # Determine which parameter corresponds to the human effect.
    params_index = list(res.params.index)

    # Preferred parameter name from the constructed contrast model
    param_name = None
    if 'IsHuman' in params_index:
        param_name = 'IsHuman'
    else:
        # try substrings (defensive): look for a Genus.*Homo or Homo.*Genus or any param containing 'Homo'
        for nm in params_index:
            if ('Homo' in nm) or ('sapiens' in nm) or ('Is_Human' in nm) or ('IsHuman' in nm):
                param_name = nm
                break
        # As an additional fallback, look for any parameter that starts with 'Genus' (and then contains Homo)
        if param_name is None:
            for nm in params_index:
                if nm.startswith('Genus') and ('Homo' in nm or 'sapiens' in nm):
                    param_name = nm
                    break

    if param_name is None:
        # Cannot find a direct human parameter: return a helpful message and the available param names
        return {
            "object": {"available_params": params_index},
            "description": ("Could not locate a model parameter corresponding to a human vs non-human contrast. "
                            "Available parameter names are returned in 'object'. If the contrast variable was named "
                            "differently, re-run the model or provide the correct parameter name.")
        }

    # Extract statistics for the chosen parameter
    coef = float(res.params[param_name])
    se = float(res.bse[param_name]) if hasattr(res, 'bse') else None
    # statsmodels provides pvalues; if not, compute z statistic and two-sided p via normal approx
    pval = None
    zstat = None
    if hasattr(res, 'pvalues') and param_name in res.pvalues.index:
        pval = float(res.pvalues[param_name])
        # try to get z from tvalues or zvalues if present
        if hasattr(res, 'tvalues') and param_name in res.tvalues.index:
            zstat = float(res.tvalues[param_name])
        elif hasattr(res, 'zvalues') and param_name in res.zvalues.index:
            zstat = float(res.zvalues[param_name])
        else:
            # approximate z
            zstat = coef / se if se not in (None, 0) else None
    else:
        # fallback: compute z and p from coef and se using normal approx
        if se not in (None, 0):
            zstat = coef / se
            from math import erf, sqrt
            # two-sided p from normal
            pval = float(2 * (1 - 0.5 * (1 + erf(abs(zstat) / sqrt(2)))))
        else:
            pval = None

    # Confidence interval
    try:
        ci = res.conf_int().loc[param_name].values.astype(float)
        ci_lower, ci_upper = float(ci[0]), float(ci[1])
    except Exception:
        ci_lower, ci_upper = None, None

    # Odds ratio and CI on OR scale
    try:
        or_val = float(np.exp(coef))
        or_ci = (float(np.exp(ci_lower)) if ci_lower is not None else None,
                 float(np.exp(ci_upper)) if ci_upper is not None else None)
    except Exception:
        or_val = None
        or_ci = (None, None)

    # Sample size / effective observations
    try:
        n_obs = int(res.nobs)
    except Exception:
        # fallback: try model.endog
        try:
            n_obs = int(res.model.endog.shape[0])
        except Exception:
            n_obs = None

    # Prepare object to return
    result_object = {
        "parameter_name": param_name,
        "coef_log_odds": coef,
        "se": se,
        "z": zstat,
        "p_value": pval,
        "conf_int_log_odds": (ci_lower, ci_upper),
        "odds_ratio": or_val,
        "odds_ratio_conf_int": or_ci,
        "n_obs": n_obs
    }

    # Short interpretation relative to the research question
    if pval is None:
        interpretation = (
            "Could not compute a p-value for the human contrast parameter. Extracted coefficient and CIs are provided "
            "in 'object'."
        )
    else:
        alpha = 0.05
        if pval < alpha:
            if coef > 0:
                interpretation = (
                    f"Yes — the model indicates a statistically significant higher AMTL in modern humans compared to "
                    f"non-human primates after controlling for age, sex, and tooth class (parameter '{param_name}': "
                    f"log-odds = {coef:.4f}, SE = {se:.4f}, z = {zstat:.3f}, p = {pval:.3e}; "
                    f"OR = {or_val:.3f}, 95% CI = [{or_ci[0]:.3f}, {or_ci[1]:.3f}])."
                )
            else:
                interpretation = (
                    f"No — the model indicates a statistically significant lower AMTL in modern humans compared to "
                    f"non-human primates (parameter '{param_name}': log-odds = {coef:.4f}, SE = {se:.4f}, "
                    f"z = {zstat:.3f}, p = {pval:.3e}; OR = {or_val:.3f}, 95% CI = [{or_ci[0]:.3f}, {or_ci[1]:.3f}])."
                )
        else:
            # not statistically significant
            interpretation = (
                f"No — there is no statistically significant difference in AMTL between modern humans and non-human "
                f"primates after controlling for age, sex, and tooth class (parameter '{param_name}': "
                f"log-odds = {coef:.4f}, SE = {se:.4f}, z = {zstat:.3f}, p = {pval:.3e}; "
                f"OR = {or_val:.3f}, 95% CI = [{or_ci[0]:.3f}, {or_ci[1]:.3f}])."
            )

    return {
        "object": result_object,
        "description": interpretation
    }