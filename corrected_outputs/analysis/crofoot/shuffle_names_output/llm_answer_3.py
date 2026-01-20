def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, z-values, p-values, 95% CIs, and odds-ratios
    for the main predictors (relative group size, proximity to focal home) and their interaction
    from the clustered-results-like object returned by the modeling function.

    Returns a dict with keys:
      - "object": dict mapping parameter short-names to statistics (coef, se, z, p, ci, odds_ratio, odds_ci)
      - "description": human-readable interpretation of the extracted statistics
    """
    import numpy as np
    import pandas as pd

    # Helper to safely obtain attributes possibly stored on wrapper or base model
    def _safe_get(attr_name):
        if hasattr(model_output, attr_name):
            return getattr(model_output, attr_name)
        # try underlying base model if available
        if hasattr(model_output, "_base") and hasattr(model_output._base, attr_name):
            return getattr(model_output._base, attr_name)
        return None

    params = _safe_get("params")
    bse = _safe_get("bse")
    pvalues = _safe_get("pvalues")
    conf_int = None
    try:
        if hasattr(model_output, "conf_int"):
            conf_int = model_output.conf_int()
        elif hasattr(model_output, "_base") and hasattr(model_output._base, "conf_int"):
            conf_int = model_output._base.conf_int()
    except Exception:
        conf_int = None

    # Ensure we have pandas Series/DataFrame for consistency
    if params is None:
        raise ValueError("Could not find 'params' on model_output.")
    # If params is numpy array convert to Series if possible (but expecting index names)
    if not isinstance(params, pd.Series):
        try:
            params = pd.Series(params)
        except Exception:
            pass

    index_names = list(params.index)

    # Utility to find parameter name by substring matching
    def find_param(containing_all):
        """Find first parameter name that contains all substrings in containing_all (case-sensitive)."""
        for name in index_names:
            if all(sub in name for sub in containing_all):
                return name
        return None

    # Identify parameter names
    name_rgs = find_param(["RelativeGroupSize_z"]) or find_param(["RelativeGroupSize"])
    name_prox = find_param(["ProximityToFocalHome_z"]) or find_param(["ProximityToFocalHome"])
    # interaction could be ":" or "*" or "RelativeGroupSize_z:ProximityToFocalHome_z"
    name_inter = None
    # Try presence of both substrings
    if name_rgs and name_prox:
        # look for a parameter name that contains both substrings
        for name in index_names:
            if name_rgs in name and name_prox in name:
                name_inter = name
                break
    # fallback: look for ":" containing both keywords
    if name_inter is None:
        for name in index_names:
            if (("RelativeGroupSize" in name and "ProximityToFocalHome" in name)
                or (":" in name and "RelativeGroupSize" in name and "ProximityToFocalHome" in name)):
                name_inter = name
                break

    # Also include intercept and controls for completeness
    name_intercept = find_param(["Intercept"]) or find_param(["const"]) or find_param(["(Intercept)"])
    name_n_focal = find_param(["n_focal"]) or find_param(["n_focal".lower()])
    name_other_males = find_param(["other_males"]) or find_param(["other_males".lower()])

    def extract_stats(param_name):
        if param_name is None:
            return None
        coef = float(params[param_name])
        se = None
        p = None
        z = None
        ci_lower = None
        ci_upper = None
        # bse might be a Series or callable; try to get value
        try:
            if bse is not None:
                se = float(bse[param_name])
        except Exception:
            se = None
        try:
            if pvalues is not None:
                p = float(pvalues[param_name])
        except Exception:
            p = None
        try:
            if se is not None:
                z = coef / se
        except Exception:
            z = None
        try:
            if conf_int is not None and param_name in conf_int.index:
                # conf_int columns might be 0 and 1
                row = conf_int.loc[param_name]
                ci_lower = float(row[0])
                ci_upper = float(row[1])
        except Exception:
            ci_lower = ci_upper = None
        # odds ratio and CI on odds ratio scale
        try:
            odds = float(np.exp(coef))
            odds_ci = (float(np.exp(ci_lower)) if ci_lower is not None else None,
                       float(np.exp(ci_upper)) if ci_upper is not None else None)
        except Exception:
            odds = None
            odds_ci = (None, None)
        return {
            "param_name": param_name,
            "coef": coef,
            "se": se,
            "z": z,
            "p_value": p,
            "ci_95": (ci_lower, ci_upper),
            "odds_ratio": odds,
            "odds_ratio_95_ci": odds_ci
        }

    results = {
        "RelativeGroupSize": extract_stats(name_rgs),
        "ProximityToFocalHome": extract_stats(name_prox),
        "Interaction_RelSize_x_Proximity": extract_stats(name_inter),
        "Intercept": extract_stats(name_intercept),
        "n_focal": extract_stats(name_n_focal),
        "other_males": extract_stats(name_other_males)
    }

    # Build a concise interpretation
    def interp(stat):
        if stat is None:
            return "Parameter not found in model output."
        p = stat.get("p_value")
        coef = stat.get("coef")
        orr = stat.get("odds_ratio")
        sig = None
        if p is None:
            sig = "p-value unavailable"
        elif p < 0.001:
            sig = "highly significant (p < 0.001)"
        elif p < 0.01:
            sig = "very significant (p < 0.01)"
        elif p < 0.05:
            sig = "statistically significant (p < 0.05)"
        else:
            sig = "not statistically significant (p >= 0.05)"
        direction = "positive" if (coef is not None and coef > 0) else ("negative" if (coef is not None and coef < 0) else "null")
        return f"Coef={coef:.3f} (odds-ratio={orr:.3f} if available). Effect is {direction}; {sig}."

    desc_lines = []
    desc_lines.append("Extracted model estimates for predictors influencing focal-group win probability (logistic GLM, cluster-robust SEs):")
    desc_lines.append("- RelativeGroupSize: " + interp(results["RelativeGroupSize"]))
    desc_lines.append("- ProximityToFocalHome: " + interp(results["ProximityToFocalHome"]))
    desc_lines.append("- Interaction (RelativeSize x Proximity): " + interp(results["Interaction_RelSize_x_Proximity"]))
    desc_lines.append("")
    desc_lines.append("Interpretation guidance:")
    desc_lines.append(" - A positive RelativeGroupSize coefficient means that when the focal group is larger than the opponent, the log-odds of the focal group winning increase.")
    desc_lines.append(" - A positive ProximityToFocalHome coefficient means contests closer to the focal group's home (relative to the opponent) increase focal win probability.")
    desc_lines.append(" - A positive interaction means the advantage of being larger is stronger when the contest is closer to the focal's home; a negative interaction means the advantage of size is weaker near the focal home.")
    desc_lines.append("")
    desc_lines.append("Returned 'object' contains numeric estimates (coef, se, z, p, 95% CI) and odds-ratio conversions for each parameter found.")

    return {"object": results, "description": "\n".join(desc_lines)}