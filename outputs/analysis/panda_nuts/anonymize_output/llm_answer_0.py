def extract_final_answer(model_output):
    """
    Extracts fixed-effect estimates, standard errors, z-stats, p-values, and 95% CIs
    from a statsmodels MixedLMResults (or wrapper) object.

    Returns:
      {
        "object": {
          "table": {param: {"coef": float, "se": float, "z": float, "p": float, "ci_lower": float, "ci_upper": float}, ...},
          "focal": {same structure but only for Age, Sex_M, Help_Binary, "Sex_M:Help_Binary", "Age:Help_Binary"}
          "significant": [list of param names with p < 0.05]
        },
        "description": "<brief explanation of what the numbers mean and how to interpret the focal effects>"
      }
    """
    import numpy as np
    import math

    # Try to get fixed-effect estimates and their standard errors in a few ways
    fe = None
    se = None
    # Preferred attributes on MixedLMResults
    if hasattr(model_output, "fe_params") and hasattr(model_output, "bse_fe"):
        fe = model_output.fe_params
        se = model_output.bse_fe
    # Fallback to generic params / bse (may include random effects in some cases)
    elif hasattr(model_output, "params") and hasattr(model_output, "bse"):
        fe = model_output.params
        se = model_output.bse
    else:
        raise AttributeError("Cannot find fixed-effect parameters and standard errors on the provided model_output object.")

    # Ensure both are pandas Series (or similar); convert to numpy arrays for computation
    try:
        index = list(fe.index)
    except Exception:
        # if fe is a numpy array, create index from positional names
        index = [f"param_{i}" for i in range(len(fe))]

    fe_arr = np.asarray(fe, dtype=float)
    se_arr = np.asarray(se, dtype=float)

    # Compute z-statistics and two-sided p-values using normal approximation.
    z_arr = fe_arr / se_arr
    try:
        # try to use scipy if available for better numeric stability
        from scipy.stats import norm
        p_arr = 2.0 * (1.0 - norm.cdf(np.abs(z_arr)))
    except Exception:
        # fallback using math.erf for the normal cdf
        def normal_cdf(x):
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        p_arr = np.array([2.0 * (1.0 - normal_cdf(abs(z))) for z in z_arr])

    # 95% Wald confidence intervals (coef +/- 1.96 * SE)
    ci_lower = fe_arr - 1.96 * se_arr
    ci_upper = fe_arr + 1.96 * se_arr

    # Build results dict (convert numpy floats to native floats)
    table = {}
    for i, name in enumerate(index):
        table[name] = {
            "coef": float(fe_arr[i]),
            "se": float(se_arr[i]),
            "z": float(z_arr[i]),
            "p": float(p_arr[i]),
            "ci_lower": float(ci_lower[i]),
            "ci_upper": float(ci_upper[i])
        }

    # Focal parameters expected from the model formula
    focal_names = ["Age", "Sex_M", "Help_Binary", "Sex_M:Help_Binary", "Age:Help_Binary"]
    # Patsy/statsmodels may use different interaction naming convention (e.g., "Sex_M:Help_Binary").
    # Create focal dict for whichever of these appear in the table.
    focal = {}
    for name in focal_names:
        if name in table:
            focal[name] = table[name]

    # Also include alternative interaction name ordering if present (just in case)
    alt_inter_1 = "Help_Binary:Sex_M"
    alt_inter_2 = "Help_Binary:Age"
    if alt_inter_1 in table and "Sex_M:Help_Binary" not in focal:
        focal["Sex_M:Help_Binary"] = table[alt_inter_1]
    if alt_inter_2 in table and "Age:Help_Binary" not in focal:
        focal["Age:Help_Binary"] = table[alt_inter_2]

    # Significant effects at alpha = 0.05
    significant = [name for name, vals in table.items() if vals["p"] < 0.05]

    description_lines = [
        "This output returns the fixed-effect estimates from the fitted mixed-effects model:",
        "- For each fixed-effect parameter: coefficient (log-scale change in nuts/min), standard error, z-statistic, two-sided p-value, and 95% Wald confidence interval.",
        "- Positive coefficient => higher log(nuts/min); negative => lower log(nuts/min).",
        "- Main-effect coefficients (Age, Sex_M, Help_Binary) are estimated at the reference level of interacting variables (e.g., Help_Binary = 0 for interactions).",
        "- Interaction terms (Sex_M:Help_Binary, Age:Help_Binary) indicate how the effect of Sex or Age differs when Help_Binary = 1 (i.e., when help was received).",
        "- 'focal' contains the subset of parameters most relevant to the research question.",
        "- 'significant' lists parameters with p < 0.05 (two-sided).",
        "",
        "Use these numbers to determine whether Age, Sex, Help, or their interactions have statistically detectable effects on nut-cracking efficiency (on the log(nuts/min) scale)."
    ]
    description = " ".join(description_lines)

    return {"object": {"table": table, "focal": focal, "significant": significant}, "description": description}