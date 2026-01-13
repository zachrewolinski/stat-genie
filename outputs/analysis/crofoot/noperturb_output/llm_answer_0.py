def extract_final_answer(model_output):
    """
    Extract key statistics from a fitted statsmodels GLMResultsWrapper (logistic) object
    for the predictors of interest:
      - size_diff_z (relative group size)
      - location_adv_z (location advantage)
      - their interaction (if present)
    Returns a dict with:
      - "object": a pandas.DataFrame with coef, SE, p-value, 95% CI, odds ratio and OR 95% CI
                 for each term of interest
      - "description": a short plain-language interpretation of what these numbers mean
    """
    import pandas as pd
    import numpy as np

    res = model_output

    # Basic checks
    if not hasattr(res, "params"):
        raise ValueError("model_output does not look like a statsmodels results object (missing .params)")

    params = res.params
    bse = getattr(res, "bse", None)
    pvalues = getattr(res, "pvalues", None)
    try:
        ci = res.conf_int()
    except Exception:
        # If conf_int fails, compute using normal approximation
        if bse is None:
            raise
        z = 1.96
        ci = pd.DataFrame({
            0: params - z * bse,
            1: params + z * bse
        }, index=params.index)

    # Identify parameter names for main effects and interaction more robustly
    def find_param(name):
        if name in params.index:
            return name
        # if exact not found, look for any param containing the token
        matches = [idx for idx in params.index if name in idx]
        return matches[0] if matches else None

    size_name = find_param("size_diff_z")
    loc_name = find_param("location_adv_z")
    # find interaction: parameter name usually contains both tokens, separated by ':'.
    inter_name = None
    for idx in params.index:
        if "size_diff_z" in idx and "location_adv_z" in idx:
            inter_name = idx
            break

    terms = []
    for term_name, pretty in [(size_name, "size_diff_z"),
                              (loc_name, "location_adv_z"),
                              (inter_name, "interaction")]:
        if term_name is None:
            continue
        coef = float(params[term_name])
        se = float(bse[term_name]) if bse is not None else np.nan
        p = float(pvalues[term_name]) if pvalues is not None else np.nan
        ci_low = float(ci.loc[term_name, 0])
        ci_high = float(ci.loc[term_name, 1])
        # odds ratio and CI (exp of log-odds coef)
        or_coef = float(np.exp(coef))
        or_low = float(np.exp(ci_low))
        or_high = float(np.exp(ci_high))

        terms.append({
            "term": pretty,
            "param_name": term_name,
            "coef (log-odds)": coef,
            "SE": se,
            "p_value": p,
            "CI_lower": ci_low,
            "CI_upper": ci_high,
            "odds_ratio": or_coef,
            "OR_CI_lower": or_low,
            "OR_CI_upper": or_high
        })

    if len(terms) == 0:
        raise ValueError("None of the expected terms (size_diff_z, location_adv_z, interaction) were found in the model parameters.")

    df_out = pd.DataFrame(terms).set_index("term")

    # Build a concise description interpreting the key results generically.
    # Note: this text will tell the user what to look for; numeric decisions (significant or not)
    # are based on p-values computed above.
    desc_lines = []
    desc_lines.append("Extracted coefficients (log-odds), standard errors, p-values, 95% CIs, and odds ratios for:")
    desc_lines.append(" - size_diff_z: relative group size (focal - other). Positive coef => being larger increases log-odds of winning.")
    desc_lines.append(" - location_adv_z: contest location advantage (positive => closer to focal home). Positive coef => being nearer increases log-odds of winning.")
    desc_lines.append(" - interaction: whether the effect of size_diff_z depends on location_adv_z.")
    desc_lines.append("")
    desc_lines.append("How to interpret the numbers in the returned table:")
    desc_lines.append(" - If coef > 0 and odds_ratio > 1, that predictor increases the odds of the focal group winning; coef < 0 and OR < 1 decreases odds.")
    desc_lines.append(" - p_value indicates statistical evidence against the null that coef == 0 (commonly using p < 0.05).")
    desc_lines.append(" - A statistically significant interaction (p < 0.05) implies the effect of relative group size on win probability depends on contest location.")
    desc_lines.append("")
    desc_lines.append("The 'object' returned is a pandas DataFrame with the numerical results for these terms.")
    description = "\n".join(desc_lines)

    return {"object": df_out, "description": description}