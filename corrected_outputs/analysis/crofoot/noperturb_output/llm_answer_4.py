def extract_final_answer(model_output):
    """
    Extracts coefficients, standard errors, p-values, 95% CIs, and odds ratios for:
      - LogRelSizeRatio_z (relative group size)
      - FocalCloser (contest location)
      - their interaction (LogRelSizeRatio_z * FocalCloser)
    from a fitted statsmodels Logit or GLM (binomial) results object.

    Returns:
      {
        "object": {
          "nobs": <int or None>,
          "terms": {
            "<term_name>": {
              "coef": float,
              "se": float,
              "p": float,
              "ci_lower": float,
              "ci_upper": float,
              "odds_ratio": float,
              "or_ci_lower": float,
              "or_ci_upper": float,
              "significant": bool
            }, ...
          }
        },
        "description": <string interpretation of the extracted stats>
      }
    """
    import numpy as np

    res = model_output

    # Safely get primary result components
    try:
        params = res.params
    except Exception:
        raise ValueError("Model output has no .params attribute")

    try:
        bse = res.bse
    except Exception:
        bse = None

    try:
        pvalues = res.pvalues
    except Exception:
        pvalues = None

    try:
        ci = res.conf_int()
    except Exception:
        ci = None

    # Helper to find the relevant term keys robustly
    idx = list(params.index)

    def find_term_key(name1, name2=None):
        # Return the first index that contains name1 (and optionally name2)
        for k in idx:
            if name1 in k and (name2 is None or name2 in k):
                return k
        return None

    rel_key = find_term_key('LogRelSizeRatio_z')
    focal_key = find_term_key('FocalCloser')
    # interaction may appear as 'LogRelSizeRatio_z:FocalCloser' or similar
    interaction_key = None
    for k in idx:
        if 'LogRelSizeRatio_z' in k and 'FocalCloser' in k:
            # ensure it's not the main effects (i.e., contains both)
            if ':' in k or ('LogRelSizeRatio_z' in k and 'FocalCloser' in k and k != rel_key and k != focal_key):
                interaction_key = k
                break

    keys_of_interest = {
        'LogRelSizeRatio_z': rel_key,
        'FocalCloser': focal_key,
        'Interaction': interaction_key
    }

    # Function to extract numeric summary for a key
    def summarize_key(k):
        if k is None:
            return None
        coef = float(params[k]) if k in params.index else None
        se = float(bse[k]) if (bse is not None and k in bse.index) else None
        p = float(pvalues[k]) if (pvalues is not None and k in pvalues.index) else None
        if ci is not None and k in ci.index:
            ci_low = float(ci.loc[k, 0])
            ci_high = float(ci.loc[k, 1])
        else:
            ci_low = ci_high = None
        or_val = float(np.exp(coef)) if coef is not None else None
        or_ci_low = float(np.exp(ci_low)) if ci_low is not None else None
        or_ci_high = float(np.exp(ci_high)) if ci_high is not None else None
        significant = (p is not None) and (p < 0.05)
        return {
            "coef": coef,
            "se": se,
            "p": p,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "odds_ratio": or_val,
            "or_ci_lower": or_ci_low,
            "or_ci_upper": or_ci_high,
            "significant": significant
        }

    terms_summary = {}
    for display_name, key in keys_of_interest.items():
        terms_summary[display_name] = summarize_key(key)

    # Model-level info
    try:
        nobs = int(res.nobs)
    except Exception:
        nobs = None

    # Build interpretation text
    def fmt(v):
        return "NA" if v is None else f"{v:.3f}"

    desc_lines = []
    desc_lines.append(f"Model nobs = {nobs}" if nobs is not None else "Model nobs = NA")

    rel = terms_summary['LogRelSizeRatio_z']
    focal = terms_summary['FocalCloser']
    inter = terms_summary['Interaction']

    if rel is not None:
        desc_lines.append(
            f"Relative group size (LogRelSizeRatio_z): coef = {fmt(rel['coef'])}, "
            f"SE = {fmt(rel['se'])}, p = {fmt(rel['p'])}. "
            f"OR = {fmt(rel['odds_ratio'])} (95% CI OR = {fmt(rel['or_ci_lower'])} - {fmt(rel['or_ci_upper'])}). "
            + ("Statistically significant; larger focal groups are more likely to win." if rel['significant']
               else "Not statistically significant.")
        )
    else:
        desc_lines.append("Relative group size term not found in model output.")

    if focal is not None:
        # Determine interpretation direction
        direction = "increases" if (focal['coef'] is not None and focal['coef'] > 0) else "decreases"
        desc_lines.append(
            f"FocalCloser (binary): coef = {fmt(focal['coef'])}, SE = {fmt(focal['se'])}, p = {fmt(focal['p'])}. "
            f"OR = {fmt(focal['odds_ratio'])} (95% CI OR = {fmt(focal['or_ci_lower'])} - {fmt(focal['or_ci_upper'])}). "
            + (f"Statistically significant; being closer to the home-range center {direction} the odds of the focal group winning."
               if focal['significant'] else "Not statistically significant.")
        )
    else:
        desc_lines.append("FocalCloser term not found in model output.")

    if inter is not None:
        # Interpret interaction: sign and significance
        inter_dir = "amplifies" if inter['coef'] is not None and inter['coef'] > 0 else "dampens"
        desc_lines.append(
            f"Interaction (LogRelSizeRatio_z:FocalCloser): coef = {fmt(inter['coef'])}, "
            f"SE = {fmt(inter['se'])}, p = {fmt(inter['p'])}. "
            f"OR = {fmt(inter['odds_ratio'])} (95% CI OR = {fmt(inter['or_ci_lower'])} - {fmt(inter['or_ci_upper'])}). "
            + (f"Statistically significant; the effect of relative group size on winning is {inter_dir} when the focal group is closer to its range center."
               if inter['significant'] else "Not statistically significant; no evidence that location moderates the size effect.")
        )
    else:
        desc_lines.append("Interaction term not found in model output.")

    description = " ".join(desc_lines)

    return {
        "object": {
            "nobs": nobs,
            "terms": terms_summary
        },
        "description": description
    }