def extract_final_answer(model_output):
    """
    Extracts genus-related coefficients from a fitted binomial GLM stored in model_output,
    computes odds ratios and 95% CIs, and returns a summary plus a brief conclusion answering:
    "Do modern humans (Homo sapiens) have higher AMTL than the listed non-human genera?"
    
    Returns:
      {
        "object": pandas.DataFrame  # rows: Pan, Pongo, Papio with coef, se, z, p, OR, OR_CI_lower, OR_CI_upper
        "description": str          # short human-readable interpretation and final yes/no conclusion
      }
    """
    import re
    import numpy as np
    import pandas as pd

    # Validate input
    if not isinstance(model_output, dict):
        raise ValueError("model_output must be a dict as returned by the modeling function.")
    # Prefer robust results if present
    results = model_output.get('glm_results_robust') or model_output.get('glm_results')
    if results is None:
        raise ValueError("model_output does not contain 'glm_results_robust' or 'glm_results'")

    # Extract parameters, standard errors, p-values, confidence intervals
    params = results.params
    bse = results.bse
    pvalues = results.pvalues
    try:
        ci = results.conf_int()  # DataFrame with columns [0,1]
    except Exception:
        # Fallback: attempt to compute approximate CIs using normal approx
        z_crit = 1.96
        ci_lower = params - z_crit * bse
        ci_upper = params + z_crit * bse
        ci = pd.DataFrame({0: ci_lower, 1: ci_upper})

    # Identify genus-related parameter names (those comparing each genus to reference Homo sapiens)
    genus_param_names = [n for n in params.index if 'genus' in n]
    if not genus_param_names:
        # Try alternate pattern: parameter names that contain "T." (treatment level) and look like Pan/Pongo/Papio
        genus_param_names = [n for n in params.index if re.search(r'\bT\.(Pan|Pongo|Papio)\b', n)]
    if not genus_param_names:
        raise ValueError("Could not find genus-related parameters in the model results index.")

    rows = []
    for name in genus_param_names:
        coef = float(params[name])
        se = float(bse[name]) if name in bse.index else np.nan
        z = coef / se if (se != 0 and not np.isnan(se)) else np.nan
        p = float(pvalues[name]) if name in pvalues.index else np.nan
        ci_lower = float(ci.loc[name, 0]) if name in ci.index else np.nan
        ci_upper = float(ci.loc[name, 1]) if name in ci.index else np.nan
        # Extract short genus label (e.g., "Pan") from the parameter name
        m = re.search(r'\bT\.([A-Za-z0-9_ -]+)\]?', name)
        if m:
            genus_label = m.group(1)
        else:
            # fallback: take substring after last dot or bracket
            genus_label = name.split('.')[-1].rstrip(']')
        or_est = float(np.exp(coef))
        or_ci_low = float(np.exp(ci_lower)) if not np.isnan(ci_lower) else np.nan
        or_ci_high = float(np.exp(ci_upper)) if not np.isnan(ci_upper) else np.nan

        rows.append({
            'param_name': name,
            'genus': genus_label,
            'coef_log_odds': coef,
            'se': se,
            'z': z,
            'p_value': p,
            'OR': or_est,
            'OR_CI_lower': or_ci_low,
            'OR_CI_upper': or_ci_high
        })

    summary_df = pd.DataFrame(rows).set_index('genus')[[
        'param_name', 'coef_log_odds', 'se', 'z', 'p_value', 'OR', 'OR_CI_lower', 'OR_CI_upper'
    ]]

    # Determine conclusion about whether Homo sapiens have higher AMTL:
    # For each non-human genus, the coefficient is (genus - Homo). If coef < 0 and significant (p < .05),
    # that means the genus has lower AMTL than Homo (i.e., Homo has higher AMTL).
    sig_mask = (summary_df['p_value'] < 0.05) & (summary_df['coef_log_odds'] < 0)
    genera_with_lower_than_homo = list(summary_df[sig_mask].index)
    genera_tested = list(summary_df.index)

    if len(genera_with_lower_than_homo) == len(genera_tested) and len(genera_tested) > 0:
        conclusion = "Yes — all tested non-human genera ({} ) have significantly LOWER AMTL than Homo sapiens; therefore Homo sapiens have higher AMTL.".format(", ".join(genera_with_lower_than_homo))
    elif len(genera_with_lower_than_homo) > 0:
        conclusion = ("Partially — the following non-human genera have significantly lower AMTL than Homo sapiens: {}. "
                      "Other genera did not differ significantly.".format(", ".join(genera_with_lower_than_homo)))
    else:
        # Check if any genus is significantly higher than Homo
        sig_higher_mask = (summary_df['p_value'] < 0.05) & (summary_df['coef_log_odds'] > 0)
        genera_higher_than_homo = list(summary_df[sig_higher_mask].index)
        if len(genera_higher_than_homo) > 0:
            conclusion = ("No — some non-human genera ({} ) have significantly HIGHER AMTL than Homo sapiens. "
                          "None had significantly lower AMTL than Homo.".format(", ".join(genera_higher_than_homo)))
        else:
            conclusion = ("No — none of the non-human genera differ significantly from Homo sapiens in AMTL after "
                          "accounting for covariates (age, sex-probability, tooth class, population).")

    # Add note about dispersion if provided
    dispersion = model_output.get('dispersion')
    dispersion_note = ""
    if dispersion is not None:
        dispersion_note = f" (model dispersion = {dispersion:.3g})."
        # For binomial GLM, dispersion substantially different from 1 may indicate over/under-dispersion;
        # robust SEs were used where available.

    # Build human readable description
    lines = []
    lines.append("Extracted genus comparisons vs reference (Homo sapiens).")
    lines.append("For each genus the table lists log-odds coef (genus minus Homo), SE, z, p-value, OR and 95% CI for OR.")
    lines.append("Interpretation rule: coef < 0 and p < 0.05 => that non-human genus has lower AMTL than Homo (so Homo higher).")
    lines.append(conclusion + dispersion_note)
    description = " ".join(lines)

    return {
        "object": summary_df,
        "description": description
    }